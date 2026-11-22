//! Atomic for the search (`Dfpn`): its board, its tactics, its guidance and
//! its replays, which made the search before it served other variants. An
//! Atomic search runs as it did then, node for node.
//!
//! Explosions in one move, either way, are settled without a search, and so
//! are checks and double threats after which every defence lets the king be
//! exploded. Forcing attacker moves start cheaper than quiet ones, so they
//! are tried first while the quiet ones stay available: a check starts at its
//! number of evasions that survive, and a threat to explode the king cheaper
//! the more ways it threatens, unless its piece is cheap to blow up, and a
//! plain move whose piece is cheap to blow up starts dearer than the others.
//! Defences unlikely to refute start harder to refute, and ties go to the
//! moves that proved or refuted most so far (the history heuristic). When a
//! defence is proven, its proof is replayed on the other defences (Kawano's
//! simulation), which proves those it also answers. So is a threat: when a
//! defender position not in check is first expanded, a win the attacker would
//! have at once if the defender passed is replayed on every defence, and only
//! those that parry it stay open.
//!
//! Children are scored from their moves' effects; a child position is built
//! only to descend into it. The attacker's plain moves, quiet ones that
//! neither check nor threaten, wait together as one child, the group, scored
//! only when the search first wants one of them (as KomoringHeights delays
//! expansions): most positions never want one. When they are scored, the
//! network evaluates the position after each, and those it likes less than
//! its best few start dearer.
//!
//! Each expanded position is evaluated once, by a hand-crafted evaluation,
//! Atomic-Stockfish's network, or a provability model, and its prospects for
//! the attacker, as a step from behind to ahead, scale its new children's
//! starting numbers (EF-PN, Winands and Schadd): lines that look winnable
//! start cheaper to prove and harder to refute.
//!
//! Proofs that come out the same whenever worked out again are not kept: the
//! attacker exploding the king at once, the defender mated, and a defence
//! after which every move lets the king be exploded (`recomputed`).

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet, TryReserveError};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Condvar, Mutex, PoisonError};

use crate::clock::Instant;

use shakmaty::{Color, uci::UciMove, variant::Atomic};

use super::{Board, Child, Dfpn, GROUP, LAST, Numbers, Store, Value, Variant, lock, mated};
use crate::board::{Effect, Move, Position};
use crate::model::{FEATURES, Model};
use crate::moves::{self, Mover};
use crate::nnue::{Accumulator, Network, RefreshCache, Transformed};
use crate::proofs::Proof;
use crate::search::{Key, Status, status};
use crate::table::Known;
use crate::tablebase::{Tablebase, Verdict};
use crate::variant;

/// The starting proof numbers of attacker moves: a check or a threat to
/// explode the king two ways starts at 1, one threat at `THREAT`, and a quiet
/// move at `QUIET`.
const THREAT: u32 = 4;
const QUIET: u32 = 9;
/// A plain move whose piece the defender can blow up without losing
/// material, after Stockfish's ordering, which ranks a quiet move lower when
/// a cheaper piece attacks where it lands.
const HANGING: u32 = 15;
/// When the plain moves' group opens, the network evaluates the position
/// after each, and all but the best `GRADED` start at `UNGRADED` times their
/// number.
const GRADED: usize = 3;
const UNGRADED: u32 = 3;
/// A quiet move's starting number when the provability model was fitted,
/// which its feature of the cheapest child's number keeps as its scale.
const FITTED_QUIET: f32 = 16.0;
/// A threat from a piece the defender can blow up without losing material.
const WEAK_THREAT: u32 = 8;
/// A check starts at its number of evasions that do not lose at once, up to
/// this many.
const EVASIONS: u32 = 8;
/// A defence unlikely to refute starts at this disproof number, so that
/// likelier ones are tried first.
const UNLIKELY: u32 = 2;
/// How many positions a replay of a sibling's proof may visit.
const SIMULATION: u32 = 16;
/// EF-PN's parameters: how much a parent's evaluation weighs on its new
/// children's starting numbers, and the evaluation that decides a position:
/// five pawns for the network, 500 for the hand-crafted evaluation.
const EF_A: u32 = 10;
const EF_B: u32 = 5;
/// How steeply a model's chance scales the starting numbers: by 10^MAGNITUDE
/// for each unit of its logit, down for proof numbers and up for disproof
/// numbers.
const MAGNITUDE: f32 = 0.25;
const DECISIVE: i32 = 5 * 208;
const HAND_DECISIVE: i32 = 500;

/// What guides the search's starting numbers.
#[derive(Clone, Copy)]
pub enum Guide<'a> {
    None,
    /// The hand-crafted evaluation, cheap.
    Hand,
    /// A network, better but dearer.
    Network(&'a Network),
    /// A provability model, with the network's evaluation among its features
    /// when there is one.
    Model(&'a Model, Option<&'a Network>),
}

impl<'a> Guide<'a> {
    fn network(&self) -> Option<&'a Network> {
        match *self {
            Guide::Network(network) | Guide::Model(_, Some(network)) => Some(network),
            _ => None,
        }
    }

    /// Whether the network grades the plain moves, as a model may say it
    /// does not.
    fn grades(&self) -> bool {
        match *self {
            Guide::Network(_) => true,
            Guide::Model(model, Some(_)) => model.grades(),
            _ => false,
        }
    }
}

/// What each thread of an Atomic search keeps: the guide and the tables it
/// reads, the network's layers along its line, its history and its
/// scratch.
pub struct State<'a> {
    tablebase: Option<&'a Tablebase>,
    guide: Guide<'a>,
    /// The network's first layer at each depth of the line, each computed
    /// from its parent's.
    frames: Vec<Accumulator>,
    /// The first layer at each king square, for the network's refreshes.
    refresh_cache: Option<RefreshCache>,
    /// A plain move's first layer while the moves are graded.
    scratch_frame: Box<Accumulator>,
    scratch_transformed: Box<Transformed>,
    /// The expanded position's first layer as the dense layers and the model
    /// read it.
    transformed: Box<Transformed>,
    /// How often each attacker move, by piece and destination, proved a
    /// position, and how often each defence refuted one.
    attack_history: Vec<u32>,
    defense_history: Vec<u32>,
    /// The legal moves being scored, the expanded position's or its passed
    /// position's, with their effects and whether the filter may hold their
    /// children.
    scratch: Vec<(Move, Effect, bool)>,
    /// The line of a replay (`simulate`), kept between replays.
    line: Vec<u64>,
}

impl Board for Position {
    fn hash(&self) -> u64 {
        Position::hash(self)
    }

    fn halfmoves(&self) -> u32 {
        Position::halfmoves(self)
    }

    fn turn(&self) -> Color {
        Position::turn(self)
    }

    fn play(&self, m: Move) -> Position {
        Position::play(self, m)
    }

    fn pieces(&self) -> u32 {
        self.occupied().count() as u32
    }

    fn uci(&self, m: Move) -> String {
        UciMove::from_standard(self.to_shakmaty(m)).to_string()
    }
}

impl Variant for variant::Atomic {
    const EPSILON: u32 = 8;

    type Root = Atomic;
    type Board = Position;
    type Shared<'a> = (Option<&'a Tablebase>, Guide<'a>);
    type State<'a> = State<'a>;

    fn state((tablebase, guide): Self::Shared<'_>) -> State<'_> {
        State {
            tablebase,
            guide,
            frames: Vec::new(),
            refresh_cache: None,
            scratch_frame: Box::default(),
            scratch_transformed: Box::default(),
            transformed: Box::default(),
            attack_history: vec![0; 7 * 64],
            defense_history: vec![0; 7 * 64],
            scratch: Vec::new(),
            line: Vec::new(),
        }
    }

    fn shared<'a>(state: &Self::State<'a>) -> Self::Shared<'a> {
        (state.tablebase, state.guide)
    }

    fn clear(state: &mut Self::State<'_>) {
        state.attack_history.fill(0);
        state.defense_history.fill(0);
    }

    fn board(root: &Atomic) -> Position {
        Position::from_atomic(root)
    }

    fn status(root: &Atomic, attacker: Color) -> Status {
        status(root, attacker)
    }

    fn children(
        search: &mut Dfpn<'_, Self>,
        pos: &Position,
        attacking: bool,
        remaining: u32,
        children: &mut Vec<Child>,
    ) -> Option<f32> {
        search.children(pos, attacking, remaining, children)
    }

    fn scored(
        search: &mut Dfpn<'_, Self>,
        pos: &Position,
        attacking: bool,
        fresh: bool,
        remaining: u32,
        children: &mut [Child],
    ) {
        if !attacking && fresh && !children.is_empty() && !pos.in_check() {
            search.replay_threat(pos, remaining, children);
        }
    }

    /// A mate is settled before the search reaches it, except at the root;
    /// anything else is a stalemate, a draw.
    fn no_moves(search: &mut Dfpn<'_, Self>, pos: &Position) -> Value {
        if pos.in_check() {
            search.mated(pos).unwrap_or(Value::DRAWN)
        } else {
            Value::DRAWN
        }
    }

    fn open_group(
        search: &mut Dfpn<'_, Self>,
        pos: &Position,
        remaining: u32,
        children: &mut Vec<Child>,
        step: Option<f32>,
    ) {
        search.open_group(pos, remaining, children, step);
    }

    /// A defence just proven may prove its siblings too: its proof is
    /// replayed on each defence still open.
    fn defence_proven(
        search: &mut Dfpn<'_, Self>,
        pos: &Position,
        next: &Position,
        best: usize,
        remaining: u32,
        children: &mut [Child],
    ) {
        for (index, sibling) in children.iter_mut().enumerate() {
            let open = sibling.value.numbers.proof != 0 && sibling.value.numbers.disproof != 0;
            if index == best || sibling.simulated || !open {
                continue;
            }
            sibling.simulated = true;
            let position = pos.play(sibling.m);
            let mut left = SIMULATION;
            let mut line = std::mem::take(&mut search.var.line);
            let reached = search.simulate(&position, next, remaining, &mut left, &mut line);
            search.var.line = line;
            if reached.is_some_and(|(length, _)| length <= remaining) {
                debug_assert!(
                    search.full
                        || search.proof_at(sibling.hash, sibling.counter).is_some()
                        || search.recomputed(&position).is_some()
                );
                sibling.value = Value::WON;
            }
        }
    }

    fn refuted(search: &mut Dfpn<'_, Self>, pos: &Position, m: Move) {
        let index = history_index(pos, m);
        let history = &mut search.var.defense_history;
        history[index] = history[index].saturating_add(1);
    }

    fn record_proof(
        search: &mut Dfpn<'_, Self>,
        pos: &Position,
        hash: u64,
        children: &[Child],
        attacking: bool,
    ) {
        search.record_proof(pos, hash, children, attacking);
    }

    fn recomputed(search: &Dfpn<'_, Self>, pos: &Position) -> Option<Proof> {
        search.recomputed(pos)
    }

    fn mates_with(search: &Dfpn<'_, Self>, pos: &Position, m: Move) -> bool {
        search.mates_with(pos, m)
    }
}

impl<'a> Dfpn<'a, variant::Atomic> {
    /// A search of `root` with a table and proofs of these megabytes.
    pub fn new(
        root: &Atomic,
        tablebase: Option<&'a Tablebase>,
        guide: Guide<'a>,
        deadline: Instant,
        memory: (usize, usize),
    ) -> Dfpn<'a> {
        Dfpn::try_new(root, tablebase, guide, deadline, memory)
            .unwrap_or_else(|error| panic!("the search's memory: {error}"))
    }

    /// As `new`, unless the memory cannot be had, as a browser may refuse it.
    pub fn try_new(
        root: &Atomic,
        tablebase: Option<&'a Tablebase>,
        guide: Guide<'a>,
        deadline: Instant,
        memory: (usize, usize),
    ) -> Result<Dfpn<'a>, TryReserveError> {
        Dfpn::try_with_threads(root, tablebase, guide, deadline, memory, 1)
    }

    /// As `try_new`, for `threads` threads that search together, sharing the
    /// table and the proofs: this search is one of them, and starts the
    /// others for each `solve`, each with a seed of its own.
    pub fn try_with_threads(
        root: &Atomic,
        tablebase: Option<&'a Tablebase>,
        guide: Guide<'a>,
        deadline: Instant,
        memory: (usize, usize),
        threads: usize,
    ) -> Result<Dfpn<'a>, TryReserveError> {
        let attacker = shakmaty::Position::turn(root);
        Dfpn::with_threads(attacker, (tablebase, guide), deadline, memory, threads)
    }

    /// The winning move for an attacker position, for reading the proof back:
    /// once the proofs have dropped their mates in one, a mate in one found
    /// again; and once they have dropped their short proofs, the move of a
    /// win of so few moves worked out again (`short_proof`), before any proof
    /// kept is read. A position may keep a longer proof than the short one
    /// its parents counted on, and a walk that followed the longer one could
    /// come back to a position on its line. Of the proofs kept, the shortest
    /// that holds at the counter, or with `furthest` the one that holds at
    /// the highest counter (`kept_proof`).
    pub fn winning_move(&self, pos: &Atomic, key: &Key, furthest: bool) -> Option<shakmaty::Move> {
        let pos = Position::from_atomic(pos);
        let m = match self.short_proof(&pos).and_then(|proof| proof.m) {
            Some(m) => m,
            None => match self.kept_proof(key.0, key.1, furthest) {
                Some(proof) => proof.m?,
                None => self.dropped_mate(&pos)?,
            },
        };
        Some(pos.to_shakmaty(m))
    }

    /// Calls `write` with `winning_move` for a proof's walk, worked out ahead
    /// by `threads - 1` other threads once the proofs have dropped their
    /// short ones, which a walk works out again at every attacker position:
    /// from each defender position the walk reaches, they take the attacker
    /// positions its replies reach, work out each one's move as the walk
    /// would, and go on below it, the most recent first, as a walk goes depth
    /// first. The walk takes a move they found, waits for one they are
    /// working out, or works one out itself, so the moves and the proof are
    /// the same whatever the threads do. Without drops, or on one thread,
    /// `write` is called with `winning_move` alone.
    pub fn with_moves_ahead<R>(
        &self,
        threads: usize,
        write: impl FnOnce(&mut dyn FnMut(&Atomic, &Key, bool) -> Option<shakmaty::Move>) -> R,
    ) -> R {
        if threads <= 1 || self.short_drops() == 0 {
            return write(&mut |pos, key, furthest| self.winning_move(pos, key, furthest));
        }
        let ahead = Ahead::default();
        let moves = WalkMoves {
            attacker: self.attacker,
            tables: self.var.tablebase.filter(|_| !self.exact),
            store: &self.store,
            filtering: self.filtering,
        };
        std::thread::scope(|scope| {
            // The threads ahead end with the walk, also when it fails.
            let _stop = StopAhead(&ahead);
            for _ in 1..threads {
                // A thread that cannot start leaves its share to the walk.
                let _ = std::thread::Builder::new().spawn_scoped(scope, || ahead.work(&moves));
            }
            write(&mut |pos, key, furthest| {
                // The walk again with the furthest proofs, after a first
                // that gave a position two moves: none of the moves worked
                // out ahead, chosen the shortest way, may serve it.
                if furthest {
                    ahead.stop();
                    return self.winning_move(pos, key, true);
                }
                let pos = Position::from_atomic(pos);
                let m = match ahead.take(key) {
                    Some(m) => m,
                    None => {
                        let m = moves.walk_move(&pos);
                        if let Some(m) = m {
                            ahead.give(pos.play(m));
                        }
                        m
                    }
                };
                Some(pos.to_shakmaty(m?))
            })
        })
    }

    /// `winning_move` on the board a proof's walk plays (`format::walk`),
    /// with no position or move to convert.
    pub fn proof_move(&self, pos: &Position, furthest: bool) -> Option<Move> {
        if let Some(m) = self.short_proof(pos).and_then(|proof| proof.m) {
            return Some(m);
        }
        match self.kept_proof(pos.hash(), pos.halfmoves(), furthest) {
            Some(proof) => proof.m,
            None => self.dropped_mate(pos),
        }
    }

    /// Scores a position's moves into `children`, with `remaining` attacker
    /// moves left, and returns the step its evaluation scaled their starting
    /// numbers by.
    pub(super) fn children(
        &mut self,
        pos: &Position,
        attacking: bool,
        remaining: u32,
        children: &mut Vec<Child>,
    ) -> Option<f32> {
        let candidates = pos.pseudo_legal();
        let mover = Mover::new(pos);
        // The attacker's plain moves wait together, except where a pawn move
        // could reach the tablebases.
        let group = attacking
            && self
                .var
                .tablebase
                .is_none_or(|tables| pos.occupied().count() > tables.max_pieces() + 1);
        let mut plain = 0;
        let left = remaining.saturating_sub(u32::from(attacking));
        if attacking && left == 0 && !self.record {
            self.last_children(pos, &mover, children);
            return None;
        }
        // First every legal move's effect and its child's bit in the filter,
        // starting the table reads of the children it may hold: the reads do
        // not wait on one another, so their latencies overlap.
        let mut legal = std::mem::take(&mut self.var.scratch);
        legal.clear();
        let filtering = self.filtering;
        for &m in candidates.as_slice() {
            if group && mover.plain(m) {
                plain += u32::from(mover.legal(m));
                continue;
            }
            if mover.legal(m) {
                let effect = mover.full_effect(m);
                self.prefetch_child(effect.hash, left, filtering);
                legal.push((m, effect, !filtering));
            }
        }
        // Then the filter's bits, their loads under way, and the table reads
        // of the children it may hold.
        if filtering {
            self.read_filter(&mut legal, left);
        }
        children.clear();
        self.mark_repetitions(pos.halfmoves());
        for &(m, effect, seen) in &legal {
            children.push(self.evaluate(pos, &mover, m, &effect, seen, attacking, left, EVASIONS));
        }
        self.var.scratch = legal;
        if plain > 0 {
            let key = self.key(pos.hash(), remaining) ^ GROUP;
            let counter = pos.halfmoves();
            let known = self
                .lookup(key, counter, |_| false)
                .filter(|value| value.numbers.disproof != 0);
            children.push(Child {
                m: Move::default(),
                hash: key,
                counter,
                value: known.unwrap_or(Value::open(Numbers {
                    proof: QUIET,
                    disproof: plain,
                })),
                exploded: false,
                winning: false,
                score: 0,
                simulated: false,
                fresh: known.is_none(),
                group: true,
            });
        }
        let step = self.step(pos, attacking, children, plain);
        scale(children, attacking, step);
        step
    }

    /// `children` for an attacker position with one move left, where only a
    /// move that wins at once counts: a plain move neither captures nor
    /// checks, so it cannot and is left out, unless a pawn move may reach a
    /// tablebase win; every other child is decided where it is scored, so
    /// nothing needs the position's evaluation. Apart from `children`, so
    /// that searches without a limit run the same code.
    #[inline(never)]
    fn last_children(&mut self, pos: &Position, mover: &Mover, children: &mut Vec<Child>) {
        let mut legal = std::mem::take(&mut self.var.scratch);
        legal.clear();
        let filtering = self.filtering;
        let tables = !self.exact
            && self
                .var
                .tablebase
                .is_some_and(|tables| pos.occupied().count() <= tables.max_pieces());
        for &m in pos.pseudo_legal().as_slice() {
            let kept =
                !mover.plain(m) || tables && pos.role_at(m.from()) == Some(shakmaty::Role::Pawn);
            if !kept || !mover.legal(m) {
                continue;
            }
            let effect = mover.full_effect(m);
            self.prefetch_child(effect.hash, 0, filtering);
            legal.push((m, effect, !filtering));
        }
        if filtering {
            self.read_filter(&mut legal, 0);
        }
        children.clear();
        self.mark_repetitions(pos.halfmoves());
        for &(m, effect, seen) in &legal {
            children.push(self.evaluate(pos, mover, m, &effect, seen, true, 0, EVASIONS));
        }
        self.var.scratch = legal;
    }

    /// The expanded position's evaluation for the attacker, squashed to a
    /// step from 0, the defender clearly ahead, to 2, the attacker clearly
    /// ahead, when an evaluation guides the search.
    fn step(
        &mut self,
        pos: &Position,
        attacking: bool,
        children: &[Child],
        plain: u32,
    ) -> Option<f32> {
        let value = self
            .var
            .guide
            .network()
            .map(|network| self.network_value(network, pos));
        if self.record {
            // The first position expanded: the root.
            self.recorded = Some(self.features(pos, attacking, children, plain, value));
            self.record = false;
        }
        let step = match self.var.guide {
            Guide::None => None,
            Guide::Hand => {
                let value = crate::eval::evaluate(pos);
                Some(self.band(pos, value, HAND_DECISIVE))
            }
            Guide::Network(_) => value.map(|value| self.band(pos, value, DECISIVE)),
            Guide::Model(model, network) => {
                let features = self.features(pos, attacking, children, plain, value);
                // The position's first layer, which `network_value` just
                // transformed.
                let layer = network
                    .filter(|_| model.reads_layer())
                    .map(|_| &*self.var.transformed);
                let logit = |chance: f32| {
                    let chance = chance.clamp(1e-6, 1.0 - 1e-6);
                    (chance / (1.0 - chance)).ln()
                };
                // Centred on the share proven among the model's own positions,
                // so that an average position takes the middle step.
                let chance = model.predict(attacking, &features, layer);
                let centred = logit(chance) - logit(model.prior(attacking));
                Some(2.0 / (1.0 + (-centred).exp()))
            }
        };
        step.map(|step| self.perturb(pos, step))
    }

    /// The step moved by up to 1% either way when a seed is set, by a hash of
    /// the position and the seed, so that every expansion of a position
    /// agrees. A model's step moves through its odds: moved itself, it could
    /// pass 2, where `scale` switches to EF-PN's formula.
    fn perturb(&self, pos: &Position, step: f32) -> f32 {
        if self.seed == 0 {
            return step;
        }
        let mixed = (pos.hash() ^ self.seed).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let unit = (mixed >> 40) as f32 / (1 << 23) as f32 - 1.0;
        if matches!(self.var.guide, Guide::Model(..)) {
            perturb_model_step(step, unit)
        } else {
            step * (1.0 + 0.01 * unit)
        }
    }

    /// The network's evaluation for the side to move, with its first layer
    /// transformed into `transformed`, where the model reads it. The line's
    /// first layers follow the search down: this depth's from its parent's,
    /// which the parent computed when it expanded.
    fn network_value(&mut self, network: &Network, pos: &Position) -> i32 {
        let level = self.path.len();
        if self.var.frames.len() <= level {
            self.var.frames.resize_with(level + 1, Accumulator::default);
        }
        let (parents, frames) = self.var.frames.split_at_mut(level);
        let frame = &mut frames[0];
        match parents.last() {
            Some(parent) => {
                let cache = self
                    .var
                    .refresh_cache
                    .get_or_insert_with(|| RefreshCache::new(network));
                network.update(parent, pos, frame, cache);
            }
            None => network.refresh(pos, frame),
        }
        network.transform(frame, pos, &mut self.var.transformed);
        let value = network.output_from(&self.var.transformed, frame, pos);
        debug_assert_eq!(value, network.evaluate(pos));
        debug_assert!(self.var.transformed.values == network.evaluate_layer(pos));
        value
    }

    /// A position's features for the provability model, from the attacker's
    /// side, given its children's starting numbers before any scaling and
    /// the network's evaluation for the side to move, if any.
    fn features(
        &self,
        pos: &Position,
        attacking: bool,
        children: &[Child],
        plain: u32,
        value: Option<i32>,
    ) -> [f32; FEATURES] {
        let for_attacker = |value: i32| if attacking { value } else { -value };
        let clip = |value: f32| value.clamp(-3.0, 3.0);
        let net = value.map(for_attacker);
        let material =
            crate::eval::material(pos, self.attacker) - crate::eval::material(pos, !self.attacker);
        let kings_touch = match (pos.king(self.attacker), pos.king(!self.attacker)) {
            (Some(ours), Some(theirs)) => shakmaty::attacks::king_attacks(ours).contains(theirs),
            _ => false,
        };
        // One pass over the children: those open, the moves, and among the
        // fresh moves the forcing, the strong, the cheapest and the good.
        let (mut open, mut moves, mut losing) = (0, plain as usize, 0);
        let (mut forcing, mut strong, mut good, mut cheapest) = (0, 0, 0, None::<u32>);
        for child in children.iter() {
            let numbers = child.value.numbers;
            open += usize::from(numbers.proof != 0 && numbers.disproof != 0);
            losing += usize::from(child.winning);
            if child.group {
                continue;
            }
            moves += 1;
            if child.fresh {
                forcing += usize::from(numbers.proof <= 8);
                strong += usize::from(numbers.proof == 1);
                good += usize::from(numbers.disproof == 1);
                cheapest = Some(cheapest.map_or(numbers.proof, |known| known.min(numbers.proof)));
            }
        }
        let (forcing, strong, cheapest, good, losing) = if attacking {
            (forcing, strong, cheapest.unwrap_or(QUIET), 0, 0.0)
        } else {
            (
                0,
                0,
                QUIET,
                good,
                losing as f32 / children.len().max(1) as f32,
            )
        };
        let ln = |count: usize| (1.0 + count as f32).ln();
        [
            net.map_or(0.0, |net| clip(net as f32 / 2080.0)),
            net.map_or(0.0, |net| f32::from(net >= DECISIVE)),
            net.map_or(0.0, |net| f32::from(net <= -DECISIVE)),
            clip(material as f32 / 3000.0),
            pos.occupied().count() as f32 / 32.0,
            f32::from(kings_touch),
            f32::from(pos.in_check()),
            ln(moves) / 4.0,
            ln(forcing) / 3.0,
            strong.min(4) as f32 / 4.0,
            if cheapest >= QUIET {
                1.0
            } else {
                cheapest as f32 / FITTED_QUIET
            },
            if attacking { 0.0 } else { ln(open) / 3.0 },
            ln(good) / 3.0,
            losing,
            pos.halfmoves() as f32 / 100.0,
        ]
    }

    /// An evaluation for the side to move, squashed for the attacker to 0
    /// when the defender is clearly ahead, 1 when neither is, or 2.
    fn band(&self, pos: &Position, value: i32, decisive: i32) -> f32 {
        let value = if pos.turn() == self.attacker {
            value
        } else {
            -value
        };
        if value >= decisive {
            2.0
        } else if value <= -decisive {
            0.0
        } else {
            1.0
        }
    }

    /// Scores the plain moves a group held, as `children` would have.
    fn open_group(
        &mut self,
        pos: &Position,
        remaining: u32,
        children: &mut Vec<Child>,
        step: Option<f32>,
    ) {
        let first = children.len();
        let mover = Mover::new(pos);
        let left = remaining.saturating_sub(1);
        // The moves first, their filter words' loads started, as in `children`.
        let mut plain = std::mem::take(&mut self.var.scratch);
        plain.clear();
        let filtering = self.filtering;
        for &m in pos.pseudo_legal().as_slice() {
            if mover.plain(m) && mover.legal(m) {
                let effect = mover.full_effect(m);
                self.prefetch_child(effect.hash, left, filtering);
                plain.push((m, effect, !filtering));
            }
        }
        if filtering {
            self.read_filter(&mut plain, left);
        }
        self.mark_repetitions(pos.halfmoves());
        for &(m, effect, seen) in &plain {
            children.push(self.evaluate(pos, &mover, m, &effect, seen, true, left, EVASIONS));
        }
        self.var.scratch = plain;
        scale(&mut children[first..], true, step);
        self.grade(pos, &mut children[first..]);
    }

    /// Raises the starting numbers of the plain moves the network likes less
    /// than its best `GRADED`, by its evaluation after each.
    fn grade(&mut self, pos: &Position, plain: &mut [Child]) {
        if !self.var.guide.grades() {
            return;
        }
        let Some(network) = self.var.guide.network() else {
            return;
        };
        // The expanded position's first layer, which `network_value` computed
        // when this visit expanded it.
        let level = self.path.len();
        if self.var.frames.len() <= level || plain.len() <= GRADED {
            return;
        }
        let pieces = pos.occupied().count() as u32;
        // The best GRADED values and their moves, earlier moves first among
        // equal values.
        let mut best: [Option<(i32, usize)>; GRADED] = [None; GRADED];
        for (index, child) in plain.iter().enumerate() {
            if !child.fresh {
                continue;
            }
            // A plain move is quiet and not a king's: its first layer follows
            // from the expanded position's without building the position.
            let (m, them) = (child.m, !pos.turn());
            network.update_quiet(
                &self.var.frames[level],
                m.from(),
                m.to(),
                &mut self.var.scratch_frame,
            );
            network.transform_for(
                &self.var.scratch_frame,
                them,
                &mut self.var.scratch_transformed,
            );
            // The defender moves next: the attacker's view is the negation.
            let value = -network.output_for(
                &self.var.scratch_transformed,
                &self.var.scratch_frame,
                pieces,
                them,
            );
            debug_assert_eq!(value, -network.evaluate(&pos.play(m)));
            // Moves come in order, so only a higher value displaces a kept
            // one.
            if let Some(place) = best
                .iter()
                .position(|kept| kept.is_none_or(|(kept, _)| value > kept))
            {
                best.copy_within(place..GRADED - 1, place + 1);
                best[place] = Some((value, index));
            }
        }
        for (index, child) in plain.iter_mut().enumerate() {
            if child.fresh && !best.iter().flatten().any(|&(_, kept)| kept == index) {
                let numbers = &mut child.value.numbers;
                numbers.proof = (numbers.proof * UNGRADED).min(1 << 24);
            }
        }
    }

    /// What is known about a child before searching it, from its move's
    /// effect, with `remaining` attacker moves left there; the child position
    /// is built only for a check, a tablebase probe, or en passant.
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &mut self,
        pos: &Position,
        mover: &Mover,
        m: Move,
        effect: &Effect,
        seen: bool,
        attacking: bool,
        remaining: u32,
        evasions: u32,
    ) -> Child {
        let counter = effect.counter;
        let hash = effect.hash;
        let key = self.key(hash, remaining);
        let history = if attacking {
            &self.var.attack_history
        } else {
            &self.var.defense_history
        };
        let score = history[history_index(pos, m)];
        let child = |value: Value, exploded: bool| Child {
            m,
            hash,
            counter,
            value,
            exploded,
            winning: false,
            score,
            simulated: false,
            fresh: false,
            group: false,
        };
        // Only a capture explodes a king, never the mover's.
        if mover.explodes_them(effect) {
            return if attacking {
                child(Value::WON, true)
            } else {
                child(Value::DRAWN, false)
            };
        }
        // A move after which the defender explodes the attacker's king at
        // once loses on any line.
        if attacking && mover.exposes(m, effect) {
            return child(Value::DRAWN, false);
        }
        if let Some(depth) = self.repeated(hash, counter) {
            let repeated = Value {
                cycle: depth,
                ..Value::DRAWN
            };
            return child(repeated, false);
        }
        // A defence after which the attacker explodes the defender's king at
        // once is won without a search, as a mate in one is, unless it
        // reaches the 50-move limit.
        if !attacking && counter <= LAST && mover.exposes(m, effect) {
            return Child {
                winning: true,
                ..child(Value::WON, false)
            };
        }
        // Most children are new: one bit of the filter says so for both tables.
        if seen && self.proof_within(hash, counter, remaining).is_some() {
            return child(Value::WON, false);
        }
        // The tables decide a child reached by a capture or pawn move, with
        // any moves left.
        if counter == 0
            && self.var.tablebase.is_some()
            && let Some(value) = self.tablebase_value(pos, m, effect, key, seen)
        {
            return child(value, false);
        }
        // Out of attacker moves, only a mate still wins.
        if remaining == 0 {
            let mated = mover.gives_check(m, effect)
                && self
                    .mated(&pos.play(m))
                    .is_some_and(|value| value.numbers.proof == 0);
            return child(if mated { Value::WON } else { Value::DRAWN }, false);
        }
        let check = mover.gives_check(m, effect);
        if counter > LAST {
            // Past the 50-move limit only a mate still counts.
            let value = match check.then(|| self.mated(&pos.play(m))).flatten() {
                Some(value) => self.settle(key, counter, value, 0),
                None => Value {
                    fifty: true,
                    ..Value::DRAWN
                },
            };
            return child(value, false);
        }
        // A mate in one the table notes is checked on the child's position,
        // made only then.
        let known = if seen {
            self.lookup(key, counter, |mate| self.mates_with(&pos.play(m), mate))
        } else {
            None
        };
        if let Some(value) = known {
            return child(value, false);
        }
        // Met for the first time: settle mates and positions the attacker
        // cannot win, and otherwise start from how forcing the move was.
        let mut check_proof = 1;
        if check {
            let next = pos.play(m);
            if attacking {
                // A check is as hard to prove as it has evasions that
                // survive, and with none it wins in two moves; with no
                // evasion at all it mates.
                let (any, safe) = moves::replies(&next, evasions);
                if !any || safe == 0 && self.exposed(&next).is_some() {
                    return child(Value::WON, false);
                }
                check_proof = safe.max(1);
            } else if let Some(value) = self.mated(&next) {
                return child(self.settle(key, counter, value, 0), false);
            }
        }
        // Only a capture or a promotion changes the material.
        if (effect.capture || m.promotion().is_some())
            && mover.might_end_material(self.attacker, effect)
            && moves::cannot_win(&pos.play(m), self.attacker)
        {
            return child(self.settle(key, counter, Value::DRAWN, 0), false);
        }
        if !attacking {
            // A defence likely to refute is tried first: a check, a blast
            // that removes an attacking piece, or kings that can touch.
            let attackers = pos.by_color(self.attacker) & !pos.by_role(shakmaty::Role::Pawn);
            let good = check
                || (effect.capture && (attackers & !effect.occupied).any())
                || mover.connectable(m, effect);
            let disproof = if good { 1 } else { UNLIKELY };
            return Child {
                fresh: true,
                ..child(Value::open(Numbers { proof: 1, disproof }), false)
            };
        }
        let threats = if check { 0 } else { mover.threats(m, effect) };
        // So does a double threat no defence parries.
        if threats >= 2 && self.exposed(&pos.play(m)).is_some() {
            return child(Value::WON, false);
        }
        let proof = if check {
            check_proof
        } else {
            match threats {
                0 if mover.plain(m) && mover.blasted_cheaply(m, effect) => HANGING,
                0 => QUIET,
                1 if mover.blasted_cheaply(m, effect) => WEAK_THREAT,
                1 => THREAT,
                _ => 1,
            }
        };
        Child {
            fresh: true,
            ..child(Value::open(Numbers { proof, disproof: 1 }), false)
        }
    }

    /// A child reached by a capture or pawn move as the tablebases score it,
    /// where they hold it: won, its proof kept, with any moves left, unless
    /// the search is exact; otherwise refuted, as the table then keeps it
    /// for these moves left, so that the tables are read once.
    fn tablebase_value(
        &mut self,
        pos: &Position,
        m: Move,
        effect: &Effect,
        key: u64,
        seen: bool,
    ) -> Option<Value> {
        let tables = self.var.tablebase?;
        if effect.occupied.count() > tables.max_pieces() {
            return None;
        }
        if seen
            && let Some(value) = self.lookup(key, 0, |_| false)
            && value.numbers.disproof == 0
        {
            return Some(value);
        }
        let verdict = tables.probe(&pos.play(m).to_atomic())?;
        let attacker_wins = if pos.turn() == self.attacker {
            verdict == Verdict::Loss
        } else {
            verdict == Verdict::Win
        };
        if !attacker_wins {
            self.store(key, 0, Value::DRAWN, 0);
            return Some(Value::DRAWN);
        }
        if self.exact {
            return None;
        }
        #[cfg(test)]
        self.check_short(&pos.play(m), Proof::TABLEBASE);
        self.add_proof(effect.hash, Proof::TABLEBASE);
        Some(Value::WON)
    }

    /// Tries to prove `pos`, attacker to move, by replaying the proof of
    /// `guide`, a similar position (Kawano's simulation): the attacker plays
    /// the guide's winning moves, and every defence must be answered by a
    /// known proof, an explosion at once, or the same defence in the guide.
    /// The proofs it finds are recorded as the search records its own, so it
    /// is exact; it gives up when the budget runs out. Under a limit, a proof
    /// must take at most `remaining` attacker moves: a known one that takes
    /// more leaves the replay to look for a shorter one.
    fn simulate(
        &mut self,
        pos: &Position,
        guide: &Position,
        remaining: u32,
        budget: &mut u32,
        line: &mut Vec<u64>,
    ) -> Option<(u32, u32)> {
        let counter = pos.halfmoves();
        if let Some(proof) = self.proof_at(pos.hash(), counter)
            && proof.length <= remaining
        {
            return Some((proof.length, proof.limit));
        }
        if counter > LAST || *budget == 0 || remaining == 0 || self.on_line(pos.hash(), line) {
            return None;
        }
        *budget -= 1;
        self.nodes += 1;
        let m = match self.proof_at(guide.hash(), guide.halfmoves()) {
            Some(proof) => proof.m?,
            None => self.explosion(guide).or_else(|| self.dropped_mate(guide))?,
        };
        if !pos.is_pseudo_legal(m) || !pos.legal(m) {
            return None;
        }
        let after = pos.play(m);
        line.push(pos.hash());
        let reached = self.simulate_defences(&after, &guide.play(m), remaining - 1, budget, line);
        line.pop();
        let (length, limit) = reached?;
        let mates = mated(length, limit);
        let limit = if after.halfmoves() == 0 {
            LAST
        } else {
            limit.saturating_sub(1).min(LAST)
        };
        let proof = Proof {
            m: Some(m),
            length: length + 1,
            limit,
        };
        // A capture that explodes the king at once is not kept.
        if after.king(!self.attacker).is_some() {
            self.keep_attacker(pos, proof, mates);
        }
        Some((proof.length, proof.limit))
    }

    /// The defender's side of `simulate`, with `remaining` attacker moves
    /// left.
    fn simulate_defences(
        &mut self,
        pos: &Position,
        guide: &Position,
        remaining: u32,
        budget: &mut u32,
        line: &mut Vec<u64>,
    ) -> Option<(u32, u32)> {
        if pos.king(!self.attacker).is_none() {
            return Some((0, u32::MAX));
        }
        let counter = pos.halfmoves();
        if let Some(proof) = self.proof_at(pos.hash(), counter)
            && proof.length <= remaining
        {
            return Some((proof.length, proof.limit));
        }
        if counter > LAST || self.on_line(pos.hash(), line) {
            return None;
        }
        let mover = Mover::new(pos);
        let mut length = 0;
        let mut limit = LAST;
        let mut defences = 0;
        line.push(pos.hash());
        let mut answered = true;
        // Whether every defence lets the king be exploded at once, which
        // makes the proof one `recomputed` finds again.
        let mut exposed = true;
        for &reply in pos.pseudo_legal().as_slice() {
            if !mover.legal(reply) {
                continue;
            }
            defences += 1;
            let effect = mover.effect(reply);
            if mover.explodes_them(&effect) {
                answered = false;
                break;
            }
            let next = pos.play(reply);
            let exposing = next.halfmoves() <= LAST && mover.exposes(reply, &effect);
            exposed &= exposing;
            let reached = if exposing {
                moves::exploding_capture(&next, self.attacker)
                    .filter(|_| remaining > 0)
                    .map(|_| (1, LAST))
            } else if let Some(proof) = self
                .proof_at(next.hash(), next.halfmoves())
                .filter(|proof| proof.length <= remaining)
            {
                Some((proof.length, proof.limit))
            } else if guide.is_pseudo_legal(reply) && guide.legal(reply) {
                self.simulate(&next, &guide.play(reply), remaining, budget, line)
            } else {
                None
            };
            let Some((reply_length, reply_limit)) = reached else {
                answered = false;
                break;
            };
            length = length.max(reply_length);
            limit = limit.min(if next.halfmoves() == 0 {
                LAST
            } else {
                reply_limit.saturating_sub(1).min(LAST)
            });
        }
        line.pop();
        // No defence at all is a mate if in check, else a stalemate.
        if !answered || (defences == 0 && !pos.in_check()) {
            return None;
        }
        let proof = if defences == 0 {
            Proof::FINISHED
        } else {
            Proof {
                m: None,
                length,
                limit,
            }
        };
        // A mate, or a defence every move of which lets the king be exploded,
        // is not kept: `recomputed` finds it again.
        if defences != 0 && !(exposed && counter < LAST) {
            #[cfg(test)]
            self.check_short(pos, proof);
            self.add_proof(pos.hash(), proof);
        }
        debug_assert!(
            defences != 0 && !(exposed && counter < LAST) || self.recomputed(pos) == Some(proof)
        );
        Some((proof.length, proof.limit))
    }

    /// The proof of a position, defender to move, in which every move lets
    /// the attacker explode the king at once, short of the 50-move limit: a
    /// win in two attacker moves, settled without a search and not kept.
    fn exposed(&self, pos: &Position) -> Option<Proof> {
        if pos.halfmoves() >= LAST {
            return None;
        }
        let mover = Mover::new(pos);
        let mut any = false;
        let mut limit = LAST;
        for &reply in pos.pseudo_legal().as_slice() {
            if !mover.legal(reply) {
                continue;
            }
            let effect = mover.effect(reply);
            if mover.explodes_them(&effect) || !mover.exposes(reply, &effect) {
                return None;
            }
            any = true;
            // Where a reply leaves the 50-move counter running, the win holds
            // at one counter less.
            if effect.counter != 0 {
                limit = LAST - 1;
            }
        }
        any.then_some(Proof {
            m: None,
            length: 1,
            limit,
        })
    }

    /// The proof of a position whose proof is not kept because it comes out
    /// the same whenever it is worked out again: the attacker exploding the
    /// king at once, the defender mated, or a defence after which every move
    /// lets the king be exploded; once the proofs have dropped them, a mate
    /// in one; and once they have dropped their short proofs, a win of so
    /// few moves (`short_proof`).
    fn recomputed(&self, pos: &Position) -> Option<Proof> {
        if pos.turn() == self.attacker {
            return match self.explosion(pos).or_else(|| self.dropped_mate(pos)) {
                Some(m) => Some(Proof {
                    m: Some(m),
                    length: 1,
                    limit: LAST,
                }),
                None => self.short_proof(pos),
            };
        }
        if pos.in_check() && !moves::has_legal_move(pos) {
            return Some(Proof::FINISHED);
        }
        self.exposed(pos).or_else(|| self.short_proof(pos))
    }

    /// Once the proofs have dropped their short proofs (`drop_short`): a
    /// position's win in at most `proofs::SHORT` moves, worked out again
    /// from the position alone (`format::short`), as the search counts a
    /// proof's length, its tablebase wins too unless it is exact. The win
    /// is found at the position's counter, so the proof holds up to there.
    fn short_proof(&self, pos: &Position) -> Option<Proof> {
        if self.short_drops() == 0 {
            return None;
        }
        let tables = self.var.tablebase.filter(|_| !self.exact);
        let (length, m) = crate::format::short(pos, self.attacker, tables)?;
        Some(Proof {
            m,
            length,
            limit: pos.halfmoves(),
        })
    }

    /// A mate in one, once the proofs have dropped them and keep none.
    fn dropped_mate(&self, pos: &Position) -> Option<Move> {
        if self.mates_kept() {
            None
        } else {
            mate_in_one(pos)
        }
    }

    /// Whether a move mates at once in an attacker position, as `mate_in_one`
    /// finds a mate: how a mate in one the table notes is checked.
    fn mates_with(&self, pos: &Position, m: Move) -> bool {
        pos.turn() == self.attacker
            && pos.halfmoves() <= LAST
            && pos.is_pseudo_legal(m)
            && mates(&Mover::new(pos), pos, m)
    }

    /// The proof of the attacker exploding the king at once.
    fn exploding(&self, pos: &Position) -> Option<Proof> {
        self.explosion(pos).map(|m| Proof {
            m: Some(m),
            length: 1,
            limit: LAST,
        })
    }

    /// Keeps an attacker position's proof, mated whether its move mates,
    /// unless it explodes the king at once: those make up most short proofs,
    /// and replays make many of them.
    fn keep_proof(&mut self, pos: &Position, proof: Proof, mated: bool) {
        if proof.length <= 1 && self.exploding(pos) == Some(proof) {
            return;
        }
        self.keep_attacker(pos, proof, mated);
    }

    /// Keeps an attacker position's proof, mated whether its move mates. Once
    /// the proofs have dropped their mates in one, a position that has one
    /// keeps no proof, and drops any it kept before: a parent may count on
    /// the mate, as `recomputed` finds it, and a longer proof found by lookup
    /// would make the parent's win longer than it says. The table notes the
    /// mate instead, so that meeting the position again costs a check of its
    /// move rather than a node.
    fn keep_attacker(&mut self, pos: &Position, proof: Proof, mated: bool) {
        #[cfg(test)]
        self.check_short(pos, proof);
        let hash = pos.hash();
        if self.mates_kept() {
            if !mated {
                self.add_proof(hash, proof);
                return;
            }
            self.filtering = self.table().note(hash, self.filtering);
            if self.proofs().add_mate(hash, proof) && self.proofs().taken() < self.drop_mates_at {
                return;
            }
            // Dropped with the others.
            self.drop_mates();
        }
        let mate = if mated { proof.m } else { mate_in_one(pos) };
        match mate {
            Some(m) => {
                self.proofs().remove(hash);
                // With a limit, the table's keys count the moves left.
                if self.most.is_none() {
                    let known = Known::Mate(m.bits());
                    self.filtering = self.table().put(hash, known, 1, self.filtering);
                    self.puts += usize::from(self.together);
                    self.refresh_filter();
                }
            }
            None => self.add_proof(hash, proof),
        }
    }

    /// Replays on a defender position's open defences the win the attacker
    /// would have at once if the defender passed: the first move the tactics
    /// that score moves find winning (an explosion, a check every evasion of
    /// loses, a double threat no defence parries, or a move into a known
    /// proof). It becomes the passed position's proof, and each defence it
    /// still answers is proven by replaying it, as a proven sibling's proof
    /// is replayed; a defence that parries it starts at half its proof
    /// number. Plain moves are not tried: they win at once only into a known
    /// proof.
    pub(super) fn replay_threat(&mut self, pos: &Position, remaining: u32, children: &mut [Child]) {
        let passed = pos.null();
        let left = remaining.saturating_sub(1);
        let counter = passed.halfmoves();
        if counter > LAST {
            return;
        }
        if self.proof_at(passed.hash(), counter).is_none() {
            let mover = Mover::new(&passed);
            let mut win = None;
            // The moves first, their table reads started, as `children` does.
            let mut moves = std::mem::take(&mut self.var.scratch);
            moves.clear();
            let filtering = self.filtering;
            for &m in passed.pseudo_legal().as_slice() {
                if mover.plain(m) || !mover.legal(m) {
                    continue;
                }
                let effect = mover.full_effect(m);
                self.prefetch_child(effect.hash, left, filtering);
                moves.push((m, effect, !filtering));
            }
            if filtering {
                self.read_filter(&mut moves, left);
            }
            self.mark_repetitions(passed.halfmoves());
            for &(m, effect, seen) in &moves {
                // Only whether the move wins at once counts here, so a check
                // needs one surviving evasion, not their number.
                let child = self.evaluate(&passed, &mover, m, &effect, seen, true, left, 1);
                if child.value.numbers.proof != 0 {
                    continue;
                }
                let (length, limit) = if child.exploded {
                    (0, u32::MAX)
                } else if let Some(proof) = self
                    .proof_at(child.hash, child.counter)
                    .or_else(|| self.recomputed(&passed.play(m)))
                {
                    (proof.length, proof.limit)
                } else {
                    continue;
                };
                let mates = !child.exploded && mated(length, limit);
                let limit = if child.counter == 0 {
                    LAST
                } else {
                    limit.saturating_sub(1).min(LAST)
                };
                let proof = Proof {
                    m: Some(m),
                    length: length + 1,
                    limit,
                };
                win = Some((proof, mates));
                break;
            }
            self.var.scratch = moves;
            let Some((proof, mates)) = win else {
                return;
            };
            self.keep_proof(&passed, proof, mates);
        }
        for child in children.iter_mut() {
            if child.value.numbers.proof == 0 || child.value.numbers.disproof == 0 {
                continue;
            }
            let position = pos.play(child.m);
            let mut budget = SIMULATION;
            let mut line = std::mem::take(&mut self.var.line);
            let reached = self.simulate(&position, &passed, remaining, &mut budget, &mut line);
            self.var.line = line;
            if reached.is_some_and(|(length, _)| length <= remaining) {
                child.value = Value::WON;
                child.simulated = true;
            } else if child.fresh {
                child.value.numbers.proof = (child.value.numbers.proof / 2).max(1);
            }
        }
    }

    /// The value of a mate, if the side to move, in check, is mated.
    fn mated(&self, pos: &Position) -> Option<Value> {
        if moves::has_legal_move(pos) {
            return None;
        }
        Some(if pos.turn() == self.attacker {
            Value::DRAWN
        } else {
            Value::WON
        })
    }

    /// Reads the filter's bit of every listed child, their words' loads under
    /// way, and starts the table reads of the children it may hold.
    fn read_filter(&self, moves: &mut [(Move, Effect, bool)], remaining: u32) {
        for (_, effect, seen) in moves.iter_mut() {
            let key = self.key(effect.hash, remaining);
            *seen = self.seen(effect.hash, key);
            if *seen {
                self.table().prefetch(key);
                self.proofs().prefetch(effect.hash);
            }
        }
    }

    /// The attacker's capture that explodes the defender's king at once before
    /// the 50-move limit: the proof of a position whose proof is not kept.
    fn explosion(&self, pos: &Position) -> Option<Move> {
        if pos.halfmoves() > LAST {
            return None;
        }
        moves::exploding_capture(pos, self.attacker)
    }

    /// Records a proven position: attacking, through the shortest winning
    /// move; defending, through every defence.
    fn record_proof(&mut self, pos: &Position, hash: u64, children: &[Child], attacking: bool) {
        // A child's proof seen from here: its length, the highest counter
        // here it holds at, which a capture or pawn move frees, and whether it
        // is the defender mated.
        let reach = |child: &Child| -> Option<(u32, u32, bool)> {
            let counter = child.counter;
            let (length, limit) = if child.exploded {
                (0, u32::MAX)
            } else if child.winning {
                (1, LAST)
            } else {
                if child.value.numbers.proof != 0 {
                    return None;
                }
                let proof = self
                    .proof_at(child.hash, counter)
                    .or_else(|| self.recomputed(&pos.play(child.m)))?;
                (proof.length, proof.limit)
            };
            let mates = !child.exploded && mated(length, limit);
            let limit = if counter == 0 {
                LAST
            } else {
                limit.saturating_sub(1).min(LAST)
            };
            Some((length, limit, mates))
        };
        let proof = if attacking {
            children
                .iter()
                .filter(|child| !child.group)
                .filter_map(|child| reach(child).map(|reached| (child.m, reached)))
                .min_by_key(|&(_, (length, limit, _))| (length, Reverse(limit)))
                .map(|(m, (length, limit, mates))| {
                    let proof = Proof {
                        m: Some(m),
                        length: length + 1,
                        limit,
                    };
                    (proof, mates)
                })
        } else {
            children
                .iter()
                .map(reach)
                .try_fold(
                    Proof {
                        m: None,
                        length: 0,
                        limit: LAST,
                    },
                    |proof, reached| {
                        let (length, limit, _) = reached?;
                        Some(Proof {
                            m: None,
                            length: proof.length.max(length),
                            limit: proof.limit.min(limit),
                        })
                    },
                )
                .map(|proof| (proof, false))
        };
        if attacking && let Some((Proof { m: Some(m), .. }, _)) = proof {
            let index = history_index(pos, m);
            self.var.attack_history[index] = self.var.attack_history[index].saturating_add(1);
        }
        if let Some((proof, mates)) = proof {
            debug_assert_eq!(hash, pos.hash());
            // An attacker's shortest win longer than one move leaves no mate
            // in one there: one would have been shorter.
            if attacking && proof.length <= 1 {
                self.keep_proof(pos, proof, mates);
            } else if attacking
                || !(pos.halfmoves() < LAST && children.iter().all(|child| child.winning))
            {
                #[cfg(test)]
                self.check_short(pos, proof);
                self.add_proof(hash, proof);
            }
        }
    }

    /// For tests that ask: fails unless a short proof the search has found
    /// is one `format::short` works out again as short, at the position's
    /// counter and at the highest one the proof holds at: what dropping the
    /// short proofs relies on (`drop_short`).
    #[cfg(test)]
    fn check_short(&mut self, pos: &Position, proof: Proof) {
        if !self.checking || proof.length > crate::proofs::SHORT {
            return;
        }
        let tables = self.var.tablebase.filter(|_| !self.exact);
        let counter = pos.halfmoves();
        for counter in [counter, proof.limit.min(LAST).max(counter)] {
            let found = crate::format::short(&pos.at_counter(counter), self.attacker, tables);
            assert!(
                found.is_some_and(|(length, _)| length <= proof.length),
                "{:?} at counter {counter}: {proof:?}, worked out {found:?}",
                pos.to_atomic()
            );
        }
        self.checked += 1;
    }
}

/// What `Dfpn::winning_move` reads once the proofs have dropped their short
/// ones, apart from the search, for the threads of `with_moves_ahead`.
struct WalkMoves<'a> {
    attacker: Color,
    tables: Option<&'a Tablebase>,
    store: &'a Store,
    filtering: bool,
}

impl WalkMoves<'_> {
    /// `winning_move` on the board: the short win worked out again, else the
    /// proof kept, else a mate in one if the proofs dropped theirs.
    fn walk_move(&self, pos: &Position) -> Option<Move> {
        if let Some(m) = crate::format::short(pos, self.attacker, self.tables).and_then(|(_, m)| m)
        {
            return Some(m);
        }
        let (hash, counter) = (pos.hash(), pos.halfmoves());
        if self.store.table.may_hold(hash, self.filtering)
            && let Some(proof) = self.store.proofs.get(hash, counter)
        {
            return proof.m;
        }
        if self.store.mates_kept.load(Relaxed) {
            None
        } else {
            mate_in_one(pos)
        }
    }
}

/// The moves `Dfpn::with_moves_ahead` works out ahead of a proof's walk,
/// and the positions waiting for them.
#[derive(Default)]
struct Ahead {
    state: Mutex<AheadState>,
    /// Signalled when a position waits, or the walk ends.
    waiting: Condvar,
    /// Signalled when a win is worked out.
    worked: Condvar,
}

#[derive(Default)]
struct AheadState {
    /// Defender positions the walk reached, and the attacker positions their
    /// replies reach, last in first out.
    queue: Vec<Position>,
    /// Attacker positions being worked out.
    running: HashSet<Key>,
    found: HashMap<Key, Option<Move>>,
    /// Positions the walk worked out itself, left alone if met in the queue.
    taken: HashSet<Key>,
    stopped: bool,
}

/// How many positions the queue and the wins found may hold, before the
/// oldest positions go, or the wins the walk never asked for.
const AHEAD: usize = 1 << 15;

impl Ahead {
    /// A defender position the walk has reached.
    fn give(&self, pos: Position) {
        let mut state = lock(&self.state);
        state.push(pos);
        drop(state);
        self.waiting.notify_one();
    }

    /// The move of an attacker position, if a thread has worked it out or is
    /// working it out; none if the walk must.
    fn take(&self, key: &Key) -> Option<Option<Move>> {
        let mut state = lock(&self.state);
        loop {
            if let Some(m) = state.found.remove(key) {
                return Some(m);
            }
            if !state.running.contains(key) {
                if state.taken.len() >= AHEAD {
                    state.taken.clear();
                }
                state.taken.insert(*key);
                return None;
            }
            state = self
                .worked
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn stop(&self) {
        lock(&self.state).stopped = true;
        self.waiting.notify_all();
    }

    /// A thread's work until the walk ends: a defender position's replies go
    /// to the queue; an attacker position's move is worked out, and the
    /// position it leads to goes to the queue.
    fn work(&self, moves: &WalkMoves) {
        let attacker = moves.attacker;
        let mut state = lock(&self.state);
        loop {
            if state.stopped {
                return;
            }
            let Some(pos) = state.queue.pop() else {
                state = self
                    .waiting
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
                continue;
            };
            if pos.king(Color::White).is_none() || pos.king(Color::Black).is_none() {
                continue;
            }
            if pos.turn() != attacker {
                // The walk takes the replies in the format's order: the first
                // goes on top.
                for &reply in crate::format::replies(&pos).iter().rev() {
                    state.push(pos.play(reply));
                }
                self.waiting.notify_all();
                continue;
            }
            let key = (pos.hash(), pos.halfmoves());
            if state.taken.remove(&key)
                || state.found.contains_key(&key)
                || !state.running.insert(key)
            {
                continue;
            }
            drop(state);
            let working = Working { ahead: self, key };
            let m = moves.walk_move(&pos);
            std::mem::forget(working);
            state = lock(&self.state);
            state.running.remove(&key);
            if state.found.len() >= AHEAD {
                state.found.clear();
            }
            state.found.insert(key, m);
            if let Some(m) = m {
                state.push(pos.play(m));
            }
            self.worked.notify_all();
        }
    }
}

impl AheadState {
    /// A position for the queue, the oldest half going once it is full.
    fn push(&mut self, pos: Position) {
        if self.queue.len() >= AHEAD {
            self.queue.drain(..AHEAD / 2);
        }
        self.queue.push(pos);
    }
}

/// Stops the threads ahead when dropped.
struct StopAhead<'a>(&'a Ahead);

impl Drop for StopAhead<'_> {
    fn drop(&mut self) {
        self.0.stop();
    }
}

/// An attacker position a thread ahead is working out. Dropped only if that
/// fails: it is no longer running, and the threads ahead stop, so that the
/// walk works it out itself rather than wait for it.
struct Working<'a> {
    ahead: &'a Ahead,
    key: Key,
}

impl Drop for Working<'_> {
    fn drop(&mut self) {
        let mut state = lock(&self.ahead.state);
        state.running.remove(&self.key);
        state.stopped = true;
        drop(state);
        self.ahead.worked.notify_all();
        self.ahead.waiting.notify_all();
    }
}

/// `x.round()`. WebAssembly has no instruction for it and calls a library
/// function, slowly; there, `round_by_truncation`.
#[cfg(target_arch = "wasm32")]
fn round(x: f32) -> f32 {
    round_by_truncation(x)
}

#[cfg(not(target_arch = "wasm32"))]
fn round(x: f32) -> f32 {
    x.round()
}

/// `x.round()` from instructions WebAssembly has: the difference from the
/// truncation is exact, and so is a step of one from it.
#[cfg(any(target_arch = "wasm32", test))]
pub(super) fn round_by_truncation(x: f32) -> f32 {
    let whole = x.trunc();
    if (x - whole).abs() >= 0.5 {
        whole + 1f32.copysign(x)
    } else {
        whole
    }
}

/// 10 to the `x`. Natively, `10f32.powf(x)` becomes the system's exp10f;
/// WebAssembly has none and calls a slow powf: there, 2 to the x log2(10) in
/// double precision, rounded once.
#[cfg(target_arch = "wasm32")]
fn exp10(x: f32) -> f32 {
    (f64::from(x) * std::f64::consts::LOG2_10).exp2() as f32
}

#[cfg(not(target_arch = "wasm32"))]
fn exp10(x: f32) -> f32 {
    10f32.powf(x)
}

/// Scales the new children's starting numbers by their parent's evaluation,
/// as EF-PN does (Winands and Schadd): `attacking` when the parent is the
/// attacker's, `step` from 0 (the defender clearly ahead) to 2. Between those,
/// a model's step scales both numbers, uncapped and alike: proof numbers
/// grow by 10^(-MAGNITUDE logit) and disproof numbers by 10^(MAGNITUDE
/// logit) from EF-PN's at an average position.
pub(super) fn scale(children: &mut [Child], attacking: bool, step: Option<f32>) {
    let Some(step) = step else {
        return;
    };
    let (a, b) = (EF_A as f32, EF_B as f32);
    let times = |number: u32, factor: f32| (round(number as f32 * factor) as u32).min(1 << 24);
    let logit = (step > 0.0 && step < 2.0).then(|| (step / (2.0 - step)).ln());
    let effort = logit.map(|logit| exp10(-MAGNITUDE * logit));
    let refute = logit.map(|logit| exp10(MAGNITUDE * logit));
    for child in children.iter_mut().filter(|child| child.fresh) {
        let numbers = &mut child.value.numbers;
        if attacking {
            numbers.proof = match effort {
                Some(effort) => times(numbers.proof, ((1.0 + b) * effort).max(1.0)),
                None => times(numbers.proof, 1.0 + b * (2.0 - step)),
            };
            numbers.disproof = match refute {
                Some(refute) => times(numbers.disproof, ((1.0 + a) * refute).max(1.0)),
                None => times(numbers.disproof, 1.0 + a * step),
            };
        } else {
            // A defence's new children start at proof number 1.
            numbers.proof = match effort {
                Some(effort) => times(numbers.proof, ((1.0 + a) * effort).max(1.0)),
                None => times(numbers.proof, 1.0) + round(a * (2.0 - step)) as u32,
            };
            numbers.disproof = match refute {
                Some(refute) => times(numbers.disproof, ((1.0 + b) * refute).max(1.0)),
                None => times(numbers.disproof, 1.0 + b * step),
            };
        }
    }
}

/// A model's step, between 0 and 2, with its odds, step / (2 - step), moved
/// by `unit` percent: computed in f64 and kept off the endpoints, which
/// `scale` treats apart.
pub(super) fn perturb_model_step(step: f32, unit: f32) -> f32 {
    debug_assert!((0.0..=2.0).contains(&step));
    debug_assert!((-1.0..=1.0).contains(&unit));
    if step == 0.0 || step == 2.0 {
        return step;
    }
    let multiplier = 1.0 + 0.01 * f64::from(unit);
    let value = f64::from(step);
    let perturbed = (2.0 * value * multiplier / (2.0 - value + value * multiplier)) as f32;
    perturbed.clamp(f32::from_bits(1), f32::from_bits(2.0f32.to_bits() - 1))
}

/// A move's place in the history tables: its piece and destination.
fn history_index(pos: &Position, m: Move) -> usize {
    let role = pos.role_at(m.from()).map_or(0, |role| role as usize);
    role * 64 + m.to() as usize
}

/// A check the defender has no move against, before the 50-move limit.
fn mate_in_one(pos: &Position) -> Option<Move> {
    if pos.halfmoves() > LAST {
        return None;
    }
    let mover = Mover::new(pos);
    pos.pseudo_legal()
        .as_slice()
        .iter()
        .copied()
        .find(|&m| mates(&mover, pos, m))
}

/// Whether a pseudo-legal move mates: a check the other side has no move
/// against, rather than an explosion of its king.
fn mates(mover: &Mover, pos: &Position, m: Move) -> bool {
    // Plain moves check nothing, and telling them costs a bit test.
    if mover.plain(m) || !mover.legal(m) {
        return false;
    }
    let effect = mover.effect(m);
    !mover.explodes_them(&effect)
        && mover.gives_check(m, &effect)
        && !moves::has_legal_move(&pos.play(m))
}
