//! Antichess's part of the search: White wins within a number of White
//! moves, always under a limit (`set_most`).
//!
//! A position after White's move starts with as many proof numbers as Black
//! has replies, and one after Black's with as many disproof numbers as White
//! has moves, so that forced lines, of which Antichess's compulsory captures
//! make many, are tried first. With Fairy-Stockfish's network (`nnue`), the
//! position after each Black reply far enough from the end (`GUIDE_FROM`)
//! also has its numbers scaled by how the network rates it for White
//! (`GUIDE_DISPROOF`). With tolius's tables (`tablebases`), a position of four
//! pieces or fewer is settled from them where they can (`settle`), rather
//! than searched.
//!
//! Repetitions need no handling: under a limit on White's moves a line ends
//! (`rules`). Proofs of finished games and of positions the tables settle are
//! not kept, as the walk of a proof works them out again (`recomputed`), and
//! the format's writer plays the tables' quickest wins.

use std::cmp::Reverse;
use std::collections::TryReserveError;
use std::time::Duration;

use shakmaty::{Color, Position as _, variant::Antichess};

use super::{Board as _, Child, Dfpn, INFINITY, LAST, Numbers, Value, Variant};
use crate::antichess::nnue::{Accumulator, Network};
use crate::antichess::rules::{self, LAST_COUNTER, Status as Rules};
use crate::antichess::tablebases::{Entry, MOST_PIECES, Tablebases};
use crate::antichess::{Board, packed};
use crate::clock::Instant;
use crate::proofs::Proof;
use crate::search::Status;
use crate::variant;

/// With a network, how far its evaluation of the position after each Black
/// reply scales that position's starting numbers, as EF-PN does (Winands and
/// Schadd): White's evaluation squashed to a step from 0 (Black far ahead)
/// to 2 (White far ahead), `2 / (1 + e^(-evaluation / GUIDE_SPREAD))`,
/// multiplies the disproof number by `1 + GUIDE_DISPROOF * step` and the
/// proof number by `1 + GUIDE_PROOF * (2 - step)`. Black's replies that
/// leave White worse off are then tried first as refutations, and White's
/// moves after which they all leave White better off look cheaper to prove.
const GUIDE_DISPROOF: f32 = 8.0;
const GUIDE_PROOF: f32 = 4.0;
const GUIDE_SPREAD: f32 = 300.0;
/// The fewest White moves left at which the network guides the search:
/// nearer the end, counting moves orders it about as well, for less.
pub const GUIDE_FROM: u32 = 12;
/// How many of the network's evaluations are remembered, a power of two.
const EVALUATIONS: usize = 1 << 18;
/// How many of the tables' answers are remembered, a power of two: 64 MB a
/// search thread.
const ANSWERS: usize = 1 << 22;

/// What every thread of an Antichess search shares: the network and the
/// tables it reads, if any, and the fewest White moves left at which the
/// network guides it.
#[derive(Clone, Copy)]
pub struct Shared<'a> {
    network: Option<&'a Network>,
    tablebases: Option<Tablebases>,
    guide_from: u32,
}

/// The network's first layer of a position on the search's line, with its
/// board, from which the next one's is updated.
struct Frame {
    acc: Accumulator,
    board: shakmaty::Board,
}

/// What each thread of an Antichess search keeps.
pub struct State<'a> {
    shared: Shared<'a>,
    /// With a network, the first layer of each position on the line from the
    /// root, as far as they have `guide_from` White moves left or more.
    frames: Vec<Frame>,
    /// The network's recent evaluations by position: a position is often
    /// met again before it is expanded. Zero stands for an empty slot, so a
    /// position keyed zero is never remembered.
    evaluations: Vec<(u64, i32)>,
    /// The tables' latest answers by position, likewise: a position is met
    /// again each time its parent is searched.
    answers: Vec<(u64, Option<Entry>)>,
    /// The legal moves being scored: each move, its child's key and counter,
    /// and whether the table's filter may hold the child.
    scratch: Vec<(shakmaty::Move, u64, u32, bool)>,
    /// The positions the tables have settled.
    hits: u64,
    /// Positions expanded by their count of pieces.
    by_pieces: [u64; 33],
}

impl<'a> State<'a> {
    fn new(shared: Shared<'a>) -> State<'a> {
        let mut state = State {
            shared,
            frames: Vec::new(),
            evaluations: Vec::new(),
            answers: Vec::new(),
            scratch: Vec::new(),
            hits: 0,
            by_pieces: [0; 33],
        };
        state.allocate();
        state
    }

    /// The memory of evaluations and answers, for a network and tables.
    fn allocate(&mut self) {
        if self.shared.network.is_some() && self.evaluations.is_empty() {
            self.evaluations = vec![(0, 0); EVALUATIONS];
        }
        if self.shared.tablebases.is_some() && self.answers.is_empty() {
            self.answers = vec![(0, None); ANSWERS];
        }
    }

    /// What the tables say of a position, by its key, as remembered.
    fn probe(&mut self, tables: Tablebases, pos: &Antichess, key: u64) -> Option<Entry> {
        let slot = &mut self.answers[key as usize & (ANSWERS - 1)];
        if slot.0 == key && key != 0 {
            return slot.1;
        }
        let entry = tables.probe(pos);
        *slot = (key, entry);
        entry
    }
}

impl Variant for variant::Antichess {
    const EPSILON: u32 = 4;

    type Root = Antichess;
    type Board = Board;
    type Shared<'a> = Shared<'a>;
    type State<'a> = State<'a>;

    fn state(shared: Self::Shared<'_>) -> Self::State<'_> {
        State::new(shared)
    }

    fn shared<'a>(state: &Self::State<'a>) -> Self::Shared<'a> {
        state.shared
    }

    fn board(root: &Antichess) -> Board {
        Board::new(root.clone())
    }

    fn status(root: &Antichess, attacker: Color) -> Status {
        match rules::status_counted(root, rules::count_moves(root)) {
            Rules::Open => Status::Open,
            Rules::WhiteWon if attacker == Color::White => Status::Won,
            Rules::BlackWon if attacker == Color::Black => Status::Won,
            _ => Status::Over,
        }
    }

    fn children(
        search: &mut Dfpn<'_, Self>,
        pos: &Board,
        attacking: bool,
        remaining: u32,
        children: &mut Vec<Child>,
    ) -> Option<f32> {
        search.children(pos, attacking, remaining, children);
        None
    }

    fn scored(_: &mut Dfpn<'_, Self>, _: &Board, _: bool, _: bool, _: u32, _: &mut [Child]) {}

    /// Unreached: a position without a legal move has ended, which its parent
    /// settles first. The side to move wins it.
    fn no_moves(search: &mut Dfpn<'_, Self>, pos: &Board) -> Value {
        if pos.turn() == search.attacker {
            Value::WON
        } else {
            Value::DRAWN
        }
    }

    fn open_group(_: &mut Dfpn<'_, Self>, _: &Board, _: u32, _: &mut Vec<Child>, _: Option<f32>) {
        unreachable!("Antichess holds no moves back");
    }

    fn defence_proven(
        _: &mut Dfpn<'_, Self>,
        _: &Board,
        _: &Board,
        _: usize,
        _: u32,
        _: &mut [Child],
    ) {
    }

    fn refuted(_: &mut Dfpn<'_, Self>, _: &Board, _: crate::board::Move) {}

    fn record_proof(
        search: &mut Dfpn<'_, Self>,
        pos: &Board,
        hash: u64,
        children: &[Child],
        attacking: bool,
    ) {
        search.record_proof(pos, hash, children, attacking);
    }

    fn recomputed(search: &Dfpn<'_, Self>, pos: &Board) -> Option<Proof> {
        search.recomputed(pos)
    }

    /// The table notes no mates in one for Antichess.
    fn mates_with(_: &Dfpn<'_, Self>, _: &Board, _: crate::board::Move) -> bool {
        false
    }

    /// The tables' settled positions and the positions expanded by their
    /// pieces count every thread's.
    fn gather<'a>(state: &mut State<'a>, helper: State<'a>) {
        state.hits += helper.hits;
        for (count, more) in state.by_pieces.iter_mut().zip(helper.by_pieces) {
            *count += more;
        }
    }
}

impl<'a> Dfpn<'a, variant::Antichess> {
    /// A search for White's wins, with a table and proofs of these
    /// megabytes, for `threads` threads that search together (`with_threads`),
    /// without a network or tables until they are set. Unless the memory
    /// cannot be had.
    pub fn antichess(
        memory: (usize, usize),
        threads: usize,
    ) -> Result<Dfpn<'a, variant::Antichess>, TryReserveError> {
        let shared = Shared {
            network: None,
            tablebases: None,
            guide_from: GUIDE_FROM,
        };
        // No deadline until one is set.
        let deadline = Instant::now() + Duration::from_secs(100 * 365 * 86_400);
        Dfpn::with_threads(Color::White, shared, deadline, memory, threads)
    }

    /// Guides later searches by Fairy-Stockfish's Antichess network.
    pub fn set_network(&mut self, network: &'a Network) {
        self.var.shared.network = Some(network);
        self.var.allocate();
    }

    /// Settles later searches' positions of four pieces or fewer from
    /// tolius's tables where they can (`settle`).
    pub fn set_tablebases(&mut self, tables: Tablebases) {
        self.var.shared.tablebases = Some(tables);
        self.var.allocate();
    }

    /// The fewest White moves left at which the network guides the search.
    pub fn set_guide_from(&mut self, moves: u32) {
        self.var.shared.guide_from = moves;
    }

    /// The tables the search reads, if any.
    pub fn tablebases(&self) -> Option<Tablebases> {
        self.var.shared.tablebases
    }

    /// The positions the tables have settled, over every search and every
    /// thread.
    pub fn tablebase_hits(&self) -> u64 {
        self.var.hits
    }

    /// Positions expanded by their count of pieces, over every search and
    /// every thread.
    pub fn by_pieces(&self) -> [u64; 33] {
        self.var.by_pieces
    }

    /// The shortest proof kept of a position, by its key (`rules::key`), that
    /// holds at this 50-move counter: none for a finished game or one the
    /// tables settled, which are not kept.
    pub fn proof(&self, key: u64, counter: u32) -> Option<Proof> {
        self.proofs().get(key, counter)
    }

    /// Scores a position's moves into `children`, with `remaining` White moves
    /// left, as `mid` reads them: what the proofs and the table know of each
    /// child, and otherwise its game's end, the tables' answer, or its
    /// starting numbers, which the network scales after a Black move.
    fn children(
        &mut self,
        pos: &Board,
        attacking: bool,
        remaining: u32,
        children: &mut Vec<Child>,
    ) {
        let board = pos.pos.board();
        self.var.by_pieces[board.occupied().count()] += 1;
        let left = remaining - u32::from(attacking);
        let guided = self.frame(pos, remaining) && !attacking;
        // First every child's key, starting the reads of the filter or of
        // what the table and the proofs hold of it: the reads do not wait on
        // one another, so their latencies overlap.
        let mut scratch = std::mem::take(&mut self.var.scratch);
        scratch.clear();
        let filtering = self.filtering;
        let next = pos.pos.halfmoves() + 1;
        for &m in &pos.pos.legal_moves() {
            let hash = rules::child_key(&pos.pos, pos.key, m);
            let counter = if m.is_zeroing() { 0 } else { next };
            self.prefetch_child(hash, left, filtering);
            scratch.push((m, hash, counter, !filtering));
        }
        if filtering {
            for (_, hash, _, seen) in &mut scratch {
                let key = self.key(*hash, left);
                *seen = self.seen(*hash, key);
                if *seen {
                    self.table().prefetch(key);
                    self.proofs().prefetch(*hash);
                }
            }
        }
        children.clear();
        for &(m, hash, counter, seen) in &scratch {
            let known = if seen && counter <= LAST {
                if self.proof_within(hash, counter, left).is_some() {
                    Some(Value::WON)
                } else {
                    self.lookup(self.key(hash, left), counter, |_| false)
                }
            } else {
                None
            };
            let value = match known {
                Some(value) => value,
                None => self.start(pos, m, hash, left, guided),
            };
            children.push(Child {
                m: packed(m),
                hash,
                counter,
                value,
                exploded: false,
                winning: false,
                score: 0,
                simulated: false,
                fresh: false,
                group: false,
            });
        }
        self.var.scratch = scratch;
    }

    /// With a network, computes the first layer of `pos`, the position at the
    /// line's end, where it has `guide_from` White moves left or more, or is
    /// the root: from its parent's, which then has one too. Returns whether
    /// it did.
    fn frame(&mut self, pos: &Board, remaining: u32) -> bool {
        let Some(network) = self.var.shared.network else {
            return false;
        };
        let level = self.path.len();
        let frames = &mut self.var.frames;
        frames.truncate(level);
        let acc = match frames.last() {
            None if level == 0 => network.refresh(&pos.pos),
            Some(parent) if frames.len() == level && remaining >= self.var.shared.guide_from => {
                network.update_board(&parent.acc, &parent.board, pos.pos.board())
            }
            _ => return false,
        };
        frames.push(Frame {
            acc,
            board: pos.pos.board().clone(),
        });
        remaining >= self.var.shared.guide_from
    }

    /// The value of a child not searched before, the position after `m`
    /// whose key is `hash`, with `left` White moves: its game's end, the
    /// tables' answer, or its starting numbers, which with `guided` the
    /// network's evaluation scales.
    fn start(
        &mut self,
        pos: &Board,
        m: shakmaty::Move,
        hash: u64,
        left: u32,
        guided: bool,
    ) -> Value {
        let mut after = pos.pos.clone();
        after.play_unchecked(m);
        let moves = rules::count_moves(&after);
        match rules::status_counted(&after, moves) {
            Rules::WhiteWon => return Value::WON,
            Rules::BlackWon | Rules::Insufficient => return Value::DRAWN,
            Rules::FiftyMoves => {
                return Value {
                    fifty: true,
                    ..Value::DRAWN
                };
            }
            Rules::Open => {}
        }
        let moves = moves as u32;
        let numbers = match after.turn() {
            Color::White if left == 0 => return Value::DRAWN,
            Color::White => Numbers {
                proof: 1,
                disproof: moves,
            },
            Color::Black => Numbers {
                proof: moves,
                disproof: 1,
            },
        };
        if let Some(tables) = self.var.shared.tablebases
            && after.board().occupied().count() <= MOST_PIECES
            && let Some(entry) = self.var.probe(tables, &after, hash)
            && let Some(settled) = settle(entry, after.turn(), after.halfmoves(), left)
        {
            self.var.hits += 1;
            return match settled {
                Settled::Won { .. } => Value::WON,
                Settled::Refuted => Value::DRAWN,
            };
        }
        let numbers = match (guided, self.var.shared.network) {
            (true, Some(network)) => {
                let level = self.path.len();
                let slot = &mut self.var.evaluations[hash as usize & (EVALUATIONS - 1)];
                if slot.0 != hash || hash == 0 {
                    let frame = &self.var.frames[level];
                    let first = network.update_board(&frame.acc, &frame.board, after.board());
                    *slot = (hash, network.evaluate(&first, &after));
                }
                guided_numbers(numbers, slot.1)
            }
            _ => numbers,
        };
        Value::open(numbers)
    }

    /// Records a proven position: White to move, through the shortest
    /// winning move, the one that holds at the highest counter of those;
    /// Black to move, through every reply.
    fn record_proof(&mut self, pos: &Board, hash: u64, children: &[Child], attacking: bool) {
        // A child's proof seen from here: its length, and the highest counter
        // here it holds at, which a capture or pawn move frees.
        let reach = |child: &Child| -> Option<(u32, u32)> {
            if child.value.numbers.proof != 0 {
                return None;
            }
            let proof = self
                .proof_at(child.hash, child.counter)
                .or_else(|| self.recomputed(&pos.play(child.m)))?;
            let limit = if child.counter == 0 {
                LAST
            } else {
                proof.limit.saturating_sub(1).min(LAST)
            };
            Some((proof.length, limit))
        };
        let proof = if attacking {
            children
                .iter()
                .filter_map(|child| reach(child).map(|reached| (child.m, reached)))
                .min_by_key(|&(_, (length, limit))| (length, Reverse(limit)))
                .map(|(m, (length, limit))| Proof {
                    m: Some(m),
                    length: length + 1,
                    limit,
                })
        } else {
            children.iter().map(reach).try_fold(
                Proof {
                    m: None,
                    length: 0,
                    limit: LAST,
                },
                |proof, reached| {
                    let (length, limit) = reached?;
                    Some(Proof {
                        m: None,
                        length: proof.length.max(length),
                        limit: proof.limit.min(limit),
                    })
                },
            )
        };
        if let Some(proof) = proof {
            self.add_proof(hash, proof);
        }
    }

    /// The proof of a won position that is not kept: a finished game, or one
    /// the tables settle, which the format's writer plays out.
    fn recomputed(&self, pos: &Board) -> Option<Proof> {
        match rules::status_counted(&pos.pos, rules::count_moves(&pos.pos)) {
            Rules::WhiteWon => return Some(Proof::FINISHED),
            Rules::Open => {}
            _ => return None,
        }
        let tables = self.var.shared.tablebases?;
        if pos.pos.board().occupied().count() > MOST_PIECES {
            return None;
        }
        let slot = self
            .var
            .answers
            .get(pos.key as usize & (ANSWERS - 1))
            .filter(|slot| slot.0 == pos.key && pos.key != 0);
        let entry = match slot {
            Some(slot) => slot.1,
            None => tables.probe(&pos.pos),
        }?;
        match settle(entry, pos.turn(), pos.halfmoves(), u32::MAX)? {
            Settled::Won { need, limit } => Some(Proof {
                m: None,
                length: need,
                limit,
            }),
            Settled::Refuted => None,
        }
    }
}

/// What a table's entry settles for White.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Settled {
    /// Won within `need` White moves at 50-move counters up to `limit`.
    Won { need: u32, limit: u32 },
    /// Not won within the moves left, at any counter.
    Refuted,
}

/// What a table's entry for a position settles for White, with `budget`
/// White moves left and its 50-move `counter`: nothing when the search must
/// go on.
pub fn settle(entry: Entry, turn: Color, counter: u32, budget: u32) -> Option<Settled> {
    // The plies to White's win. A draw or a loss without the 50-move rule is
    // no win with it.
    let plies = match (entry, turn) {
        (Entry::Win(plies), Color::White) | (Entry::Loss(plies), Color::Black) => plies,
        _ => return Some(Settled::Refuted),
    };
    // White's moves in so many plies, the first of them where White is to
    // move.
    let moves = |plies: u32| match turn {
        Color::White => plies.div_ceil(2),
        Color::Black => plies / 2,
    };
    // No win is quicker. Up to 100 plies the tables' depth is the quickest
    // win's, whether they count it within the 50-move rule from counter 0 or
    // without it; beyond, it shows only that every win takes more than 100.
    if moves(plies.min(101)) > budget {
        return Some(Settled::Refuted);
    }
    // Every line then ends before the 50-move rule could draw it, and none
    // repeats a position, as the depth falls with every ply. Further, the
    // search goes on: a win may reset the counter on its way.
    let last = LAST_COUNTER + 1;
    (counter + plies <= last).then(|| Settled::Won {
        need: moves(plies),
        limit: last - plies,
    })
}

/// Starting numbers scaled by White's evaluation of the position, for White
/// to move (`GUIDE_DISPROOF`).
fn guided_numbers(numbers: Numbers, evaluation: i32) -> Numbers {
    let step = 2.0 / (1.0 + (-(evaluation as f32) / GUIDE_SPREAD).exp());
    let times =
        |number: u32, factor: f32| ((number as f32 * factor).round() as u32).min(INFINITY - 1);
    Numbers {
        proof: times(numbers.proof, 1.0 + GUIDE_PROOF * (2.0 - step)),
        disproof: times(numbers.disproof, 1.0 + GUIDE_DISPROOF * step),
    }
}
