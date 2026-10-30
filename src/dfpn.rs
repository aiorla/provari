//! Depth-first proof-number search (df-pn, Nagai 2002) for Atomic wins.
//!
//! Every position has a proof number, how many unsettled leaves must still be
//! won to win it, and a disproof number, how many must fail to refute it. The
//! search always descends into the child that looks cheapest to settle, within
//! thresholds that keep it depth-first, and uses the 1+ε trick against
//! thrashing. Explosions in one move, either way, are settled without a search,
//! and so are checks and double threats after which every defence lets the
//! king be exploded.
//! Forcing attacker moves start cheaper than quiet ones, so they are tried
//! first while the quiet ones stay available: a check starts at its number of
//! evasions that survive, and a threat to explode the king cheaper the more
//! ways it threatens, unless its piece is cheap to blow up, and a plain move
//! whose piece is cheap to blow up starts dearer than the others. Defences
//! unlikely to refute start harder to refute, and ties go to the moves that
//! proved or refuted most so far (the history heuristic). When a defence is
//! proven, its proof is replayed on the other defences (Kawano's simulation),
//! which proves those it also answers. So is a threat: when a defender
//! position not in check is first expanded, a win the attacker would have at
//! once if the defender passed is replayed on every defence, and only those
//! that parry it stay open.
//!
//! A position keeps its children's numbers while it is searched and updates
//! only the child it searched. Children are scored from their moves' effects;
//! a child position is built only to descend into it. The attacker's plain
//! moves, quiet ones that neither check nor threaten, wait together as one
//! child, scored only when the search first wants one of them (as
//! KomoringHeights delays expansions): most positions never want one. When
//! they are scored, the network evaluates the position after each, and those
//! it likes less than its best few start dearer.
//!
//! Each expanded position is evaluated once, by a hand-crafted evaluation,
//! Atomic-Stockfish's network, or a provability model, and its prospects for
//! the attacker, as a step from behind to ahead, scale its new children's
//! starting numbers (EF-PN, Winands and Schadd): lines that look winnable
//! start cheaper to prove and harder to refute.
//!
//! Positions are keyed without the 50-move counter, so that move orders share
//! their work. A proof records its length in attacker moves and the highest
//! counter it holds at, and only a proof that is shorter or holds further
//! joins it, so lengths shrink along every line of a proof and none can repeat
//! a position. Proofs that come out the same whenever worked out again, such
//! as an explosion at once, are not kept. Repeating a position of the current
//! line, the 50-move limit, and stalemate count as draws. A refutation that
//! relies on a repetition applies only while the repeated position is on the
//! line, and one that relies on the 50-move limit only from the same counter
//! upward.
//!
//! A search may be limited to wins in at most so many attacker moves. Proofs
//! know their lengths, so one counts wherever it fits the moves left; the
//! table keeps what it learns apart for each number of moves left, since a
//! position refuted for lack of moves may be won with more. A tablebase win
//! says nothing of its length, so a limited search plays it out.
//!
//! Threads may search a position together (`try_with_threads`), as chess
//! engines' threads do (Lazy SMP): they share the table and the proofs, each
//! searches from the root with a seed of its own, which moves its path, and
//! a thread that has searched 16 nodes since it last read its position's
//! open children reads them again from the store before its next descent
//! (`REFRESH_AFTER`), so that it builds on what the others have found. A
//! thread also sees a child as dearer the more of the others search below
//! it, as Kaneko's virtual numbers do (2010), so that the threads spread out
//! over the tree rather than follow one line. The first to settle the root
//! stops the others. Dropping the mates in one and rebuilding the table's
//! filter move what every thread reads, so one thread makes them while the
//! others wait (`pause`).
//!
//! A search can report how it goes (`set_progress`): besides its nodes, its
//! line, the first plies of the path it is on, which is the attacker's
//! cheapest move, the defender's likeliest refutation, and so on; the moves
//! still open where each was played, for the defender the defences not yet
//! beaten; the root move searched most; and the root's numbers rebuilt from
//! the line. The root's own are combined only when the search comes back to
//! it, which it may not do for millions of nodes, so each level of the line
//! keeps its other children's numbers as it last combined them, and a report
//! combines them again from the bottom up. That costs some 1.5% of a
//! search's time, so a search keeps its line only while it reports, or when
//! asked (`keep_line`).

use std::cmp::Reverse;
use std::collections::TryReserveError;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering, Ordering::Relaxed};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use crate::clock::Instant;

use shakmaty::{Color, uci::UciMove, variant::Atomic};

use crate::board::{Effect, Move, Position};
use crate::model::{FEATURES, Model};
use crate::moves::{self, Mover};
use crate::nnue::{Accumulator, Network, RefreshCache, Transformed};
use crate::pause::Pause;
use crate::proofs::{Proof, Proofs};
use crate::search::{Key, Status, status};
use crate::table::{self, Known, Table};
use crate::tablebase::{Tablebase, Verdict};

const INFINITY: u32 = 1 << 30;
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
/// The highest 50-move counter at which the game goes on.
const LAST: u32 = 99;
/// The depth of a refutation's repetition when it relies on none.
const ANY_LINE: u32 = u32::MAX;
/// Where the table keeps a position's plain moves together.
const GROUP: u64 = 0x2545_f491_4f6c_dd1d;
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

/// The plies of its line a search keeps for its progress lines.
const LINE: usize = 16;

/// What spreads a seed over all bits (`set_seed`).
const SEED_SPREAD: u64 = 0xD1B5_4A32_D192_ED03;
/// The bits of a position's hash that pick its slot among the threads' marks
/// (`Store::busy`).
const BUSY_BITS: u32 = 18;
/// How much a child's deciding number grows, in percent of it and by at least
/// 1, for each other thread searching below it (`crowd`): at 8 threads, 1
/// proved 1.31 times as fast as 10.
const SPREAD: u64 = 1;
/// Together: the nodes a thread searches after reading what the others found
/// of a position's children before it reads them again (`refresh`). A short
/// search leaves the others little time to find anything, and a child the
/// search goes on to is read anew there. Counted at 4 threads, 16 cut the
/// reads by 85% with as many nodes to a proof; at 8 threads, it proved 1.08
/// times as fast as a read after every child.
const REFRESH_AFTER: u64 = 16;

/// What guides the search's starting numbers.
#[derive(Clone, Copy)]
pub enum Guide<'a> {
    None,
    /// The hand-crafted evaluation, cheap.
    Hand,
    /// Atomic-Stockfish's network, better but dearer.
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
}

/// Where progress lines go instead of standard error.
pub type ProgressSink = Box<dyn FnMut(&serde_json::Value)>;

/// A search's progress lines: how often, the next one, when the search
/// started, the time and nodes at the last line, and where they go if not
/// to standard error.
struct Progress {
    every: Duration,
    next: Instant,
    started: Instant,
    last: (Instant, u64),
    sink: Option<ProgressSink>,
}

/// A position on the search's line, as it was when the search last chose
/// its move there: the move, whether the attacker's, the child's numbers
/// then, the other children's deciding numbers at their least and the
/// others summed (attacking, proof and disproof numbers; defending, the
/// other way round), and the children still open, the chosen one among them.
#[derive(Clone, Copy, Default)]
struct Level {
    m: Move,
    attacking: bool,
    taken: Numbers,
    least: u32,
    sum: u32,
    open: u32,
}

/// Positions a search expands near its root, kept at random for labelling.
pub struct Sampler {
    pub max_depth: usize,
    pub min_pieces: usize,
    /// How many to keep of the attacker's positions, and of the defender's.
    pub keep: usize,
    seen: [u64; 2],
    rng: u64,
    pub kept: [Vec<(Position, usize)>; 2],
}

impl Sampler {
    pub fn new(max_depth: usize, min_pieces: usize, keep: usize, seed: u64) -> Sampler {
        Sampler {
            max_depth,
            min_pieces,
            keep,
            seen: [0; 2],
            rng: seed | 1,
            kept: [Vec::new(), Vec::new()],
        }
    }

    /// Reservoir sampling, for each side apart.
    fn offer(&mut self, pos: &Position, attacking: bool, depth: usize) {
        if depth > self.max_depth || pos.occupied().count() < self.min_pieces {
            return;
        }
        let side = usize::from(attacking);
        self.seen[side] += 1;
        if self.kept[side].len() < self.keep {
            self.kept[side].push((pos.clone(), depth));
            return;
        }
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let slot = (self.rng % self.seen[side]) as usize;
        if slot < self.keep {
            self.kept[side][slot] = (pos.clone(), depth);
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Numbers {
    proof: u32,
    disproof: u32,
}

const WON: Numbers = Numbers {
    proof: 0,
    disproof: INFINITY,
};
const LOST: Numbers = Numbers {
    proof: INFINITY,
    disproof: 0,
};

/// What the search knows about a position on the current line.
#[derive(Clone, Copy)]
struct Value {
    numbers: Numbers,
    /// For a refutation: the depth on the line of the earliest repeated
    /// position it relies on, or `ANY_LINE`.
    cycle: u32,
    /// For a refutation: whether it relies on the 50-move limit.
    fifty: bool,
}

impl Value {
    const WON: Value = Value::open(WON);
    const DRAWN: Value = Value::open(LOST);

    const fn open(numbers: Numbers) -> Value {
        Value {
            numbers,
            cycle: ANY_LINE,
            fifty: false,
        }
    }
}

struct Child {
    m: Move,
    hash: u64,
    counter: u32,
    value: Value,
    /// The move exploded the defender's king.
    exploded: bool,
    /// A defence after which the attacker explodes the king at once.
    winning: bool,
    /// How often the move proved a position, attacking, or refuted one,
    /// defending: the history heuristic, which breaks ties.
    score: u32,
    /// A defence a sibling's proof has been replayed on.
    simulated: bool,
    /// Starting numbers, which the parent's evaluation scales, rather than
    /// known ones.
    fresh: bool,
    /// Not a move: the attacker's plain moves together, scored only when the
    /// search first wants one of them.
    group: bool,
}

/// What `Dfpn::shorten` found: the shortest win's length, the passes it ran,
/// and whether that length is known to be the shortest, by the search's
/// rules.
pub struct Shortening {
    pub length: u32,
    pub passes: u32,
    pub shortest: bool,
}

/// What a search keeps of what it learns, its table and its proofs, with
/// what the threads that search a position together share about how the
/// search goes (`try_with_threads`).
struct Store {
    table: Table,
    /// The proven positions, in a fixed amount of memory. A proof that comes
    /// out the same whenever it is worked out again (`recomputed`) is not
    /// kept: the attacker exploding the king at once, the defender mated, or
    /// a defence after which every move lets the king be exploded. The search
    /// spots those whenever it scores the move leading there, and proofs
    /// leave out the explosions for the checkers to find. Once the proofs
    /// first fill, a mate in one is not kept either (`drop_mates`).
    proofs: Proofs,
    /// Whether the proofs keep mates in one: until they first fill, when they
    /// drop them so that the search goes on (`drop_mates`).
    mates_kept: AtomicBool,
    /// How many threads search together: one for a search alone, which needs
    /// none of the rest.
    threads: usize,
    /// The nodes the threads have searched, as each adds its own now and then.
    nodes: AtomicU64,
    /// Stops every thread at its next look at the clock: set when one settles
    /// the root, or the budget, the deadline, the stop flag or the proofs end
    /// the search.
    stop: AtomicBool,
    /// Whether the proofs filled their memory, which stops the search.
    full: AtomicBool,
    /// The nodes searched when the proofs dropped their mates in one.
    mates_dropped: Mutex<Option<u64>>,
    /// The root's numbers as the thread that settled it left them.
    settled: Mutex<Option<Numbers>>,
    /// Lets a thread drop the mates in one, or rebuild or drop the table's
    /// filter, while the others wait, as these move what they read.
    pause: Pause,
    /// Together: how many threads search below each position, by its hash
    /// (`crowded`); none for a search alone.
    busy: Vec<AtomicU16>,
}

/// A thread's part in a search together, from before it starts: when it
/// ends, the pause no longer waits for it, and if it panicked, the others
/// stop too, and the search fails with it once they end.
struct Part<'s>(&'s Store);

impl<'s> Part<'s> {
    fn new(store: &'s Store) -> Part<'s> {
        store.pause.enter();
        Part(store)
    }
}

impl Drop for Part<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.stop.store(true, Relaxed);
        }
        self.0.pause.leave();
    }
}

/// What a thread that helps a search takes from it: all but its own state.
#[derive(Clone, Copy)]
struct Settings<'a> {
    attacker: Color,
    tablebase: Option<&'a Tablebase>,
    guide: Guide<'a>,
    deadline: Instant,
    stop: Option<&'a AtomicBool>,
    limit: u64,
    drop_mates_at: usize,
    most: Option<u32>,
    root_attacking: bool,
}

pub struct Dfpn<'a> {
    attacker: Color,
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
    /// The table and the proofs, which other threads may share.
    store: Arc<Store>,
    /// Whether other threads search with this one, sharing its store.
    together: bool,
    /// Together: the nodes this thread has added to the store's, and the
    /// entries it has stored in the table since it last added them (`flush`).
    flushed: u64,
    puts: usize,
    /// Whether the table's filter is read, as this thread last took it from
    /// the table (`Table::filtering`): after each change it made, and after
    /// waiting while another thread made one.
    filtering: bool,
    path: Vec<u64>,
    /// How often each attacker move, by piece and destination, proved a
    /// position, and how often each defence refuted one.
    attack_history: Vec<u32>,
    defense_history: Vec<u32>,

    /// Each depth's children, kept between expansions to spare allocations.
    buffers: Vec<Vec<Child>>,
    /// The legal moves being scored, the expanded position's or its passed
    /// position's, with their effects and whether the filter may hold their
    /// children.
    scratch: Vec<(Move, Effect, bool)>,
    /// The line of a replay (`simulate`), kept between replays.
    line: Vec<u64>,
    /// One bit, by hash, for each position on the line that the moves being
    /// scored could repeat (`mark_repetitions`): a clear bit spares
    /// `repeated` its walk back along the line.
    repetitions: [u64; 16],
    deadline: Instant,
    /// Set from outside to stop the search, as its deadline does.
    stop: Option<&'a AtomicBool>,
    /// The node budget, checked at every node so that it repeats exactly.
    limit: u64,
    /// The proofs filled their memory, which stopped the search.
    full: bool,
    /// The proofs taken at which they drop their mates in one, if before they
    /// fill: for tests of the search after.
    drop_mates_at: usize,
    /// When and how to print progress lines, if at all.
    progress: Option<Progress>,
    /// The root's proof and disproof numbers as the search last combined them.
    root: (u32, u32),
    /// Whether the search keeps its line and its root moves' nodes: while it
    /// reports, or when asked (`keep_line`).
    keeping: bool,
    /// The first `LINE` plies of the search's line, `line_len` of them
    /// current, from `line_root` (`Level`).
    levels: [Level; LINE],
    line_len: usize,
    line_root: Option<Position>,
    /// This thread's nodes below each root move searched.
    root_moves: Vec<(Move, u64)>,
    /// The highest root proof number a progress line has reported.
    peak: u32,
    /// The perturbation's seed, spread over all bits; 0 for none.
    seed: u64,
    /// The most attacker moves a win may take, if limited.
    most: Option<u32>,
    /// Whether the attacker is to move at the root, which tells the line's
    /// attacker moves from its plies.
    root_attacking: bool,
    pub nodes: u64,
    pub stopped: bool,
    /// The nodes searched when the proofs dropped their mates in one.
    pub mates_dropped: Option<u64>,
    /// The root's proof and disproof numbers when the search last stopped.
    pub root_numbers: Option<(u32, u32)>,
    /// Expanded positions kept for labelling, when sampling.
    pub sampler: Option<Sampler>,
    /// Whether to keep the features of the next expanded position, once.
    pub record: bool,
    pub recorded: Option<[f32; FEATURES]>,
}

impl<'a> Dfpn<'a> {
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
        (table_mb, proofs_mb): (usize, usize),
        threads: usize,
    ) -> Result<Dfpn<'a>, TryReserveError> {
        let threads = threads.max(1);
        let shared = threads > 1;
        let store = Store {
            table: Table::new(table_mb, shared)?,
            proofs: Proofs::new(proofs_mb, shared)?,
            mates_kept: AtomicBool::new(true),
            threads,
            nodes: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            full: AtomicBool::new(false),
            mates_dropped: Mutex::new(None),
            settled: Mutex::new(None),
            pause: Pause::new(),
            busy: if shared { busy_slots()? } else { Vec::new() },
        };
        let settings = Settings {
            attacker: shakmaty::Position::turn(root),
            tablebase,
            guide,
            deadline,
            stop: None,
            limit: u64::MAX,
            drop_mates_at: usize::MAX,
            most: None,
            root_attacking: true,
        };
        Ok(Dfpn::with_store(Arc::new(store), settings))
    }

    /// A search with these settings and a state of its own, which keeps what
    /// it learns in `store`.
    fn with_store(store: Arc<Store>, settings: Settings<'a>) -> Dfpn<'a> {
        Dfpn {
            attacker: settings.attacker,
            tablebase: settings.tablebase,
            guide: settings.guide,
            frames: Vec::new(),
            refresh_cache: None,
            scratch_frame: Box::default(),
            scratch_transformed: Box::default(),
            transformed: Box::default(),
            together: store.threads > 1,
            filtering: store.table.filtering(),
            store,
            flushed: 0,
            puts: 0,
            path: Vec::new(),
            attack_history: vec![0; 7 * 64],
            defense_history: vec![0; 7 * 64],

            buffers: Vec::new(),
            scratch: Vec::new(),
            line: Vec::new(),
            repetitions: [0; 16],
            deadline: settings.deadline,
            stop: settings.stop,
            limit: settings.limit,
            full: false,
            drop_mates_at: settings.drop_mates_at,
            progress: None,
            root: (1, 1),
            keeping: false,
            levels: [Level::default(); LINE],
            line_len: 0,
            line_root: None,
            root_moves: Vec::new(),
            peak: 0,
            seed: 0,
            most: settings.most,
            root_attacking: settings.root_attacking,
            nodes: 0,
            stopped: false,
            mates_dropped: None,
            root_numbers: None,
            sampler: None,
            record: false,
            recorded: None,
        }
    }

    /// What a thread that helps this search takes from it.
    fn settings(&self) -> Settings<'a> {
        Settings {
            attacker: self.attacker,
            tablebase: self.tablebase,
            guide: self.guide,
            deadline: self.deadline,
            stop: self.stop,
            limit: self.limit,
            drop_mates_at: self.drop_mates_at,
            most: self.most,
            root_attacking: self.root_attacking,
        }
    }

    /// Searches for a win of `attacker`, who need not be the side to move.
    pub fn set_attacker(&mut self, attacker: Color) {
        self.attacker = attacker;
    }

    /// Perturbs the search: each position's step moves by up to 1% either
    /// way, by the position and the seed (a model's through its odds). 0
    /// leaves it alone.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed.wrapping_mul(SEED_SPREAD);
    }

    /// Stops the search after `nodes` nodes as well as at the deadline: unlike
    /// a time limit, a node budget gives the same search on any machine.
    pub fn set_limit(&mut self, nodes: u64) {
        self.limit = nodes;
    }

    /// Stops the search at `deadline` instead of the one it was made with.
    pub fn set_deadline(&mut self, deadline: Instant) {
        self.deadline = deadline;
    }

    /// Stops the search once `stop` is set, as at the deadline: for a search
    /// that another thread ends.
    pub fn set_stop(&mut self, stop: &'a AtomicBool) {
        self.stop = Some(stop);
    }

    /// Accepts only wins in at most `moves` attacker moves.
    pub fn set_most(&mut self, moves: u32) {
        self.most = Some(moves);
    }

    /// Prints a progress line every `every` (see `report`), keeping the
    /// search's line for it.
    pub fn set_progress(&mut self, every: Duration) {
        let now = Instant::now();
        self.keeping = true;
        self.progress = Some(Progress {
            every,
            next: now + every,
            started: now,
            last: (now, self.nodes),
            sink: None,
        });
    }

    /// Hands each progress line to `sink` instead of printing it: the
    /// browser build passes them to its page.
    pub fn set_progress_sink(&mut self, every: Duration, sink: ProgressSink) {
        self.set_progress(every);
        if let Some(progress) = &mut self.progress {
            progress.sink = Some(sink);
        }
    }

    /// Keeps the search's line without reporting it, for `line` once the
    /// search stops.
    pub fn keep_line(&mut self) {
        self.keeping = true;
    }

    /// How many attacker moves the proven win takes, if one is found in time.
    pub fn solve(&mut self, root: &Atomic) -> Option<u32> {
        self.line_len = 0;
        self.root_moves.clear();
        self.peak = 0;
        // The search settles a position that has ended before reaching it;
        // a root that has ended, such as a sampled defender position, is
        // settled here, for the attacker as set.
        match status(root, self.attacker) {
            Status::Won => {
                self.add_proof(Position::from_atomic(root).hash(), Proof::FINISHED);
                self.root_numbers = Some((0, INFINITY));
                return Some(0);
            }
            Status::Over => {
                self.root_numbers = Some((INFINITY, 0));
                return None;
            }
            Status::Open => {}
        }
        self.root_attacking = shakmaty::Position::turn(root) == self.attacker;
        let root = Position::from_atomic(root);
        self.line_root = Some(root.clone());
        let numbers = if self.together {
            self.search_together(&root)
        } else {
            self.mid(&root, INFINITY, INFINITY).numbers
        };
        self.root_numbers = Some((numbers.proof, numbers.disproof));
        // Once the proofs are full, a win may rest on a proof never kept.
        if numbers.proof != 0 || self.full {
            return None;
        }
        self.proof_at(root.hash(), root.halfmoves())
            .or_else(|| self.recomputed(&root))
            .map(|proof| proof.length)
    }

    /// Searches the root with the other threads until one of them settles it
    /// or they stop: this thread with its own seed, each other with another.
    /// Returns the root's numbers as the thread that settled it left them,
    /// else as this one did.
    fn search_together(&mut self, root: &Position) -> Numbers {
        let store = Arc::clone(&self.store);
        store.nodes.store(self.nodes, Relaxed);
        store.stop.store(false, Relaxed);
        *lock(&store.settled) = None;
        self.flushed = self.nodes;
        self.puts = 0;
        let settings = self.settings();
        let numbers = std::thread::scope(|scope| {
            for thread in 1..store.threads {
                let part = Part::new(&store);
                let (shared, seed) = (Arc::clone(&store), self.seed);
                let helper = std::thread::Builder::new().stack_size(crate::STACK);
                // A thread that cannot start drops its part: the others search
                // without it.
                let _ = helper.spawn_scoped(scope, move || {
                    let _part = part;
                    let mut search = Dfpn::with_store(shared, settings);
                    search.seed = seed.wrapping_add(SEED_SPREAD.wrapping_mul(thread as u64));
                    search.search_root(root)
                });
            }
            let _part = Part::new(&store);
            self.search_root(root)
        });
        // The others may have changed the filter after this thread left.
        self.filtering = store.table.filtering();
        self.nodes = store.nodes.load(Relaxed);
        self.full = store.full.load(Relaxed);
        if self.mates_dropped.is_none() {
            self.mates_dropped = *lock(&store.mates_dropped);
        }
        let settled = lock(&store.settled).take();
        self.stopped = settled.is_none();
        settled.unwrap_or(numbers)
    }

    /// One thread's search of the root among others: once it settles the
    /// root, they stop.
    fn search_root(&mut self, root: &Position) -> Numbers {
        let numbers = self.mid(root, INFINITY, INFINITY).numbers;
        self.flush();
        if numbers.proof == 0 || numbers.disproof == 0 {
            lock(&self.store.settled).get_or_insert(numbers);
            self.store.stop.store(true, Relaxed);
        }
        numbers
    }

    /// Together: adds this thread's nodes since it last did to the store's,
    /// and its entries to the table's count of them, and returns every
    /// thread's nodes.
    fn flush(&mut self) -> u64 {
        let new = self.nodes - self.flushed;
        self.flushed = self.nodes;
        let puts = std::mem::take(&mut self.puts);
        self.table().count(puts);
        self.store.nodes.fetch_add(new, Relaxed) + new
    }

    /// After a win of `length`, searches again with a limit one below the
    /// last win's length, so that the tables and proofs carry over: a
    /// position with so many moves left is the same problem whatever the
    /// root's limit, and stored proofs count wherever they fit. Calls
    /// `improved` after each shorter win, with the search and its length.
    /// Stops when the budget ends, the proofs fill, or a pass settles that no
    /// shorter win exists.
    pub fn shorten(
        &mut self,
        root: &Atomic,
        mut length: u32,
        mut improved: impl FnMut(&Self, u32),
    ) -> Shortening {
        let mut passes = 0;
        while length > 1 {
            self.set_most(length - 1);
            passes += 1;
            match self.solve(root) {
                Some(shorter) => {
                    length = shorter;
                    improved(self, shorter);
                }
                None => {
                    let shortest = !self.stopped
                        && !self.full
                        && self.root_numbers.is_some_and(|(_, disproof)| disproof == 0);
                    return Shortening {
                        length,
                        passes,
                        shortest,
                    };
                }
            }
        }
        // No win is shorter than one move.
        Shortening {
            length,
            passes,
            shortest: true,
        }
    }

    /// The winning move for an attacker position, for reading the proof back:
    /// once the proofs have dropped their mates in one, a mate in one found
    /// again.
    pub fn winning_move(&self, pos: &Atomic, key: &Key) -> Option<shakmaty::Move> {
        let pos = Position::from_atomic(pos);
        let m = match self.proof_at(key.0, key.1) {
            Some(proof) => proof.m?,
            None => self.dropped_mate(&pos)?,
        };
        Some(pos.to_shakmaty(m))
    }

    /// Has the proofs drop their mates in one once they hold this share of
    /// their room, rather than when they fill: for tests of the search after.
    pub fn set_drop_mates_at(&mut self, share: f64) {
        self.drop_mates_at = (self.proofs().room() as f64 * share) as usize;
        if self.drop_mates_at == 0 {
            self.drop_mates();
        }
    }

    /// Whether the search stopped because its proofs filled their memory.
    pub fn proofs_full(&self) -> bool {
        self.full
    }

    fn mid(&mut self, pos: &Position, proof_limit: u32, disproof_limit: u32) -> Value {
        let start = self.nodes;
        let hash = pos.hash();
        let counter = pos.halfmoves();
        let remaining = self.remaining(self.path.len());
        // A transposition may have settled the position, or taken it past the
        // limits, since its parent looked.
        if self.proof_within(hash, counter, remaining).is_some() {
            return Value::WON;
        }
        // Out of attacker moves, a game that goes on is lost to the limit.
        if remaining == 0 {
            return Value::DRAWN;
        }
        let key = self.key(hash, remaining);
        let known = self.lookup(key, counter, |m| self.mates_with(pos, m));
        // A mate in one the table notes comes back won, its disproof number
        // past every limit.
        if let Some(value) = known {
            let numbers = value.numbers;
            if numbers.disproof == 0
                || numbers.proof >= proof_limit
                || numbers.disproof >= disproof_limit
            {
                return value;
            }
        }
        self.tick();
        let attacking = pos.turn() == self.attacker;
        let level = self.path.len();
        if self.buffers.len() <= level {
            self.buffers.resize_with(level + 1, Vec::new);
        }
        let mut children = std::mem::take(&mut self.buffers[level]);
        let step = self.children(pos, attacking, remaining, &mut children);
        if !attacking && known.is_none() && !children.is_empty() && !pos.in_check() {
            self.replay_threat(pos, remaining, &mut children);
        }
        if let Some(sampler) = &mut self.sampler {
            sampler.offer(pos, attacking, self.path.len());
        }
        if children.is_empty() {
            self.buffers[level] = children;
            // A mate is settled before the search reaches it, except at the
            // root; anything else is a stalemate, a draw.
            let value = if pos.in_check() {
                self.mated(pos).unwrap_or(Value::DRAWN)
            } else {
                Value::DRAWN
            };
            return self.settle(key, counter, value, 1);
        }
        let depth = self.path.len() as u32;
        self.path.push(hash);
        let mut opened = None;
        // Together: this thread's nodes when it last read the children.
        let mut read = self.nodes;
        let (value, best) = loop {
            let (value, best, second) = if !self.together {
                combine(&children, attacking, |_, deciding| deciding)
            } else {
                combine(&children, attacking, |child, deciding| {
                    self.crowded(child, deciding)
                })
            };
            let numbers = value.numbers;
            if depth == 0 {
                self.root = (numbers.proof, numbers.disproof);
            }
            if numbers.proof == 0
                || numbers.disproof == 0
                || numbers.proof >= proof_limit
                || numbers.disproof >= disproof_limit
                || self.stopped
            {
                break (value, best);
            }
            if children[best].group {
                children.swap_remove(best);
                opened = Some(children.len());
                self.path.pop();
                self.open_group(pos, remaining, &mut children, step);
                self.path.push(hash);
                continue;
            }
            let current = children[best].value.numbers;
            let (child_proof, child_disproof) = if attacking {
                (
                    proof_limit.min(widen(second)),
                    share(disproof_limit, numbers.disproof, current.disproof),
                )
            } else {
                (
                    share(proof_limit, numbers.proof, current.proof),
                    disproof_limit.min(widen(second)),
                )
            };
            if self.keeping && (depth as usize) < LINE {
                self.keep_level(depth as usize, attacking, &children, best);
            }
            let next = pos.play(children[best].m);
            debug_assert_eq!(next.hash(), children[best].hash);
            // Together, the other threads see this one below the child while
            // it searches there.
            let busy = self.together.then(|| busy_slot(children[best].hash));
            if let Some(slot) = busy {
                self.store.busy[slot].fetch_add(1, Relaxed);
            }
            let before = self.nodes;
            let value = self.mid(&next, child_proof, child_disproof);
            if let Some(slot) = busy {
                self.store.busy[slot].fetch_sub(1, Relaxed);
            }
            if self.keeping && depth == 0 {
                self.searched(children[best].m, self.nodes - before);
            }
            children[best].value = value;
            // A defence just proven may prove its siblings too: replay its
            // proof on each defence still open.
            if !attacking && value.numbers.proof == 0 {
                for (index, sibling) in children.iter_mut().enumerate() {
                    let open =
                        sibling.value.numbers.proof != 0 && sibling.value.numbers.disproof != 0;
                    if index == best || sibling.simulated || !open {
                        continue;
                    }
                    sibling.simulated = true;
                    let position = pos.play(sibling.m);
                    let mut left = SIMULATION;
                    let mut line = std::mem::take(&mut self.line);
                    let reached = self.simulate(&position, &next, remaining, &mut left, &mut line);
                    self.line = line;
                    if reached.is_some_and(|(length, _)| length <= remaining) {
                        debug_assert!(
                            self.full
                                || self.proof_at(sibling.hash, sibling.counter).is_some()
                                || self.recomputed(&position).is_some()
                        );
                        sibling.value = Value::WON;
                    }
                }
            }
            if self.together && !self.stopped && self.nodes - read >= REFRESH_AFTER {
                read = self.nodes;
                self.refresh(pos, &mut children);
            }
        };
        self.path.pop();
        if !attacking && value.numbers.disproof == 0 {
            let index = history_index(pos, children[best].m);
            self.defense_history[index] = self.defense_history[index].saturating_add(1);
        }
        // Repeating this position, or one after it, refutes it on any line.
        let value = if value.cycle >= depth {
            Value {
                cycle: ANY_LINE,
                ..value
            }
        } else {
            value
        };
        if let Some(first) = opened
            && value.numbers.proof != 0
            && !self.stopped
        {
            let plain = &children[first..];
            let proof = plain.iter().map(|child| child.value.numbers.proof).min();
            let disproof = plain
                .iter()
                .map(|child| child.value.numbers.disproof)
                .fold(0, u32::saturating_add);
            if let Some(proof) = proof
                && proof != 0
                && proof < INFINITY
                && disproof != 0
            {
                let numbers = Numbers {
                    proof,
                    disproof: disproof.min(INFINITY - 1),
                };
                self.store(
                    key ^ GROUP,
                    counter,
                    Value::open(numbers),
                    self.nodes - start,
                );
            }
        }
        if value.numbers.proof == 0 {
            self.record_proof(pos, hash, &children, attacking);
            debug_assert!(
                self.full
                    || self.proof_at(hash, counter).is_some()
                    || self.recomputed(pos).is_some(),
                "a win without a proof"
            );
        } else if !self.stopped {
            self.store(key, counter, value, self.nodes - start);
        }
        self.buffers[level] = children;
        value
    }

    /// Together: what the store now knows of the open children, which other
    /// threads may have searched since this one looked, as `mid` would find
    /// it on reaching each, with this position on the line.
    fn refresh(&self, pos: &Position, children: &mut [Child]) {
        let remaining = self.remaining(self.path.len());
        let filtering = self.filtering;
        let open =
            |child: &Child| child.value.numbers.proof != 0 && child.value.numbers.disproof != 0;
        let moves = |child: &&Child| open(child) && !child.group;
        // The reads started first, as `children` starts them: the filter's
        // words, then the entries of the children it may hold.
        for child in children.iter().filter(moves) {
            self.prefetch_child(child.hash, remaining, filtering);
        }
        if filtering {
            for child in children.iter().filter(moves) {
                let key = self.key(child.hash, remaining);
                if self.seen(child.hash, key) {
                    self.table().prefetch(key);
                    self.proofs().prefetch(child.hash);
                }
            }
        }
        for child in children.iter_mut().filter(|child| open(child)) {
            let known = if child.group {
                self.lookup(child.hash, child.counter, |_| false)
                    .filter(|value| value.numbers.disproof != 0)
            } else if self
                .proof_within(child.hash, child.counter, remaining)
                .is_some()
            {
                Some(Value::WON)
            } else {
                let key = self.key(child.hash, remaining);
                self.lookup(key, child.counter, |mate| {
                    self.mates_with(&pos.play(child.m), mate)
                })
            };
            if let Some(value) = known {
                child.value = value;
                child.fresh = false;
            }
        }
    }

    /// Together: a child's deciding number as this thread sees it, grown for
    /// each other thread searching below it (`crowd`). The plain moves'
    /// group is never searched below, so none is.
    fn crowded(&self, child: &Child, deciding: u32) -> u32 {
        if child.group {
            return deciding;
        }
        let busy = self.store.busy[busy_slot(child.hash)].load(Relaxed);
        crowd(deciding, busy)
    }

    /// Scores a position's moves into `children`, with `remaining` attacker
    /// moves left, and returns the step its evaluation scaled their starting
    /// numbers by.
    fn children(
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
        let mut legal = std::mem::take(&mut self.scratch);
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
        self.scratch = legal;
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
    /// checks, so it cannot and is left out, and every other child is decided
    /// where it is scored, so nothing needs the position's evaluation. Apart
    /// from `children`, so that searches without a limit run the same code.
    #[inline(never)]
    fn last_children(&mut self, pos: &Position, mover: &Mover, children: &mut Vec<Child>) {
        let mut legal = std::mem::take(&mut self.scratch);
        legal.clear();
        let filtering = self.filtering;
        for &m in pos.pseudo_legal().as_slice() {
            if mover.plain(m) || !mover.legal(m) {
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
        self.scratch = legal;
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
            .guide
            .network()
            .map(|network| self.network_value(network, pos));
        if self.record {
            // The first position expanded: the root.
            self.recorded = Some(self.features(pos, attacking, children, plain, value));
            self.record = false;
        }
        let step = match self.guide {
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
                    .map(|_| &*self.transformed);
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
        if matches!(self.guide, Guide::Model(..)) {
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
        if self.frames.len() <= level {
            self.frames.resize_with(level + 1, Accumulator::default);
        }
        let (parents, frames) = self.frames.split_at_mut(level);
        let frame = &mut frames[0];
        match parents.last() {
            Some(parent) => {
                let cache = self
                    .refresh_cache
                    .get_or_insert_with(|| RefreshCache::new(network));
                network.update(parent, pos, frame, cache);
            }
            None => network.refresh(pos, frame),
        }
        network.transform(frame, pos, &mut self.transformed);
        network.output_from(&self.transformed, frame, pos)
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
        let mut plain = std::mem::take(&mut self.scratch);
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
        self.scratch = plain;
        scale(&mut children[first..], true, step);
        self.grade(pos, &mut children[first..]);
    }

    /// Raises the starting numbers of the plain moves the network likes less
    /// than its best `GRADED`, by its evaluation after each.
    fn grade(&mut self, pos: &Position, plain: &mut [Child]) {
        let Some(network) = self.guide.network() else {
            return;
        };
        // The expanded position's first layer, which `network_value` computed
        // when this visit expanded it.
        let level = self.path.len();
        if self.frames.len() <= level || plain.len() <= GRADED {
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
                &self.frames[level],
                m.from(),
                m.to(),
                &mut self.scratch_frame,
            );
            network.transform_for(&self.scratch_frame, them, &mut self.scratch_transformed);
            // The defender moves next: the attacker's view is the negation.
            let value =
                -network.output_for(&self.scratch_transformed, &self.scratch_frame, pieces, them);
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
            &self.attack_history
        } else {
            &self.defense_history
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
        if let Some(value) = known
            && value.numbers.disproof == 0
        {
            return child(value, false);
        }
        if counter == 0
            && let Some(tables) = self.tablebase
            && effect.occupied.count() <= tables.max_pieces()
            && let Some(verdict) = tables.probe(&pos.play(m).to_atomic())
        {
            let attacker_wins = if pos.turn() == self.attacker {
                verdict == Verdict::Loss
            } else {
                verdict == Verdict::Win
            };
            if !attacker_wins {
                self.store(key, counter, Value::DRAWN, 0);
                return child(Value::DRAWN, false);
            }
            if self.most.is_none() {
                self.add_proof(hash, Proof::TABLEBASE);
                return child(Value::WON, false);
            }
        }
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

    /// The depth of a child's position on the current line, if it repeats
    /// one. The line ends with the child's parent's parent, and only positions
    /// since the last capture or pawn move can recur, with the same side to
    /// move: an even number of plies back.
    fn repeated(&self, hash: u64, counter: u32) -> Option<u32> {
        let bit = (hash >> 54) as usize;
        if self.repetitions[bit >> 6] & (1 << (bit & 63)) == 0 {
            return None;
        }
        let len = self.path.len();
        let back = (counter as usize).min(len + 1);
        (2..=back)
            .step_by(2)
            .map(|plies| len + 1 - plies)
            .find(|&depth| self.path[depth] == hash)
            .map(|depth| depth as u32)
    }

    /// Sets the bits `repeated` reads for the moves of a position whose
    /// 50-move counter is `counter`: their children's counters are at most one
    /// more, so only the line's positions that many plies back, with the
    /// children's side to move, could repeat.
    fn mark_repetitions(&mut self, counter: u32) {
        self.repetitions = [0; 16];
        let len = self.path.len();
        let back = (counter as usize + 1).min(len + 1);
        for plies in (2..=back).step_by(2) {
            let bit = (self.path[len + 1 - plies] >> 54) as usize;
            self.repetitions[bit >> 6] |= 1 << (bit & 63);
        }
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
    /// lets the king be exploded; and once the proofs have dropped them, a
    /// mate in one.
    fn recomputed(&self, pos: &Position) -> Option<Proof> {
        if pos.turn() == self.attacker {
            let m = self.explosion(pos).or_else(|| self.dropped_mate(pos))?;
            return Some(Proof {
                m: Some(m),
                length: 1,
                limit: LAST,
            });
        }
        if pos.in_check() && !moves::has_legal_move(pos) {
            return Some(Proof::FINISHED);
        }
        self.exposed(pos)
    }

    /// A check the defender has no move against, before the 50-move limit.
    fn mate_in_one(&self, pos: &Position) -> Option<Move> {
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

    /// A mate in one, once the proofs have dropped them and keep none.
    fn dropped_mate(&self, pos: &Position) -> Option<Move> {
        if self.mates_kept() {
            None
        } else {
            self.mate_in_one(pos)
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
        let mate = if mated {
            proof.m
        } else {
            self.mate_in_one(pos)
        };
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
    fn replay_threat(&mut self, pos: &Position, remaining: u32, children: &mut [Child]) {
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
            let mut moves = std::mem::take(&mut self.scratch);
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
            self.scratch = moves;
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
            let mut line = std::mem::take(&mut self.line);
            let reached = self.simulate(&position, &passed, remaining, &mut budget, &mut line);
            self.line = line;
            if reached.is_some_and(|(length, _)| length <= remaining) {
                child.value = Value::WON;
                child.simulated = true;
            } else if child.fresh {
                child.value.numbers.proof = (child.value.numbers.proof / 2).max(1);
            }
        }
    }

    /// Whether a position is on the search's line or the replay's.
    fn on_line(&self, hash: u64, line: &[u64]) -> bool {
        self.path.contains(&hash) || line.contains(&hash)
    }

    /// Remembers a finished game that is not won, as a refutation: a won one
    /// is the defender mated, which `recomputed` finds again.
    fn settle(&mut self, hash: u64, counter: u32, value: Value, work: u64) -> Value {
        if value.numbers.proof != 0 {
            self.store(hash, counter, value, work);
        }
        value
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

    /// The attacker moves left `depth` plies from the root.
    fn remaining(&self, depth: usize) -> u32 {
        match self.most {
            None => u32::MAX,
            Some(most) => {
                let made = (depth + usize::from(self.root_attacking)) / 2;
                most.saturating_sub(u32::try_from(made).unwrap_or(u32::MAX))
            }
        }
    }

    /// A position's key in the table with `remaining` attacker moves left:
    /// its hash, apart for each number of moves left under a limit.
    fn key(&self, hash: u64, remaining: u32) -> u64 {
        match self.most {
            None => hash,
            Some(_) => hash ^ (u64::from(remaining) + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        }
    }

    /// Whether the proofs or the table may hold a child: its hash for the
    /// proofs, its key for the table.
    fn seen(&self, hash: u64, key: u64) -> bool {
        self.table().may_hold(hash, self.filtering)
            || (key != hash && self.table().may_hold(key, self.filtering))
    }

    /// Starts loading what a child's scoring reads: its filter words while the
    /// filter is read, else its table cluster and proof slot at once.
    fn prefetch_child(&self, hash: u64, remaining: u32, filtering: bool) {
        if filtering {
            self.prefetch_seen(hash, remaining);
        } else {
            self.table().prefetch(self.key(hash, remaining));
            self.proofs().prefetch(hash);
        }
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

    /// Starts loading the filter's words for a child: for its hash, and for
    /// its key under a limit on moves.
    fn prefetch_seen(&self, hash: u64, remaining: u32) {
        self.table().prefetch_filter(hash);
        let key = self.key(hash, remaining);
        if key != hash {
            self.table().prefetch_filter(key);
        }
    }

    /// The shortest proof of a position that holds at this counter, if it
    /// takes at most `remaining` attacker moves.
    fn proof_within(&self, hash: u64, counter: u32, remaining: u32) -> Option<Proof> {
        self.proof_at(hash, counter)
            .filter(|proof| proof.length <= remaining)
    }

    /// What the table knows about a position at this counter on this line: a
    /// mate in one only where `mates` finds that its move mates.
    fn lookup(&self, hash: u64, counter: u32, mates: impl FnOnce(Move) -> bool) -> Option<Value> {
        match self.table().get(hash, self.filtering)? {
            Known::Mate(bits) => mates(Move::from_bits(bits)).then_some(Value::WON),
            Known::Open { proof, disproof } => Some(Value::open(Numbers { proof, disproof })),
            Known::Refuted { cycle, from } => {
                if counter < from {
                    return None;
                }
                let cycle = match cycle {
                    None => ANY_LINE,
                    Some(repeated) => self
                        .path
                        .iter()
                        .position(|&on_line| table::check(on_line) == repeated)?
                        as u32,
                };
                Some(Value {
                    numbers: LOST,
                    cycle,
                    fifty: from > 0,
                })
            }
        }
    }

    /// Stores a value with the work that found it, which decides how long
    /// the table keeps it. Out of line, with the table's `put` in it: the
    /// other way round costs the search's loops more instructions.
    #[inline(never)]
    fn store(&mut self, hash: u64, counter: u32, value: Value, work: u64) {
        let known = if value.numbers.disproof != 0 {
            Known::Open {
                proof: value.numbers.proof,
                disproof: value.numbers.disproof,
            }
        } else {
            Known::Refuted {
                cycle: (value.cycle != ANY_LINE)
                    .then(|| table::check(self.path[value.cycle as usize])),
                from: if value.fifty { counter } else { 0 },
            }
        };
        self.filtering = self.table().put(hash, known, work, self.filtering);
        self.puts += usize::from(self.together);
        self.refresh_filter();
    }

    /// Rebuilds the table's filter once it is stale; together, the threads
    /// count their entries now and then, and `tick_together` rebuilds it or
    /// drops it.
    #[inline(always)]
    fn refresh_filter(&mut self) {
        if !self.together && self.table().filter_is_stale() {
            self.filtering = self
                .table()
                .rebuild_filter(self.proofs().keys(), self.proofs().taken());
        }
    }

    /// Together: rebuilds the table's filter once it is stale, or drops it
    /// once it is dense, while the other threads wait, unless one did
    /// meanwhile; returns whether the filter is read.
    #[cold]
    fn change_filter_alone(&self) -> bool {
        let store = &*self.store;
        store.pause.alone(|| {
            if store.table.filter_is_stale() {
                store
                    .table
                    .rebuild_filter(store.proofs.keys(), store.proofs.taken());
            } else if store.table.filter_is_dense() {
                store.table.drop_filter();
            }
        });
        store.table.filtering()
    }

    /// The table, which other threads may share.
    fn table(&self) -> &Table {
        &self.store.table
    }

    /// The proofs, which other threads may share.
    fn proofs(&self) -> &Proofs {
        &self.store.proofs
    }

    /// Whether the proofs keep mates in one (`drop_mates`).
    fn mates_kept(&self) -> bool {
        self.store.mates_kept.load(Relaxed)
    }

    /// The shortest proof of a position that holds at this counter.
    fn proof_at(&self, hash: u64, counter: u32) -> Option<Proof> {
        if !self.table().may_hold(hash, self.filtering) {
            return None;
        }
        self.proofs().get(hash, counter)
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
            self.attack_history[index] = self.attack_history[index].saturating_add(1);
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
                self.add_proof(hash, proof);
            }
        }
    }

    /// Keeps a proof. When the proofs fill, the first time they drop their
    /// mates in one (`drop_mates`), and the second the search stops.
    fn add_proof(&mut self, hash: u64, proof: Proof) {
        self.filtering = self.table().note(hash, self.filtering);
        if self.proofs().add(hash, proof) {
            if self.mates_kept() && self.proofs().taken() >= self.drop_mates_at {
                self.drop_mates();
            }
            return;
        }
        if self.mates_kept() {
            self.drop_mates();
            if self.proofs().add(hash, proof) {
                return;
            }
        }
        self.full = true;
        self.stopped = true;
        if self.together {
            self.store.full.store(true, Relaxed);
            self.store.stop.store(true, Relaxed);
        }
    }

    /// Drops the mates in one from the proofs, so that the search goes on once
    /// they fill: from then on it keeps none (`keep_attacker`), and works each
    /// out again where it needs one (`recomputed`).
    fn drop_mates(&mut self) {
        if self.together {
            return self.drop_mates_alone();
        }
        self.store.mates_kept.store(false, Relaxed);
        self.proofs().drop_mates();
        self.mates_dropped = Some(self.nodes);
    }

    /// Together: `drop_mates` while the other threads wait, unless one did
    /// meanwhile, as they do when the proofs fill for them too.
    #[cold]
    fn drop_mates_alone(&mut self) {
        let nodes = self.flush();
        let store = &*self.store;
        store.pause.alone(|| {
            if store.mates_kept.load(Relaxed) {
                store.mates_kept.store(false, Relaxed);
                store.proofs.drop_mates();
                *lock(&store.mates_dropped) = Some(nodes);
            }
        });
        // Another thread's change may have come first.
        self.filtering = store.table.filtering();
        if self.mates_dropped.is_none() {
            self.mates_dropped = *lock(&store.mates_dropped);
        }
    }

    fn tick(&mut self) {
        self.nodes += 1;
        if self.together {
            return self.tick_together();
        }
        if self.nodes >= self.limit {
            self.stopped = true;
        } else if self.nodes.is_multiple_of(1024) {
            let now = Instant::now();
            if now >= self.deadline || self.stop.is_some_and(|stop| stop.load(Ordering::Relaxed)) {
                self.stopped = true;
            }
            if let Some(progress) = &mut self.progress
                && now >= progress.next
            {
                progress.next = now + progress.every;
                self.report(now);
            }
        }
    }

    /// `tick` for a thread among others: it waits while another changes the
    /// store alone, and every 1,024 nodes adds its nodes to the store's, stops
    /// every thread at the budget, the deadline or the stop flag, stops itself
    /// once another has, and rebuilds a stale filter or drops a dense one.
    fn tick_together(&mut self) {
        if self.store.pause.check() {
            self.filtering = self.table().filtering();
        }
        // By the nodes since the last flush, not a multiple of 1,024, which
        // the nodes that simulations count between ticks can step over.
        if self.nodes - self.flushed < 1024 {
            return;
        }
        let nodes = self.flush();
        let now = Instant::now();
        let store = &*self.store;
        if nodes >= self.limit
            || now >= self.deadline
            || self.stop.is_some_and(|stop| stop.load(Relaxed))
        {
            store.stop.store(true, Relaxed);
        }
        if store.stop.load(Relaxed) {
            self.stopped = true;
        }
        if store.table.filter_is_stale() || store.table.filter_is_dense() {
            self.filtering = self.change_filter_alone();
        }
        if let Some(progress) = &mut self.progress
            && now >= progress.next
        {
            progress.next = now + progress.every;
            self.report(now);
        }
    }

    /// Keeps a level of the search's line (`Level`), at `depth`, before the
    /// search descends into its chosen child.
    fn keep_level(&mut self, depth: usize, attacking: bool, children: &[Child], best: usize) {
        let mut level = Level {
            m: children[best].m,
            attacking,
            taken: children[best].value.numbers,
            least: INFINITY,
            sum: 0,
            open: 0,
        };
        for (index, child) in children.iter().enumerate() {
            let numbers = child.value.numbers;
            level.open += u32::from(numbers.proof != 0 && numbers.disproof != 0);
            if index == best {
                continue;
            }
            let (deciding, summed) = if attacking {
                (numbers.proof, numbers.disproof)
            } else {
                (numbers.disproof, numbers.proof)
            };
            level.least = level.least.min(deciding);
            level.sum = level.sum.saturating_add(summed);
        }
        self.levels[depth] = level;
        self.line_len = depth + 1;
    }

    /// Counts `nodes` searched below the root move `m`.
    fn searched(&mut self, m: Move, nodes: u64) {
        match self.root_moves.iter_mut().find(|(known, _)| *known == m) {
            Some((_, searched)) => *searched += nodes,
            None => self.root_moves.push((m, nodes)),
        }
    }

    /// The root's numbers now, rebuilt from the line: the deepest level's
    /// chosen child as it was chosen, and at each level above, the numbers
    /// below combined with the other children's, as `combine` does. Without
    /// a line, as the root last combined them.
    fn root_now(&self) -> (u32, u32) {
        let mut below: Option<Numbers> = None;
        for level in self.levels[..self.line_len].iter().rev() {
            let child = below.unwrap_or(level.taken);
            // A sum of open children's numbers stays short of infinity, which
            // would read as settled.
            let sum = |number: u32| level.sum.saturating_add(number).min(INFINITY - 1);
            below = Some(if level.attacking {
                Numbers {
                    proof: level.least.min(child.proof),
                    disproof: sum(child.disproof),
                }
            } else {
                Numbers {
                    proof: sum(child.proof),
                    disproof: level.least.min(child.disproof),
                }
            });
        }
        below.map_or(self.root, |numbers| (numbers.proof, numbers.disproof))
    }

    /// The search's line, its first `LINE` plies at most, in UCI's notation,
    /// with the moves still open where each was played: for the defender,
    /// the defences not yet beaten; for the attacker, the moves not yet
    /// refuted, its quiet moves counted as one until the search opens them
    /// (`open_group`). When the search has stopped, the line it was on.
    /// Empty unless the search keeps its line (`keep_line`).
    pub fn line(&self) -> (Vec<String>, Vec<u32>) {
        let levels = &self.levels[..self.line_len];
        let Some(mut pos) = self.line_root.clone() else {
            return (Vec::new(), Vec::new());
        };
        let mut moves = Vec::with_capacity(levels.len());
        for level in levels {
            moves.push(UciMove::from_standard(pos.to_shakmaty(level.m)).to_string());
            pos = pos.play(level.m);
        }
        (moves, levels.iter().map(|level| level.open).collect())
    }

    /// The root move this thread has searched the most nodes below, in UCI's
    /// notation.
    fn most_searched(&self) -> Option<String> {
        let root = self.line_root.as_ref()?;
        let &(m, _) = self
            .root_moves
            .iter()
            .rev()
            .max_by_key(|(_, nodes)| *nodes)?;
        Some(UciMove::from_standard(root.to_shakmaty(m)).to_string())
    }

    /// Prints a line on how the search goes, as JSON to standard error, or
    /// hands it to the sink: the seconds and nodes so far, the nodes a second
    /// since the last line, the root's proof and disproof numbers rebuilt
    /// from the line (`root_now`), the highest proof number reported so far
    /// (`peak`), the line and the moves open along it (`line`, `open`), the
    /// root move searched most (`most_searched`, once one is), the share of
    /// the table's slots taken (from a sample), how full the proofs are, 1
    /// being where they fill, and the nodes searched when they dropped their
    /// mates in one, once they have. Together, the line and the moves are
    /// this thread's.
    fn report(&mut self, now: Instant) {
        if self.progress.is_none() {
            return;
        }
        let (proof, disproof) = self.root_now();
        self.peak = self.peak.max(proof);
        let (line, open) = self.line();
        let most = self.most_searched();
        // Together, every thread's nodes, which `tick_together` has just
        // added up.
        let nodes = if self.together {
            self.store.nodes.load(Relaxed)
        } else {
            self.nodes
        };
        let table = (self.store.table.fill() * 1000.0).round() / 1000.0;
        let proofs = (self.store.proofs.fill() * 1000.0).round() / 1000.0;
        let dropped = self.mates_dropped;
        let peak = self.peak;
        let Some(progress) = &mut self.progress else {
            return;
        };
        let seconds = now.duration_since(progress.started).as_secs_f64();
        let recent = now.duration_since(progress.last.0).as_secs_f64();
        let rate = (nodes - progress.last.1) as f64 / recent.max(1e-9);
        progress.last = (now, nodes);
        let mut record = serde_json::json!({
            "seconds": seconds.round() as u64,
            "nodes": nodes,
            "rate": rate.round() as u64,
            "proof": proof,
            "disproof": disproof,
            "peak": peak,
            "line": line,
            "open": open,
            "table": table,
            "proofs": proofs,
        });
        if let Some(most) = most {
            record["most_searched"] = serde_json::json!(most);
        }
        if let Some(nodes) = dropped {
            record["mates_dropped"] = serde_json::json!(nodes);
        }
        match &mut progress.sink {
            Some(sink) => sink(&record),
            None => eprintln!("{record}"),
        }
    }
}

/// A lock's value, also after a thread panicked holding it.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Whether a child's proof, by its length and limit, is a finished game: for
/// a defender position, mated.
fn mated(length: u32, limit: u32) -> bool {
    length == Proof::FINISHED.length && limit == Proof::FINISHED.limit
}

impl Proof {
    /// A finished game: won at any counter.
    const FINISHED: Proof = Proof {
        m: None,
        length: 0,
        limit: u32::MAX,
    };
    /// A tablebase win, trusted only right after a capture or pawn move.
    const TABLEBASE: Proof = Proof {
        m: None,
        length: 0,
        limit: 0,
    };
}

/// How `--memory` divides between the table and the proofs: the proofs get
/// what the search's budget needs (`proofs_needed`), up to three quarters,
/// and at least what is left after the table's largest power of two up to two
/// thirds; the table gets the rest. Proofs that fill stop a search, once their
/// mates in one are dropped, while a full table costs little. Each takes at
/// most `MOST_ONE` megabytes.
pub fn split(memory_mb: usize, needed_mb: usize) -> (usize, usize) {
    let table = match memory_mb.saturating_mul(2) / 3 {
        0 => 0,
        most => (1 << most.ilog2()).min(MOST_ONE),
    };
    let proofs = needed_mb
        .min(memory_mb - memory_mb / 4)
        .max(memory_mb - table)
        .min(MOST_ONE);
    ((memory_mb - proofs).min(MOST_ONE), proofs)
}

/// The proofs' megabytes a search's budget needs, the less of what its time
/// limit and its nodes need.
pub fn proofs_needed(seconds: Option<f64>, nodes: Option<u64>) -> usize {
    let for_time = seconds.map(|seconds| (seconds * PROOF_MB_A_SECOND).ceil() as usize);
    let for_nodes =
        nodes.map(|nodes| nodes.saturating_mul(PROOF_BYTES_A_NODE).div_ceil(1 << 20) as usize);
    match (for_nodes, for_time) {
        (Some(nodes), Some(time)) => nodes.min(time),
        (nodes, time) => nodes.or(time).unwrap_or(0),
    }
}

/// Proof memory a search may need for each second of its time limit and each
/// node of its budget, at 12 bytes a proof.
const PROOF_MB_A_SECOND: f64 = 1.5;
const PROOF_BYTES_A_NODE: u64 = 3;

/// The most megabytes one table may take: no allocation reaches `isize::MAX`
/// bytes, which on 32-bit targets such as WebAssembly is 2 GB.
const MOST_ONE: usize = isize::MAX as usize >> 20;

/// `x.round()`. WebAssembly has no instruction for it and calls a library
/// function, slowly; there, `round_by_truncation`.
#[cfg(target_arch = "wasm32")]
fn round(x: f32) -> f32 {
    round_by_truncation(x)
}

/// `x.round()` from instructions WebAssembly has: the difference from the
/// truncation is exact, and so is a step of one from it.
#[cfg(any(target_arch = "wasm32", test))]
fn round_by_truncation(x: f32) -> f32 {
    let whole = x.trunc();
    if (x - whole).abs() >= 0.5 {
        whole + 1f32.copysign(x)
    } else {
        whole
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn round(x: f32) -> f32 {
    x.round()
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
fn scale(children: &mut [Child], attacking: bool, step: Option<f32>) {
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

/// Combines the children: the position's value, the child to search, and the
/// second-best child's deciding number, each child's deciding number as
/// `seen` sees it from its own (`crowded`); the other numbers are summed as
/// they are.
fn combine(
    children: &[Child],
    attacking: bool,
    mut seen: impl FnMut(&Child, u32) -> u32,
) -> (Value, usize, u32) {
    let mut best_score = 0;
    let mut best = 0;
    let mut first = u32::MAX;
    let mut second = u32::MAX;
    let mut sum = 0u32;
    // With every child refuted, a refutation relies on what any of them does.
    let mut cycle = ANY_LINE;
    let mut fifty = false;
    // With one refuted, it relies only on what the sturdiest one does.
    let mut sturdiest: Option<(u32, bool)> = None;
    for (index, child) in children.iter().enumerate() {
        let numbers = child.value.numbers;
        // Attacking, a child is best when cheapest to prove; defending, when
        // cheapest to refute.
        let (deciding, summed) = if attacking {
            (numbers.proof, numbers.disproof)
        } else {
            (numbers.disproof, numbers.proof)
        };
        sum = sum.saturating_add(summed);
        // A child seen no cheaper than it is, already dearer than the second
        // best, can be neither: it needs no look.
        let deciding = if deciding > second {
            deciding
        } else {
            seen(child, deciding)
        };
        // Ties go to the move with more history.
        if deciding < first || (deciding == first && child.score > best_score) {
            second = first;
            first = deciding;
            best = index;
            best_score = child.score;
        } else if deciding < second {
            second = deciding;
        }
        if numbers.disproof == 0 {
            cycle = cycle.min(child.value.cycle);
            fifty |= child.value.fifty;
            let sturdiness = (child.value.cycle, !child.value.fifty);
            if sturdiest.is_none_or(|known| sturdiness > known) {
                sturdiest = Some(sturdiness);
            }
        }
    }
    let (first, second) = (first.min(INFINITY), second.min(INFINITY));
    // Only a settled child makes a sum infinite: sums over many open ones
    // grow fast, and one that reached infinity would read as settled.
    let sum = if sum >= INFINITY
        && !children.iter().any(|child| {
            let numbers = child.value.numbers;
            (if attacking {
                numbers.disproof
            } else {
                numbers.proof
            }) >= INFINITY
        }) {
        INFINITY - 1
    } else {
        sum.min(INFINITY)
    };
    let numbers = if attacking {
        Numbers {
            proof: first,
            disproof: sum,
        }
    } else {
        Numbers {
            proof: sum,
            disproof: first,
        }
    };
    let value = if numbers.disproof != 0 {
        Value::open(numbers)
    } else if attacking {
        Value {
            numbers,
            cycle,
            fifty,
        }
    } else {
        let (cycle, sturdy) = sturdiest.unwrap_or((ANY_LINE, true));
        Value {
            numbers,
            cycle,
            fifty: !sturdy,
        }
    };
    (value, best, second)
}

/// A deciding number grown by `SPREAD` percent of it, and at least 1, for
/// each of `busy` threads searching below its position, as Kaneko's virtual
/// numbers count them (2010), and kept below infinity, which would read as
/// settled. A settled position's stays as it is.
fn crowd(deciding: u32, busy: u16) -> u32 {
    if busy == 0 || deciding == 0 || deciding >= INFINITY {
        return deciding;
    }
    let each = (u64::from(deciding) * SPREAD).div_ceil(100).max(1);
    let crowded = u64::from(deciding) + each * u64::from(busy);
    crowded.min(u64::from(INFINITY - 1)) as u32
}

/// A position's slot among the threads' marks (`Store::busy`).
fn busy_slot(hash: u64) -> usize {
    (hash >> (64 - BUSY_BITS)) as usize
}

/// The threads' marks, all clear, unless the memory cannot be had.
fn busy_slots() -> Result<Vec<AtomicU16>, TryReserveError> {
    let mut busy = Vec::new();
    busy.try_reserve_exact(1 << BUSY_BITS)?;
    busy.resize_with(1 << BUSY_BITS, AtomicU16::default);
    Ok(busy)
}

/// A model's step, between 0 and 2, with its odds, step / (2 - step), moved
/// by `unit` percent: computed in f64 and kept off the endpoints, which
/// `scale` treats apart.
fn perturb_model_step(step: f32, unit: f32) -> f32 {
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

/// A child's limit on the number its parent sums: its own number plus what
/// the sum may still grow by. A parent without a limit gives none: its sum may
/// stand just below infinity (see `combine`), and taking that from infinity
/// would leave the child a margin of one, sending it back after every step.
fn share(limit: u32, sum: u32, own: u32) -> u32 {
    if limit >= INFINITY {
        INFINITY
    } else {
        capped(limit - sum + own)
    }
}

fn capped(value: u32) -> u32 {
    value.min(INFINITY)
}

/// The 1+ε trick, ε = 1/8: a child may run until it is somewhat worse than
/// the second-best one, which spares many switches between near-equal
/// children.
fn widen(second: u32) -> u32 {
    if second >= INFINITY {
        INFINITY
    } else {
        capped(second + 1 + second / 8)
    }
}

/// A move's place in the history tables: its piece and destination.
fn history_index(pos: &Position, m: Move) -> usize {
    let role = pos.role_at(m.from()).map_or(0, |role| role as usize);
    role * 64 + m.to() as usize
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::{Duration, Instant};

    use shakmaty::{Position as _, variant::Atomic};

    use super::{
        Child, Dfpn, Guide, INFINITY, Level, Numbers, Proofs, Value, combine, crowd,
        perturb_model_step, scale, share, split,
    };
    use crate::board::from_fen;

    /// The length of a search's win as `provari prove` writes its proof, in
    /// the compact format, which the checker confirms.
    fn checked_length(search: &Dfpn, root: &Atomic) -> u32 {
        let (header, indices) = crate::format::walk(root, root.turn(), None, |pos| {
            let key = (pos.hash(), pos.halfmoves());
            search
                .winning_move(&pos.to_atomic(), &key)
                .and_then(|m| crate::format::board_move(pos, &m))
        })
        .unwrap();
        let bytes = crate::format::file(&header, &indices).unwrap();
        let checked = crate::check::check_proof("proof", &bytes, None);
        assert_eq!(checked["valid"], true, "{checked}");
        header.moves
    }

    #[test]
    fn terminal_roots_use_the_fixed_attacker() {
        use shakmaty::Color;

        for (fen, winner) in [
            ("8/8/8/8/8/8/P7/4K3 b - - 0 1", Color::White),
            ("4k3/p7/8/8/8/8/8/8 w - - 0 1", Color::Black),
            // An already completed mate takes precedence over the draw clock.
            ("3qkb2/3pRp2/8/8/8/8/8/K7 b - - 100 1", Color::White),
        ] {
            let root = crate::position(fen).unwrap();
            for attacker in [winner, !winner] {
                let mut search = Dfpn::new(
                    &root,
                    None,
                    Guide::None,
                    Instant::now() + Duration::from_secs(1),
                    split(1, 0),
                );
                search.set_attacker(attacker);
                search.record = true;
                search.set_limit(1);
                assert_eq!(search.solve(&root), (attacker == winner).then_some(0));
                assert_eq!(search.nodes, 0);
                assert_eq!(search.recorded, None);
                assert_eq!(
                    search.root_numbers,
                    Some(if attacker == winner {
                        (0, super::INFINITY)
                    } else {
                        (super::INFINITY, 0)
                    })
                );
            }
        }
    }

    #[test]
    fn a_root_draw_cannot_be_reset_by_a_winning_capture() {
        for clock in [99, 100] {
            let root = crate::position(&format!(
                "rnbqkb1r/pp1p3p/2pBppp1/8/3PP3/2N5/PPP2P1P/R2QKB1R w KQ - {clock} 2"
            ))
            .unwrap();
            let mut search = Dfpn::new(
                &root,
                None,
                Guide::None,
                Instant::now() + Duration::from_secs(1),
                split(1, 0),
            );
            assert_eq!(search.solve(&root), (clock == 99).then_some(1));
            if clock == 100 {
                assert_eq!(search.root_numbers, Some((super::INFINITY, 0)));
                assert_eq!(search.nodes, 0);
            }
        }
    }

    /// The browser build's rounding gives what `round` does, bit for bit:
    /// at every half and quarter in a wide range, around the largest values
    /// with fractions, and at random bit patterns of every kind.
    #[test]
    fn rounds_by_truncation_as_round_does() {
        let mut values = vec![0.49999997, -0.49999997, 8_388_607.5, -8_388_607.5];
        values.extend((-400_000..400_000).map(|quarter| quarter as f32 / 4.0));
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..200_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values.push(f32::from_bits(state as u32));
        }
        for x in values {
            let (ours, theirs) = (super::round_by_truncation(x), x.round());
            assert!(
                ours.to_bits() == theirs.to_bits() || (ours.is_nan() && theirs.is_nan()),
                "{x}: {ours} for {theirs}"
            );
        }
    }

    /// A child's limit leaves its parent's sum room to grow to the parent's
    /// limit, and none under a parent without one, whose sum may stand just
    /// below infinity.
    #[test]
    fn a_child_shares_only_a_limit() {
        assert_eq!(share(1000, 900, 5), 105);
        assert_eq!(share(1000, 1000, 5), 5);
        assert_eq!(share(INFINITY, INFINITY - 1, 5), INFINITY);
        assert_eq!(share(INFINITY, 7, 5), INFINITY);
    }

    #[test]
    fn model_noise_preserves_domain_direction_and_endpoints() {
        let extra = [
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            f32::from_bits(2.0f32.to_bits() - 1),
        ];
        let steps = (0..=65536).map(|i| 2.0 * i as f32 / 65536.0).chain(extra);
        for step in steps {
            assert_eq!(perturb_model_step(step, 0.0), step);
            for unit in [-1.0, -0.5, 0.5, 1.0] {
                let changed = perturb_model_step(step, unit);
                if step == 0.0 || step == 2.0 {
                    assert_eq!(changed, step);
                } else {
                    assert!(changed > 0.0 && changed < 2.0);
                    assert!((unit < 0.0 && changed <= step) || (unit > 0.0 && changed >= step));
                    assert!((changed - step).abs() <= 0.010001 * step + f32::MIN_POSITIVE);
                }
                if (0.01..1.99).contains(&step) {
                    let odds = |value: f32| f64::from(value) / (2.0 - f64::from(value));
                    assert!(
                        (odds(changed) / odds(step) - (1.0 + 0.01 * f64::from(unit))).abs()
                            < 0.00002
                    );
                }
            }
        }
    }

    #[test]
    fn positive_model_noise_does_not_reduce_disproof_estimate() {
        // A step near 2, raised by positive noise: it stays below 2, where
        // the formula would change, so the disproof estimate only grows.
        let step = 1.991_977_6;
        let unit = 0.812_429_8;
        let scaled = |step| {
            let mut children = [Child {
                m: Default::default(),
                hash: 0,
                counter: 0,
                value: Value::open(Numbers {
                    proof: 1,
                    disproof: 1,
                }),
                exploded: false,
                winning: false,
                score: 0,
                simulated: false,
                fresh: true,
                group: false,
            }];
            scale(&mut children, true, Some(step));
            children[0].value.numbers
        };
        let native = scaled(step);
        let changed = scaled(perturb_model_step(step, unit));
        assert_eq!((native.proof, native.disproof), (1, 263));
        assert_eq!((changed.proof, changed.disproof), (1, 264));
    }

    /// Short wins replayed near the 50-move limit, where a defence that lets
    /// the king be exploded reaches the limit and draws. Debug builds check,
    /// as the search goes, that every position it calls won has a proof.
    #[test]
    fn calls_won_only_what_it_proves() {
        let wins = [
            "r1b1kbnr/pp1pq2p/2n1ppp1/1B6/3PPB2/5N2/PPP2PPP/R2QK2R w KQkq -",
            "r1bqkbnr/1p1p2pp/p1n2p2/1B2p3/3PP3/2P2N2/PP3PPP/R1BQK2R w KQkq -",
            "rnb1kb1r/pp1p3p/4pp1n/q5p1/3N1P2/2P1P3/PP1P2PP/R1BQKB1R w KQkq -",
        ];
        for win in wins {
            for clock in [0, 96, 98] {
                let root = from_fen(&format!("{win} {clock} 1")).unwrap().to_atomic();
                let deadline = Instant::now() + Duration::from_secs(2);
                let mut search = Dfpn::new(&root, None, Guide::None, deadline, split(4, 0));
                let found = search.solve(&root);
                if clock == 0 {
                    assert!(found.is_some(), "{win}");
                }
            }
        }
    }

    /// Defender positions after quiet threats: if the defender could pass,
    /// the attacker would win at once, and replaying that win proves most
    /// defences, each with a proof.
    #[test]
    fn replays_a_threat_on_the_defences() {
        for fen in [
            "rnb1k3/p3p2P/2p1P3/qp1p4/3P1pp1/2N5/PPP2PP1/R1BQKB1R b KQq - 0 6",
            "r1bk1bnr/pp1p3p/2nBppp1/qB6/3PP3/2P2N2/PP3PPP/R2QK2R b KQ - 1 1",
        ] {
            let pos = from_fen(fen).unwrap();
            let root = pos.to_atomic();
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut search = Dfpn::new(&root, None, Guide::None, deadline, split(4, 0));
            search.set_attacker(!pos.turn());
            let mut children = Vec::new();
            search.children(&pos, false, u32::MAX, &mut children);
            let open = |children: &[Child]| {
                children
                    .iter()
                    .filter(|child| child.value.numbers.proof != 0)
                    .count()
            };
            let before = open(&children);
            search.replay_threat(&pos, u32::MAX, &mut children);
            let after = open(&children);
            assert!(after * 2 < before, "{fen}: {before} open, then {after}");
            for child in children.iter().filter(|child| child.simulated) {
                assert!(
                    search.proof_at(child.hash, child.counter).is_some(),
                    "{fen}"
                );
            }
        }
    }

    /// Shortening brings a first win down to the shortest one the exhaustive
    /// baseline finds, and says it is the shortest, as its proof reads back;
    /// also once the proofs have dropped their mates in one, which leaves the
    /// first win the shortest here.
    #[test]
    fn shortens_to_the_shortest_win() {
        let root = from_fen("r1bqkbnr/1p1p2pp/p1n2p2/1B2p3/3PP3/2P2N2/PP3PPP/R1BQK2R w KQkq - 0 1")
            .unwrap()
            .to_atomic();
        let deadline = Instant::now() + Duration::from_secs(60);
        for dropped in [false, true] {
            let mut search = Dfpn::new(&root, None, Guide::None, deadline, (16, 8));
            if dropped {
                search.set_drop_mates_at(0.0);
            }
            let first = search.solve(&root).unwrap();
            let first_nodes = search.nodes;
            let mut wins = Vec::new();
            let shortening = search.shorten(&root, first, |search, length| {
                wins.push((length, search.nodes));
            });
            let fewest = crate::search::Search::new(&root, None, deadline, 1 << 20)
                .solve(&root, first)
                .unwrap();
            assert_eq!((shortening.length, shortening.shortest), (fewest, true));
            assert_eq!(checked_length(&search, &root), fewest);
            assert_eq!(search.mates_dropped.is_some(), dropped);
            if dropped {
                assert_eq!((first, shortening.passes), (fewest, 1));
                continue;
            }
            assert!(
                fewest < first,
                "a first win of {first} that is already the shortest"
            );
            // At least one pass that shortened and the one that settled it.
            assert!(shortening.passes >= 2, "{}", shortening.passes);
            assert_eq!(wins.last().map(|win| win.0), Some(fewest));
            assert!(wins.iter().all(|&(_, nodes)| nodes > first_nodes));
        }
    }

    /// With one attacker move left only a move that wins at once counts, and
    /// with two the search looks no further: it agrees with the exhaustive
    /// baseline on whether a win exists within one move and within two, with
    /// its mates in one kept or dropped.
    #[test]
    fn decides_the_last_moves_as_the_baseline() {
        for fen in [
            // The queen takes next to the king, which explodes.
            "4k3/3p4/8/8/8/8/8/3QK3 w - - 0 1",
            // No capture reaches the king's neighbours, and no check mates.
            "4k3/3p4/8/8/8/8/8/2Q1K3 w - - 0 1",
            // The shortest win takes two moves.
            "r1b1kbnr/pp1pq2p/2n1ppp1/1B6/3PPB2/5N2/PPP2PPP/R2QK2R w KQkq - 0 1",
        ] {
            let root = from_fen(fen).unwrap().to_atomic();
            for (most, dropped) in [(1, false), (2, false), (1, true), (2, true)] {
                let deadline = Instant::now() + Duration::from_secs(10);
                let baseline =
                    crate::search::Search::new(&root, None, deadline, 1 << 16).solve(&root, most);
                let mut search = Dfpn::new(&root, None, Guide::Hand, deadline, split(4, 0));
                if dropped {
                    search.set_drop_mates_at(0.0);
                }
                search.set_most(most);
                let found = search.solve(&root);
                assert_eq!(found.is_some(), baseline.is_some(), "{fen} within {most}");
                assert!(
                    found.is_none_or(|length| length <= most),
                    "{fen} within {most}"
                );
                match found {
                    Some(_) => assert!(checked_length(&search, &root) <= most, "{fen}"),
                    None => {
                        let refuted = search
                            .root_numbers
                            .is_some_and(|(_, disproof)| disproof == 0);
                        assert!(refuted, "{fen} within {most}");
                    }
                }
            }
        }
    }

    /// Under a limit, a search proves a win in as many attacker moves as the
    /// exhaustive baseline's fewest, as its proof reads back, and refutes the
    /// position with one less; also with its mates in one dropped.
    #[test]
    fn wins_within_the_most_moves() {
        for fen in [
            "r1b1kbnr/pp1pq2p/2n1ppp1/1B6/3PPB2/5N2/PPP2PPP/R2QK2R w KQkq - 0 1",
            "r1bqkbnr/1p1p2pp/p1n2p2/1B2p3/3PP3/2P2N2/PP3PPP/R1BQK2R w KQkq - 0 1",
        ] {
            let root = from_fen(fen).unwrap().to_atomic();
            let deadline = Instant::now() + Duration::from_secs(60);
            let fewest = crate::search::Search::new(&root, None, deadline, 1 << 20)
                .solve(&root, 8)
                .unwrap();
            let solve = |most, dropped| {
                let mut search = Dfpn::new(&root, None, Guide::Hand, deadline, split(4, 0));
                if dropped {
                    search.set_drop_mates_at(0.0);
                }
                search.set_most(most);
                let found = search.solve(&root);
                let written = found.map(|_| checked_length(&search, &root));
                (found, written, search.root_numbers)
            };
            for dropped in [false, true] {
                let (found, written, _) = solve(fewest, dropped);
                assert_eq!((found, written), (Some(fewest), Some(fewest)), "{fen}");
                let (found, _, numbers) = solve(fewest - 1, dropped);
                assert_eq!(found, None, "{fen}");
                assert_eq!(numbers.map(|(_, disproof)| disproof), Some(0), "{fen}");
            }
        }
    }

    /// A search whose proofs fill their memory drops its mates in one, and
    /// once they fill again stops, and claims no win.
    #[test]
    fn stops_when_its_proofs_are_full() {
        let root = from_fen("r1bqkbnr/1p1p2pp/p1n2p2/1B2p3/3PP3/2P2N2/PP3PPP/R1BQK2R w KQkq - 0 1")
            .unwrap()
            .to_atomic();
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut search = Dfpn::new(&root, None, Guide::Hand, deadline, split(4, 0));
        assert!(search.solve(&root).is_some() && !search.proofs_full());
        let mut search = Dfpn::new(&root, None, Guide::Hand, deadline, split(4, 0));
        Arc::get_mut(&mut search.store).unwrap().proofs = Proofs::new(0, false).unwrap();
        assert_eq!(search.solve(&root), None);
        assert!(search.proofs_full() && search.stopped && search.mates_dropped.is_some());
    }

    /// A search told to stop stops at its next look at the clock, within
    /// 1024 nodes, as at its deadline, and claims no win.
    #[test]
    fn stops_when_told() {
        let root = Atomic::default();
        let deadline = Instant::now() + Duration::from_secs(60);
        let stop = std::sync::atomic::AtomicBool::new(true);
        let mut search = Dfpn::new(&root, None, Guide::Hand, deadline, split(4, 0));
        search.set_stop(&stop);
        assert_eq!(search.solve(&root), None);
        assert!(search.stopped && search.nodes < 2048, "{}", search.nodes);
    }

    /// Proofs that fill drop their mates in one, and the search goes on to the
    /// win it finds with room to spare, in nearly as many nodes, its proof
    /// whole; where they keep up, nothing is dropped. The drop can be forced
    /// at any share of the proofs.
    #[test]
    fn drops_its_mates_and_goes_on() {
        let root = from_fen("r1b1kb1r/pp1p1qp1/2n1p3/1B3P1p/3PP3/2P5/PP4PP/R2QK2R w KQkq - 0 1")
            .unwrap()
            .to_atomic();
        let solve = |slots, share: Option<f64>| {
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut search = Dfpn::new(&root, None, Guide::Hand, deadline, split(4, 0));
            Arc::get_mut(&mut search.store).unwrap().proofs = Proofs::sized(slots, false);
            if let Some(share) = share {
                search.set_drop_mates_at(share);
            }
            let found = search.solve(&root);
            let written = found.map(|_| checked_length(&search, &root));
            (
                found,
                written,
                search.nodes,
                search.mates_dropped,
                search.proofs_full(),
            )
        };
        let (found, written, nodes, dropped, full) = solve(1024, None);
        assert_eq!((dropped, full), (None, false));
        assert!(found.is_some() && written <= found, "{found:?} {written:?}");
        for (slots, share) in [(512, None), (1024, Some(0.2)), (1024, Some(0.0))] {
            let (again, written, more, dropped, full) = solve(slots, share);
            assert_eq!((again, full), (found, false), "{slots} {share:?}");
            assert!(dropped.is_some_and(|at| at < more), "{slots} {share:?}");
            assert!(written <= again, "{slots} {share:?}");
            assert!(
                more <= nodes + nodes / 10,
                "{slots} {share:?}: {more} nodes"
            );
        }
    }

    /// A node budget stops the search where it says, and the same budget
    /// gives the same search again.
    #[test]
    fn repeats_under_a_node_budget() {
        let root = from_fen("r1b1kb1r/pp1p1qp1/2n1p3/1B3P1p/3PP3/2P5/PP4PP/R2QK2R w KQkq - 0 1")
            .unwrap()
            .to_atomic();
        let run = || {
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut search = Dfpn::new(&root, None, Guide::Hand, deadline, split(4, 0));
            search.set_limit(300);
            let found = search.solve(&root);
            (found, search.nodes, search.stopped)
        };
        let first = run();
        assert_eq!(first, run());
        assert!(first.1 >= 300 && first.2, "{first:?}");
    }

    /// A seed perturbs the search's path, not what it proves.
    #[test]
    fn proves_the_same_under_a_seed() {
        let root = from_fen("r1b1kb1r/pp1p1qp1/2n1p3/1B3P1p/3PP3/2P5/PP4PP/R2QK2R w KQkq - 0 1")
            .unwrap()
            .to_atomic();
        let solve = |seed| {
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut search = Dfpn::new(&root, None, Guide::Hand, deadline, split(4, 0));
            search.set_seed(seed);
            (search.solve(&root), search.nodes)
        };
        let (plain, seeded) = (solve(0), solve(1));
        assert!(
            plain.0.is_some() && seeded.0.is_some(),
            "{plain:?} {seeded:?}"
        );
        assert_ne!(plain.1, seeded.1, "the seed left the search unchanged");
    }

    /// A search with `threads` threads together, with a table and proofs of
    /// so many megabytes.
    fn together<'a>(root: &Atomic, threads: usize, memory: (usize, usize)) -> Dfpn<'a> {
        let deadline = Instant::now() + Duration::from_secs(60);
        Dfpn::try_with_threads(root, None, Guide::Hand, deadline, memory, threads).unwrap()
    }

    /// Threads that search together prove the wins one thread proves, each
    /// with a proof the checker confirms, also under a limit on moves, and
    /// settle a root with no win so short as refuted; they leave no mark of
    /// a thread below any position.
    #[test]
    fn threads_prove_what_one_proves() {
        for fen in [
            "r1b1kbnr/pp1pq2p/2n1ppp1/1B6/3PPB2/5N2/PPP2PPP/R2QK2R w KQkq - 0 1",
            "r1bqkbnr/1p1p2pp/p1n2p2/1B2p3/3PP3/2P2N2/PP3PPP/R1BQK2R w KQkq - 0 1",
            "r1b1kb1r/pp1p1qp1/2n1p3/1B3P1p/3PP3/2P5/PP4PP/R2QK2R w KQkq - 0 1",
        ] {
            let root = from_fen(fen).unwrap().to_atomic();
            for threads in [2, 4] {
                let mut search = together(&root, threads, split(4, 0));
                assert!(search.solve(&root).is_some(), "{fen} with {threads}");
                assert!(!search.stopped && !search.proofs_full());
                checked_length(&search, &root);
                let busy = &search.store.busy;
                assert!(busy.iter().all(|slot| slot.load(Relaxed) == 0));
            }
        }
        let root = from_fen("r1bqkbnr/1p1p2pp/p1n2p2/1B2p3/3PP3/2P2N2/PP3PPP/R1BQK2R w KQkq - 0 1")
            .unwrap()
            .to_atomic();
        let deadline = Instant::now() + Duration::from_secs(60);
        let fewest = crate::search::Search::new(&root, None, deadline, 1 << 20)
            .solve(&root, 8)
            .unwrap();
        let mut search = together(&root, 3, split(4, 0));
        search.set_most(fewest);
        assert_eq!(search.solve(&root), Some(fewest));
        assert_eq!(checked_length(&search, &root), fewest);
        let mut search = together(&root, 3, split(4, 0));
        search.set_most(fewest - 1);
        assert_eq!(search.solve(&root), None);
        assert_eq!(search.root_numbers.map(|(_, disproof)| disproof), Some(0));
        assert!(!search.stopped);
    }

    /// Threads stop together at their node budget, which counts every
    /// thread's nodes, and once told to stop.
    #[test]
    fn threads_stop_together() {
        let root = Atomic::default();
        let mut search = together(&root, 3, split(4, 0));
        search.set_limit(100_000);
        assert_eq!(search.solve(&root), None);
        assert!(search.stopped, "{}", search.nodes);
        assert!(
            (100_000..100_000 + 3 * 4096).contains(&search.nodes),
            "{}",
            search.nodes
        );
        let stop = std::sync::atomic::AtomicBool::new(true);
        let mut search = together(&root, 3, split(4, 0));
        search.set_stop(&stop);
        assert_eq!(search.solve(&root), None);
        assert!(
            search.stopped && search.nodes < 3 * 4096,
            "{}",
            search.nodes
        );
    }

    /// Threads whose proofs fill drop their mates in one, once, while the
    /// others wait, and go on to the win; proofs too small for it fill again
    /// and stop them all, with no win claimed. A table of one cluster, whose
    /// filter goes stale at every count, has them rebuild it while the
    /// others wait, again and again.
    #[test]
    fn threads_share_their_changes() {
        let root = from_fen("r1b1kb1r/pp1p1qp1/2n1p3/1B3P1p/3PP3/2P5/PP4PP/R2QK2R w KQkq - 0 1")
            .unwrap()
            .to_atomic();
        let mut search = together(&root, 4, split(4, 0));
        Arc::get_mut(&mut search.store).unwrap().proofs = Proofs::sized(4096, true);
        search.set_drop_mates_at(0.05);
        assert!(search.solve(&root).is_some());
        assert!(search.mates_dropped.is_some() && !search.proofs_full());
        checked_length(&search, &root);
        let mut search = together(&root, 4, split(4, 0));
        Arc::get_mut(&mut search.store).unwrap().proofs = Proofs::new(0, true).unwrap();
        assert_eq!(search.solve(&root), None);
        assert!(search.proofs_full() && search.stopped && search.mates_dropped.is_some());
        let mut search = together(&root, 4, (0, 4));
        assert!(search.solve(&root).is_some());
        checked_length(&search, &root);
    }

    /// Threads that find the table's filter a quarter set drop it while the
    /// others wait, and each stops reading it.
    #[test]
    fn threads_drop_a_dense_filter() {
        let root = Atomic::default();
        let mut search = together(&root, 4, split(4, 0));
        // A third of the filter's 2^20 bits, each position's check bits a bit
        // of their own.
        for i in 0..(1u32 << 20) / 3 {
            let check = u64::from(i.wrapping_mul(0x9e37_79b9));
            search.table().note(check << 32, true);
        }
        assert!(search.table().filter_is_dense() && search.filtering);
        search.set_limit(20_000);
        assert_eq!(search.solve(&root), None);
        assert!(search.stopped && !search.table().filtering() && !search.filtering);
    }

    /// A seed for each thread: the threads' paths differ, and so do their
    /// nodes, while what they prove does not.
    #[test]
    fn threads_search_with_seeds_of_their_own() {
        let root = from_fen("r1b1kb1r/pp1p1qp1/2n1p3/1B3P1p/3PP3/2P5/PP4PP/R2QK2R w KQkq - 0 1")
            .unwrap()
            .to_atomic();
        for seed in [0, 7] {
            let mut search = together(&root, 2, split(4, 0));
            search.set_seed(seed);
            assert!(search.solve(&root).is_some(), "{seed}");
            checked_length(&search, &root);
        }
    }

    /// A busy position's number grows by 1% for each thread below it, by at
    /// least 1, and stays below infinity; a settled one's, or a number no
    /// thread is below, stays as it is.
    #[test]
    fn crowding_grows_a_busy_number() {
        assert_eq!(crowd(40, 0), 40);
        assert_eq!(crowd(40, 1), 41);
        assert_eq!(crowd(40, 3), 43);
        assert_eq!(crowd(1, 2), 3);
        assert_eq!(crowd(250, 1), 253);
        assert_eq!(crowd(250, 2), 256);
        assert_eq!(crowd(0, 3), 0);
        assert_eq!(crowd(INFINITY, 3), INFINITY);
        assert_eq!(crowd(INFINITY - 1, 1), INFINITY - 1);
        assert_eq!(crowd(INFINITY / 2, 100), INFINITY - 1);
    }

    /// Together, a child is chosen, and its parent numbered, by the
    /// children's deciding numbers as seen, while the other numbers sum as
    /// they are; a child already dearer than the second best is not looked
    /// at.
    #[test]
    fn combine_chooses_by_the_numbers_seen() {
        let child = |proof, disproof| Child {
            m: Default::default(),
            hash: u64::from(proof),
            counter: 0,
            value: Value::open(Numbers { proof, disproof }),
            exploded: false,
            winning: false,
            score: 0,
            simulated: false,
            fresh: false,
            group: false,
        };
        let children = [child(10, 5), child(11, 7), child(30, 1)];
        let (value, best, second) = combine(&children, true, |_, deciding| deciding);
        assert_eq!((value.numbers.proof, value.numbers.disproof), (10, 13));
        assert_eq!((best, second), (0, 11));
        // Two threads below the first child: 10 is seen as 12.
        let mut looked = Vec::new();
        let (value, best, second) = combine(&children, true, |child, deciding| {
            looked.push(deciding);
            crowd(deciding, if child.hash == 10 { 2 } else { 0 })
        });
        assert_eq!((value.numbers.proof, value.numbers.disproof), (11, 13));
        assert_eq!((best, second), (1, 12));
        assert_eq!(looked, [10, 11]);
        // Defending, the disproof numbers decide.
        let (value, best, second) = combine(&children, false, |child, deciding| {
            crowd(deciding, u16::from(child.hash == 30))
        });
        assert_eq!((value.numbers.proof, value.numbers.disproof), (51, 2));
        assert_eq!((best, second), (2, 5));
    }

    /// The root's numbers rebuilt from the line combine each level's other
    /// children with the numbers below, from the deepest level's child as it
    /// was chosen up, as `combine` does: the attacker's least proof number
    /// and summed disproof numbers, the defender's the other way round,
    /// sums short of infinity; without a line, the root's own.
    #[test]
    fn rebuilds_the_roots_numbers_from_the_line() {
        let root = Atomic::default();
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut search = Dfpn::new(&root, None, Guide::Hand, deadline, split(4, 0));
        search.root = (3, 4);
        assert_eq!(search.root_now(), (3, 4));
        let level = |attacking, proof, disproof, least, sum| Level {
            m: Default::default(),
            attacking,
            taken: Numbers { proof, disproof },
            least,
            sum,
            open: 2,
        };
        search.levels[0] = level(true, 50, 9, 7, 10);
        search.levels[1] = level(false, 40, 9, 6, 8);
        // Kept when the search went down, and replaced by the level below.
        search.levels[2] = level(true, 2, 4, INFINITY, 0);
        search.line_len = 3;
        // Below: (2, 4); defending: (8 + 2, min(6, 4)) = (10, 4); attacking:
        // (min(7, 10), 10 + 4) = (7, 14).
        assert_eq!(search.root_now(), (7, 14));
        search.levels[0] = level(true, 50, 9, 7, INFINITY - 2);
        assert_eq!(search.root_now(), (7, INFINITY - 1));
    }

    /// A search's progress lines carry its line, the moves open along it, the
    /// root move it has searched most and the root's numbers rebuilt from the
    /// line with their peak; the search itself goes as it would without them,
    /// and keeps no line unless it reports or is asked to.
    #[test]
    fn reports_its_line_and_numbers() {
        let root = Atomic::default();
        let run = |progress: bool, keep: bool| {
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut search = Dfpn::new(&root, None, Guide::Hand, deadline, split(4, 0));
            search.set_limit(20_000);
            let records = Arc::new(std::sync::Mutex::new(Vec::new()));
            if progress {
                let kept = records.clone();
                search.set_progress_sink(
                    Duration::ZERO,
                    Box::new(move |record| kept.lock().unwrap().push(record.clone())),
                );
            }
            if keep {
                search.keep_line();
            }
            let found = search.solve(&root);
            let records = records.lock().unwrap().clone();
            (found, search.nodes, search.line(), records)
        };
        let (found, nodes, line, records) = run(true, false);
        let (kept_found, kept_nodes, kept_line, _) = run(false, true);
        assert_eq!((found, nodes, &line), (kept_found, kept_nodes, &kept_line));
        let (quiet_found, quiet_nodes, quiet_line, _) = run(false, false);
        assert_eq!((found, nodes), (quiet_found, quiet_nodes));
        assert!(quiet_line.0.is_empty() && quiet_line.1.is_empty());
        // A report at most every 1,024 nodes, where the clock is read.
        assert!(records.len() >= 10, "{}", records.len());
        let mut peak = 0;
        for record in &records {
            let moves = record["line"].as_array().unwrap();
            let open = record["open"].as_array().unwrap();
            assert!(!moves.is_empty() && moves.len() <= super::LINE, "{record}");
            assert_eq!(moves.len(), open.len(), "{record}");
            let mut pos = root.clone();
            for (uci, open) in moves.iter().zip(open) {
                let m = uci
                    .as_str()
                    .unwrap()
                    .parse::<shakmaty::uci::UciMove>()
                    .unwrap();
                pos.play_unchecked(m.to_move(&pos).unwrap());
                assert!(open.as_u64().unwrap() >= 1, "{record}");
            }
            let most = record["most_searched"].as_str().unwrap();
            assert!(
                root.legal_moves()
                    .iter()
                    .any(|&m| shakmaty::uci::UciMove::from_standard(m).to_string() == most),
                "{record}"
            );
            let proof = record["proof"].as_u64().unwrap();
            assert!(proof > 0 && proof < u64::from(INFINITY), "{record}");
            peak = peak.max(proof);
            assert_eq!(record["peak"].as_u64(), Some(peak), "{record}");
        }
        // Stopped, the search tells the line it was on.
        assert!(
            !line.0.is_empty() && line.0.len() == line.1.len(),
            "{line:?}"
        );
    }
}
