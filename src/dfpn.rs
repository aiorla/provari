//! Depth-first proof-number search (df-pn, Nagai 2002) for wins in chess
//! variants: the search itself, which every variant shares, and the
//! `Variant` trait, through which a variant says what it alone knows. Atomic
//! is the variant provari proves (`atomic`).
//!
//! Every position has a proof number, how many unsettled leaves must still be
//! won to win it, and a disproof number, how many must fail to refute it. The
//! search always descends into the child that looks cheapest to settle, within
//! thresholds that keep it depth-first, and uses the 1+ε trick against
//! thrashing. The variant scores a position's children before the search
//! descends: which are settled at once, and the starting numbers of the
//! others, as its rules and its guide judge them.
//!
//! A variant is a type parameter of the search, so that each variant compiles
//! to its own search and none tests at run time which variant it searches: what
//! one variant does at every node costs another nothing.
//!
//! A position keeps its children's numbers while it is searched and updates
//! only the child it searched. A variant may hold some moves back as one
//! child, a group, opened only when the search first wants one of them.
//!
//! Positions are keyed without the 50-move counter, so that move orders share
//! their work. A proof records its length in attacker moves and the highest
//! counter it holds at, and only a proof that is shorter or holds further
//! joins it, so lengths shrink along every line of a proof and none can repeat
//! a position. A position may so keep a short proof up to a low counter and a
//! longer one, with another move, up to a higher one; a proof's walk that
//! would write both writes the one that holds furthest instead (`kept_proof`),
//! as the format lets no two moves of a position link positions in a circle.
//! Proofs that come out the same whenever worked out again are not kept; when
//! the kept ones fill their memory, the mates in one go too, and each time
//! they fill after that, every proof of at most two moves (`drop_short`): the
//! search then works one out again where it needs it, and stops only once the
//! longer proofs alone fill the memory. Repeating a position of the current
//! line, the 50-move limit, and stalemate count as draws. A refutation that
//! relies on a repetition applies only while the repeated position is on the
//! line, and one that relies on the 50-move limit only from the same counter
//! upward.
//!
//! A search may be limited to wins in at most so many attacker moves. Proofs
//! know their lengths, so one counts wherever it fits the moves left; the
//! table keeps what it learns apart for each number of moves left, since a
//! position refuted for lack of moves may be won with more. A tablebase win
//! counts as won where it is reached, as a proof's length counts it
//! (FORMAT.md), with any moves left; an exact search plays it out instead,
//! so that its every line ends in a finished game (`set_exact`).
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
//! stops the others. Dropping the mates in one or the short proofs and
//! rebuilding the table's filter move what every thread reads, so one thread
//! makes them while the others wait (`pause`).
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

use std::collections::TryReserveError;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering, Ordering::Relaxed};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use crate::clock::Instant;

use shakmaty::Color;

use crate::board::{Move, Position};
use crate::model::FEATURES;
use crate::pause::Pause;
use crate::proofs::{Proof, Proofs};
use crate::search::Status;
use crate::table::{self, Known, Table};
use crate::variant;

const INFINITY: u32 = 1 << 30;

/// The highest 50-move counter at which the game goes on.
const LAST: u32 = 99;

/// The depth of a refutation's repetition when it relies on none.
const ANY_LINE: u32 = u32::MAX;

/// Where the table keeps a position's plain moves together.
const GROUP: u64 = 0x2545_f491_4f6c_dd1d;

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

#[cfg(feature = "antichess")]
pub mod antichess;
mod atomic;

pub use atomic::Guide;

/// What the search needs of a variant's board: its own board, built for
/// speed, whose moves fit in 16 bits as `Move` holds them.
pub trait Board: Clone + Send + Sync {
    /// The position's Zobrist hash, its 50-move counter left out.
    fn hash(&self) -> u64;
    /// The 50-move counter.
    fn halfmoves(&self) -> u32;
    fn turn(&self) -> Color;
    /// The position after a legal move.
    fn play(&self, m: Move) -> Self;
    /// The position after a legal move whose child's hash is known.
    #[inline(always)]
    fn play_known(&self, m: Move, _hash: u64) -> Self {
        self.play(m)
    }
    /// How many pieces stand on the board.
    fn pieces(&self) -> u32;
    /// A legal move in UCI's notation.
    fn uci(&self, m: Move) -> String;
}

/// What a variant tells the search, as a type parameter of it (`Dfpn`): its
/// positions, the state each search thread keeps for it, and what the search
/// asks of it while it searches. Each function that takes the search is
/// called at one place of `Dfpn::mid` or `Dfpn::solve`, and may read and
/// change the search as `mid` would there.
pub trait Variant: Sized + 'static {
    /// A position as the program reads it, from shakmaty.
    type Root: shakmaty::Position;
    /// A position as the search plays it.
    type Board: Board;
    /// The 1+ε trick's ε, as its inverse: a child may run until it is this
    /// share dearer than the second best (`widen`).
    const EPSILON: u32;

    /// What every thread of a search shares and reads, such as the guide
    /// and the tablebases.
    type Shared<'a>: Copy + Send;
    /// What each thread keeps for the variant as it searches, which a
    /// helper thread hands back as it ends (`gather`).
    type State<'a>: Send;

    /// A thread's state, from what the threads share.
    fn state(shared: Self::Shared<'_>) -> Self::State<'_>;

    /// What the threads share, from a thread's state.
    fn shared<'a>(state: &Self::State<'a>) -> Self::Shared<'a>;

    /// The search's board of a position.
    fn board(root: &Self::Root) -> Self::Board;

    /// How a root has ended for `attacker`, if it has.
    fn status(root: &Self::Root, attacker: Color) -> Status;

    /// Scores a position's moves into `children`, with `remaining` attacker
    /// moves left, and returns the step its evaluation scaled their
    /// starting numbers by, which a group opened later reuses.
    fn children(
        search: &mut Dfpn<'_, Self>,
        pos: &Self::Board,
        attacking: bool,
        remaining: u32,
        children: &mut Vec<Child>,
    ) -> Option<f32>;

    /// After a position's children are scored, `fresh` when the table knew
    /// nothing of it: what the variant learns from them all at once.
    fn scored(
        search: &mut Dfpn<'_, Self>,
        pos: &Self::Board,
        attacking: bool,
        fresh: bool,
        remaining: u32,
        children: &mut [Child],
    );

    /// The value of a position with no legal move.
    fn no_moves(search: &mut Dfpn<'_, Self>, pos: &Self::Board) -> Value;

    /// Scores the moves a group held into `children`, after the others, as
    /// `children` would have with this `step`.
    fn open_group(
        search: &mut Dfpn<'_, Self>,
        pos: &Self::Board,
        remaining: u32,
        children: &mut Vec<Child>,
        step: Option<f32>,
    );

    /// After the defence `children[best]`, which leads to `next`, is proven.
    fn defence_proven(
        search: &mut Dfpn<'_, Self>,
        pos: &Self::Board,
        next: &Self::Board,
        best: usize,
        remaining: u32,
        children: &mut [Child],
    );

    /// After the defence `m` refuted the position.
    fn refuted(search: &mut Dfpn<'_, Self>, pos: &Self::Board, m: Move);

    /// Records a proven position: attacking, through the shortest winning
    /// move; defending, through every defence.
    fn record_proof(
        search: &mut Dfpn<'_, Self>,
        pos: &Self::Board,
        hash: u64,
        children: &[Child],
        attacking: bool,
    );

    /// The proof of a won position whose proof is not kept, because it comes
    /// out the same whenever it is worked out again.
    fn recomputed(search: &Dfpn<'_, Self>, pos: &Self::Board) -> Option<Proof>;

    /// Whether `m` wins at once, as the table's notes of mates in one say,
    /// in a position the table notes one for.
    fn mates_with(search: &Dfpn<'_, Self>, pos: &Self::Board, m: Move) -> bool;

    /// Takes in a helper thread's state once its part of a search together
    /// ends (`search_together`), into the state of the thread that started
    /// it: for what the variant counts over every thread.
    #[inline(always)]
    fn gather<'a>(_state: &mut Self::State<'a>, _helper: Self::State<'a>) {}

    /// Resets the variant's per-thread search state when `Dfpn::clear` forgets
    /// the search's tables between solves.
    #[inline(always)]
    fn clear(_state: &mut Self::State<'_>) {}
}

/// Where progress lines go instead of standard error.
pub type ProgressSink = Box<dyn FnMut(&serde_json::Value)>;

/// What runs with the search after each progress line while it goes on
/// (`set_progress_hook`).
pub type ProgressHook<'a, V = variant::Atomic> = Box<dyn FnMut(&Dfpn<'_, V>) + 'a>;

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
pub struct Sampler<P = Position> {
    pub max_depth: usize,
    pub min_pieces: usize,
    /// How many to keep of the attacker's positions, and of the defender's.
    pub keep: usize,
    seen: [u64; 2],
    rng: u64,
    pub kept: [Vec<(P, usize)>; 2],
}

impl<P: Board> Sampler<P> {
    pub fn new(max_depth: usize, min_pieces: usize, keep: usize, seed: u64) -> Sampler<P> {
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
    fn offer(&mut self, pos: &P, attacking: bool, depth: usize) {
        if depth > self.max_depth || (pos.pieces() as usize) < self.min_pieces {
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
pub struct Numbers {
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
pub struct Value {
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

pub struct Child {
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
    /// first fill, a mate in one is not kept either (`drop_mates`), and each
    /// time they fill after that, they drop every proof of at most
    /// `proofs::SHORT` moves (`drop_short`).
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
    /// How many times the proofs have dropped their short proofs
    /// (`drop_short`): once they have, the search and a proof's walk work
    /// one out again where they need it (`short_proof`).
    short_drops: AtomicU64,
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
struct Settings<'a, V: Variant> {
    attacker: Color,
    variant: V::Shared<'a>,
    deadline: Instant,
    stop: Option<&'a AtomicBool>,
    limit: u64,
    drop_mates_at: usize,
    most: Option<u32>,
    exact: bool,
    root_attacking: bool,
}

// By hand, as a derive would ask the variant itself to be `Copy`.
impl<V: Variant> Clone for Settings<'_, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: Variant> Copy for Settings<'_, V> {}

pub struct Dfpn<'a, V: Variant = variant::Atomic> {
    attacker: Color,
    /// What this thread keeps for the variant.
    var: V::State<'a>,
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
    /// Each depth's children, kept between expansions to spare allocations.
    buffers: Vec<Vec<Child>>,
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
    hook: Option<ProgressHook<'a, V>>,
    /// The root's proof and disproof numbers as the search last combined them.
    root: (u32, u32),
    /// Whether the search keeps its line and its root moves' nodes: while it
    /// reports, or when asked (`keep_line`).
    keeping: bool,
    /// The first `LINE` plies of the search's line, `line_len` of them
    /// current, from `line_root` (`Level`).
    levels: [Level; LINE],
    line_len: usize,
    line_root: Option<V::Board>,
    /// This thread's nodes below each root move searched.
    root_moves: Vec<(Move, u64)>,
    /// The highest root proof number a progress line has reported.
    peak: u32,
    /// The perturbation's seed, spread over all bits; 0 for none.
    seed: u64,
    /// The most attacker moves a win may take, if limited.
    most: Option<u32>,
    /// Whether tablebase wins are played out (`set_exact`).
    exact: bool,
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
    pub sampler: Option<Sampler<V::Board>>,
    /// Whether to keep the features of the next expanded position, once.
    pub record: bool,
    pub recorded: Option<[f32; FEATURES]>,
    /// For tests: whether each short proof found is checked against the
    /// search that works one out again (`check_short`), and how many were.
    #[cfg(test)]
    checking: bool,
    #[cfg(test)]
    checked: u64,
}

impl<'a, V: Variant> Dfpn<'a, V> {
    /// A search for `attacker`'s wins, with a table and proofs of these
    /// megabytes, for `threads` threads that search together, sharing the
    /// table and the proofs: this search is one of them, and starts the
    /// others for each `solve`, each with a seed of its own. Unless the
    /// memory cannot be had, as a browser may refuse it.
    fn with_threads(
        attacker: Color,
        variant: V::Shared<'a>,
        deadline: Instant,
        (table_mb, proofs_mb): (usize, usize),
        threads: usize,
    ) -> Result<Dfpn<'a, V>, TryReserveError> {
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
            short_drops: AtomicU64::new(0),
            settled: Mutex::new(None),
            pause: Pause::new(),
            busy: if shared { busy_slots()? } else { Vec::new() },
        };
        let settings = Settings {
            attacker,
            variant,
            deadline,
            stop: None,
            limit: u64::MAX,
            drop_mates_at: usize::MAX,
            most: None,
            exact: false,
            root_attacking: true,
        };
        Ok(Dfpn::with_store(Arc::new(store), settings))
    }

    /// A search with these settings and a state of its own, which keeps what
    /// it learns in `store`.
    fn with_store(store: Arc<Store>, settings: Settings<'a, V>) -> Dfpn<'a, V> {
        Dfpn {
            attacker: settings.attacker,
            var: V::state(settings.variant),
            together: store.threads > 1,
            filtering: store.table.filtering(),
            store,
            flushed: 0,
            puts: 0,
            path: Vec::new(),
            buffers: Vec::new(),
            repetitions: [0; 16],
            deadline: settings.deadline,
            stop: settings.stop,
            limit: settings.limit,
            full: false,
            drop_mates_at: settings.drop_mates_at,
            progress: None,
            hook: None,
            root: (1, 1),
            keeping: false,
            levels: [Level::default(); LINE],
            line_len: 0,
            line_root: None,
            root_moves: Vec::new(),
            peak: 0,
            seed: 0,
            most: settings.most,
            exact: settings.exact,
            root_attacking: settings.root_attacking,
            nodes: 0,
            stopped: false,
            mates_dropped: None,
            root_numbers: None,
            sampler: None,
            record: false,
            recorded: None,
            #[cfg(test)]
            checking: false,
            #[cfg(test)]
            checked: 0,
        }
    }

    /// What a thread that helps this search takes from it.
    fn settings(&self) -> Settings<'a, V> {
        Settings {
            attacker: self.attacker,
            variant: V::shared(&self.var),
            deadline: self.deadline,
            stop: self.stop,
            limit: self.limit,
            drop_mates_at: self.drop_mates_at,
            most: self.most,
            exact: self.exact,
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

    /// Plays tablebase wins out rather than count them as won where they are
    /// reached, so that every line of a win ends in a mate or an explosion
    /// and its length is exact, as `go mate` asks; the tables still refute.
    pub fn set_exact(&mut self) {
        self.exact = true;
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

    /// Runs `hook` with the search after each progress line, unless the
    /// search has stopped: for what reads its proofs while it goes on, as
    /// the browser build writes a shorter win's proof for its page.
    pub fn set_progress_hook(&mut self, hook: ProgressHook<'a, V>) {
        self.hook = Some(hook);
    }

    /// Keeps the search's line without reporting it, for `line` once the
    /// search stops.
    pub fn keep_line(&mut self) {
        self.keeping = true;
    }

    /// How many attacker moves the proven win takes, if one is found in time.
    pub fn solve(&mut self, root: &V::Root) -> Option<u32> {
        self.line_len = 0;
        self.root_moves.clear();
        self.peak = 0;
        // The search settles a position that has ended before reaching it;
        // a root that has ended, such as a sampled defender position, is
        // settled here, for the attacker as set.
        match V::status(root, self.attacker) {
            Status::Won => {
                self.add_proof(V::board(root).hash(), Proof::FINISHED);
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
        let root = V::board(root);
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
            .or_else(|| V::recomputed(self, &root))
            .map(|proof| proof.length)
    }

    /// Searches the root with the other threads until one of them settles it
    /// or they stop: this thread with its own seed, each other with another.
    /// Returns the root's numbers as the thread that settled it left them,
    /// else as this one did.
    fn search_together(&mut self, root: &V::Board) -> Numbers {
        let store = Arc::clone(&self.store);
        store.nodes.store(self.nodes, Relaxed);
        store.stop.store(false, Relaxed);
        *lock(&store.settled) = None;
        self.flushed = self.nodes;
        self.puts = 0;
        let settings = self.settings();
        let numbers = std::thread::scope(|scope| {
            let mut helpers = Vec::new();
            for thread in 1..store.threads {
                let part = Part::new(&store);
                let (shared, seed) = (Arc::clone(&store), self.seed);
                let helper = std::thread::Builder::new().stack_size(crate::STACK);
                // A thread that cannot start drops its part: the others search
                // without it.
                let started = helper.spawn_scoped(scope, move || {
                    let _part = part;
                    let mut search = Dfpn::with_store(shared, settings);
                    search.seed = seed.wrapping_add(SEED_SPREAD.wrapping_mul(thread as u64));
                    let numbers = search.search_root(root);
                    (numbers, search.var)
                });
                helpers.extend(started.ok());
            }
            let numbers = {
                let _part = Part::new(&store);
                self.search_root(root)
            };
            // Joined, as the scope's end does not: it waits for the threads'
            // work, and a thread joined has exited too, which the browser
            // build's next spawn may wait for (web/worker.js).
            for helper in helpers {
                match helper.join() {
                    Ok((_, state)) => V::gather(&mut self.var, state),
                    Err(panic) => std::panic::resume_unwind(panic),
                }
            }
            numbers
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
    fn search_root(&mut self, root: &V::Board) -> Numbers {
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
        root: &V::Root,
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

    /// A position's kept proof for a walk at this counter: the shortest that
    /// holds there, or with `furthest` the one that holds at the highest
    /// counter. A position proven at a low counter and again, by a longer
    /// proof with another move, at a higher one keeps both, and a walk that
    /// meets it at both counters writes both moves when it takes the shortest
    /// each time: every line it walks then shrinks, but the two moves of one
    /// position with the two of another can link them in a circle, which the
    /// format forbids, as it compares positions without their counters
    /// (FORMAT.md). The proof that holds furthest is the same at every
    /// counter, so a walk that takes it writes one move a position; and the
    /// positions its defences reach hold further still, or come after a
    /// capture or pawn move, so its lines end.
    fn kept_proof(&self, hash: u64, counter: u32, furthest: bool) -> Option<Proof> {
        let mut proof = self.proof_at(hash, counter)?;
        while furthest
            && proof.limit < LAST
            && let Some(further) = self.proofs().get(hash, proof.limit + 1)
        {
            proof = further;
        }
        Some(proof)
    }

    /// Forgets everything the search has learned, its table and its proofs,
    /// so that the next `solve` starts afresh: between solves, when no other
    /// thread shares the store.
    pub fn clear(&mut self) {
        let store = Arc::get_mut(&mut self.store).expect("no thread searches between solves");
        store.table.clear();
        store.proofs.clear();
        *store.mates_kept.get_mut() = true;
        *store.full.get_mut() = false;
        *store.short_drops.get_mut() = 0;
        *lock(&store.mates_dropped) = None;
        self.filtering = store.table.filtering();
        self.full = false;
        self.stopped = false;
        self.mates_dropped = None;
        self.root_numbers = None;
        V::clear(&mut self.var);
    }

    /// The proofs kept, and the most the proofs' memory takes.
    pub fn proofs_kept(&self) -> (usize, usize) {
        (self.proofs().taken(), self.proofs().room())
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

    /// How many times the proofs have dropped their short proofs
    /// (`drop_short`).
    pub fn short_drops(&self) -> u64 {
        self.store.short_drops.load(Relaxed)
    }

    fn mid(&mut self, pos: &V::Board, proof_limit: u32, disproof_limit: u32) -> Value {
        let start = self.nodes;
        let hash = pos.hash();
        let counter = pos.halfmoves();
        let remaining = self.remaining(self.path.len());
        // A transposition may have settled the position, or taken it past the
        // limits, since its parent looked.
        if self.proof_within(hash, counter, remaining).is_some() {
            return Value::WON;
        }
        // Out of attacker moves, the attacker to move has lost to the limit.
        // The defender to move may still lose at once, as an Antichess
        // defence can; Atomic's are settled where they are scored.
        if remaining == 0 && pos.turn() == self.attacker {
            return Value::DRAWN;
        }
        let key = self.key(hash, remaining);
        let known = self.lookup(key, counter, |m| V::mates_with(self, pos, m));
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
        let step = V::children(self, pos, attacking, remaining, &mut children);
        V::scored(
            self,
            pos,
            attacking,
            known.is_none(),
            remaining,
            &mut children,
        );
        if let Some(sampler) = &mut self.sampler {
            sampler.offer(pos, attacking, self.path.len());
        }
        if children.is_empty() {
            self.buffers[level] = children;
            let value = V::no_moves(self, pos);
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
                V::open_group(self, pos, remaining, &mut children, step);
                self.path.push(hash);
                continue;
            }
            let current = children[best].value.numbers;
            let (child_proof, child_disproof) = if attacking {
                (
                    proof_limit.min(widen(second, V::EPSILON)),
                    share(disproof_limit, numbers.disproof, current.disproof),
                )
            } else {
                (
                    share(proof_limit, numbers.proof, current.proof),
                    disproof_limit.min(widen(second, V::EPSILON)),
                )
            };
            if self.keeping && (depth as usize) < LINE {
                self.keep_level(depth as usize, attacking, &children, best);
            }
            let next = pos.play_known(children[best].m, children[best].hash);
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
            if !attacking && value.numbers.proof == 0 {
                V::defence_proven(self, pos, &next, best, remaining, &mut children);
            }
            if self.together && !self.stopped && self.nodes - read >= REFRESH_AFTER {
                read = self.nodes;
                self.refresh(pos, &mut children);
            }
        };
        self.path.pop();
        if !attacking && value.numbers.disproof == 0 {
            V::refuted(self, pos, children[best].m);
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
            V::record_proof(self, pos, hash, &children, attacking);
            debug_assert!(
                self.full
                    || self.proof_at(hash, counter).is_some()
                    || V::recomputed(self, pos).is_some(),
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
    fn refresh(&self, pos: &V::Board, children: &mut [Child]) {
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
                    V::mates_with(self, &pos.play(child.m), mate)
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
                .rebuild_filter(self.proofs().checks(), self.proofs().taken());
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
                    .rebuild_filter(store.proofs.checks(), store.proofs.taken());
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

    /// Keeps a proof. When the proofs fill, the first time they drop their
    /// mates in one (`drop_mates`), and each time after that their short
    /// proofs (`drop_short`), until that no longer lets the search go on,
    /// which then stops.
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
        if self.drop_short() && self.proofs().add(hash, proof) {
            return;
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

    /// Drops every proof of at most `proofs::SHORT` moves from the proofs, so
    /// that the search goes on when they fill without their mates in one:
    /// most proofs are that short, and the search works one out again where
    /// it needs it (`short_proof`), as a proof's walk does (`winning_move`).
    /// The longer ones are never dropped, so the lengths still shrink along
    /// every line of a proof. Returns whether the search goes on: not once
    /// the longer proofs take more than nine tenths of the room, when the
    /// drops would soon come one after another; and not under a limit on
    /// moves, where nothing is dropped: a longer proof a position keeps could
    /// then stand in for its short one and pass the limit.
    fn drop_short(&mut self) -> bool {
        if self.most.is_some() {
            return false;
        }
        if self.together {
            return self.drop_short_alone();
        }
        self.proofs().drop_short();
        self.store.short_drops.fetch_add(1, Relaxed);
        self.proofs().taken() * 10 <= self.proofs().room() * 9
    }

    /// Together: `drop_short` while the other threads wait, unless the proofs
    /// have room again, as they do when another thread's drop came first.
    #[cold]
    fn drop_short_alone(&mut self) -> bool {
        let store = &*self.store;
        store.pause.alone(|| {
            let proofs = &store.proofs;
            if proofs.taken() >= proofs.room() {
                proofs.drop_short();
                store.short_drops.fetch_add(1, Relaxed);
                if proofs.taken() * 10 > proofs.room() * 9 {
                    store.full.store(true, Relaxed);
                }
            }
        });
        // Another thread's change may have come first.
        self.filtering = store.table.filtering();
        !store.full.load(Relaxed)
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
            moves.push(pos.uci(level.m));
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
        Some(root.uci(m))
    }

    /// Prints a line on how the search goes, as JSON to standard error, or
    /// hands it to the sink: the seconds and nodes so far, the nodes a second
    /// since the last line, the root's proof and disproof numbers rebuilt
    /// from the line (`root_now`), the highest proof number reported so far
    /// (`peak`), the line and the moves open along it (`line`, `open`), the
    /// root move searched most (`most_searched`, once one is), the share of
    /// the table's slots taken (from a sample), how full the proofs are, 1
    /// being where they fill, the nodes searched when they dropped their
    /// mates in one, once they have, and how many times they have dropped
    /// their short proofs since. Together, the line and the moves are this
    /// thread's.
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
        let dropped = self.mates_dropped.or_else(|| {
            self.together
                .then(|| *lock(&self.store.mates_dropped))
                .flatten()
        });
        let short_drops = self.short_drops();
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
        if short_drops > 0 {
            record["short_drops"] = serde_json::json!(short_drops);
        }
        match &mut progress.sink {
            Some(sink) => sink(&record),
            None => eprintln!("{record}"),
        }
        if !self.stopped
            && let Some(mut hook) = self.hook.take()
        {
            hook(self);
            self.hook = Some(hook);
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
/// node of its budget, at 10 bytes a proof.
const PROOF_MB_A_SECOND: f64 = 1.5;

const PROOF_BYTES_A_NODE: u64 = 3;

/// The most megabytes one table may take: no allocation reaches `isize::MAX`
/// bytes, which on 32-bit targets such as WebAssembly is 2 GB.
const MOST_ONE: usize = isize::MAX as usize >> 20;

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

/// The 1+ε trick, ε = 1/`epsilon` (`Variant::EPSILON`): a child may run
/// until it is somewhat worse than the second-best one, which spares many
/// switches between near-equal children.
#[inline(always)]
fn widen(second: u32, epsilon: u32) -> u32 {
    if second >= INFINITY {
        INFINITY
    } else {
        capped(second + 1 + second / epsilon)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::{Duration, Instant};

    use shakmaty::{Position as _, uci::UciMove, variant::Atomic};

    use super::atomic::{perturb_model_step, scale};
    use super::{
        Child, Dfpn, Guide, INFINITY, Level, Numbers, Proof, Proofs, Value, combine, crowd, share,
        split,
    };
    use crate::board::{Move, Position, from_fen};
    use crate::search::Key;
    use crate::tablebase::Tablebase;

    /// The length of a search's win as `provari prove` writes its proof, in
    /// the compact format, which the checker confirms.
    fn checked_length(search: &Dfpn, root: &Atomic) -> u32 {
        checked_length_with(search, root, None)
    }

    /// `checked_length` with tablebases: the prover's, and the same as the
    /// checker reads them.
    fn checked_length_with(
        search: &Dfpn,
        root: &Atomic,
        tables: Option<(&Tablebase, &shakmaty_syzygy::Tablebase<Atomic>)>,
    ) -> u32 {
        let prover = tables.map(|(prover, _)| prover);
        let write =
            |winning_move: &mut dyn FnMut(&Atomic, &Key, bool) -> Option<shakmaty::Move>| {
                crate::format::walk_search(root, root.turn(), prover, |pos, furthest| {
                    let key = (pos.hash(), pos.halfmoves());
                    winning_move(&pos.to_atomic(), &key, furthest)
                        .and_then(|m| crate::format::board_move(pos, &m))
                })
                .unwrap()
            };
        let (header, indices, _) =
            write(&mut |pos, key, furthest| search.winning_move(pos, key, furthest));
        // Short wins worked out ahead by other threads give the same proof.
        if search.short_drops() > 0 {
            let (_, ahead, _) = search.with_moves_ahead(4, write);
            assert_eq!(ahead, indices);
        }
        let bytes = crate::format::file(&header, &indices).unwrap();
        let checker = tables.map(|(_, checker)| checker);
        let checked = crate::check::check_proof("proof", &bytes, checker);
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
            let (ours, theirs) = (super::atomic::round_by_truncation(x), x.round());
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

    /// With the tablebases, a win counts as won where it reaches one: under a
    /// limit the search agrees with the exhaustive baseline, which counts
    /// them the same way, on whether a win exists within one, two and three
    /// moves, and its proof reads back no longer; shortening brings a first
    /// win down to the baseline's fewest. An exact search agrees with the
    /// baseline without the tables, which plays every line out. In
    /// positions of five pieces past a capture or pawn move, where the
    /// tables decide what captures and pawn moves reach, quiet pawn moves in
    /// the last; with the tables up to five pieces in `PROVARI_TABLEBASES`
    /// (a list as `PATH` is):
    /// `cargo test --release -- --ignored counts_tablebase_wins_as_reached`.
    #[test]
    #[ignore = "needs the tablebases"]
    fn counts_tablebase_wins_as_reached() {
        let path = std::env::var_os("PROVARI_TABLEBASES").expect("PROVARI_TABLEBASES");
        let tables = Tablebase::open(std::path::Path::new(&path)).unwrap();
        let mut checker = shakmaty_syzygy::Tablebase::<Atomic>::new();
        for directory in std::env::split_paths(&path) {
            checker.add_directory(directory).unwrap();
        }
        for fen in [
            // A pawn takes the knight, which leaves the queen.
            "7k/8/8/3n4/4P3/8/8/K5Q1 w - - 3 1",
            // A pawn takes the rook.
            "8/8/5k2/8/2r5/3P4/8/K3Q3 w - - 3 1",
            // Quiet pawn moves reach the tables.
            "k7/1n6/8/8/8/8/6P1/K2Q4 w - - 3 1",
        ] {
            let root = from_fen(fen).unwrap().to_atomic();
            let deadline = Instant::now() + Duration::from_secs(60);
            for most in 1..=3 {
                for exact in [false, true] {
                    let seen = (!exact).then_some(&tables);
                    let baseline = crate::search::Search::new(&root, seen, deadline, 1 << 20)
                        .solve(&root, most);
                    let mut search =
                        Dfpn::new(&root, Some(&tables), Guide::Hand, deadline, split(16, 0));
                    if exact {
                        search.set_exact();
                    }
                    search.set_most(most);
                    let found = search.solve(&root);
                    let case = format!("{fen} within {most}, exact {exact}");
                    assert_eq!(found.is_some(), baseline.is_some(), "{case}");
                    if found.is_some() {
                        let read = seen.map(|tables| (tables, &checker));
                        assert!(checked_length_with(&search, &root, read) <= most, "{case}");
                    }
                }
            }
            let mut search = Dfpn::new(&root, Some(&tables), Guide::Hand, deadline, split(16, 0));
            let first = search.solve(&root).unwrap();
            let shortening = search.shorten(&root, first, |_, _| {});
            let fewest = crate::search::Search::new(&root, Some(&tables), deadline, 1 << 20)
                .solve(&root, first)
                .unwrap();
            assert_eq!(
                (shortening.length, shortening.shortest),
                (fewest, true),
                "{fen}"
            );
            let read = Some((&tables, &checker));
            assert_eq!(checked_length_with(&search, &root, read), fewest, "{fen}");
        }
    }

    /// A search whose proofs fill their memory drops its mates in one, then
    /// its short proofs each time they fill again, and stops once the longer
    /// ones alone leave too little room, claiming no win. Under a limit on
    /// moves it drops no short proofs, and stops when they fill again.
    #[test]
    fn stops_when_its_proofs_are_full() {
        let root = from_fen("r1b1kb1r/pp1p1qp1/2n1p3/1B3P1p/3PP3/2P5/PP4PP/R2QK2R w KQkq - 0 1")
            .unwrap()
            .to_atomic();
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut search = Dfpn::new(&root, None, Guide::Hand, deadline, split(4, 0));
        let found = search.solve(&root);
        assert!(found.is_some() && !search.proofs_full());
        for most in [None, found] {
            let mut search = Dfpn::new(&root, None, Guide::Hand, deadline, split(4, 0));
            Arc::get_mut(&mut search.store).unwrap().proofs = Proofs::sized(16, false);
            if let Some(most) = most {
                search.set_most(most);
            }
            assert_eq!(search.solve(&root), None, "{most:?}");
            assert!(search.proofs_full() && search.stopped && search.mates_dropped.is_some());
            assert_eq!(search.short_drops() > 0, most.is_none(), "{most:?}");
        }
    }

    /// Proofs that fill again once their mates in one are gone drop their
    /// short proofs, as often as they fill, and the search goes on to a win,
    /// its proof whole, where it used to stop: the smaller the proofs, the
    /// more often. With room to spare nothing is dropped. Every short proof
    /// these searches find is one the search for short wins works out again
    /// (`check_short`), which the drops rely on.
    #[test]
    fn drops_its_short_proofs_and_goes_on() {
        let solve = |fen: &str, places: Option<usize>| {
            let root = from_fen(fen).unwrap().to_atomic();
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut search = Dfpn::new(&root, None, Guide::Hand, deadline, split(4, 0));
            if let Some(places) = places {
                Arc::get_mut(&mut search.store).unwrap().proofs = Proofs::sized(places, false);
            }
            search.checking = true;
            let found = search.solve(&root);
            let written = found.map(|_| checked_length(&search, &root));
            assert!(
                found.is_some() && written <= found && !search.proofs_full(),
                "{fen} with {places:?}: {found:?} {written:?}"
            );
            assert!(search.checked > 0, "{fen} with {places:?}");
            search.short_drops()
        };
        let three = "r1bqkbnr/1p1p2pp/p1n2p2/1B2p3/3PP3/2P2N2/PP3PPP/R1BQK2R w KQkq - 0 1";
        let eight = "r1b1kb1r/pp1p1qp1/2n1p3/1B3P1p/3PP3/2P5/PP4PP/R2QK2R w KQkq - 0 1";
        assert_eq!((solve(three, None), solve(eight, None)), (0, 0));
        assert!(solve(three, Some(4)) > 0);
        let (few, many) = (solve(eight, Some(256)), solve(eight, Some(64)));
        assert!(0 < few && few < many, "{few} {many}");
    }

    /// A position that keeps a short proof up to a low counter and a longer
    /// one, with another move, up to a higher one gives a walk the shortest
    /// that holds at its counter, or when asked the one that holds furthest,
    /// the same move at every counter (`kept_proof`): `format::CIRCLE`'s
    /// first rook position, with its two moves.
    #[test]
    fn gives_the_proof_that_holds_furthest() {
        let pos = from_fen("2N1kN2/2R5/3PP3/7p/7P/8/7K/8 w - - 2 2").unwrap();
        let root = pos.to_atomic();
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut search = Dfpn::new(&root, None, Guide::None, deadline, (16, 8));
        let uci =
            |m: Option<Move>| m.map(|m| UciMove::from_standard(pos.to_shakmaty(m)).to_string());
        let m = |uci: &str| {
            let m = uci.parse::<UciMove>().unwrap().to_move(&root).unwrap();
            crate::format::board_move(&pos, &m)
        };
        for (uci, length, limit) in [("c7f7", 3, 4), ("c7e7", 5, 40)] {
            let proof = Proof {
                m: m(uci),
                length,
                limit,
            };
            search.add_proof(pos.hash(), proof);
        }
        for (counter, shortest, furthest) in [
            (2, Some("c7f7"), Some("c7e7")),
            (4, Some("c7f7"), Some("c7e7")),
            (5, Some("c7e7"), Some("c7e7")),
            (40, Some("c7e7"), Some("c7e7")),
            (41, None, None),
        ] {
            let at = pos.at_counter(counter);
            let moves = [false, true].map(|furthest| uci(search.proof_move(&at, furthest)));
            assert_eq!(
                moves,
                [shortest, furthest].map(|m| m.map(str::to_owned)),
                "{counter}"
            );
            let key = (at.hash(), counter);
            let (atomic, m) = (at.to_atomic(), search.proof_move(&at, true));
            let written = search.winning_move(&atomic, &key, true);
            assert_eq!(written, m.map(|m| at.to_shakmaty(m)), "{counter}");
        }
    }

    /// With the tablebases, the short proofs dropped include the wins the
    /// tables give where they are reached, worked out again from the tables:
    /// searches of eight pieces whose proofs fill time and again go on to
    /// their wins, each short proof found one the search for short wins
    /// works out again, and each win's proof whole as the checker reads it
    /// with the tables. An exact search, which plays tablebase wins out,
    /// works its short proofs out again without the tables. With the tables
    /// up to five pieces in `PROVARI_TABLEBASES` (a list as `PATH` is):
    /// `cargo test --release -- --ignored drops_short_proofs_with_tablebases`.
    #[test]
    #[ignore = "needs the tablebases"]
    fn drops_short_proofs_with_tablebases() {
        let path = std::env::var_os("PROVARI_TABLEBASES").expect("PROVARI_TABLEBASES");
        let tables = Tablebase::open(std::path::Path::new(&path)).unwrap();
        let mut checker = shakmaty_syzygy::Tablebase::<Atomic>::new();
        for directory in std::env::split_paths(&path) {
            checker.add_directory(directory).unwrap();
        }
        // A pawn takes the rook, which leaves a queen the tables hold: a
        // win in one move with them, in none where they hold the position
        // itself, its counter at 0, and no short win without them.
        let pawn = from_fen("8/8/5k2/8/2r5/3P4/8/K3Q3 w - - 3 1").unwrap();
        let short = |pos: &Position, tables| {
            crate::format::short(pos, shakmaty::Color::White, tables)
                .map(|(length, m)| (length, m.is_some()))
        };
        assert_eq!(short(&pawn, Some(&tables)), Some((1, true)));
        assert_eq!(short(&pawn.at_counter(0), Some(&tables)), Some((0, false)));
        assert_eq!(short(&pawn, None), None);
        for (fen, exact, places) in [
            ("4k3/pp6/8/8/8/8/PP6/RB2K3 w - - 0 1", false, 1 << 12),
            ("4k3/pp6/8/8/8/8/PP6/RN2K3 w - - 0 1", false, 1 << 14),
            ("4k3/1p6/8/8/8/8/1P6/RN2K3 w - - 0 1", true, 1 << 13),
        ] {
            let root = from_fen(fen).unwrap().to_atomic();
            let deadline = Instant::now() + Duration::from_secs(120);
            let mut search = Dfpn::new(&root, Some(&tables), Guide::Hand, deadline, split(64, 0));
            Arc::get_mut(&mut search.store).unwrap().proofs = Proofs::sized(places, false);
            search.checking = true;
            if exact {
                search.set_exact();
            }
            let found = search.solve(&root);
            assert!(found.is_some() && search.short_drops() > 0, "{fen}");
            assert!(search.checked > 0, "{fen}");
            let read = (!exact).then_some((&tables, &checker));
            assert!(
                Some(checked_length_with(&search, &root, read)) <= found,
                "{fen}"
            );
        }
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

    /// Threads whose proofs fill again once their mates in one are gone drop
    /// their short proofs while the others wait, as often as they fill, and
    /// go on to the win, its proof whole. Two threads' lines keep more long
    /// proofs than one's, which at this size now and then leaves too little
    /// room: the search then stops, claiming no win, and another is run.
    #[test]
    fn threads_drop_their_short_proofs() {
        let root = from_fen("r1b1kb1r/pp1p1qp1/2n1p3/1B3P1p/3PP3/2P5/PP4PP/R2QK2R w KQkq - 0 1")
            .unwrap()
            .to_atomic();
        let won = (0..20).any(|_| {
            let mut search = together(&root, 2, split(4, 0));
            Arc::get_mut(&mut search.store).unwrap().proofs = Proofs::sized(256, true);
            search.checking = true;
            let found = search.solve(&root);
            assert_eq!(found.is_none(), search.proofs_full());
            if found.is_some() {
                checked_length(&search, &root);
            }
            found.is_some() && search.short_drops() > 0
        });
        assert!(won);
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

    /// A search with the network and the model takes every position's value
    /// and first layer from its parent's, and grades a plain move without
    /// its position: both as computed from scratch, which `network_value`
    /// and `grade` assert at each position and move where debug assertions
    /// are on, as in tests. The searches open with pieces to blow up, kings
    /// that move and an ending; the test passes without checking when
    /// `data/network.nnue` or `data/model.json` is absent.
    #[test]
    fn a_search_values_positions_as_from_scratch() {
        use std::path::Path;

        let (network, model) = (Path::new("data/network.nnue"), Path::new("data/model.json"));
        if !cfg!(debug_assertions) || !network.exists() || !model.exists() {
            return;
        }
        let network = crate::nnue::Network::load(network).unwrap();
        let model = crate::model::Model::load(model).unwrap();
        let mut nodes = 0;
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rn2k2r/pp3ppp/2p5/3pP3/8/2N5/PPP2PPP/R3K2R w KQkq d6 0 1",
            "r3k2r/1P4pp/8/2pP4/8/8/p5PP/R3K2R w KQkq c6 0 1",
            "r1b1kb1r/pp1p1qp1/2n1p3/1B3P1p/3PP3/2P5/PP4PP/R2QK2R w KQkq - 0 1",
            "8/1p4pp/2pk4/8/P6P/7B/1PP5/6K1 w - - 1 20",
        ] {
            let root = from_fen(fen).unwrap().to_atomic();
            let mut search = Dfpn::new(
                &root,
                None,
                Guide::Model(&model, Some(&network)),
                Instant::now() + Duration::from_secs(600),
                split(64, 0),
            );
            search.set_limit(50_000);
            search.solve(&root);
            nodes += search.nodes;
        }
        assert!(nodes > 150_000, "{nodes}");
    }

    /// The same of a network in varifish's format, SFNNv20: the test's own,
    /// of generated weights, with a model of weights as arbitrary, since
    /// the values only need to be those from scratch.
    #[test]
    fn a_search_values_an_sfnnv20_networks_positions_as_from_scratch() {
        if !cfg!(debug_assertions) {
            return;
        }
        let network = crate::nnue::tests::sfnnv20();
        let mut random = crate::nnue::tests::Random(0x2545_f491_4f6c_dd1d);
        let mut side = |bias: f32| {
            let mut weights = |count: usize, scale: f32| -> Vec<f32> {
                (0..count)
                    .map(|_| ((random.next() % 2001) as f32 - 1000.0) * scale)
                    .collect()
            };
            serde_json::json!({
                "bias": bias,
                "prior": 0.4,
                "weights": weights(crate::model::FEATURES, 1e-3),
                "layer": weights(crate::model::LAYER, 2e-4),
            })
        };
        let model = serde_json::json!({
            "features": crate::model::NAMES,
            "attacker": side(-0.5),
            "defender": side(0.5),
        });
        let model = crate::model::Model::from_text(&model.to_string()).unwrap();
        let mut nodes = 0;
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rn2k2r/pp3ppp/2p5/3pP3/8/2N5/PPP2PPP/R3K2R w KQkq d6 0 1",
            "r3k2r/1P4pp/8/2pP4/8/8/p5PP/R3K2R w KQkq c6 0 1",
            "r1b1kb1r/pp1p1qp1/2n1p3/1B3P1p/3PP3/2P5/PP4PP/R2QK2R w KQkq - 0 1",
            "8/1p4pp/2pk4/8/P6P/7B/1PP5/6K1 w - - 1 20",
        ] {
            let root = from_fen(fen).unwrap().to_atomic();
            let mut search = Dfpn::new(
                &root,
                None,
                Guide::Model(&model, Some(&network)),
                Instant::now() + Duration::from_secs(600),
                split(64, 0),
            );
            search.set_limit(50_000);
            search.solve(&root);
            nodes += search.nodes;
        }
        assert!(nodes > 150_000, "{nodes}");
    }

    /// `clear` resets `stopped` and Atomic's history tables so that reusing a
    /// `Dfpn` after a stopped search behaves as a fresh one of the same size.
    #[test]
    fn clear_resets_stopped_and_history_for_next_solve() {
        let root = from_fen("1k1r4/ppp5/8/8/8/3Q4/PPP5/1K6 w - - 0 1")
            .unwrap()
            .to_atomic();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut fresh = Dfpn::new(&root, None, Guide::None, deadline, (16, 8));
        let fresh_length = fresh.solve(&root).unwrap();
        let fresh_nodes = fresh.nodes;

        let mut reused = Dfpn::new(&root, None, Guide::None, deadline, (16, 8));
        reused.set_limit(1);
        assert_eq!(reused.solve(&root), None);
        assert!(reused.stopped);

        reused.clear();
        reused.nodes = 0;
        reused.set_limit(u64::MAX);
        assert!(!reused.stopped);
        assert_eq!(reused.solve(&root), Some(fresh_length));
        assert_eq!(reused.nodes, fresh_nodes);
    }
}
