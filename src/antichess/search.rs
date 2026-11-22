//! The search for White wins within a number of White moves, as a prover
//! runs it: the depth-first proof-number search (`crate::dfpn`) with its
//! Antichess variant (`crate::dfpn::antichess`), with a node budget and a
//! deadline for each search and its outcome in Antichess's terms.
//!
//! The White moves left are part of what is searched: a position with `k`
//! left is a different problem from the same one with more, and the table
//! keeps them apart. Positions are keyed without the 50-move counter, so that
//! move orders share their work: a proof records the highest counter it holds
//! at, and a refutation that relies on the 50-move rule the counter it holds
//! from. Repetitions need no handling (`rules`).

use std::time::{Duration, Instant};

use serde_json::{Value as Json, json};
use shakmaty::{Color, Position, variant::Antichess};

use crate::antichess::format::{self, Written};
use crate::antichess::nnue::Network;
use crate::antichess::rules;
use crate::antichess::tablebases::Tablebases;
use crate::dfpn::Dfpn as Search;
use crate::proofs::Proof;
use crate::variant;

pub use crate::dfpn::antichess::{Settled, settle};

/// The most White moves a proof may need: what its 10 bits hold.
pub const MOST_NEED: u32 = (1 << 10) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    Nodes,
    Time,
    /// The proofs filled their memory.
    Memory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// White wins within `need` moves.
    Won { need: u32 },
    /// White does not win within the moves given.
    NotWithin,
    /// The search stopped first.
    Unknown(Stop),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Limits {
    pub nodes: Option<u64>,
    pub deadline: Option<Instant>,
}

pub struct Dfpn {
    search: Search<'static, variant::Antichess>,
    /// The threads that search together.
    pub threads: usize,
    /// Positions expanded, over every search and thread.
    pub nodes: u64,
    /// Positions expanded by their count of pieces, over every search and
    /// thread.
    pub by_pieces: [u64; 33],
}

impl Dfpn {
    /// A search with a table of `table_mb` and proofs of `proofs_mb`, which
    /// later searches share until `clear`. Panics if the memory cannot be had.
    pub fn new(table_mb: usize, proofs_mb: usize) -> Dfpn {
        Dfpn::with_threads(table_mb, proofs_mb, 1)
    }

    /// `new`'s search, run by `threads` threads that share its table and
    /// proofs (`crate::dfpn`'s threads): one is `new`'s, node for node.
    pub fn with_threads(table_mb: usize, proofs_mb: usize, threads: usize) -> Dfpn {
        let search = Search::antichess((table_mb, proofs_mb), threads)
            .unwrap_or_else(|error| panic!("no memory for the search: {error}"));
        Dfpn {
            search,
            threads: threads.max(1),
            nodes: 0,
            by_pieces: [0; 33],
        }
    }

    /// Guides later searches by Fairy-Stockfish's Antichess network
    /// (`nnue`), loaded for the whole process.
    pub fn set_network(&mut self, network: &'static Network) {
        self.search.set_network(network);
    }

    /// The fewest White moves left at which the network guides the search.
    pub fn set_guide_from(&mut self, moves: u32) {
        self.search.set_guide_from(moves);
    }

    /// Settles later searches' positions of four pieces or fewer from
    /// tolius's tables where they can (`settle`).
    pub fn set_tablebases(&mut self, tables: Tablebases) {
        self.search.set_tablebases(tables);
    }

    /// The tables the search reads, if any.
    pub fn tablebases(&self) -> Option<Tablebases> {
        self.search.tablebases()
    }

    /// The positions the tables have settled, over every search and thread.
    pub fn tablebase_hits(&self) -> u64 {
        self.search.tablebase_hits()
    }

    /// The proofs kept.
    pub fn proofs_len(&self) -> usize {
        self.search.proofs_kept().0
    }

    /// The most proofs the memory takes.
    pub fn proofs_capacity(&self) -> usize {
        self.search.proofs_kept().1
    }

    /// Forgets everything learned, to start afresh.
    pub fn clear(&mut self) {
        self.search.clear();
    }

    /// The shortest proof kept of a position, by its key (`rules::key`), that
    /// holds at `counter` within `budget` White moves: none for a finished
    /// game or one the tables settled, which are not kept.
    pub fn proof(&self, key: u64, counter: u32, budget: u32) -> Option<Proof> {
        self.search
            .proof(key, counter)
            .filter(|proof| proof.length <= budget)
    }

    /// Whether White, to move in `root`, wins within `most` White moves, with
    /// these limits counted from now.
    pub fn solve(&mut self, root: &Antichess, most: u32, limits: Limits) -> Outcome {
        assert_eq!(root.turn(), Color::White, "the prover proves White wins");
        let search = &mut self.search;
        search.set_most(most);
        let limit = limits
            .nodes
            .map_or(u64::MAX, |nodes| search.nodes.saturating_add(nodes));
        search.set_limit(limit);
        search.stopped = false;
        // A deadline far off stands for none.
        let far = Instant::now() + Duration::from_secs(100 * 365 * 86_400);
        search.set_deadline(limits.deadline.unwrap_or(far));
        let found = search.solve(root);
        self.nodes = search.nodes;
        self.by_pieces = search.by_pieces();
        if let Some(need) = found {
            return Outcome::Won { need };
        }
        if search.proofs_full() {
            Outcome::Unknown(Stop::Memory)
        } else if !search.stopped {
            Outcome::NotWithin
        } else if search.nodes >= limit {
            Outcome::Unknown(Stop::Nodes)
        } else {
            Outcome::Unknown(Stop::Time)
        }
    }
}

/// The most White moves a win may take when nothing says.
pub const DEFAULT_MOST: u32 = 100;
/// A search's memory, its table and its proofs, when nothing says.
pub const DEFAULT_MEMORY_MB: usize = 256;

/// A budget of nodes and seconds, counted from now; one too long to reach is
/// none.
pub fn limits(nodes: Option<u64>, seconds: Option<f64>) -> Limits {
    Limits {
        nodes,
        deadline: seconds.and_then(|seconds| {
            Instant::now().checked_add(Duration::try_from_secs_f64(seconds).ok()?)
        }),
    }
}

/// A bound on a win's White moves, as long as a proof can record.
pub fn most(most: u32) -> Result<u32, String> {
    if most <= MOST_NEED {
        Ok(most)
    } else {
        Err(format!("a win may take at most {MOST_NEED} White moves"))
    }
}

/// A position to search, from its FEN: White must be to move.
pub fn root(fen: &str) -> Result<Antichess, String> {
    let pos = crate::antichess::rules::parse_fen(fen)?;
    if pos.turn() != Color::White {
        return Err(format!(
            "{fen}: the prover proves White wins, so White must be to move"
        ));
    }
    Ok(pos)
}

/// Searches one position for a win within `most` White moves, and on a win
/// writes its proof: the result as a JSON object, with the nodes, time,
/// threads and tables' work of this search alone, and the proof.
pub fn search(
    dfpn: &mut Dfpn,
    pos: &Antichess,
    most: u32,
    limits: Limits,
) -> (Json, Option<Written>) {
    let started = Instant::now();
    let (before, hits, by_pieces) = (dfpn.nodes, dfpn.tablebase_hits(), dfpn.by_pieces);
    let outcome = dfpn.solve(pos, most, limits);
    // No position below the root has more pieces than it.
    let pieces = pos.board().occupied().count();
    let mut result = json!({
        "most": most,
        "nodes": dfpn.nodes - before,
        "ms": started.elapsed().as_millis() as u64,
        "proofs_kept": dfpn.proofs_len(),
        "nodes_by_pieces": (0..=pieces)
            .map(|n| dfpn.by_pieces[n] - by_pieces[n])
            .collect::<Vec<u64>>(),
    });
    if dfpn.threads > 1 {
        result["threads"] = json!(dfpn.threads);
    }
    if let Some(tables) = dfpn.tablebases() {
        result["tablebase_hits"] = json!(dfpn.tablebase_hits() - hits);
        // For the whole process, over its searches so far.
        result["tablebase_memory_mb"] = json!(tables.kept() >> 20);
    }
    let mut written = None;
    match outcome {
        Outcome::Won { need } => match format::from_search(dfpn, pos, most) {
            Ok(proof) => {
                result["found"] = json!(true);
                result["need"] = json!(need);
                // The highest 50-move counter the root's proof holds at, or
                // null for any.
                let root = dfpn.proof(rules::key(pos), pos.halfmoves(), most);
                result["limit"] = match root.map(|root| root.limit) {
                    Some(limit) if limit <= rules::LAST_COUNTER => json!(limit),
                    _ => Json::Null,
                };
                result["proof_positions"] = json!(proof.walked);
                written = Some(proof);
            }
            Err(error) => {
                result["found"] = json!(false);
                result["error"] = json!(error);
            }
        },
        Outcome::NotWithin => {
            result["found"] = json!(false);
            result["refuted"] = json!(true);
        }
        Outcome::Unknown(stop) => {
            result["found"] = json!(false);
            result["stopped"] = json!(match stop {
                Stop::Nodes => "nodes",
                Stop::Time => "time",
                Stop::Memory => "memory",
            });
        }
    }
    (result, written)
}

/// `search` of a position from nothing: what the search learned from earlier
/// positions is forgotten first, so that no search depends on those before.
/// With `share` it is kept instead, its proofs and table guiding this search
/// too, until the proofs fill half their room. `limits` gives the search its
/// budget once it starts.
pub fn search_next(
    dfpn: &mut Dfpn,
    pos: &Antichess,
    most: u32,
    share: bool,
    limits: impl FnOnce() -> Limits,
) -> (Json, Option<Written>) {
    let half_full = dfpn.proofs_len() > dfpn.proofs_capacity() / 2;
    if dfpn.nodes > 0 && (!share || half_full) {
        dfpn.clear();
    }
    let before = dfpn.proofs_len();
    let (mut result, proof) = search(dfpn, pos, most, limits());
    if share {
        result["proofs_before"] = json!(before);
    }
    (result, proof)
}

#[cfg(test)]
mod tests {
    use super::{Dfpn, Limits, Outcome, root, search};
    use crate::antichess::format;

    /// After 1. e3 d6, which White wins in 14 moves and not in 8.
    const D6: &str = "rnbqkbnr/ppp1pppp/3p4/8/8/4P3/PPPP1PPP/RNBQKBNR w - - 0 2";

    /// Threads that search together win and refute where one thread does,
    /// write proofs that read back, and count every thread's nodes.
    #[test]
    fn threads_prove_and_refute_what_one_does() {
        let pos = root(D6).unwrap();
        for threads in [1, 2, 4] {
            let mut dfpn = Dfpn::with_threads(4, 12, threads);
            let (result, proof) = search(&mut dfpn, &pos, 14, Limits::default());
            assert_eq!(result["found"], true, "{threads} threads");
            let proof = proof.unwrap();
            let (header, indices) = format::decode(&proof.bytes).unwrap();
            assert!(header.moves <= 14);
            assert!(!format::steps(&header, &indices).unwrap().is_empty());
            let by_pieces: u64 = result["nodes_by_pieces"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap())
                .sum();
            assert_eq!(by_pieces, result["nodes"].as_u64().unwrap());
            assert_eq!(result.get("threads").is_some(), threads > 1);
            dfpn.clear();
            assert_eq!(dfpn.solve(&pos, 8, Limits::default()), Outcome::NotWithin);
        }
    }
}
