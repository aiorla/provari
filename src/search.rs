//! Exhaustive search for a win in N attacker moves, deepened one move at a
//! time: the win-in-N baseline (`--search exhaustive`).
//!
//! A position is won when the attacker has a move after which every defence
//! loses in fewer moves. Repeating a position on the current line counts as a
//! draw, and so does reaching the 50-move limit, so every proof also holds
//! under those rules.

use crate::clock::Instant;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

use shakmaty::{
    Color, EnPassantMode, KnownOutcome, Move, MoveList, Outcome, Position, Role, fen::Fen,
    variant::Atomic, zobrist::Zobrist64,
};

use crate::tablebase::{Tablebase, Verdict};

/// A position and its 50-move counter, which can turn a win into a draw.
pub type Key = (u64, u32);

/// Keys are Zobrist hashes already, so hashing them again is wasted work.
#[derive(Default)]
pub struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(*byte);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 ^= value;
    }

    fn write_u32(&mut self, value: u32) {
        self.0 ^= u64::from(value).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

pub type Table<V> = HashMap<Key, V, BuildHasherDefault<KeyHasher>>;

pub fn hash(pos: &Atomic) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

pub fn key(pos: &Atomic) -> Key {
    (hash(pos), pos.halfmoves())
}

#[derive(Clone, Copy)]
enum Bound {
    /// Won within this many attacker moves.
    Won(u32),
    /// Not won within this many.
    NotWithin(u32),
}

pub enum Status {
    /// The game ended with the attacker winning.
    Won,
    /// The game ended otherwise.
    Over,
    Open,
}

pub struct Search<'a> {
    attacker: Color,
    tablebase: Option<&'a Tablebase>,
    known: Table<Bound>,
    /// The winning move of every won attacker position, with its bound.
    strategy: Table<(Move, u32)>,
    path: Vec<u64>,
    capacity: usize,
    deadline: Instant,
    pub nodes: u64,
    pub stopped: bool,
}

impl<'a> Search<'a> {
    pub fn new(
        root: &Atomic,
        tablebase: Option<&'a Tablebase>,
        deadline: Instant,
        capacity: usize,
    ) -> Search<'a> {
        Search {
            attacker: root.turn(),
            tablebase,
            known: Table::default(),
            strategy: Table::default(),
            path: Vec::new(),
            capacity,
            deadline,
            nodes: 0,
            stopped: false,
        }
    }

    /// The fewest attacker moves that win, if a win within `most` is found in
    /// time.
    pub fn solve(&mut self, root: &Atomic, most: u32) -> Option<u32> {
        match status(root, self.attacker) {
            Status::Won => return Some(0),
            Status::Over => return None,
            Status::Open => {}
        }
        for moves in 1..=most {
            if self.attacker_to_move(root, moves) {
                return Some(moves);
            }
            if self.stopped {
                break;
            }
        }
        None
    }

    fn attacker_to_move(&mut self, pos: &Atomic, moves: u32) -> bool {
        if self.tick() {
            return false;
        }
        let key = key(pos);
        match self.known.get(&key) {
            Some(Bound::Won(within)) if *within <= moves => return true,
            Some(Bound::NotWithin(within)) if *within >= moves => return false,
            _ => {}
        }
        match self.probe(pos) {
            Some(Verdict::Win) => {
                self.known.insert(key, Bound::Won(0));
                return true;
            }
            Some(_) => {
                self.known.insert(key, Bound::NotWithin(u32::MAX));
                return false;
            }
            None => {}
        }
        let legal = pos.legal_moves();
        // A move that ends the game is tried before any deeper line.
        for m in &legal {
            let mut child = pos.clone();
            child.play_unchecked(*m);
            if matches!(status(&child, self.attacker), Status::Won) {
                self.record(key, *m, 1);
                return true;
            }
        }
        if moves > 1 {
            self.path.push(key.0);
            for m in attacker_order(legal) {
                let mut child = pos.clone();
                child.play_unchecked(m);
                if !matches!(status(&child, self.attacker), Status::Open) || self.repeats(&child) {
                    continue;
                }
                if self.defender_to_move(&child, moves - 1) {
                    self.path.pop();
                    self.record(key, m, moves);
                    return true;
                }
                if self.stopped {
                    break;
                }
            }
            self.path.pop();
        }
        if !self.stopped {
            self.remember(key, Bound::NotWithin(moves));
        }
        false
    }

    /// Every defence loses within `moves` attacker moves.
    fn defender_to_move(&mut self, pos: &Atomic, moves: u32) -> bool {
        if self.tick() {
            return false;
        }
        if let Some(verdict) = self.probe(pos) {
            return verdict == Verdict::Loss;
        }
        let legal = pos.legal_moves();
        if legal.is_empty() {
            return false;
        }
        self.path.push(key(pos).0);
        let mut holds = true;
        for m in defender_order(legal) {
            let mut child = pos.clone();
            child.play_unchecked(m);
            let lost = match status(&child, self.attacker) {
                Status::Won => true,
                Status::Over => false,
                Status::Open => !self.repeats(&child) && self.attacker_to_move(&child, moves),
            };
            if !lost {
                holds = false;
                break;
            }
        }
        self.path.pop();
        holds
    }

    fn probe(&self, pos: &Atomic) -> Option<Verdict> {
        self.tablebase?.probe(pos)
    }

    fn repeats(&self, pos: &Atomic) -> bool {
        self.path.contains(&key(pos).0)
    }

    /// Keeps the shortest win, so a proof's bounds shrink along every line.
    fn record(&mut self, key: Key, m: Move, within: u32) {
        let shorter = self
            .strategy
            .get(&key)
            .is_none_or(|(_, known)| within < *known);
        if shorter {
            self.strategy.insert(key, (m, within));
        }
        self.remember(key, Bound::Won(within.min(self.strategy[&key].1)));
    }

    fn remember(&mut self, key: Key, bound: Bound) {
        if self.known.len() >= self.capacity {
            self.known.clear();
        }
        self.known.insert(key, bound);
    }

    fn tick(&mut self) -> bool {
        self.nodes += 1;
        if self.nodes.is_multiple_of(4096) && Instant::now() >= self.deadline {
            self.stopped = true;
        }
        self.stopped
    }

    /// The winning move found for a won attacker position.
    pub fn winning_move(&self, key: &Key) -> Option<Move> {
        self.strategy.get(key).map(|(m, _)| *m)
    }
}

/// How a game stands after a move, for the attacker.
pub fn status(pos: &Atomic, attacker: Color) -> Status {
    match pos.variant_outcome() {
        Outcome::Known(KnownOutcome::Decisive { winner }) if winner == attacker => {
            return Status::Won;
        }
        Outcome::Known(_) => return Status::Over,
        Outcome::Unknown => {}
    }
    // Only a position in check can be mate; a stalemate stays open and
    // fails when the search finds no move in it.
    if pos.is_check() && pos.legal_moves().is_empty() {
        return if pos.turn() != attacker {
            Status::Won
        } else {
            Status::Over
        };
    }
    if pos.is_insufficient_material() || pos.halfmoves() >= 100 {
        return Status::Over;
    }
    Status::Open
}

/// Reads a proof back from the root, handing `emit` every attacker position
/// the win reaches with the move a search recorded for it, or none for a
/// tablebase win, in the order a checker meets them; except where a capture
/// wins at once (an explosion of the king, as a rule), which the checkers find
/// for themselves. Returns how many positions it handed over.
pub fn read_proof(
    root: &Atomic,
    tablebase: Option<&Tablebase>,
    mut winning_move: impl FnMut(&Atomic, &Key) -> Option<Move>,
    mut emit: impl FnMut(&Atomic, Option<Move>) -> Result<(), String>,
) -> Result<usize, String> {
    let attacker = root.turn();
    match status(root, attacker) {
        Status::Won => return Ok(0),
        Status::Over => return Err("the root has ended without the attacker winning".into()),
        Status::Open => {}
    }
    let probe = |pos: &Atomic| tablebase.and_then(|tables| tables.probe(pos));
    let mut count = 0;
    let mut seen = HashSet::new();
    let mut pending = vec![root.clone()];
    while let Some(pos) = pending.pop() {
        let key = key(&pos);
        if !seen.insert(key) {
            continue;
        }
        if probe(&pos) == Some(Verdict::Win) {
            emit(&pos, None)?;
            count += 1;
            continue;
        }
        if wins_by_capture(&pos, attacker) {
            continue;
        }
        let m = winning_move(&pos, &key).ok_or_else(|| {
            let fen = Fen::from_position(&pos, EnPassantMode::Legal);
            format!("no winning move recorded for {fen}")
        })?;
        emit(&pos, Some(m))?;
        count += 1;
        let mut after = pos.clone();
        after.play_unchecked(m);
        if !matches!(status(&after, attacker), Status::Open) || probe(&after) == Some(Verdict::Loss)
        {
            continue;
        }
        for reply in after.legal_moves() {
            let mut next = after.clone();
            next.play_unchecked(reply);
            if matches!(status(&next, attacker), Status::Open) {
                pending.push(next);
            }
        }
    }
    Ok(count)
}

/// Whether a capture wins at once, as the checkers look for one: an explosion
/// of the king, en passant included, or a mate.
fn wins_by_capture(pos: &Atomic, attacker: Color) -> bool {
    pos.legal_moves().iter().any(|m| {
        let mut next = pos.clone();
        next.play_unchecked(*m);
        m.is_capture() && matches!(status(&next, attacker), Status::Won)
    })
}

/// Captures first: in Atomic every capture explodes.
fn attacker_order(legal: MoveList) -> Vec<Move> {
    let mut moves: Vec<Move> = legal.into_iter().collect();
    moves.sort_by_key(|m| if m.is_capture() { 0 } else { 1 });
    moves
}

/// Captures first, then king moves: the likeliest escapes.
fn defender_order(legal: MoveList) -> Vec<Move> {
    let mut moves: Vec<Move> = legal.into_iter().collect();
    moves.sort_by_key(|m| {
        if m.is_capture() {
            0
        } else if m.role() == Role::King {
            1
        } else {
            2
        }
    });
    moves
}

#[cfg(test)]
mod tests {
    use super::{Search, read_proof};
    use shakmaty::variant::Atomic;
    use std::time::{Duration, Instant};

    #[test]
    fn a_root_draw_precedes_a_capture_that_would_win() {
        for clock in [99, 100] {
            let root = crate::position(&format!(
                "rnbqkb1r/pp1p3p/2pBppp1/8/3PP3/2N5/PPP2P1P/R2QKB1R w KQ - {clock} 2"
            ))
            .unwrap();
            let mut search = Search::new(&root, None, Instant::now() + Duration::from_secs(1), 100);
            assert_eq!(search.solve(&root, 1), (clock == 99).then_some(1));
            let winning_move = |pos: &Atomic, _: &_| {
                "d6f8"
                    .parse::<shakmaty::uci::UciMove>()
                    .unwrap()
                    .to_move(pos)
                    .ok()
            };
            let proof = read_proof(&root, None, winning_move, |_, _| Ok(()));
            assert_eq!(proof.is_ok(), clock == 99);
        }
    }
}
