//! A proof's walk: a proof here is a map from attacker boards to their
//! winning moves, one move a board as the prover keeps them, and walking it
//! measures each attacker position's subproof, its tree and its length. A
//! board met again at another 50-move counter is the same position for a
//! repetition, so the walk catches a repetition as a cycle of boards. A tree
//! counts the attacker positions of the strategy tree, a position reached
//! along two lines twice, as proof numbers count leaves.

use std::collections::{HashMap, HashSet};

use shakmaty::{Color, Move, Position as _, variant::Atomic};

use crate::search::{Status, hash, status};
use crate::tablebase::{Tablebase, Verdict};

/// Attacker boards and their winning moves; none for a tablebase win.
pub type Moves = HashMap<u64, Option<Move>>;

/// A subproof's tree size and length in attacker moves.
#[derive(Clone, Copy, Debug)]
pub struct Measure {
    pub tree: u64,
    pub length: u32,
}

/// A capture that wins at once, which the checkers find for themselves.
pub fn winning_capture(pos: &Atomic, attacker: Color) -> Option<Move> {
    pos.legal_moves().into_iter().find(|m| {
        m.is_capture() && {
            let mut next = pos.clone();
            next.play_unchecked(*m);
            matches!(status(&next, attacker), Status::Won)
        }
    })
}

fn wins_by_capture(pos: &Atomic, attacker: Color) -> bool {
    winning_capture(pos, attacker).is_some()
}

/// What a walk of a proof learns: each attacker position's subproof and the
/// position itself, and the line being walked, to catch a repetition.
#[derive(Default)]
pub struct Walk {
    pub measures: HashMap<u64, Measure>,
    pub positions: HashMap<u64, Atomic>,
    active: HashSet<u64>,
    /// Whether to leave `positions` empty, for walks that only measure.
    measures_only: bool,
}

impl Walk {
    /// A walk that keeps the measures but not the positions.
    pub fn measures_only() -> Walk {
        Walk {
            measures_only: true,
            ..Walk::default()
        }
    }
}

/// Measures the subproof of an attacker position, memoised by position; an
/// error for a missing move or a cycle.
pub fn measure(
    pos: &Atomic,
    attacker: Color,
    moves: &Moves,
    tablebase: Option<&Tablebase>,
    walk: &mut Walk,
) -> Result<Measure, String> {
    let here = hash(pos);
    if walk.active.contains(&here) {
        return Err("the proof repeats a position".into());
    }
    if let Some(measure) = walk.measures.get(&here) {
        return Ok(*measure);
    }
    let probe = |pos: &Atomic| tablebase.and_then(|tables| tables.probe(pos));
    let measure = if probe(pos) == Some(Verdict::Win) {
        Measure { tree: 1, length: 0 }
    } else if wins_by_capture(pos, attacker) {
        Measure { tree: 0, length: 1 }
    } else {
        let m = moves
            .get(&here)
            .copied()
            .flatten()
            .ok_or("a position of the proof has no winning move")?;
        walk.active.insert(here);
        let mut after = pos.clone();
        after.play_unchecked(m);
        let replies = measure_after(&after, attacker, moves, tablebase, walk);
        // Off the line whether or not it failed, so that the walk can go on.
        walk.active.remove(&here);
        let replies = replies?;
        if !walk.measures_only {
            walk.positions.insert(here, pos.clone());
        }
        Measure {
            tree: replies.tree.saturating_add(1),
            length: replies.length + 1,
        }
    };
    walk.measures.insert(here, measure);
    Ok(measure)
}

/// The replies' subproofs after an attacker's move: their tree sizes summed
/// and their longest length.
pub fn measure_after(
    after: &Atomic,
    attacker: Color,
    moves: &Moves,
    tablebase: Option<&Tablebase>,
    walk: &mut Walk,
) -> Result<Measure, String> {
    let mut total = Measure { tree: 0, length: 0 };
    if !matches!(status(after, attacker), Status::Open)
        || tablebase.and_then(|tables| tables.probe(after)) == Some(Verdict::Loss)
    {
        return Ok(total);
    }
    for reply in after.legal_moves() {
        let mut next = after.clone();
        next.play_unchecked(reply);
        if matches!(status(&next, attacker), Status::Open) {
            let child = measure(&next, attacker, moves, tablebase, walk)?;
            total.tree = total.tree.saturating_add(child.tree);
            total.length = total.length.max(child.length);
        }
    }
    Ok(total)
}
