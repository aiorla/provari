//! Exhaustive search for the fewest White moves that win, one length at a
//! time: the baseline that the proof-number search must agree with. Only
//! small positions and lengths are within its reach.

use std::collections::HashMap;

use shakmaty::{Color, Position, variant::Antichess};

use crate::antichess::rules::{self, Status};

/// The fewest White moves, up to `most`, in which White, to move, wins.
pub fn fewest(root: &Antichess, most: u32) -> Option<u32> {
    let mut known = HashMap::new();
    (0..=most).find(|&moves| white_wins(root, moves, &mut known))
}

type Known = HashMap<(u64, u32, u32), bool>;

/// White, to move, wins within `moves`.
fn white_wins(pos: &Antichess, moves: u32, known: &mut Known) -> bool {
    debug_assert_eq!(pos.turn(), Color::White);
    let legal = pos.legal_moves();
    match rules::status(pos, &legal) {
        Status::WhiteWon => return true,
        Status::Open if moves > 0 => {}
        _ => return false,
    }
    let entry = (rules::key(pos), pos.halfmoves(), moves);
    if let Some(&won) = known.get(&entry) {
        return won;
    }
    let won = legal.iter().any(|&m| {
        let mut after = pos.clone();
        after.play_unchecked(m);
        black_loses(&after, moves - 1, known)
    });
    known.insert(entry, won);
    won
}

/// Black, to move, loses whatever it plays, White then winning within `moves`.
fn black_loses(pos: &Antichess, moves: u32, known: &mut Known) -> bool {
    let legal = pos.legal_moves();
    if rules::status(pos, &legal) != Status::Open {
        return false;
    }
    legal.iter().all(|&m| {
        let mut after = pos.clone();
        after.play_unchecked(m);
        white_wins(&after, moves, known)
    })
}
