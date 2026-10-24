//! A hand-crafted Atomic evaluation after Multi-Variant Stockfish's Atomic
//! terms, to guide the search as the network does at a fraction of its cost:
//! material at Atomic values, then, unless the kings touch, explosion
//! targets, pieces beside the enemy king, attacks around it, and checks.

use shakmaty::{Bitboard, Color, Role, attacks};

use crate::board::Position;

/// Piece values by role, between Multi-Variant Stockfish's Atomic midgame and
/// endgame values: pawns survive blasts, so the pieces are worth fewer pawns.
const VALUES: [i32; 7] = [0, 305, 545, 634, 930, 1670, 0];
/// One of their pieces beside their king that we attack: a threat to explode
/// the king; two are nearly decisive.
const TARGET: i32 = 400;
const TARGETS: i32 = 1000;
/// One of their pieces beside their king, which a capture would explode.
const FUSE: i32 = 60;
/// Each of our attacks on a square beside their king.
const PRESSURE: i32 = 20;
/// Each of our pieces that can check next move.
const CHECKER: i32 = 60;

/// The evaluation for the side to move.
pub fn evaluate(pos: &Position) -> i32 {
    let us = pos.turn();
    terms(pos, us).score() - terms(pos, !us).score()
}

/// What one side has: its material and its attack on the other king.
#[derive(Clone, Copy, Default)]
pub struct Terms {
    pub material: i32,
    /// Their pieces beside their king that we attack, up to two.
    pub targets: u32,
    /// Their pieces beside their king.
    pub fuses: u32,
    /// Our attacks on the squares beside their king.
    pub pressure: u32,
    /// Our pieces that can check next move.
    pub checkers: u32,
}

impl Terms {
    fn score(&self) -> i32 {
        let targets = match self.targets {
            0 => 0,
            1 => TARGET,
            _ => TARGETS,
        };
        self.material
            + targets
            + FUSE * self.fuses as i32
            + PRESSURE * self.pressure as i32
            + CHECKER * self.checkers as i32
    }
}

/// One side's material at Atomic values, without its attack terms.
pub fn material(pos: &Position, color: Color) -> i32 {
    (pos.by_color(color) & !pos.by_role(Role::King))
        .into_iter()
        .map(|square| pos.role_at(square).map_or(0, |role| VALUES[role as usize]))
        .sum()
}

pub fn terms(pos: &Position, color: Color) -> Terms {
    let ours = pos.by_color(color);
    let theirs = pos.by_color(!color);
    let occupied = pos.occupied();
    let kings = pos.by_role(Role::King);
    let mut terms = Terms::default();
    for square in ours & !kings {
        terms.material += pos.role_at(square).map_or(0, |role| VALUES[role as usize]);
    }
    let (Some(their_king), Some(our_king)) = (pos.king(!color), pos.king(color)) else {
        return terms;
    };
    let ring = attacks::king_attacks(their_king);
    // Touching kings can be neither checked nor exploded.
    if ring.contains(our_king) {
        return terms;
    }
    let targets = ring & theirs & !attacks::king_attacks(our_king);
    let mut attacked = Bitboard::EMPTY;
    let mut pressure = 0u32;
    let mut checkers = 0u32;
    for square in ours & !kings {
        let Some(role) = pos.role_at(square) else {
            continue;
        };
        let reach = attacks::attacks(square, role.of(color), occupied);
        attacked |= reach;
        pressure += (reach & ring).count() as u32;
        // Squares from which this role would check their king.
        let checks = attacks::attacks(their_king, role.of(!color), occupied);
        if (reach & checks & !ours).any() {
            checkers += 1;
        }
    }
    terms.targets = (attacked & targets).count().min(2) as u32;
    terms.fuses = (ring & theirs).count() as u32;
    terms.pressure = pressure;
    terms.checkers = checkers;
    terms
}
