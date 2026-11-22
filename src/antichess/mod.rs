//! Antichess under Lichess's rules, as provari proves it (`dfpn::antichess`):
//! its rules on shakmaty's (`rules`), Fairy-Stockfish's network (`nnue`) and
//! tolius's depth-to-win tables (`tablebases`), and the board the search plays
//! on (`Board`).

pub mod exhaustive;
pub mod format;
pub mod nnue;
pub mod rules;
pub mod search;
pub mod tablebases;

use shakmaty::{Color, Position, Role, Square, variant::Antichess};

use crate::board::Move;

/// An Antichess position as the search plays it: shakmaty's, with its key
/// (`rules::key`), which each move updates from what it changes
/// (`rules::child_key`) rather than working it out afresh.
#[derive(Clone)]
pub struct Board {
    pub pos: Antichess,
    pub key: u64,
}

impl Board {
    pub fn new(pos: Antichess) -> Board {
        let key = rules::key(&pos);
        Board { pos, key }
    }

    /// The shakmaty move a 16-bit one stands for, legal or not.
    pub fn to_shakmaty(&self, m: Move) -> shakmaty::Move {
        let (from, to) = (m.from(), m.to());
        if m.is_en_passant() {
            return shakmaty::Move::EnPassant { from, to };
        }
        let board = self.pos.board();
        shakmaty::Move::Normal {
            role: board.role_at(from).unwrap_or(Role::Pawn),
            from,
            capture: board.role_at(to),
            to,
            promotion: m.promotion(),
        }
    }

    /// The position after a shakmaty move, legal here, with its key.
    pub fn after(&self, m: shakmaty::Move) -> Board {
        let key = rules::child_key(&self.pos, self.key, m);
        let mut pos = self.pos.clone();
        pos.play_unchecked(m);
        Board { pos, key }
    }
}

impl crate::dfpn::Board for Board {
    fn hash(&self) -> u64 {
        self.key
    }

    fn halfmoves(&self) -> u32 {
        self.pos.halfmoves()
    }

    fn turn(&self) -> Color {
        self.pos.turn()
    }

    fn play(&self, m: Move) -> Board {
        self.after(self.to_shakmaty(m))
    }

    fn play_known(&self, m: Move, key: u64) -> Board {
        let mut pos = self.pos.clone();
        pos.play_unchecked(self.to_shakmaty(m));
        Board { pos, key }
    }

    fn pieces(&self) -> u32 {
        self.pos.board().occupied().count() as u32
    }

    fn uci(&self, m: Move) -> String {
        rules::uci(self.to_shakmaty(m))
    }
}

/// The 16-bit move of a shakmaty Antichess move: no castling, no drops.
pub fn packed(m: shakmaty::Move) -> Move {
    let from = m.from().unwrap_or(Square::A1);
    match m {
        shakmaty::Move::EnPassant { .. } => Move::en_passant(from, m.to()),
        _ => match m.promotion() {
            Some(role) => Move::promote(from, m.to(), role),
            None => Move::normal(from, m.to()),
        },
    }
}
