//! Lichess Antichess as the prover plays it, on shakmaty's rules.
//!
//! The side to move wins with no pieces left or no legal move, and that ends
//! the game before any draw: insufficient material, as Lichess defines it
//! (`insufficient_material`), or the 50-move rule, which draws once a
//! position's counter reaches 100.
//! Repetitions are no concern of the search: a win within so many White moves
//! exists exactly when one exists that ignores them, and the proofs it builds
//! never repeat a position, since every line's remaining length shrinks.

use shakmaty::{
    Bitboard, CastlingMode, Color, EnPassantMode, File, Move, MoveList, Piece, Position,
    PositionError, Rank, Role, Square, attacks,
    fen::Fen,
    uci::UciMove,
    variant::Antichess,
    zobrist::{Zobrist64, Zobrist128, ZobristValue},
};

/// The highest 50-move counter at which a game that has not ended goes on.
pub const LAST_COUNTER: u32 = 99;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// White is to move with no piece or no legal move: White has won.
    WhiteWon,
    /// Black is to move with no piece or no legal move: Black has won.
    BlackWon,
    /// Insufficient material: drawn whatever the counter.
    Insufficient,
    /// The 50-move rule: the counter has passed `LAST_COUNTER`.
    FiftyMoves,
    Open,
}

/// The state of a position whose legal moves are `legal`.
pub fn status(pos: &Antichess, legal: &MoveList) -> Status {
    status_counted(pos, legal.len())
}

/// The state of a position with `moves` legal moves.
pub fn status_counted(pos: &Antichess, moves: usize) -> Status {
    if pos.us().is_empty() || moves == 0 {
        return match pos.turn() {
            Color::White => Status::WhiteWon,
            Color::Black => Status::BlackWon,
        };
    }
    if insufficient_material(pos) {
        Status::Insufficient
    } else if pos.halfmoves() > LAST_COUNTER {
        Status::FiftyMoves
    } else {
        Status::Open
    }
}

/// Whether Lichess draws for insufficient material: only bishops and pawns,
/// each side's bishops on one color of square and the two colors different,
/// and every pawn stuck behind a pawn, on its own bishops' color, with no en
/// passant capture to make. shakmaty's rule leaves out the pawns.
pub fn insufficient_material(pos: &Antichess) -> bool {
    let board = pos.board();
    let pawns = board.pawns();
    if board.occupied() != board.bishops() | pawns {
        return false;
    }
    // The color of square all of a side's bishops stand on.
    let color = |side: Color| {
        let bishops = board.bishops() & board.by_color(side);
        [Bitboard::LIGHT_SQUARES, Bitboard::DARK_SQUARES]
            .into_iter()
            .find(|&squares| bishops.any() && bishops.is_subset(squares))
    };
    let (Some(white), Some(black)) = (color(Color::White), color(Color::Black)) else {
        return false;
    };
    let white_pawns = pawns & board.white();
    let black_pawns = pawns & board.black();
    // On their own bishops' color, pawns can neither capture nor be captured.
    white != black
        && white_pawns.is_subset(white)
        && black_pawns.is_subset(black)
        && white_pawns.shift(8).is_subset(pawns)
        && black_pawns.shift(-8).is_subset(pawns)
        && pos.ep_square(EnPassantMode::Legal).is_none()
}

/// A position's key: its pieces, side to move and a capturable en passant
/// square, without the move counters.
pub fn key(pos: &Antichess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

/// A position's key in 128 bits, for tables too large for `key` to keep
/// apart: on a billion positions, two 64-bit keys meet about once in forty.
pub fn key128(pos: &Antichess) -> u128 {
    pos.zobrist_hash::<Zobrist128>(EnPassantMode::Legal).0
}

/// How many legal moves a position has, counted without listing them: with
/// no castling and no check in Antichess, the moves are the captures if
/// there are any, and otherwise every other move. A promotion counts once
/// for each of the five pieces a pawn may become.
pub fn count_moves(pos: &Antichess) -> usize {
    let board = pos.board();
    let turn = pos.turn();
    let us = pos.us();
    let them = pos.them();
    let occupied = board.occupied();
    let empty = !occupied;
    let pawns = board.pawns() & us;
    let promotions = |targets: Bitboard| {
        (targets & !Bitboard::BACKRANKS).count() + 5 * (targets & Bitboard::BACKRANKS).count()
    };
    // Each piece's attacks once, for its captures and its other moves, a
    // queen's as a bishop's and a rook's.
    let (mut captures, mut quiet) = (0, 0);
    let mut add = |attacked: Bitboard| {
        captures += (attacked & them).count();
        quiet += (attacked & empty).count();
    };
    for from in board.knights() & us {
        add(attacks::knight_attacks(from));
    }
    for from in (board.bishops() | board.queens()) & us {
        add(attacks::bishop_attacks(from, occupied));
    }
    for from in (board.rooks() | board.queens()) & us {
        add(attacks::rook_attacks(from, occupied));
    }
    for from in board.kings() & us {
        add(attacks::king_attacks(from));
    }
    // Every pawn's captures toward each side at once.
    let west = pawns & !Bitboard::from_file(File::A);
    let east = pawns & !Bitboard::from_file(File::H);
    let (west, east) = match turn {
        Color::White => (west.shift(7), east.shift(9)),
        Color::Black => (west.shift(-9), east.shift(-7)),
    };
    captures += promotions(west & them) + promotions(east & them);
    if let Some(to) = pos.maybe_ep_square() {
        captures += (pawns & attacks::pawn_attacks(!turn, to)).count();
    }
    if captures > 0 {
        return captures;
    }
    let forward = turn.fold_wb(8, -8);
    let single = pawns.shift(forward) & empty;
    let fourth = Bitboard::from_rank(turn.fold_wb(Rank::Fourth, Rank::Fifth));
    quiet + promotions(single) + (single.shift(forward) & empty & fourth).count()
}

/// The key of the position after `m` from `parent`, whose key is
/// `parent_key`: the same as `key` of that position, from what the move
/// changes, without playing it.
pub fn child_key(parent: &Antichess, parent_key: u64, m: Move) -> u64 {
    let piece = |square: Square, color: Color, role: Role| {
        Zobrist64::zobrist_for_piece(square, Piece { color, role }).0
    };
    let us = parent.turn();
    let mut key = parent_key ^ Zobrist64::zobrist_for_white_turn().0;
    match m {
        Move::Normal {
            role,
            from,
            capture,
            to,
            promotion,
        } => {
            key ^= piece(from, us, role) ^ piece(to, us, promotion.unwrap_or(role));
            if let Some(captured) = capture {
                key ^= piece(to, !us, captured);
            }
        }
        Move::EnPassant { from, to } => {
            let captured = Square::from_coords(to.file(), from.rank());
            key ^= piece(from, us, Role::Pawn) ^ piece(to, us, Role::Pawn);
            key ^= piece(captured, !us, Role::Pawn);
        }
        Move::Castle { .. } | Move::Put { .. } => unreachable!("not an Antichess move: {m:?}"),
    }
    key ^ en_passant_key(parent) ^ pushed_key(parent, m)
}

/// What a capturable en passant square adds to a position's key.
fn en_passant_key(pos: &Antichess) -> u64 {
    match pos.maybe_ep_square() {
        Some(square) => capturable_key(square, !pos.turn(), pos.our(Role::Pawn)),
        None => 0,
    }
}

/// What the en passant square of a pawn's double step `m` adds to the key of
/// the position after it.
fn pushed_key(parent: &Antichess, m: Move) -> u64 {
    match m {
        Move::Normal {
            role: Role::Pawn,
            from,
            to,
            ..
        } if from.rank().distance(to.rank()) == 2 => {
            let square = Square::new((from.to_u32() + to.to_u32()) / 2);
            capturable_key(square, parent.turn(), parent.their(Role::Pawn))
        }
        _ => 0,
    }
}

/// What en passant `square`, behind a pawn of `pushed`'s color, adds to a
/// key when one of `capturers` can take there.
fn capturable_key(square: Square, pushed: Color, capturers: Bitboard) -> u64 {
    if (attacks::pawn_attacks(pushed, square) & capturers).any() {
        Zobrist64::zobrist_for_en_passant_file(square.file()).0
    } else {
        0
    }
}

/// A move in 16 bits: its origin, its destination, and a promotion. That is
/// all an Antichess move needs, as there is no castling and nothing is dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Packed(u16);

impl Packed {
    pub fn new(m: Move) -> Packed {
        let from = m.from().map_or(0, |square| square as u16);
        let to = m.to() as u16;
        let promotion = m.promotion().map_or(0, |role| role as u16);
        Packed(from | to << 6 | promotion << 12)
    }

    /// Its 15 bits, never all zero.
    pub fn bits(self) -> u16 {
        self.0
    }

    pub fn from_bits(bits: u16) -> Packed {
        Packed(bits)
    }

    /// The legal move this one stands for.
    pub fn find(self, legal: &MoveList) -> Option<Move> {
        legal.iter().copied().find(|&m| Packed::new(m) == self)
    }
}

pub fn uci(m: Move) -> String {
    m.to_uci(CastlingMode::Standard).to_string()
}

/// The legal move written in UCI.
pub fn parse_uci(pos: &Antichess, text: &str) -> Result<Move, String> {
    let uci: UciMove = text
        .parse()
        .map_err(|_| format!("{text:?} is not a UCI move"))?;
    uci.to_move(pos)
        .map_err(|_| format!("{text} is not legal in {}", fen(pos)))
}

pub fn fen(pos: &Antichess) -> String {
    Fen::from_position(pos, EnPassantMode::Legal).to_string()
}

/// The position part of a FEN, without the move counters.
pub fn short_fen(pos: &Antichess) -> String {
    fen(pos).split(' ').take(4).collect::<Vec<_>>().join(" ")
}

/// The position a FEN describes. Antichess has no castling, so castling
/// rights a FEN claims are dropped.
pub fn parse_fen(text: &str) -> Result<Antichess, String> {
    let fen: Fen = text
        .parse()
        .map_err(|err| format!("bad FEN {text:?}: {err}"))?;
    fen.into_position(CastlingMode::Standard)
        .or_else(PositionError::ignore_invalid_castling_rights)
        .map_err(|err| format!("illegal Antichess position {text:?}: {err}"))
}
