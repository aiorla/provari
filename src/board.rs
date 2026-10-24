//! An Atomic position built the way engines build theirs, after
//! Atomic-Stockfish: bitboards with a mailbox, 16-bit moves, and legality
//! worked out from occupancy, without playing the move. Positions come from
//! shakmaty, their hashes equal shakmaty's, and the tests check the rules
//! against shakmaty's.
//!
//! Castling is standard: the kings start on the e-file and the rooks in the
//! corners.

use std::mem::MaybeUninit;

use shakmaty::{
    Bitboard, CastlingSide, Color, Piece, Position as _, Role, Setup, Square, attacks,
    variant::Atomic,
    zobrist::{Zobrist64, ZobristValue},
};

/// A move in 16 bits: from, to, and kind. Castling goes from the king to the
/// rook, as shakmaty's does.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Move(u16);

const NORMAL: u16 = 0;
const EN_PASSANT: u16 = 1;
const CASTLE: u16 = 2;
/// Promotions to the knight, bishop, rook, and queen follow.
const PROMOTION: u16 = 3;

impl Move {
    const fn new(from: Square, to: Square, kind: u16) -> Move {
        Move(from as u16 | (to as u16) << 6 | kind << 12)
    }

    fn promoting(from: Square, to: Square, role: Role) -> Move {
        Move::new(from, to, PROMOTION + role as u16 - Role::Knight as u16)
    }

    /// The move's bits, 15 at most and never 0: no move goes from a square to
    /// itself.
    pub fn bits(self) -> u16 {
        self.0
    }

    pub fn from_bits(bits: u16) -> Move {
        Move(bits)
    }

    pub fn from(self) -> Square {
        Square::new(u32::from(self.0 & 63))
    }

    pub fn to(self) -> Square {
        Square::new(u32::from((self.0 >> 6) & 63))
    }

    fn kind(self) -> u16 {
        self.0 >> 12
    }

    pub fn promotion(self) -> Option<Role> {
        let kind = self.kind();
        (kind >= PROMOTION)
            .then(|| Role::ALL[(Role::Knight as u16 + kind - PROMOTION) as usize - 1])
    }

    pub fn is_en_passant(self) -> bool {
        self.kind() == EN_PASSANT
    }

    pub fn is_castle(self) -> bool {
        self.kind() == CASTLE
    }
}

/// What a move changes, worked out without playing it.
#[derive(Clone, Copy)]
pub struct Effect {
    /// The child's hash, but for the en passant file.
    pub hash: u64,
    /// A double pawn push beside an enemy pawn: the child's hash counts the
    /// en passant file only if taking is legal, which needs the child.
    pub en_passant: bool,
    /// The child's 50-move counter.
    pub counter: u32,
    /// The board's occupancy after the move.
    pub occupied: Bitboard,
    pub capture: bool,
}

/// Moves of one position, on the stack. Only the moves pushed are written,
/// so the list is not zeroed each time.
pub struct Moves {
    moves: [MaybeUninit<Move>; 256],
    len: usize,
}

impl Moves {
    fn new() -> Moves {
        Moves {
            moves: [const { MaybeUninit::uninit() }; 256],
            len: 0,
        }
    }

    fn push(&mut self, m: Move) {
        self.moves[self.len].write(m);
        self.len += 1;
    }

    pub fn as_slice(&self) -> &[Move] {
        // SAFETY: the first `len` moves were written by `push`, and a
        // `MaybeUninit<Move>` has a `Move`'s layout.
        unsafe { std::slice::from_raw_parts(self.moves.as_ptr().cast(), self.len) }
    }
}

/// Castling rights as bits: the white king side, the white queen side, then
/// the black ones; each belongs to a corner.
const RIGHTS: [(Color, CastlingSide, Square); 4] = [
    (Color::White, CastlingSide::KingSide, Square::H1),
    (Color::White, CastlingSide::QueenSide, Square::A1),
    (Color::Black, CastlingSide::KingSide, Square::H8),
    (Color::Black, CastlingSide::QueenSide, Square::A8),
];

fn color_rights(color: Color) -> u8 {
    if color.is_white() { 0b0011 } else { 0b1100 }
}

fn corner_right(square: Square) -> u8 {
    match square {
        Square::H1 => 0b0001,
        Square::A1 => 0b0010,
        Square::H8 => 0b0100,
        Square::A8 => 0b1000,
        _ => 0,
    }
}

fn castling_hash(rights: u8) -> u64 {
    let mut hash = 0;
    for (bit, (color, side, _)) in RIGHTS.iter().enumerate() {
        if rights & (1 << bit) != 0 {
            hash ^= Zobrist64::zobrist_for_castling_right(*color, *side).0;
        }
    }
    hash
}

fn piece_hash(square: Square, piece: Piece) -> u64 {
    Zobrist64::zobrist_for_piece(square, piece).0
}

fn side(color: Color) -> usize {
    color.is_white() as usize
}

#[derive(Clone)]
pub struct Position {
    /// The piece on each square: 0 for none, else the role plus 8 for white.
    squares: [u8; 64],
    roles: [Bitboard; 7],
    colors: [Bitboard; 2],
    occupied: Bitboard,
    turn: Color,
    castling: u8,
    /// The square a pawn passed over moving two squares, if one just did.
    ep: Option<Square>,
    halfmoves: u32,
    hash: u64,
    /// The en passant part of the hash: the file, when taking there is legal.
    ep_hash: u64,
}

impl Position {
    pub fn from_atomic(pos: &Atomic) -> Position {
        let mut squares = [0; 64];
        let mut roles = [Bitboard::EMPTY; 7];
        let mut colors = [Bitboard::EMPTY; 2];
        for (square, piece) in pos.board().iter() {
            squares[square as usize] = code(piece);
            roles[piece.role as usize].add(square);
            colors[side(piece.color)].add(square);
        }
        let mut castling = 0;
        for (bit, (color, castling_side, _)) in RIGHTS.iter().enumerate() {
            if pos.castles().has(*color, *castling_side) {
                castling |= 1 << bit;
            }
        }
        let mut position = Position {
            squares,
            roles,
            colors,
            occupied: pos.board().occupied(),
            turn: pos.turn(),
            castling,
            ep: pos.maybe_ep_square(),
            halfmoves: pos.halfmoves(),
            hash: 0,
            ep_hash: 0,
        };
        position.ep_hash = position.legal_ep_hash();
        position.hash = position.full_hash();
        position
    }

    pub fn to_atomic(&self) -> Atomic {
        let mut setup = Setup::empty();
        for square in self.occupied {
            if let Some(piece) = self.piece_at(square) {
                setup.board.set_piece_at(square, piece);
            }
        }
        setup.turn = self.turn;
        setup.castling_rights = RIGHTS
            .iter()
            .enumerate()
            .filter(|(bit, _)| self.castling & (1 << bit) != 0)
            .map(|(_, (_, _, rook))| Bitboard::from(*rook))
            .fold(Bitboard::EMPTY, |rights, rook| rights | rook);
        setup.ep_square = self.ep;
        setup.halfmoves = self.halfmoves;
        // Read as the root was, which may hold more material than a game can.
        crate::from_setup(setup).expect("a position the rules reached")
    }

    /// The piece on each square, as a code: 0 for none, else the role plus
    /// 8 for white.
    pub fn squares(&self) -> &[u8; 64] {
        &self.squares
    }

    pub fn turn(&self) -> Color {
        self.turn
    }

    pub fn halfmoves(&self) -> u32 {
        self.halfmoves
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub fn occupied(&self) -> Bitboard {
        self.occupied
    }

    pub fn by_color(&self, color: Color) -> Bitboard {
        self.colors[side(color)]
    }

    pub fn by_role(&self, role: Role) -> Bitboard {
        self.roles[role as usize]
    }

    pub fn by_piece(&self, piece: Piece) -> Bitboard {
        self.roles[piece.role as usize] & self.colors[side(piece.color)]
    }

    pub fn piece_at(&self, square: Square) -> Option<Piece> {
        decode(self.squares[square as usize])
    }

    pub fn role_at(&self, square: Square) -> Option<Role> {
        self.piece_at(square).map(|piece| piece.role)
    }

    pub fn king(&self, color: Color) -> Option<Square> {
        (self.roles[Role::King as usize] & self.colors[side(color)]).first()
    }

    /// The pieces of every color attacking a square, with sliders seeing
    /// through `occupied`.
    pub fn attackers_to(&self, square: Square, occupied: Bitboard) -> Bitboard {
        let roles = &self.roles;
        let diagonal = roles[Role::Bishop as usize] | roles[Role::Queen as usize];
        let straight = roles[Role::Rook as usize] | roles[Role::Queen as usize];
        let pawns = roles[Role::Pawn as usize];
        (attacks::bishop_attacks(square, occupied) & diagonal)
            | (attacks::rook_attacks(square, occupied) & straight)
            | (attacks::knight_attacks(square) & roles[Role::Knight as usize])
            | (attacks::king_attacks(square) & roles[Role::King as usize])
            | (attacks::pawn_attacks(Color::White, square) & pawns & self.colors[0])
            | (attacks::pawn_attacks(Color::Black, square) & pawns & self.colors[1])
    }

    /// Whether `attacker` checks the king on `king`, given what is left on the
    /// board after a move: `occupied`, with every piece outside it gone.
    /// Touching kings are never in check.
    pub fn checked_after(&self, king: Square, attacker: Color, occupied: Bitboard) -> bool {
        let theirs = self.colors[side(attacker)] & occupied;
        let their_kings = theirs & self.roles[Role::King as usize];
        their_kings.any()
            && (attacks::king_attacks(king) & their_kings).is_empty()
            && (self.attackers_to(king, occupied) & theirs & !self.roles[Role::King as usize]).any()
    }

    /// The squares a capture on `to` empties: the capturer, the captured piece,
    /// and every piece but pawns beside them.
    pub fn blast(&self, to: Square) -> Bitboard {
        Bitboard::from(to)
            | (attacks::king_attacks(to) & self.occupied & !self.roles[Role::Pawn as usize])
    }

    /// The board's occupancy after a move, and whether it captures.
    pub fn occupied_after(&self, m: Move) -> (Bitboard, bool) {
        let (from, to) = (m.from(), m.to());
        if m.is_castle() {
            let castling_side = CastlingSide::from_queen_side(to < from);
            let king_to = castling_side.king_to(self.turn);
            let rook_to = castling_side.rook_to(self.turn);
            let occupied = self
                .occupied
                .without(from)
                .without(to)
                .with(king_to)
                .with(rook_to);
            return (occupied, false);
        }
        let mut occupied = self.occupied.without(from).with(to);
        if m.is_en_passant() {
            occupied.discard(Square::from_coords(to.file(), from.rank()));
            return (occupied & !self.blast(to), true);
        }
        if self.occupied.contains(to) {
            return (occupied & !self.blast(to), true);
        }
        (occupied, false)
    }

    /// Whether a pseudo-legal move is legal: the mover's king survives, and is
    /// not left in check unless the other king exploded, as Atomic-Stockfish
    /// judges it from occupancy.
    pub fn legal(&self, m: Move) -> bool {
        let us = self.turn;
        let (occupied, _) = self.occupied_after(m);
        let from = m.from();
        let moved_king = self.roles[Role::King as usize].contains(from);
        let our_king = if m.is_castle() {
            CastlingSide::from_queen_side(m.to() < from).king_to(us)
        } else if moved_king {
            m.to()
        } else {
            match self.king(us) {
                Some(king) => king,
                None => return false,
            }
        };
        if !occupied.contains(our_king) {
            return false;
        }
        let Some(their_king) = self.king(!us) else {
            return true;
        };
        if !occupied.contains(their_king) {
            return true;
        }
        !self.checked_after(our_king, !us, occupied)
    }

    /// Whether `pseudo_legal` lists a move, without listing them all, as a
    /// replay needs for the one move it follows.
    pub fn is_pseudo_legal(&self, m: Move) -> bool {
        let us = self.turn;
        let (from, to) = (m.from(), m.to());
        let Some(piece) = self.piece_at(from) else {
            return false;
        };
        if piece.color != us {
            return false;
        }
        let kind = m.kind();
        match kind {
            EN_PASSANT => {
                piece.role == Role::Pawn
                    && self.ep == Some(to)
                    && attacks::pawn_attacks(!us, to).contains(from)
            }
            // Rare enough to list.
            CASTLE => self.pseudo_legal().as_slice().contains(&m),
            _ if kind > PROMOTION + Role::Queen as u16 - Role::Knight as u16 => false,
            _ if self.by_color(us).contains(to) => false,
            _ => {
                let promotion = kind >= PROMOTION;
                match piece.role {
                    // A pawn reaching the back rank promotes, and only there.
                    Role::Pawn if Bitboard::BACKRANKS.contains(to) != promotion => false,
                    Role::Pawn => {
                        if attacks::pawn_attacks(us, from).contains(to) {
                            return self.by_color(!us).contains(to);
                        }
                        let forward = us.fold_wb(8, -8);
                        let single = from as i32 + forward;
                        if self.occupied.contains(Square::new(single as u32)) {
                            return false;
                        }
                        to as i32 == single
                            || to as i32 == single + forward
                                && us.fold_wb(Bitboard::SOUTH, Bitboard::NORTH).contains(to)
                                && !self.occupied.contains(to)
                    }
                    _ if promotion => false,
                    Role::Knight => attacks::knight_attacks(from).contains(to),
                    Role::Bishop => attacks::bishop_attacks(from, self.occupied).contains(to),
                    Role::Rook => attacks::rook_attacks(from, self.occupied).contains(to),
                    Role::Queen => attacks::queen_attacks(from, self.occupied).contains(to),
                    // A king cannot capture: it would explode.
                    Role::King => {
                        attacks::king_attacks(from).contains(to) && !self.occupied.contains(to)
                    }
                }
            }
        }
    }

    /// Every move that is legal but for where it leaves the mover's king, in
    /// shakmaty's order.
    pub fn pseudo_legal(&self) -> Moves {
        let mut moves = Moves::new();
        let us = self.turn;
        let ours = self.by_color(us);
        let occupied = self.occupied;
        if let Some(to) = self.ep {
            for from in self.by_piece(us.pawn()) & attacks::pawn_attacks(!us, to) {
                moves.push(Move::new(from, to, EN_PASSANT));
            }
        }
        self.pawn_moves(&mut moves);
        let target = !ours;
        for from in self.by_piece(us.knight()) {
            push_all(&mut moves, from, attacks::knight_attacks(from) & target);
        }
        for from in self.by_piece(us.bishop()) {
            push_all(
                &mut moves,
                from,
                attacks::bishop_attacks(from, occupied) & target,
            );
        }
        for from in self.by_piece(us.rook()) {
            push_all(
                &mut moves,
                from,
                attacks::rook_attacks(from, occupied) & target,
            );
        }
        for from in self.by_piece(us.queen()) {
            push_all(
                &mut moves,
                from,
                attacks::queen_attacks(from, occupied) & target,
            );
        }
        // A king cannot capture: it would explode.
        for from in self.by_piece(us.king()) {
            push_all(&mut moves, from, attacks::king_attacks(from) & !occupied);
        }
        if let Some(king) = self.king(us) {
            for castling_side in [CastlingSide::KingSide, CastlingSide::QueenSide] {
                self.castling_move(king, castling_side, &mut moves);
            }
        }
        moves
    }

    fn pawn_moves(&self, moves: &mut Moves) {
        let us = self.turn;
        let pawns = self.by_piece(us.pawn()).0;
        let theirs = self.by_color(!us);
        const FILE_A: u64 = 0x0101_0101_0101_0101;
        // Captures towards the a-file, then towards the h-file, as offsets.
        let (west, east) = if us.is_white() {
            (((pawns & !FILE_A) << 7, 7), ((pawns << 9) & !FILE_A, 9))
        } else {
            (((pawns & !FILE_A) >> 9, -9), ((pawns >> 7) & !FILE_A, -7))
        };
        for (captures, offset) in [west, east] {
            pawn_moves_to(Bitboard(captures) & theirs, offset, moves);
        }
        let forward = us.fold_wb(8, -8);
        let single = Bitboard(pawns).shift(forward) & !self.occupied;
        pawn_moves_to(single, forward, moves);
        let double =
            single.shift(forward) & us.fold_wb(Bitboard::SOUTH, Bitboard::NORTH) & !self.occupied;
        pawn_moves_to(double, 2 * forward, moves);
    }

    /// Castling as shakmaty allows it: the right, an empty path, and no check
    /// on the king's way or where it lands; checks beside the other king do
    /// not count.
    fn castling_move(&self, king: Square, castling_side: CastlingSide, moves: &mut Moves) {
        let us = self.turn;
        let index =
            if us.is_white() { 0 } else { 2 } + (castling_side == CastlingSide::QueenSide) as usize;
        let (_, _, rook) = RIGHTS[index];
        if self.castling & (1 << index) == 0 {
            return;
        }
        let king_to = castling_side.king_to(us);
        let rook_to = castling_side.rook_to(us);
        let path = (attacks::between(king, king_to) | attacks::between(rook, rook_to))
            .with(king_to)
            .with(rook_to)
            .without(king)
            .without(rook);
        if (path & self.occupied).any() {
            return;
        }
        let landing = self.occupied ^ king ^ rook ^ rook_to;
        if self.king_attacked(king_to, !us, landing) {
            return;
        }
        for square in attacks::between(king, king_to).with(king) {
            if self.king_attacked(square, !us, self.occupied ^ king) {
                return;
            }
        }
        moves.push(Move::new(king, rook, CASTLE));
    }

    /// shakmaty's Atomic `king_attackers` test: no attack counts beside the
    /// attacker's king.
    fn king_attacked(&self, square: Square, attacker: Color, occupied: Bitboard) -> bool {
        let kings = self.by_piece(attacker.king());
        kings.any()
            && (attacks::king_attacks(square) & kings).is_empty()
            && (self.attackers_to(square, occupied) & self.by_color(attacker)).any()
    }

    /// The position with the other side to move, as if the side to move had
    /// passed: what a threat would win.
    pub fn null(&self) -> Position {
        let mut passed = self.clone();
        passed.hash = self.child_base();
        passed.ep = None;
        passed.ep_hash = 0;
        passed.turn = !self.turn;
        passed.halfmoves = self.halfmoves + 1;
        passed
    }

    /// The hash every child's hash starts from: this one for the other side
    /// to move, without en passant.
    pub fn child_base(&self) -> u64 {
        self.hash ^ Zobrist64::zobrist_for_white_turn().0 ^ self.ep_hash
    }

    /// What a legal move changes, as `apply` would change it, from
    /// `child_base`.
    pub fn effect(&self, m: Move, base: u64) -> Effect {
        let us = self.turn;
        let (from, to) = (m.from(), m.to());
        let mut hash = base;
        let mut rights = self.castling;
        let moved = self.piece_at(from).expect("a piece to move");
        let mut en_passant = false;
        let (occupied, capture) = if m.is_castle() {
            let castling_side = CastlingSide::from_queen_side(to < from);
            let king_to = castling_side.king_to(us);
            let rook_to = castling_side.rook_to(us);
            hash ^= piece_hash(from, us.king()) ^ piece_hash(to, us.rook());
            hash ^= piece_hash(rook_to, us.rook()) ^ piece_hash(king_to, us.king());
            rights &= !color_rights(us);
            let occupied = self
                .occupied
                .without(from)
                .without(to)
                .with(king_to)
                .with(rook_to);
            (occupied, false)
        } else {
            hash ^= piece_hash(from, moved);
            if moved.role == Role::King {
                rights &= !color_rights(us);
            } else if moved.role == Role::Rook {
                rights &= !corner_right(from);
            }
            let taken = if m.is_en_passant() {
                Some(Square::from_coords(to.file(), from.rank()))
            } else {
                self.occupied.contains(to).then_some(to)
            };
            match taken {
                None => {
                    let placed = m.promotion().map_or(moved, |role| role.of(us));
                    hash ^= piece_hash(to, placed);
                    if moved.role == Role::Pawn && (to as i32 - from as i32).abs() == 16 {
                        let passed = Square::new((from as u32 + to as u32) / 2);
                        let takers =
                            attacks::pawn_attacks(us, passed) & self.by_piece((!us).pawn());
                        en_passant = takers.any();
                    }
                    (self.occupied.without(from).with(to), false)
                }
                Some(square) => {
                    let captured = self.piece_at(square).expect("a piece to take");
                    hash ^= piece_hash(square, captured);
                    if captured.role == Role::Rook {
                        rights &= !corner_right(to);
                    }
                    let left = self.occupied.without(from).without(square);
                    let blast = attacks::king_attacks(to) & left & !self.roles[Role::Pawn as usize];
                    if (blast & self.by_piece((!us).king())).any() {
                        rights &= !color_rights(!us);
                    }
                    for exploded in blast {
                        if let Some(piece) = self.piece_at(exploded) {
                            hash ^= piece_hash(exploded, piece);
                        }
                        rights &= !corner_right(exploded);
                    }
                    (left & !blast, true)
                }
            }
        };
        if rights != self.castling {
            hash ^= castling_hash(self.castling) ^ castling_hash(rights);
        }
        let counter = if capture || moved.role == Role::Pawn {
            0
        } else {
            self.halfmoves + 1
        };
        Effect {
            hash,
            en_passant,
            counter,
            occupied,
            capture,
        }
    }

    /// The en passant part of this position's hash.
    pub fn ep_hash(&self) -> u64 {
        self.ep_hash
    }

    /// The position after a legal move, with its hash.
    pub fn play(&self, m: Move) -> Position {
        let mut next = self.clone();
        next.apply(m);
        next
    }

    fn apply(&mut self, m: Move) {
        let us = self.turn;
        let (from, to) = (m.from(), m.to());
        let mut hash = self.hash ^ Zobrist64::zobrist_for_white_turn().0 ^ self.ep_hash;
        let rights = self.castling;
        let moved = self.piece_at(from).expect("a piece to move");
        let captured = if m.is_en_passant() {
            Some((!us).pawn())
        } else if m.is_castle() {
            None
        } else {
            self.piece_at(to)
        };
        self.ep = None;
        self.halfmoves = if captured.is_some() || moved.role == Role::Pawn {
            0
        } else {
            self.halfmoves + 1
        };
        if m.is_castle() {
            let castling_side = CastlingSide::from_queen_side(to < from);
            let king_to = castling_side.king_to(us);
            let rook_to = castling_side.rook_to(us);
            hash ^= self.remove(from) ^ self.remove(to);
            hash ^= self.put(rook_to, us.rook()) ^ self.put(king_to, us.king());
            self.castling &= !color_rights(us);
        } else {
            if moved.role == Role::King {
                self.castling &= !color_rights(us);
            } else if moved.role == Role::Rook {
                self.castling &= !corner_right(from);
            }
            if captured.is_some_and(|piece| piece.role == Role::Rook) {
                self.castling &= !corner_right(to);
            }
            hash ^= self.remove(from);
            match captured {
                None => {
                    let placed = m.promotion().map_or(moved, |role| role.of(us));
                    hash ^= self.put(to, placed);
                    if moved.role == Role::Pawn && (to as i32 - from as i32).abs() == 16 {
                        self.ep = Some(Square::new((from as u32 + to as u32) / 2));
                    }
                }
                Some(_) => {
                    if m.is_en_passant() {
                        hash ^= self.remove(Square::from_coords(to.file(), from.rank()));
                    } else {
                        hash ^= self.remove(to);
                    }
                    // The blast; like shakmaty, a king in it costs the side
                    // to move its castling rights.
                    let blast = attacks::king_attacks(to)
                        & self.occupied
                        & !self.roles[Role::Pawn as usize];
                    if (blast & self.by_piece((!us).king())).any() {
                        self.castling &= !color_rights(!us);
                    }
                    for square in blast {
                        hash ^= self.remove(square);
                        self.castling &= !corner_right(square);
                    }
                }
            }
        }
        if self.castling != rights {
            hash ^= castling_hash(rights) ^ castling_hash(self.castling);
        }
        self.turn = !us;
        self.ep_hash = self.legal_ep_hash();
        self.hash = hash ^ self.ep_hash;
    }

    fn remove(&mut self, square: Square) -> u64 {
        match self.piece_at(square) {
            Some(piece) => {
                self.squares[square as usize] = 0;
                self.roles[piece.role as usize].discard(square);
                self.colors[side(piece.color)].discard(square);
                self.occupied.discard(square);
                piece_hash(square, piece)
            }
            None => 0,
        }
    }

    fn put(&mut self, square: Square, piece: Piece) -> u64 {
        self.squares[square as usize] = code(piece);
        self.roles[piece.role as usize].add(square);
        self.colors[side(piece.color)].add(square);
        self.occupied.add(square);
        piece_hash(square, piece)
    }

    /// The en passant file's part of the hash, which counts only when taking
    /// en passant is legal.
    fn legal_ep_hash(&self) -> u64 {
        let Some(to) = self.ep else {
            return 0;
        };
        let us = self.turn;
        let legal = (self.by_piece(us.pawn()) & attacks::pawn_attacks(!us, to))
            .into_iter()
            .any(|from| self.legal(Move::new(from, to, EN_PASSANT)));
        if legal {
            Zobrist64::zobrist_for_en_passant_file(to.file()).0
        } else {
            0
        }
    }

    fn full_hash(&self) -> u64 {
        let mut hash = castling_hash(self.castling) ^ self.ep_hash;
        for square in self.occupied {
            if let Some(piece) = self.piece_at(square) {
                hash ^= piece_hash(square, piece);
            }
        }
        if self.turn.is_white() {
            hash ^= Zobrist64::zobrist_for_white_turn().0;
        }
        hash
    }

    /// Whether either side may still castle.
    pub fn has_castling_rights(&self) -> bool {
        self.castling != 0
    }

    /// Whether the side to move is in check.
    pub fn in_check(&self) -> bool {
        self.king(self.turn)
            .is_some_and(|king| self.checked_after(king, !self.turn, self.occupied))
    }

    /// A capture from one square to another, a pawn reaching the last rank
    /// promoting to a queen.
    pub fn capture_move(&self, from: Square, to: Square) -> Move {
        if self.roles[Role::Pawn as usize].contains(from) && Bitboard::BACKRANKS.contains(to) {
            Move::promoting(from, to, Role::Queen)
        } else {
            Move::new(from, to, NORMAL)
        }
    }

    /// The move as shakmaty writes it.
    pub fn to_shakmaty(&self, m: Move) -> shakmaty::Move {
        let (from, to) = (m.from(), m.to());
        if m.is_castle() {
            return shakmaty::Move::Castle {
                king: from,
                rook: to,
            };
        }
        if m.is_en_passant() {
            return shakmaty::Move::EnPassant { from, to };
        }
        shakmaty::Move::Normal {
            role: self.role_at(from).expect("a piece to move"),
            from,
            capture: self.role_at(to),
            to,
            promotion: m.promotion(),
        }
    }
}

fn code(piece: Piece) -> u8 {
    piece.role as u8 | if piece.color.is_white() { 8 } else { 0 }
}

fn decode(code: u8) -> Option<Piece> {
    let role = match code & 7 {
        1 => Role::Pawn,
        2 => Role::Knight,
        3 => Role::Bishop,
        4 => Role::Rook,
        5 => Role::Queen,
        6 => Role::King,
        _ => return None,
    };
    let color = Color::from_white(code & 8 != 0);
    Some(Piece { color, role })
}

fn push_all(moves: &mut Moves, from: Square, targets: Bitboard) {
    for to in targets {
        moves.push(Move::new(from, to, NORMAL));
    }
}

/// Pawn moves by one offset: the plain ones, then the promotions, queen first.
fn pawn_moves_to(targets: Bitboard, offset: i32, moves: &mut Moves) {
    let from = |to: Square| Square::new((to as i32 - offset) as u32);
    for to in targets & !Bitboard::BACKRANKS {
        moves.push(Move::new(from(to), to, NORMAL));
    }
    for to in targets & Bitboard::BACKRANKS {
        for role in [Role::Queen, Role::Rook, Role::Bishop, Role::Knight] {
            moves.push(Move::promoting(from(to), to, role));
        }
    }
}

/// A position read from FEN.
#[cfg(test)]
pub fn from_fen(fen: &str) -> Result<Position, String> {
    Ok(Position::from_atomic(&crate::position(fen)?))
}

#[cfg(test)]
mod tests {
    use shakmaty::{EnPassantMode, Position as _, variant::Atomic, zobrist::Zobrist64};

    use shakmaty::Square;

    use super::{Move, Position, from_fen};

    /// Random games, comparing every position's moves, and every position they
    /// lead to, with shakmaty's.
    #[test]
    fn matches_shakmaty() {
        let starts = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rn2k2r/pp3ppp/2p5/3pP3/8/2N5/PPP2PPP/R3K2R w KQkq d6 0 1",
            "r3k2r/1P4pp/8/2pP4/8/8/p5PP/R3K2R w KQkq c6 0 1",
            "rnbqkbnr/ppppp1p1/5p2/7p/8/4PN2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
            "r3k2r/8/8/3qQ3/8/8/8/R3K2R b KQkq - 0 1",
            "8/2P5/8/3k4/8/3K4/5p2/8 w - - 0 1",
            "r1bqBb2/pp1pk3/8/4P1pp/8/2P5/PP3P2/R1BK3R b - - 1 11",
        ];
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut positions = 0;
        for start in starts {
            for _ in 0..150 {
                let mut ours: Position = from_fen(start).unwrap();
                let mut theirs: Atomic = ours.to_atomic();
                for _ in 0..150 {
                    let hash: Zobrist64 = theirs.zobrist_hash(EnPassantMode::Legal);
                    assert_eq!(ours.hash(), hash.0);
                    let expected: Vec<shakmaty::Move> = theirs.legal_moves().into_iter().collect();
                    let found: Vec<_> = ours
                        .pseudo_legal()
                        .as_slice()
                        .iter()
                        .copied()
                        .filter(|m| ours.legal(*m))
                        .collect();
                    let converted: Vec<shakmaty::Move> =
                        found.iter().map(|m| ours.to_shakmaty(*m)).collect();
                    assert_eq!(converted, expected);
                    positions += 1;
                    // Single moves tested against the list, on one position
                    // in 400 and the position passed to the other side: from
                    // the side to move's pieces every kind of move (castling
                    // only from the king), from other squares plain ones.
                    if positions % 400 == 0 {
                        let passed = (!ours.in_check()).then(|| ours.null());
                        for pos in std::iter::once(&ours).chain(passed.as_ref()) {
                            let mut listed = vec![false; 1 << 15];
                            for m in pos.pseudo_legal().as_slice() {
                                listed[usize::from(m.bits())] = true;
                            }
                            let mine = pos.by_color(pos.turn());
                            let king = pos.king(pos.turn()).map(|king| king as u16);
                            for from in 0..64u16 {
                                let own = mine.contains(Square::new(u32::from(from)));
                                for kind in if own { 0..8 } else { 0..1 } {
                                    if kind == 2 && king != Some(from) {
                                        continue;
                                    }
                                    for to in 0..64u16 {
                                        let bits = from | to << 6 | kind << 12;
                                        let m = Move::from_bits(bits);
                                        let expected = listed[usize::from(bits)];
                                        assert_eq!(pos.is_pseudo_legal(m), expected, "{m:?}");
                                    }
                                }
                            }
                        }
                    }
                    if found.is_empty() || theirs.is_variant_end() {
                        break;
                    }
                    if !ours.in_check() {
                        let passed = ours.null();
                        let swapped = theirs.clone().swap_turn().expect("a side to pass");
                        let hash: Zobrist64 = swapped.zobrist_hash(EnPassantMode::Legal);
                        assert_eq!(passed.hash(), hash.0);
                        assert_eq!(passed.turn(), swapped.turn());
                    }
                    let base = ours.child_base();
                    for m in &found {
                        let next = ours.play(*m);
                        let effect = ours.effect(*m, base);
                        assert_eq!(effect.hash ^ next.ep_hash(), next.hash(), "{m:?}");
                        assert!(effect.en_passant || next.ep_hash() == 0, "{m:?}");
                        assert_eq!(effect.counter, next.halfmoves(), "{m:?}");
                        assert_eq!(effect.occupied, next.occupied(), "{m:?}");
                        let mut played = theirs.clone();
                        played.play_unchecked(ours.to_shakmaty(*m));
                        assert_eq!(next.to_atomic().board(), played.board(), "{m:?}");
                        assert_eq!(next.halfmoves(), played.halfmoves());
                        let hash: Zobrist64 = played.zobrist_hash(EnPassantMode::Legal);
                        assert_eq!(next.hash(), hash.0, "{m:?}");
                    }
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let m = found[(seed % found.len() as u64) as usize];
                    theirs.play_unchecked(ours.to_shakmaty(m));
                    ours = ours.play(m);
                }
            }
        }
        assert!(positions > 20_000, "{positions}");
    }
}
