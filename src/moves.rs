//! What the search needs to know of each move of a position, worked out from
//! the position and the move as Atomic-Stockfish works it out, without
//! playing it: legality, checks, and whether either side could then explode
//! the other king at once. Per position an engine keeps pins, the squares
//! that check the other king, and attack maps, which settle most moves with
//! a bit test; the rest are judged from the occupancy the move leaves.

use shakmaty::{Bitboard, ByRole, CastlingSide, Color, Role, Square, attacks};

use crate::board::{Effect, Move, Position};

/// Works out one position's moves.
pub struct Mover<'a> {
    pos: &'a Position,
    us: Color,
    /// The hash every child's hash starts from.
    base: u64,
    our_king: Option<Square>,
    their_king: Option<Square>,
    /// Our pieces whose quiet moves need a legality test: every piece while
    /// our king is in check, the pinned ones otherwise, none while the kings
    /// touch, since touching kings are never in check.
    tested: Bitboard,
    /// The kings touch, so no move but a king's checks the other one.
    touching: bool,
    /// For each role, the squares from which it checks the other king.
    check_squares: ByRole<Bitboard>,
    /// Our pieces alone between one of our sliders and the other king.
    uncovering: Bitboard,
    ours: Bitboard,
    theirs: Bitboard,
    /// The squares beside our king and away from theirs, where a piece of
    /// ours explodes our king when taken, and our pieces on them.
    beside_us: Bitboard,
    exposed: Bitboard,
    beside_them: Bitboard,
    /// What their pawns and knights attack, what their sliders reach, and
    /// their diagonal and straight sliders.
    their_steps: Bitboard,
    their_reach: Bitboard,
    their_diagonal: Bitboard,
    their_straight: Bitboard,
    /// Their pieces beside their king and away from ours, and our pieces
    /// alone between one of our sliders and one of those.
    targets: Bitboard,
    uncovering_targets: Bitboard,
}

impl<'a> Mover<'a> {
    pub fn new(pos: &'a Position) -> Mover<'a> {
        let us = pos.turn();
        let ours = pos.by_color(us);
        let theirs = pos.by_color(!us);
        let occupied = pos.occupied();
        let our_king = pos.king(us);
        let their_king = pos.king(!us);
        let touching = matches!((our_king, their_king),
            (Some(ours), Some(theirs)) if attacks::king_attacks(ours).contains(theirs));
        let tested = match our_king {
            _ if touching => Bitboard::EMPTY,
            Some(king) if pos.checked_after(king, !us, occupied) => ours,
            Some(king) => blockers(pos, king, !us) & ours,
            None => ours,
        };
        let mut check_squares = ByRole {
            pawn: Bitboard::EMPTY,
            knight: Bitboard::EMPTY,
            bishop: Bitboard::EMPTY,
            rook: Bitboard::EMPTY,
            queen: Bitboard::EMPTY,
            king: Bitboard::EMPTY,
        };
        let mut uncovering = Bitboard::EMPTY;
        if let Some(king) = their_king
            && !touching
        {
            check_squares.pawn = attacks::pawn_attacks(!us, king);
            check_squares.knight = attacks::knight_attacks(king);
            check_squares.bishop = attacks::bishop_attacks(king, occupied);
            check_squares.rook = attacks::rook_attacks(king, occupied);
            check_squares.queen = check_squares.bishop | check_squares.rook;
            uncovering = blockers(pos, king, us) & ours;
        }
        let beside = |king: Option<Square>| king.map_or(Bitboard::EMPTY, attacks::king_attacks);
        let beside_us = beside(our_king) & !beside(their_king);
        let targets = beside(their_king) & !beside(our_king) & theirs;
        let (their_steps, their_reach) = attack_map(pos, !us);
        let mut uncovering_targets = Bitboard::EMPTY;
        for target in targets {
            uncovering_targets |= blockers(pos, target, us) & ours;
        }
        let diagonal = pos.by_role(Role::Bishop) | pos.by_role(Role::Queen);
        let straight = pos.by_role(Role::Rook) | pos.by_role(Role::Queen);
        Mover {
            pos,
            us,
            base: pos.child_base(),
            our_king,
            their_king,
            tested,
            touching,
            check_squares,
            uncovering,
            ours,
            theirs,
            beside_us,
            exposed: beside_us & ours,
            beside_them: beside(their_king),
            their_steps,
            their_reach,
            their_diagonal: diagonal & theirs,
            their_straight: straight & theirs,
            targets,
            uncovering_targets,
        }
    }

    pub fn effect(&self, m: Move) -> Effect {
        self.pos.effect(m, self.base)
    }

    /// A legal move's effect with the child's whole hash: the en passant
    /// file, which `effect` leaves out, counts only if taking there is legal,
    /// which needs the child.
    pub fn full_effect(&self, m: Move) -> Effect {
        let mut effect = self.effect(m);
        if effect.en_passant {
            effect.hash ^= self.pos.play(m).ep_hash();
        }
        effect
    }

    /// The squares of a quiet move by a piece other than the king.
    fn quiet(&self, m: Move) -> Option<(Square, Square)> {
        let (from, to) = (m.from(), m.to());
        let quiet = !m.is_castle()
            && !m.is_en_passant()
            && !self.pos.occupied().contains(to)
            && Some(from) != self.our_king;
        quiet.then_some((from, to))
    }

    /// Whether a pseudo-legal move is legal: a quiet move by a piece that is
    /// neither the king nor tested always is.
    pub fn legal(&self, m: Move) -> bool {
        if let Some((from, _)) = self.quiet(m)
            && !self.tested.contains(from)
        {
            return true;
        }
        self.pos.legal(m)
    }

    /// Whether a legal move explodes the other king.
    pub fn explodes_them(&self, effect: &Effect) -> bool {
        effect.capture
            && self
                .their_king
                .is_some_and(|king| !effect.occupied.contains(king))
    }

    /// Where our king stands after a legal move.
    fn our_king_after(&self, m: Move) -> Option<Square> {
        let (from, to) = (m.from(), m.to());
        if m.is_castle() {
            Some(CastlingSide::from_queen_side(to < from).king_to(self.us))
        } else if Some(from) == self.our_king {
            Some(to)
        } else {
            self.our_king
        }
    }

    /// Our pieces after a legal move, by the board's occupancy after it.
    fn ours_after(&self, m: Move, effect: &Effect) -> Bitboard {
        let (from, to) = (m.from(), m.to());
        if m.is_castle() {
            let side = CastlingSide::from_queen_side(to < from);
            return self
                .ours
                .without(from)
                .without(to)
                .with(side.king_to(self.us))
                .with(side.rook_to(self.us));
        }
        if effect.capture {
            return self.ours & effect.occupied;
        }
        self.ours.without(from).with(to)
    }

    /// Whether a legal move checks the other king: for a quiet move other
    /// than a king's or a promotion, from the check squares and the uncovering
    /// pieces; otherwise from the occupancy it leaves.
    pub fn gives_check(&self, m: Move, effect: &Effect) -> bool {
        let Some(their_king) = self.their_king else {
            return false;
        };
        let (from, to) = (m.from(), m.to());
        if let Some((from, to)) = self.quiet(m)
            && m.promotion().is_none()
        {
            let role = self.pos.role_at(from).expect("a piece to move");
            return !self.touching
                && (self.check_squares.get(role).contains(to)
                    || (self.uncovering.contains(from)
                        && !attacks::aligned(from, to, their_king)));
        }
        // A king checks only by uncovering a slider, unless it leaves the
        // other king, whose attackers then count again.
        if Some(from) == self.our_king && !m.is_castle() && !self.touching {
            return !attacks::king_attacks(to).contains(their_king)
                && self.uncovering.contains(from)
                && !attacks::aligned(from, to, their_king);
        }
        let occupied = effect.occupied;
        if !occupied.contains(their_king) {
            return false;
        }
        let our_king = self.our_king_after(m);
        if our_king.is_some_and(|king| attacks::king_attacks(king).contains(their_king)) {
            return false;
        }
        // The pieces that stayed where they were, then the one that landed.
        let landed = if m.is_castle() {
            let side = CastlingSide::from_queen_side(to < from);
            Some((side.rook_to(self.us), Role::Rook))
        } else if effect.capture || Some(from) == self.our_king {
            None
        } else {
            let role = self.pos.role_at(from).expect("a piece to move");
            Some((to, m.promotion().unwrap_or(role)))
        };
        let mut vacated = Bitboard::from(from);
        if m.is_castle() {
            vacated.add(to);
        }
        let stayed = self.ours & occupied & !vacated & !self.pos.by_role(Role::King);
        if (self.pos.attackers_to(their_king, occupied) & stayed).any() {
            return true;
        }
        landed.is_some_and(|(square, role)| {
            attacks::attacks(square, role.of(self.us), occupied).contains(their_king)
        })
    }

    /// Whether, after a legal move, the other side could explode our king at
    /// once. A quiet move takes nothing, so their pawns and knights attack
    /// what they did, and a slider's attacks change only through the squares
    /// the move touches.
    pub fn exposes(&self, m: Move, effect: &Effect) -> bool {
        let (from, to) = (m.from(), m.to());
        let occupied = effect.occupied;
        if !effect.capture && !m.is_castle() {
            let exposed = if Some(from) == self.our_king {
                attacks::king_attacks(to) & self.ours.without(from) & !self.beside_them
            } else {
                self.exposed.without(from) | (self.beside_us & Bitboard::from(to))
            };
            if (exposed & self.their_steps).any() {
                return true;
            }
            if !self.their_reach.contains(from) && !self.their_reach.contains(to) {
                return (exposed & self.their_reach).any();
            }
            return exposed.into_iter().any(|square| {
                (attacks::bishop_attacks(square, occupied) & self.their_diagonal).any()
                    || (attacks::rook_attacks(square, occupied) & self.their_straight).any()
            });
        }
        // A capture or castling: judge the board it leaves.
        let Some(our_king) = self
            .our_king_after(m)
            .filter(|king| occupied.contains(*king))
        else {
            return false;
        };
        let Some(their_king) = self.their_king.filter(|king| occupied.contains(*king)) else {
            return false;
        };
        let ours = self.ours_after(m, effect);
        let exposed = attacks::king_attacks(our_king) & ours & !attacks::king_attacks(their_king);
        let capturers = self.theirs & occupied & !self.pos.by_role(Role::King);
        exposed
            .into_iter()
            .any(|square| (self.pos.attackers_to(square, occupied) & capturers).any())
    }

    /// Whether after a legal move their king touches ours, or can step beside
    /// it, where neither king can be checked or exploded.
    pub fn connectable(&self, m: Move, effect: &Effect) -> bool {
        let (Some(their_king), Some(our_king)) = (self.their_king, self.our_king_after(m)) else {
            return false;
        };
        let theirs = attacks::king_attacks(their_king);
        theirs.contains(our_king)
            || (theirs & attacks::king_attacks(our_king) & !effect.occupied).any()
    }

    /// Whether a pseudo-legal move is plain: quiet, by a piece other than the
    /// king, neither promoting nor uncovering anything, and neither checking
    /// nor threatening to explode their king. Exact, and much cheaper than
    /// scoring the move.
    pub fn plain(&self, m: Move) -> bool {
        let Some((from, to)) = self.quiet(m) else {
            return false;
        };
        if m.promotion().is_some() || (self.uncovering | self.uncovering_targets).contains(from) {
            return false;
        }
        let role = self.pos.role_at(from).expect("a piece to move");
        if !self.touching && self.check_squares.get(role).contains(to) {
            return false;
        }
        let occupied = self.pos.occupied() ^ Bitboard::from(from) ^ Bitboard::from(to);
        (attacks::attacks(to, role.of(self.us), occupied) & self.targets).is_empty()
    }

    /// Whether the defender can blow up the piece a quiet move just moved
    /// without losing material, which makes the threat it brings cheap to
    /// parry and a plain move a gift: their cheapest capture on its square,
    /// blast and all.
    pub fn blasted_cheaply(&self, m: Move, effect: &Effect) -> bool {
        let Some((from, to)) = self.quiet(m) else {
            return false;
        };
        let occupied = effect.occupied;
        let ring = attacks::king_attacks(to) & occupied;
        // Taking beside one's own king is suicide, and beside ours a win the
        // search has seen already.
        if self.their_king.is_some_and(|king| ring.contains(king)) {
            return false;
        }
        let theirs = self.theirs & occupied;
        let capturers =
            self.pos.attackers_to(to, occupied) & theirs & !self.pos.by_role(Role::King);
        if capturers.is_empty() {
            return false;
        }
        let role = m
            .promotion()
            .unwrap_or_else(|| self.pos.role_at(from).expect("a piece to move"));
        let destroyed = ring & !self.pos.by_role(Role::Pawn);
        let mut gain = piece_value(role);
        for square in destroyed {
            let value = self.pos.role_at(square).map_or(0, piece_value);
            gain += if theirs.contains(square) {
                -value
            } else {
                value
            };
        }
        // A capturer inside the blast is lost anyway; one outside costs itself.
        let cost = capturers
            .into_iter()
            .map(|square| {
                if destroyed.contains(square) {
                    0
                } else {
                    self.pos.role_at(square).map_or(0, piece_value)
                }
            })
            .min()
            .unwrap_or(0);
        gain - cost >= 0
    }

    /// After a legal move, how many of their pieces we could take to explode
    /// their king at once. A position with such a threat already wins at
    /// once, which makes the answer moot there; elsewhere a quiet move other
    /// than a king's threatens only with the moved piece, or by uncovering a
    /// slider.
    pub fn threats(&self, m: Move, effect: &Effect) -> u32 {
        if let Some((from, to)) = self.quiet(m)
            && !self.uncovering_targets.contains(from)
        {
            let role = m
                .promotion()
                .unwrap_or_else(|| self.pos.role_at(from).expect("a piece to move"));
            let attacked = attacks::attacks(to, role.of(self.us), effect.occupied);
            return (attacked & self.targets).count() as u32;
        }
        // Otherwise judge the board the move leaves.
        let occupied = effect.occupied;
        let Some(their_king) = self.their_king.filter(|king| occupied.contains(*king)) else {
            return 0;
        };
        let near_ours = self
            .our_king_after(m)
            .map_or(Bitboard::EMPTY, attacks::king_attacks);
        let targets = attacks::king_attacks(their_king) & self.theirs & occupied & !near_ours;
        if targets.is_empty() {
            return 0;
        }
        let (from, to) = (m.from(), m.to());
        let mut vacated = Bitboard::from(from);
        if m.is_castle() {
            vacated.add(to);
        }
        let stayed = self.ours & occupied & !vacated & !self.pos.by_role(Role::King);
        let landed = if m.is_castle() {
            let side = CastlingSide::from_queen_side(to < from);
            Some((side.rook_to(self.us), Role::Rook))
        } else if effect.capture || Some(from) == self.our_king {
            None
        } else {
            let role = self.pos.role_at(from).expect("a piece to move");
            Some((to, m.promotion().unwrap_or(role)))
        };
        let landed_attacks = landed.map_or(Bitboard::EMPTY, |(square, role)| {
            attacks::attacks(square, role.of(self.us), occupied)
        });
        targets
            .into_iter()
            .filter(|&target| {
                landed_attacks.contains(target)
                    || (self.pos.attackers_to(target, occupied) & stayed).any()
            })
            .count() as u32
    }

    /// Whether `color` might no longer win after a capture or promotion: only
    /// with at most two pieces besides its king, or bishops alone on the board,
    /// can it be so. Anything else keeps a way to explode the other king.
    pub fn might_end_material(&self, color: Color, effect: &Effect) -> bool {
        let kings = self.pos.by_role(Role::King);
        let pieces = self.pos.by_color(color) & effect.occupied & !kings;
        let bishops_only = (effect.occupied & !kings & !self.pos.by_role(Role::Bishop)).is_empty();
        pieces.count() <= 2 || bishops_only
    }
}

/// Atomic-Stockfish's piece values for blasts.
fn piece_value(role: Role) -> i32 {
    match role {
        Role::Pawn => 301,
        Role::Knight => 702,
        Role::Bishop => 738,
        Role::Rook => 1079,
        Role::Queen => 1862,
        Role::King => 0,
    }
}

/// The pieces alone between a square and a slider of `sliders` aimed at it.
fn blockers(pos: &Position, square: Square, sliders: Color) -> Bitboard {
    let diagonal = pos.by_role(Role::Bishop) | pos.by_role(Role::Queen);
    let straight = pos.by_role(Role::Rook) | pos.by_role(Role::Queen);
    let aimed = (attacks::rook_attacks(square, Bitboard::EMPTY) & straight)
        | (attacks::bishop_attacks(square, Bitboard::EMPTY) & diagonal);
    let mut blockers = Bitboard::EMPTY;
    for slider in aimed & pos.by_color(sliders) {
        let between = attacks::between(square, slider) & pos.occupied();
        if !between.more_than_one() {
            blockers |= between;
        }
    }
    blockers
}

/// What a side's pawns and knights attack, and what its sliders reach.
fn attack_map(pos: &Position, side: Color) -> (Bitboard, Bitboard) {
    let occupied = pos.occupied();
    let mut steps = Bitboard::EMPTY;
    for from in pos.by_piece(side.pawn()) {
        steps |= attacks::pawn_attacks(side, from);
    }
    for from in pos.by_piece(side.knight()) {
        steps |= attacks::knight_attacks(from);
    }
    let mut reach = Bitboard::EMPTY;
    for from in pos.by_piece(side.bishop()) {
        reach |= attacks::bishop_attacks(from, occupied);
    }
    for from in pos.by_piece(side.rook()) {
        reach |= attacks::rook_attacks(from, occupied);
    }
    for from in pos.by_piece(side.queen()) {
        reach |= attacks::queen_attacks(from, occupied);
    }
    (steps, reach)
}

/// A capture by `side`, as if it were to move, that explodes the other king:
/// of a piece beside that king, by a piece other than a king, and away from
/// its own king, which the blast would take too. Such a capture is legal
/// whatever else it exposes, since the game ends with it.
pub fn exploding_capture(pos: &Position, side: Color) -> Option<Move> {
    let king = pos.king(!side)?;
    let own = pos
        .king(side)
        .map_or(Bitboard::EMPTY, attacks::king_attacks);
    let targets = attacks::king_attacks(king) & pos.by_color(!side) & !own;
    let capturers = pos.by_color(side) & !pos.by_role(Role::King);
    targets.into_iter().find_map(|to| {
        let from = (pos.attackers_to(to, pos.occupied()) & capturers).first()?;
        Some(pos.capture_move(from, to))
    })
}

/// Whether `color` can no longer win, by shakmaty's Atomic rule.
pub fn cannot_win(pos: &Position, color: Color) -> bool {
    let kings = pos.by_role(Role::King);
    let mine = pos.by_color(color);
    let other = pos.by_color(!color);
    if (other & kings).is_empty() {
        return false;
    }
    if (mine & !kings).is_empty() {
        return true;
    }
    // While the other side has pieces, one of them may explode beside its king,
    // unless only bishops that cannot reach one another remain.
    if (other & !kings).any() {
        let bishops = pos.by_role(Role::Bishop);
        if pos.occupied() == kings | bishops {
            let white = bishops & pos.by_color(Color::White);
            let black = bishops & pos.by_color(Color::Black);
            if (white & Bitboard::DARK_SQUARES).is_empty() {
                return (black & Bitboard::LIGHT_SQUARES).is_empty();
            }
            if (white & Bitboard::LIGHT_SQUARES).is_empty() {
                return (black & Bitboard::DARK_SQUARES).is_empty();
            }
        }
        return false;
    }
    if pos.by_role(Role::Queen).any() || pos.by_role(Role::Pawn).any() {
        return false;
    }
    let knights = pos.by_role(Role::Knight);
    if (knights | pos.by_role(Role::Bishop) | pos.by_role(Role::Rook)).count() == 1 {
        return true;
    }
    if pos.occupied() == kings | knights {
        return knights.count() <= 2;
    }
    false
}

/// Whether the side to move has a legal move, and how many of its legal
/// moves do not let the other side explode its king at once, counting up to
/// `most`: what a check's scoring needs, in one pass over the moves.
pub fn replies(pos: &Position, most: u32) -> (bool, u32) {
    let mover = Mover::new(pos);
    let mut any = false;
    let mut safe = 0;
    for &m in pos.pseudo_legal().as_slice() {
        if mover.legal(m) {
            any = true;
            if !mover.exposes(m, &mover.effect(m)) {
                safe += 1;
                if safe >= most {
                    break;
                }
            }
        }
    }
    (any, safe)
}

/// Whether the side to move has a legal move, trying king moves first.
pub fn has_legal_move(pos: &Position) -> bool {
    let mover = Mover::new(pos);
    pos.pseudo_legal()
        .as_slice()
        .iter()
        .rev()
        .any(|m| mover.legal(*m))
}

#[cfg(test)]
mod tests {
    use shakmaty::{Color, EnPassantMode, Position as _, fen::Fen};

    use shakmaty::{Bitboard, Role, attacks};

    use super::{Mover, cannot_win, exploding_capture, has_legal_move, replies};
    use crate::board::{Position, from_fen};

    /// How many of the other side's pieces `side` could take to explode its
    /// king: the plain count the scoring has to match.
    fn exploding_captures(pos: &Position, side: shakmaty::Color) -> u32 {
        let Some(king) = pos.king(!side) else {
            return 0;
        };
        let own = pos
            .king(side)
            .map_or(Bitboard::EMPTY, attacks::king_attacks);
        let capturers = pos.by_color(side) & !pos.by_role(Role::King);
        (attacks::king_attacks(king) & pos.by_color(!side) & !own)
            .into_iter()
            .filter(|&target| (pos.attackers_to(target, pos.occupied()) & capturers).any())
            .count() as u32
    }

    /// Random games, checking every move's scoring against the position it
    /// leads to, and that against shakmaty.
    #[test]
    fn matches_the_played_positions() {
        let starts = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rn2k2r/pp3ppp/2p5/3pP3/8/2N5/PPP2PPP/R3K2R w KQkq d6 0 1",
            "r3k2r/1P4pp/8/2pP4/8/8/p5PP/R3K2R w KQkq c6 0 1",
            "rnbqkbnr/ppppp1p1/5p2/7p/8/4PN2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
            "r3k2r/8/8/3qQ3/8/8/8/R3K2R b KQkq - 0 1",
            "8/2P5/8/3k4/8/3K4/5p2/8 w - - 0 1",
            "rnb1kb1r/p1qppp1p/1pp5/6p1/P2P1QnP/N1P2P2/1P2P1P1/R1B1KBNR b KQkq - 1 10",
        ];
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut moves = 0;
        for start in starts {
            for _ in 0..150 {
                let mut pos: Position = from_fen(start).unwrap();
                for _ in 0..150 {
                    let mover = Mover::new(&pos);
                    let us = pos.turn();
                    let threat_before = exploding_capture(&pos, us).is_some();
                    let fen = Fen::from_position(&pos.to_atomic(), EnPassantMode::Legal);
                    let mut legal = Vec::new();
                    for &m in pos.pseudo_legal().as_slice() {
                        assert_eq!(mover.legal(m), pos.legal(m), "{m:?} in {fen}");
                        if !pos.legal(m) {
                            continue;
                        }
                        legal.push(m);
                        moves += 1;
                        let effect = mover.effect(m);
                        let next = pos.play(m);
                        assert_eq!(mover.full_effect(m).hash, next.hash(), "{m:?} in {fen}");
                        let exploded = next.king(!us).is_none();
                        assert_eq!(mover.explodes_them(&effect), exploded, "{m:?} in {fen}");
                        if exploded {
                            continue;
                        }
                        let theirs = next.to_atomic();
                        let check = mover.gives_check(m, &effect);
                        assert_eq!(check, theirs.is_check(), "{m:?} in {fen}");
                        let exposed = exploding_capture(&next, !us).is_some();
                        assert_eq!(mover.exposes(m, &effect), exposed, "{m:?} in {fen}");
                        if !threat_before {
                            let threats = exploding_captures(&next, us);
                            assert_eq!(mover.threats(m, &effect), threats, "{m:?} in {fen}");
                        }
                        // A plain move neither checks nor adds a threat.
                        if mover.plain(m) {
                            assert!(!check, "{m:?} in {fen}");
                            let threats = exploding_captures(&next, us);
                            assert!(threat_before || threats == 0, "{m:?} in {fen}");
                        }
                        for color in Color::ALL {
                            let dead = theirs.has_insufficient_material(color);
                            assert_eq!(cannot_win(&next, color), dead, "{m:?} in {fen}");
                            let changed = effect.capture || m.promotion().is_some();
                            if changed && !mover.might_end_material(color, &effect) {
                                assert!(!dead, "{m:?} in {fen}");
                            }
                        }
                        let any = !theirs.legal_moves().is_empty();
                        assert_eq!(has_legal_move(&next), any, "{m:?} in {fen}");
                        // After a check, where the search counts them: the
                        // replies that do not let us explode their king at
                        // once, or that explode ours, as the threat replay
                        // and the scoring count them.
                        if check {
                            let safe = theirs
                                .legal_moves()
                                .iter()
                                .filter(|&reply| {
                                    let after = theirs.clone().play(*reply).unwrap();
                                    let after = Position::from_atomic(&after);
                                    after.king(us).is_none()
                                        || exploding_capture(&after, us).is_none()
                                })
                                .count() as u32;
                            for most in [1, 8] {
                                let counted = (any, safe.min(most));
                                assert_eq!(replies(&next, most), counted, "{m:?} in {fen}");
                            }
                        }
                    }
                    if legal.is_empty() || pos.king(!us).is_none() {
                        break;
                    }
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    pos = pos.play(legal[(seed % legal.len() as u64) as usize]);
                }
            }
        }
        assert!(moves > 200_000, "{moves}");
    }
}
