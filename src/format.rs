//! The compact proof format of `FORMAT.md`, the prover's side: a
//! proof written from the search's moves, and read back for `line` and the
//! browser. Both walk it with the prover's own board; the checker walks it
//! with its own code (`check`), so that a mistake in either shows as a proof
//! it rejects rather than one it accepts unseen.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use serde_json::{Value, json};
use shakmaty::{CastlingSide, Color, EnPassantMode, Role, fen::Fen, variant::Atomic};

use crate::board::{Move, Position};
use crate::moves::{self, Mover};
use crate::proofs::SHORT;
use crate::tablebase::{Tablebase, Verdict};

/// The format's version.
const VERSION: u64 = 1;
/// The zstd level proofs are written at: smaller than the fast levels, and
/// still quick.
const LEVEL: i32 = 19;
/// The file extension of a proof in this format.
pub const EXTENSION: &str = "pvr";

/// A compact proof's header.
pub struct Header {
    pub root: String,
    /// The side the proof wins for, named only where it is not the side to
    /// move at the root.
    pub attacker: Option<Color>,
    /// The most pieces a tablebase verdict may have, 0 for none.
    pub tablebases: usize,
    /// The proof's length in attacker moves.
    pub moves: u32,
    /// The moves written.
    pub positions: u64,
}

/// The move of every attacker position a proof proves, by hash, as
/// `walk::Moves` keeps them: none for a tablebase win.
pub type Strategy = HashMap<u64, Option<shakmaty::Move>>;

/// How a game stands for the attacker.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Won,
    /// Over otherwise, or at the 50-move limit.
    Failed,
    Open,
}

/// How the game stands, as shakmaty's rules have it: a king exploded, a
/// mate, a stalemate, too little material for either side to win, the
/// 50-move limit. `legal` holds the side to move's legal moves.
fn judge(pos: &Position, attacker: Color, legal: &[Move]) -> State {
    if pos.king(!attacker).is_none() {
        return State::Won;
    }
    if pos.king(attacker).is_none() {
        return State::Failed;
    }
    if legal.is_empty() {
        return if pos.in_check() && pos.turn() != attacker {
            State::Won
        } else {
            State::Failed
        };
    }
    if moves::cannot_win(pos, Color::White) && moves::cannot_win(pos, Color::Black) {
        return State::Failed;
    }
    if pos.halfmoves() >= 100 {
        return State::Failed;
    }
    State::Open
}

#[cfg(test)]
fn state(pos: &Position, attacker: Color) -> State {
    judge(pos, attacker, &legal_moves(pos))
}

/// Whether the attacker's move just played won at once: the defender's king
/// exploded, which only a capture does, or the defender mated, which only a
/// check does.
fn won_at_once(next: &Position, attacker: Color, capture: bool, check: bool) -> bool {
    (capture && next.king(!attacker).is_none()) || (check && !moves::has_legal_move(next))
}

fn legal_moves(pos: &Position) -> Vec<Move> {
    let pseudo = pos.pseudo_legal();
    let mut legal = Vec::with_capacity(pseudo.as_slice().len());
    legal.extend(pseudo.as_slice().iter().copied().filter(|&m| pos.legal(m)));
    legal
}

/// A move's number in the format's order: the square it leaves, the square it
/// reaches (the king's, castling), its promotion.
fn number(pos: &Position, m: Move) -> u32 {
    let to = if m.is_castle() {
        CastlingSide::from_queen_side(m.to() < m.from()).king_to(pos.turn())
    } else {
        m.to()
    };
    let promotion = match m.promotion() {
        None => 0,
        Some(Role::Knight) => 1,
        Some(Role::Bishop) => 2,
        Some(Role::Rook) => 3,
        Some(_) => 4,
    };
    (m.from() as u32 * 64 + to as u32) * 5 + promotion
}

fn is_capture(pos: &Position, m: Move) -> bool {
    m.is_en_passant()
        || pos
            .piece_at(m.to())
            .is_some_and(|piece| piece.color != pos.turn())
}

/// A defender position's legal moves in the order a walk takes them.
pub fn replies(pos: &Position) -> Vec<Move> {
    let mut replies = legal_moves(pos);
    replies.sort_unstable_by_key(|&m| number(pos, m));
    replies
}

/// The prover's move for a shakmaty one, if legal here.
pub fn board_move(pos: &Position, m: &shakmaty::Move) -> Option<Move> {
    legal_moves(pos)
        .into_iter()
        .find(|&legal| pos.to_shakmaty(legal) == *m)
}

/// Where a walk's moves come from: the search's when writing, the file's
/// when reading. `choose` returns the index of the move in `moves`, the
/// position's legal moves in the format's order.
trait Chooser {
    fn choose(&mut self, pos: &Position, moves: &[Move]) -> Result<usize, String>;
}

/// What a walk keeps of a position it has proven: the highest counter yet,
/// its length then, and the move written for it then, 0 for a position
/// settled with nothing written (no move goes from a square to itself).
#[derive(Clone, Copy)]
struct Proven {
    length: u32,
    counter: u8,
    written: u16,
}

struct Walk<'a, C> {
    attacker: Color,
    tablebase: Option<&'a Tablebase>,
    pieces: usize,
    chooser: C,
    /// Each position proven, by hash, at the highest counter yet.
    proven: HashMap<u64, Proven>,
    /// Whether a position was written again at a higher counter with
    /// another move: without one, the moves written cannot link positions
    /// in a circle (`walk_search`).
    mixed: bool,
    /// The move of each position proven at its highest counter, if kept.
    strategy: Option<Strategy>,
    /// The most pieces of a position the tables gave a win or a loss, 0 for
    /// none: the walk with only the tables up to that many pieces would be
    /// the same.
    decided: Cell<usize>,
    /// The positions of the line being walked, by hash: one met again ends
    /// the walk, as a proof cannot repeat a position.
    line: HashSet<u64>,
}

impl<C: Chooser> Walk<'_, C> {
    /// The tablebase verdict for the side to move, where the proof allows
    /// one: at most its pieces, the counter at 0, no castling rights.
    fn verdict(&self, pos: &Position) -> Result<Option<Verdict>, String> {
        let Some(tables) = self.tablebase else {
            return Ok(None);
        };
        if pos.halfmoves() != 0 || pos.has_castling_rights() || pos.occupied().count() > self.pieces
        {
            return Ok(None);
        }
        let verdict = tables.probe_held(&pos.to_atomic())?;
        if verdict != Verdict::Other {
            self.decided
                .set(self.decided.get().max(pos.occupied().count()));
        }
        Ok(Some(verdict))
    }

    fn root(&mut self, root: &Position) -> Result<u32, String> {
        let legal = legal_moves(root);
        match judge(root, self.attacker, &legal) {
            State::Won => Ok(0),
            State::Failed => Err(format!(
                "the root is over or at the 50-move limit: {}",
                fen(root)
            )),
            State::Open if root.turn() == self.attacker => self.attacker_to_move(root, legal),
            State::Open => self.defender_to_move(root, legal),
        }
    }

    fn attacker_to_move(&mut self, pos: &Position, legal: Vec<Move>) -> Result<u32, String> {
        let (hash, counter) = (pos.hash(), pos.halfmoves());
        if let Some(proven) = self.proven.get(&hash)
            && counter <= u32::from(proven.counter)
        {
            return Ok(proven.length);
        }
        if !self.line.insert(hash) {
            return Err(format!("a line repeats {}", fen(pos)));
        }
        let (length, m, written) = self.prove(pos, legal)?;
        self.line.remove(&hash);
        let before = self.proven.get(&hash).copied();
        if before.is_none_or(|proven| counter > u32::from(proven.counter)) {
            let written = if written { m.map_or(0, Move::bits) } else { 0 };
            self.mixed |= before.is_some_and(|proven| {
                proven.written != 0 && written != 0 && proven.written != written
            });
            self.proven.insert(
                hash,
                Proven {
                    length,
                    counter: counter as u8,
                    written,
                },
            );
            if let Some(strategy) = &mut self.strategy {
                strategy.insert(hash, m.map(|m| pos.to_shakmaty(m)));
            }
        }
        Ok(length)
    }

    /// Proves an attacker position: its length, its move, none for a
    /// tablebase win, and whether the move is written rather than one the
    /// position is settled by.
    fn prove(
        &mut self,
        pos: &Position,
        legal: Vec<Move>,
    ) -> Result<(u32, Option<Move>, bool), String> {
        if self.verdict(pos)? == Some(Verdict::Win) {
            return Ok((0, None, false));
        }
        // Checks first, then the format's order, which tells every move
        // apart. The positions after the moves are made again where needed,
        // which costs less than keeping them all.
        let mut moves: Vec<(bool, u32, Move)> = Vec::with_capacity(legal.len());
        for &m in &legal {
            let next = pos.play(m);
            let check = next.in_check();
            if won_at_once(&next, self.attacker, is_capture(pos, m), check) {
                return Ok((1, Some(m), false));
            }
            moves.push((!check, number(pos, m), m));
        }
        moves.sort_unstable_by_key(|&(quiet, number, _)| (quiet, number));
        for &(_, _, m) in moves.iter().take_while(|&&(quiet, ..)| !quiet) {
            let decided = self.decided.get();
            if let Some(length) = self.wins_in_two(&pos.play(m))? {
                return Ok((length, Some(m), false));
            }
            // A check that does not win in two would not with fewer tables
            // either, so the verdicts met in its lines are not the proof's.
            self.decided.set(decided);
        }
        let mut ordered = legal;
        ordered.clear();
        ordered.extend(moves.iter().map(|&(.., m)| m));
        let m = ordered[self.chooser.choose(pos, &ordered)?];
        let next = pos.play(m);
        let replies = legal_moves(&next);
        Ok((1 + self.defender_to_move(&next, replies)?, Some(m), true))
    }

    /// Whether a check leaves the defender lost at once whatever it does,
    /// and the length then.
    fn wins_in_two(&self, pos: &Position) -> Result<Option<u32>, String> {
        let replies = legal_moves(pos);
        if judge(pos, self.attacker, &replies) != State::Open {
            return Ok(None);
        }
        if self.verdict(pos)? == Some(Verdict::Loss) {
            return Ok(Some(1));
        }
        let mut longest = 0;
        for reply in replies {
            let next = pos.play(reply);
            let legal = legal_moves(&next);
            if judge(&next, self.attacker, &legal) != State::Open {
                return Ok(None);
            }
            if self.verdict(&next)? == Some(Verdict::Win) {
                continue;
            }
            if !self.capture_wins(&next, &legal) {
                return Ok(None);
            }
            longest = 1;
        }
        Ok(Some(1 + longest))
    }

    fn capture_wins(&self, pos: &Position, legal: &[Move]) -> bool {
        legal.iter().any(|&m| {
            is_capture(pos, m) && {
                let next = pos.play(m);
                won_at_once(&next, self.attacker, true, next.in_check())
            }
        })
    }

    fn defender_to_move(&mut self, pos: &Position, replies: Vec<Move>) -> Result<u32, String> {
        match judge(pos, self.attacker, &replies) {
            State::Won => return Ok(0),
            State::Failed => {
                return Err(format!(
                    "the line ends without the attacker winning: {}",
                    fen(pos)
                ));
            }
            State::Open => {}
        }
        if self.verdict(pos)? == Some(Verdict::Loss) {
            return Ok(0);
        }
        if !self.line.insert(pos.hash()) {
            return Err(format!("a line repeats {}", fen(pos)));
        }
        let mut replies = replies;
        // No two moves share a number.
        replies.sort_unstable_by_key(|&m| number(pos, m));
        let mut longest = 0;
        for reply in replies {
            let next = pos.play(reply);
            let legal = legal_moves(&next);
            if judge(&next, self.attacker, &legal) != State::Open {
                return Err(format!(
                    "a defence ends the line without the attacker winning: {}",
                    fen(pos)
                ));
            }
            longest = longest.max(self.attacker_to_move(&next, legal)?);
        }
        self.line.remove(&pos.hash());
        Ok(longest)
    }
}

/// A win of at most `SHORT` attacker moves in an open position, worked out
/// from the position alone: its length as the search counts one, a tablebase
/// win counting none where it is reached, and the attacker's move where it
/// is to move, none for a tablebase win. The shortest comes first, so that
/// the lengths shrink along every line these moves follow. For a search whose
/// proofs have dropped their short ones (`Proofs::drop_short`): it works them
/// out again here, and so do the moves it gives a proof's walk.
pub fn short(
    pos: &Position,
    attacker: Color,
    tablebase: Option<&Tablebase>,
) -> Option<(u32, Option<Move>)> {
    if pos.turn() != attacker {
        return (0..=SHORT)
            .find(|&moves| lost_within(pos, attacker, tablebase, moves))
            .map(|moves| (moves, None));
    }
    let legal = legal_moves(pos);
    if judge(pos, attacker, &legal) != State::Open {
        return None;
    }
    if held_verdict(pos, tablebase) == Some(Verdict::Win) {
        return Some((0, None));
    }
    (1..=SHORT).find_map(|moves| {
        win_within(pos, attacker, tablebase, moves, &legal).map(|m| (moves, Some(m)))
    })
}

/// A move with which the attacker wins within `moves` of its moves: at once,
/// or into a position the defender is lost in within one move less.
fn win_within(
    pos: &Position,
    attacker: Color,
    tablebase: Option<&Tablebase>,
    moves: u32,
    legal: &[Move],
) -> Option<Move> {
    legal.iter().copied().find(|&m| {
        let next = pos.play(m);
        won_at_once(&next, attacker, is_capture(pos, m), next.in_check())
            || lost_within(&next, attacker, tablebase, moves - 1)
    })
}

/// Whether the defender, to move in an open position, is lost within `moves`
/// more attacker moves: the tables give the position lost, or its every move
/// reaches a tablebase win or a position the attacker wins from in so many.
/// The answer is a plain truth about the position, so the replies go in any
/// order and each is made only when the ones before have not refuted the
/// win, which the first one usually does.
fn lost_within(pos: &Position, attacker: Color, tablebase: Option<&Tablebase>, moves: u32) -> bool {
    if moves == 0 {
        return tablebase.is_some_and(|tables| lost_at_once(pos, attacker, tables));
    }
    if !open_but_moves(pos, attacker) {
        return false;
    }
    if held_verdict(pos, tablebase) == Some(Verdict::Loss) {
        return moves::has_legal_move(pos);
    }
    let mut any = false;
    for &reply in pos.pseudo_legal().as_slice() {
        if !pos.legal(reply) {
            continue;
        }
        any = true;
        if !wins_within(&pos.play(reply), attacker, tablebase, moves) {
            return false;
        }
    }
    // With no move, the game is over.
    any
}

/// Whether the attacker, to move, has the game open and wins within `moves`
/// of its moves: by the tables, at once, or into a position the defender is
/// lost in within one move less.
fn wins_within(pos: &Position, attacker: Color, tablebase: Option<&Tablebase>, moves: u32) -> bool {
    if moves == 1 {
        return wins_within_one(pos, attacker, tablebase);
    }
    let legal = legal_moves(pos);
    judge(pos, attacker, &legal) == State::Open
        && (held_verdict(pos, tablebase) == Some(Verdict::Win)
            || win_within(pos, attacker, tablebase, moves, &legal).is_some())
}

/// `wins_within` in one move, which a short win's every line asks of its
/// last attacker move, without the moves made: a capture that explodes the
/// king is found from the board, and a move that neither captures nor
/// checks wins only where the tables can come into it.
fn wins_within_one(pos: &Position, attacker: Color, tablebase: Option<&Tablebase>) -> bool {
    if !open_but_moves(pos, attacker) {
        return false;
    }
    // A move can reach the tables, the attacker's capture and then each
    // defence taking ten pieces off the board at most (`lost_at_once`); a
    // plain move takes none.
    let pieces = pos.occupied().count();
    let tables = tablebase.filter(|tables| pieces <= tables.max_pieces() + 20);
    let plain_tables = tables.filter(|tables| pieces <= tables.max_pieces() + 10);
    if tables.is_some_and(|tables| pieces <= tables.max_pieces()) {
        if !moves::has_legal_move(pos) {
            return false;
        }
        if held_verdict(pos, tablebase) == Some(Verdict::Win) {
            return true;
        }
    }
    // Such a capture is legal, so the game is open.
    if moves::exploding_capture(pos, attacker).is_some() {
        return true;
    }
    let mover = Mover::new(pos);
    pos.pseudo_legal().as_slice().iter().any(|&m| {
        let plain = mover.plain(m);
        if (plain && plain_tables.is_none()) || !mover.legal(m) {
            return false;
        }
        let effect = mover.effect(m);
        mover.explodes_them(&effect)
            || (!plain && mover.gives_check(m, &effect) && !moves::has_legal_move(&pos.play(m)))
            || tables.is_some_and(|tables| lost_at_once(&pos.play(m), attacker, tables))
    })
}

/// Whether the defender, to move, is lost to the tables with no attacker
/// move left: the position open and lost, or its every move a capture into
/// a position the tables win. A move that is not such a capture, the usual
/// first, settles it without a probe.
fn lost_at_once(pos: &Position, attacker: Color, tables: &Tablebase) -> bool {
    // A capture takes ten pieces off the board at most, and no other move
    // any.
    let pieces = pos.occupied().count();
    if pieces > tables.max_pieces() + 10 {
        return false;
    }
    if pieces > tables.max_pieces() {
        let base = pos.child_base();
        for &reply in pos.pseudo_legal().as_slice() {
            if !pos.legal(reply) {
                continue;
            }
            let effect = pos.effect(reply, base);
            if !effect.capture || effect.occupied.count() > tables.max_pieces() {
                return false;
            }
        }
    }
    let replies = legal_moves(pos);
    if judge(pos, attacker, &replies) != State::Open {
        return false;
    }
    if held_verdict(pos, Some(tables)) == Some(Verdict::Loss) {
        return true;
    }
    replies.into_iter().all(|reply| {
        let next = pos.play(reply);
        let legal = legal_moves(&next);
        judge(&next, attacker, &legal) == State::Open
            && held_verdict(&next, Some(tables)) == Some(Verdict::Win)
    })
}

/// Whether the game is open by `judge` but for its legal moves: both kings
/// on the board, either side able to win, the 50-move limit not reached.
fn open_but_moves(pos: &Position, attacker: Color) -> bool {
    pos.halfmoves() < 100
        && pos.king(!attacker).is_some()
        && pos.king(attacker).is_some()
        && !(moves::cannot_win(pos, Color::White) && moves::cannot_win(pos, Color::Black))
}

/// The tablebase verdict for the side to move, where a proof written with
/// these tables allows one (`Walk::verdict`).
fn held_verdict(pos: &Position, tablebase: Option<&Tablebase>) -> Option<Verdict> {
    let tables = tablebase?;
    if pos.halfmoves() != 0
        || pos.has_castling_rights()
        || pos.occupied().count() > tables.max_pieces()
    {
        return None;
    }
    tables.probe_held(&pos.to_atomic()).ok()
}

fn fen(pos: &Position) -> String {
    Fen::from_position(&pos.to_atomic(), EnPassantMode::Legal).to_string()
}

/// The search's moves, written as their indices.
struct FromSearch<F> {
    source: F,
    indices: Vec<u32>,
}

impl<F: FnMut(&Position) -> Option<Move>> Chooser for FromSearch<F> {
    fn choose(&mut self, pos: &Position, moves: &[Move]) -> Result<usize, String> {
        let m = (self.source)(pos)
            .ok_or_else(|| format!("no winning move recorded for {}", fen(pos)))?;
        let index = moves
            .iter()
            .position(|legal| *legal == m)
            .ok_or_else(|| format!("the recorded move is not legal in {}", fen(pos)))?;
        self.indices.push(index as u32);
        Ok(index)
    }
}

/// A file's moves, read in order.
struct FromFile<'a> {
    indices: &'a [u32],
    next: usize,
}

impl Chooser for FromFile<'_> {
    fn choose(&mut self, pos: &Position, moves: &[Move]) -> Result<usize, String> {
        let index = *self
            .indices
            .get(self.next)
            .ok_or("the proof's moves run out")? as usize;
        self.next += 1;
        if index >= moves.len() {
            return Err(format!("move {index} of {} in {}", moves.len(), fen(pos)));
        }
        Ok(index)
    }
}

/// Writes a proof's file through a temporary one, so that a proof there is
/// always whole.
pub fn save(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let error = |error: std::io::Error| format!("cannot write the proof: {error}");
    let partial = path.with_extension("partial");
    let mut out = File::create(&partial).map_err(error)?;
    out.write_all(bytes).map_err(error)?;
    out.sync_all().map_err(error)?;
    fs::rename(&partial, path).map_err(error)
}

/// A proof of `root` as the bytes of its file, and its header.
pub fn encode(
    root: &Atomic,
    attacker: Color,
    tablebase: Option<&Tablebase>,
    source: impl FnMut(&Position) -> Option<Move>,
) -> Result<(Header, Vec<u8>), String> {
    let (header, indices) = walk(root, attacker, tablebase, source)?;
    let bytes = file(&header, &indices)?;
    Ok((header, bytes))
}

/// Walks a proof of `attacker`'s win from `root`, whoever is to move there,
/// with the move `source` gives in each attacker position that needs one:
/// its header, length included, and its moves.
pub fn walk(
    root: &Atomic,
    attacker: Color,
    tablebase: Option<&Tablebase>,
    source: impl FnMut(&Position) -> Option<Move>,
) -> Result<(Header, Vec<u32>), String> {
    walk_once(root, attacker, tablebase, source).map(|(header, indices, _)| (header, indices))
}

/// `walk` for a search's proof, whose `source` is asked for a position's
/// shortest proof at its counter, or with `true` for the one that holds
/// furthest (`Dfpn::proof_move`). The walk takes the shortest; if that
/// writes some position at two counters with two moves, it walks again
/// taking the ones that hold furthest, and fails if that still does; it
/// tells whether the proof is the second walk's, which can be longer. A
/// position's moves at different counters, each sound on its own lines, can
/// together link positions in a circle, which the format forbids. A walk
/// that ends with one move a position has none: a circle's moves are all
/// quiet, as no capture or pawn move is undone, and each of its positions
/// had its move last walked at the highest counter it was written at, where
/// the next position of the circle, two plies on, was proven at that
/// counter plus two or written at a higher one; round the circle the
/// counters would rise without end.
pub fn walk_search(
    root: &Atomic,
    attacker: Color,
    tablebase: Option<&Tablebase>,
    mut source: impl FnMut(&Position, bool) -> Option<Move>,
) -> Result<(Header, Vec<u32>, bool), String> {
    let (header, indices, mixed) = walk_once(root, attacker, tablebase, |pos| source(pos, false))?;
    if !mixed {
        return Ok((header, indices, false));
    }
    let (header, indices, mixed) = walk_once(root, attacker, tablebase, |pos| source(pos, true))?;
    if mixed {
        return Err(TWO_MOVES.into());
    }
    Ok((header, indices, true))
}

/// Why a proof is not written when even the proofs that hold furthest give
/// some position two moves: its lines could repeat a position.
pub const TWO_MOVES: &str = "a position is proven with two moves, and a line could repeat it";

/// `walk`, and whether it wrote some position at two counters with two
/// moves.
fn walk_once(
    root: &Atomic,
    attacker: Color,
    tablebase: Option<&Tablebase>,
    source: impl FnMut(&Position) -> Option<Move>,
) -> Result<(Header, Vec<u32>, bool), String> {
    let board = Position::from_atomic(root);
    let pieces = tablebase.map_or(0, Tablebase::max_pieces);
    let mut walk = Walk {
        attacker,
        tablebase,
        pieces,
        chooser: FromSearch {
            source,
            indices: Vec::new(),
        },
        proven: HashMap::new(),
        mixed: false,
        strategy: None,
        decided: Cell::new(0),
        line: HashSet::new(),
    };
    let length = walk.root(&board)?;
    let mixed = walk.mixed;
    let indices = walk.chooser.indices;
    // Tables the proof never needed are not asked of its checkers.
    let header = Header {
        root: Fen::from_position(root, EnPassantMode::Legal).to_string(),
        attacker: (attacker != board.turn()).then_some(attacker),
        tablebases: walk.decided.get(),
        moves: length,
        positions: indices.len() as u64,
    };
    Ok((header, indices, mixed))
}

/// A proof's file: its header line, then its moves in a zstd frame.
pub fn file(header: &Header, indices: &[u32]) -> Result<Vec<u8>, String> {
    let mut body = Vec::with_capacity(indices.len());
    for &index in indices {
        if index < 128 {
            body.push(index as u8);
        } else {
            body.push((index & 127) as u8 | 128);
            body.push((index >> 7) as u8);
        }
    }
    // With zstd's checksum, so that a damaged file shows as damaged.
    let body = zstd::bulk::Compressor::new(LEVEL)
        .and_then(|mut compressor| {
            compressor.include_checksum(true)?;
            compressor.compress(&body)
        })
        .map_err(|e| format!("cannot compress: {e}"))?;
    let mut bytes = header_line(header).into_bytes();
    bytes.push(b'\n');
    bytes.extend(body);
    Ok(bytes)
}

/// A proof in JSON lines as the bytes of a compact one, its moves taken by
/// position and counter, as the lines list them; JSON lines prove wins for
/// the side to move.
pub fn encode_jsonl(
    text: &str,
    tablebase: Option<&Tablebase>,
) -> Result<(Header, Vec<u8>), String> {
    let parse = |fen: &str| -> Result<Atomic, String> {
        let setup: Fen = fen.parse().map_err(|e| format!("a bad FEN {fen}: {e}"))?;
        crate::from_setup(setup.into_setup())
            .map_err(|flaws| format!("an illegal position {fen}: {flaws}"))
    };
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let head: Value =
        serde_json::from_str(lines.next().ok_or("an empty proof")?).map_err(|e| e.to_string())?;
    let root = parse(head["root"].as_str().ok_or("a proof without its root")?)?;
    let mut moves = HashMap::new();
    for line in lines {
        let entry: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let Some(uci) = entry["move"].as_str() else {
            continue;
        };
        let pos = parse(entry["fen"].as_str().ok_or("a line without its position")?)?;
        let m = uci
            .parse::<shakmaty::uci::UciMove>()
            .map_err(|e| e.to_string())?
            .to_move(&pos)
            .map_err(|e| e.to_string())?;
        let board = Position::from_atomic(&pos);
        let m = board_move(&board, &m).ok_or(format!("{uci} is not legal"))?;
        moves.insert((board.hash(), board.halfmoves()), m);
    }
    let attacker = shakmaty::Position::turn(&root);
    encode(&root, attacker, tablebase, |pos| {
        moves.get(&(pos.hash(), pos.halfmoves())).copied()
    })
}

/// The header line, `provari` first so that a file shows what it is, and the
/// attacker only where it is not the side to move.
fn header_line(header: &Header) -> String {
    let attacker = header.attacker.map_or(String::new(), |attacker| {
        format!(r#","attacker":"{attacker}""#)
    });
    format!(
        r#"{{"provari":{VERSION},"variant":"atomic","root":{}{attacker},"tablebases":{},"moves":{},"positions":{}}}"#,
        json!(header.root),
        header.tablebases,
        header.moves,
        header.positions
    )
}

/// A proof in this format: its header and moves; none for a proof in JSON
/// lines.
pub fn read(path: &Path) -> Result<Option<(Header, Vec<u32>)>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode(&bytes)
}

/// A proof's header and moves from the bytes of its file; none for JSON
/// lines.
pub fn decode(bytes: &[u8]) -> Result<Option<(Header, Vec<u32>)>, String> {
    let Some(end) = bytes.iter().take(4096).position(|&byte| byte == b'\n') else {
        return Ok(None);
    };
    let Ok(head) = serde_json::from_slice::<Value>(&bytes[..end]) else {
        return Ok(None);
    };
    if head.get("provari").is_none() {
        return Ok(None);
    }
    if head["provari"] != VERSION || head["variant"] != "atomic" {
        return Err(format!("not a proof this version reads: {head}"));
    }
    let number = |key: &str| head[key].as_u64().ok_or(format!("a bad {key}"));
    let attacker = head
        .get("attacker")
        .map(|attacker| {
            attacker
                .as_str()
                .and_then(|name| name.parse::<Color>().ok())
                .ok_or("a bad attacker")
        })
        .transpose()?;
    let header = Header {
        root: head["root"].as_str().ok_or("a bad root")?.to_owned(),
        attacker,
        tablebases: number("tablebases")? as usize,
        moves: number("moves")? as u32,
        positions: number("positions")?,
    };
    let body = zstd::bulk::decompress(&bytes[end + 1..], header.positions as usize * 2 + 1)
        .map_err(|e| format!("a damaged proof: {e}"))?;
    let mut indices = Vec::with_capacity(header.positions as usize);
    let mut body = body.iter();
    while let Some(&first) = body.next() {
        if first < 128 {
            indices.push(u32::from(first));
        } else {
            let second = *body.next().ok_or("a damaged proof: a move cut short")?;
            indices.push(u32::from(first & 127) | u32::from(second) << 7);
        }
    }
    if indices.len() as u64 != header.positions {
        return Err("a damaged proof: the moves do not match the header".into());
    }
    Ok(Some((header, indices)))
}

/// Walks a proof back into its root, its attacker, and the move of every
/// attacker position it proves: the moves written and those its settled
/// positions take.
pub fn strategy(
    header: &Header,
    indices: &[u32],
    tablebase: Option<&Tablebase>,
) -> Result<(Atomic, Color, Strategy), String> {
    let fen: Fen = header
        .root
        .parse()
        .map_err(|e| format!("a bad root: {e}"))?;
    let root = crate::from_setup(fen.into_setup()).map_err(|e| format!("an illegal root: {e}"))?;
    if header.tablebases > 0
        && tablebase.is_none_or(|tables| tables.max_pieces() < header.tablebases)
    {
        return Err(format!(
            "the proof needs the tablebases up to {} pieces",
            header.tablebases
        ));
    }
    let board = Position::from_atomic(&root);
    let attacker = match header.attacker {
        None => board.turn(),
        Some(attacker) if attacker != board.turn() => attacker,
        Some(_) => return Err("the header names the side to move as the attacker".into()),
    };
    let mut walk = Walk {
        attacker,
        tablebase: tablebase.filter(|_| header.tablebases > 0),
        pieces: header.tablebases,
        chooser: FromFile { indices, next: 0 },
        proven: HashMap::new(),
        mixed: false,
        strategy: Some(Strategy::new()),
        decided: Cell::new(0),
        line: HashSet::new(),
    };
    let length = walk.root(&board)?;
    if walk.chooser.next != indices.len() || length != header.moves {
        return Err("the proof does not match its header".into());
    }
    Ok((root, attacker, walk.strategy.unwrap_or_default()))
}

/// A win whose two rook positions, A (rook c7, king e8) and B (rook f7, king
/// d8), each reach the other by a rook move and the king's only reply, and
/// each win at once by a check and a mate. The king has d8 and e8 alone, and
/// Black's pawn makes the tree: its move to h5 comes at the root or a move
/// later, so White meets A and B twice, at the counters 2 and 4 and again,
/// after its rook has gone round, at 10 and 8.
#[cfg(test)]
pub const CIRCLE: &str = "2Nk1N2/2R4p/3PP3/8/7P/8/8/7K b - - 0 1";

/// White's move in `CIRCLE`'s win, for a position's FEN: below the counter 6
/// A goes to B and B checks; from 6 on, and with `furthest`, as a search
/// asked for the moves that hold furthest gives them, B goes to A and A
/// checks. With White's king on g1 and White to move, a root for proofs of
/// the side to move, which reaches `CIRCLE`.
#[cfg(test)]
pub fn circle_move(fen: &str, furthest: bool) -> Option<&'static str> {
    let mut fields = fen.split(' ');
    let board = fields.next()?;
    let high = furthest || fields.nth(3)?.parse::<u32>().ok()? >= 6;
    Some(match board {
        "2Nk1N2/2R4p/3PP3/8/7P/8/8/6K1" => "g1h1",
        // The king moved first, or after the pawn's single step: the check
        // and the mate.
        "2N1kN2/2R4p/3PP3/8/7P/8/8/7K" | "2N1kN2/2R5/3PP2p/8/7P/8/7K/8" => "c7e7",
        // After the pawn's single step and after its double one, a king
        // move, which tells the two ways to h5 apart.
        "2Nk1N2/2R5/3PP2p/8/7P/8/8/7K" | "2Nk1N2/2R5/3PP3/7p/7P/8/8/7K" => "h1h2",
        // A and B.
        "2N1kN2/2R5/3PP3/7p/7P/8/7K/8" if high => "c7e7",
        "2N1kN2/2R5/3PP3/7p/7P/8/7K/8" => "c7f7",
        "2Nk1N2/5R2/3PP3/7p/7P/8/7K/8" if high => "f7c7",
        "2Nk1N2/5R2/3PP3/7p/7P/8/7K/8" => "f7d7",
        // The rook's way round to B: b7, a7, g7, f7.
        "2Nk1N2/2R5/3PP3/7p/7P/8/7K/8" => "c7b7",
        "2N1kN2/1R6/3PP3/7p/7P/8/7K/8" => "b7a7",
        "2Nk1N2/R7/3PP3/7p/7P/8/7K/8" => "a7g7",
        "2N1kN2/6R1/3PP3/7p/7P/8/7K/8" => "g7f7",
        // The mates, which only proofs in JSON lines write.
        _ if board.starts_with("2Nk1N2/4R") => "e7e8",
        _ if board.starts_with("2N1kN2/3R") => "d7d8",
        _ => return None,
    })
}

/// A mate in 4, in JSON lines: a quiet knight move first, so that the
/// compact format writes 6 moves.
#[cfg(test)]
pub const SMALL: &str = r#"{"moves":4,"root":"rnbqkb1r/1pp1n1pp/p2p4/3P1pN1/4p2P/2N5/PPP1PPP1/R1BQKB1R w KQkq - 0 1"}
{"fen":"rnbqkb1r/1pp1n1pp/p2p4/3P1pN1/4p2P/2N5/PPP1PPP1/R1BQKB1R w KQkq - 0 1","move":"g5f7"}
{"fen":"rnb1kb1r/1ppqnNpp/p2p4/3P1p2/4p2P/2N5/PPP1PPP1/R1BQKB1R w KQkq - 2 2","move":"c1g5"}
{"fen":"rnb1kbnr/1ppq1Npp/p2p4/3P1pB1/4p2P/2N5/PPP1PPP1/R2QKB1R w KQkq - 4 3","move":"g5e7"}
{"fen":"rnb1kb1r/1ppq1Npp/p2p2n1/3P1pB1/4p2P/2N5/PPP1PPP1/R2QKB1R w KQkq - 4 3","move":"g5e7"}
{"fen":"rnb1kb1r/1ppq1Npp/p1np4/3P1pB1/4p2P/2N5/PPP1PPP1/R2QKB1R w KQkq - 4 3","move":"g5e7"}
{"fen":"rnb1kb1r/1ppq1Npp/p2p4/5pB1/4p2P/2N5/PPP1PPP1/R2QKB1R w KQkq - 0 3","move":"g5e7"}
"#;

/// SMALL's win from the position after the attacker's first move, with the
/// defender to move there, so that the header names the attacker.
#[cfg(test)]
pub fn small_defended() -> (Header, Vec<u8>) {
    let (_, bytes) = encode_jsonl(SMALL, None).unwrap();
    let (header, indices) = decode(&bytes).unwrap().unwrap();
    let (root, attacker, moves) = strategy(&header, &indices, None).unwrap();
    let mut after = root.clone();
    shakmaty::Position::play_unchecked(&mut after, moves[&crate::search::hash(&root)].unwrap());
    encode(&after, attacker, None, |pos| {
        moves
            .get(&pos.hash())
            .copied()
            .flatten()
            .and_then(|m| board_move(pos, &m))
    })
    .unwrap()
}

#[cfg(test)]
mod tests {
    use shakmaty::{Color, KnownOutcome, Outcome, Position as _, uci::UciMove, variant::Atomic};

    use super::{
        CIRCLE, SMALL, State, circle_move, decode, encode_jsonl, fen, file, is_capture,
        legal_moves, number, short, small_defended, state, strategy, walk, walk_once, walk_search,
        won_at_once,
    };
    use crate::board::{Move, Position, from_fen};
    use crate::search::hash;
    use crate::tablebase::Tablebase;

    fn circle(pos: &Position, furthest: bool) -> Option<Move> {
        let uci = circle_move(&fen(pos), furthest)?;
        legal_moves(pos)
            .into_iter()
            .find(|&m| UciMove::from_standard(pos.to_shakmaty(m)).to_string() == uci)
    }

    /// A position written at two counters with two moves: each line of the
    /// walk wins, but A's first move and B's second link A to B to A, and
    /// the checker refuses the proof. `walk_search` sees the two moves and
    /// walks again with the moves that hold furthest, one a position, which
    /// the checker accepts.
    #[test]
    fn a_search_s_walk_writes_one_move_a_position() {
        let root: Atomic = crate::position(CIRCLE).unwrap();
        let white = Color::White;
        let checked = |header, indices: &[u32]| {
            let bytes = file(header, indices).unwrap();
            crate::check::check_proof("proof", &bytes, None)
        };
        let (header, indices, mixed) =
            walk_once(&root, white, None, |pos| circle(pos, false)).unwrap();
        assert!(mixed);
        assert_eq!((header.moves, header.positions), (8, 12));
        let refused = checked(&header, &indices);
        assert_eq!(refused["valid"], false, "{refused}");
        let error = refused["error"].as_str().unwrap();
        assert!(
            error.starts_with("a line can repeat the position"),
            "{error}"
        );

        let (header, indices, rewritten) = walk_search(&root, white, None, circle).unwrap();
        assert!(rewritten);
        let (again, same, mixed) = walk_once(&root, white, None, |pos| circle(pos, true)).unwrap();
        assert!(!mixed);
        assert_eq!((header.moves, &indices), (again.moves, &same));
        assert_eq!((header.moves, header.positions), (8, 11));
        let accepted = checked(&header, &indices);
        assert_eq!(accepted["valid"], true, "{accepted}");
        strategy(&header, &indices, None).unwrap();

        // A walk whose every position has one move is not walked again.
        let mut asked = 0;
        let (once, _, rewritten) = walk_search(&root, white, None, |pos, furthest| {
            asked += 1;
            assert!(!furthest);
            circle(pos, true)
        })
        .unwrap();
        assert_eq!((once.positions, asked, rewritten), (11, 11, false));
        // The plain walk takes its source's moves as they come.
        let (plain, _) = walk(&root, white, None, |pos| circle(pos, false)).unwrap();
        assert_eq!(plain.positions, 12);
        // A source that gives two moves even when asked for the ones that
        // hold furthest leaves no proof.
        let failed = walk_search(&root, white, None, |pos, _| circle(pos, false));
        assert_eq!(failed.err().unwrap(), super::TWO_MOVES);
    }

    #[test]
    fn reads_back_what_it_writes() {
        let (header, bytes) = encode_jsonl(SMALL, None).unwrap();
        assert_eq!((header.moves, header.positions), (4, 6));
        let (read, indices) = decode(&bytes).unwrap().unwrap();
        assert_eq!((read.moves, read.positions), (4, 6));
        assert_eq!(read.root, header.root);
        // The side to move attacks, and the header leaves it unnamed.
        assert_eq!(read.attacker, None);
        let (root, attacker, moves) = strategy(&read, &indices, None).unwrap();
        assert_eq!(attacker, Color::White);
        let first = moves[&hash(&root)].unwrap();
        assert_eq!(UciMove::from_standard(first).to_string(), "g5f7");
    }

    /// A short win is worked out from the position alone, with its first
    /// move: SMALL's win in two, a bishop move, a reply and a capture that
    /// explodes the king, up to the last counter its three plies fit under;
    /// after the bishop move, the one attacker move left, and after each
    /// reply the capture. SMALL's wins in four and in three are not short.
    #[test]
    fn works_out_short_wins() {
        let white = Color::White;
        let at = |fen: &str, counter| from_fen(fen).unwrap().at_counter(counter);
        for fen in [
            "rnbqkb1r/1pp1n1pp/p2p4/3P1pN1/4p2P/2N5/PPP1PPP1/R1BQKB1R w KQkq - 0 1",
            "rnb1kb1r/1ppqnNpp/p2p4/3P1p2/4p2P/2N5/PPP1PPP1/R1BQKB1R w KQkq - 2 2",
        ] {
            assert_eq!(short(&at(fen, 0), white, None), None, "{fen}");
        }
        let two = "rnb1kbnr/1ppq1Npp/p2p4/3P1pB1/4p2P/2N5/PPP1PPP1/R2QKB1R w KQkq - 4 3";
        for counter in [0, 4, 97] {
            let pos = at(two, counter);
            let (length, m) = short(&pos, white, None).unwrap();
            let m = m.unwrap();
            let uci = UciMove::from_standard(pos.to_shakmaty(m)).to_string();
            assert_eq!((length, uci.as_str()), (2, "g5e7"), "{counter}");
            let after = pos.play(m);
            assert_eq!(short(&after, white, None), Some((1, None)), "{counter}");
            for reply in legal_moves(&after) {
                let last = after.play(reply);
                let (length, m) = short(&last, white, None).unwrap();
                let over = last.play(m.unwrap());
                assert!(length == 1 && over.king(Color::Black).is_none());
            }
        }
        for counter in [98, 99] {
            assert_eq!(short(&at(two, counter), white, None), None, "{counter}");
        }
    }

    /// A proof of a win for the side not to move: the header names the
    /// attacker, the walk starts with the defences, and the moves are the
    /// side to move's proof's but its first.
    #[test]
    fn proves_a_win_for_the_side_not_to_move() {
        let (header, bytes) = small_defended();
        assert_eq!(
            (header.attacker, header.moves, header.positions),
            (Some(Color::White), 3, 5)
        );
        let line = r#"{"provari":1,"variant":"atomic","root":"rnbqkb1r/1pp1nNpp/p2p4/3P1p2/4p2P/2N5/PPP1PPP1/R1BQKB1R b KQkq - 1 1","attacker":"white","tablebases":0,"moves":3,"positions":5}"#;
        assert!(bytes.starts_with(format!("{line}\n").as_bytes()));
        let (read, indices) = decode(&bytes).unwrap().unwrap();
        assert_eq!(read.attacker, Some(Color::White));
        let (root, attacker, moves) = strategy(&read, &indices, None).unwrap();
        assert_eq!((root.turn(), attacker), (Color::Black, Color::White));
        let (_, first) = encode_jsonl(SMALL, None).unwrap();
        let (first, first_indices) = decode(&first).unwrap().unwrap();
        let (first_root, _, first_moves) = strategy(&first, &first_indices, None).unwrap();
        assert!(!moves.contains_key(&hash(&first_root)));
        assert_eq!(moves.len(), first_moves.len() - 1);
        assert!(moves.iter().all(|(key, m)| first_moves.get(key) == Some(m)));
        // The side to move named as the attacker, or a side that is none.
        let mut named = read;
        named.attacker = Some(Color::Black);
        let error = strategy(&named, &indices, None).err();
        assert_eq!(
            error.as_deref(),
            Some("the header names the side to move as the attacker")
        );
        let mut bad = line.replace("white", "red").into_bytes();
        bad.extend(&bytes[line.len()..]);
        assert_eq!(decode(&bad).err().as_deref(), Some("a bad attacker"));
    }

    /// The format's order of moves, and how the game stands, as the prover's
    /// board works them out, against shakmaty over random games.
    #[test]
    fn orders_and_judges_as_shakmaty() {
        let starts = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rn2k2r/pp3ppp/2p5/3pP3/8/2N5/PPP2PPP/R3K2R w KQkq d6 0 1",
            "r3k2r/1P4pp/8/2pP4/8/8/p5PP/R3K2R w KQkq c6 0 1",
            "rnbqkbnr/ppppp1p1/5p2/7p/8/4PN2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
            "r3k2r/8/8/3qQ3/8/8/8/R3K2R b KQkq - 0 1",
            "8/2P5/8/3k4/8/3K4/5p2/8 w - - 0 1",
            "rnb1kb1r/p1qppp1p/1pp5/6p1/P2P1QnP/N1P2P2/1P2P1P1/R1B1KBNR b KQkq - 1 10",
            "8/8/8/8/8/2k5/8/K1B5 w - - 98 80",
        ];
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        for start in starts {
            for _ in 0..20 {
                let mut pos: Position = from_fen(start).unwrap();
                for _ in 0..80 {
                    let atomic = pos.to_atomic();
                    let mut ours: Vec<_> = legal_moves(&pos)
                        .into_iter()
                        .map(|m| ((!pos.play(m).in_check(), number(&pos, m)), m))
                        .collect();
                    ours.sort_by_key(|(key, _)| *key);
                    let ours: Vec<String> = ours
                        .iter()
                        .map(|(_, m)| UciMove::from_standard(pos.to_shakmaty(*m)).to_string())
                        .collect();
                    assert_eq!(ours, shakmaty_order(&atomic), "{start}");
                    for &m in &legal_moves(&pos) {
                        let next = pos.play(m);
                        assert_eq!(
                            won_at_once(&next, pos.turn(), is_capture(&pos, m), next.in_check()),
                            state(&next, pos.turn()) == State::Won,
                            "{start}"
                        );
                    }
                    for attacker in [Color::White, Color::Black] {
                        let expected = match atomic.outcome() {
                            Outcome::Known(KnownOutcome::Decisive { winner })
                                if winner == attacker =>
                            {
                                State::Won
                            }
                            Outcome::Known(_) => State::Failed,
                            Outcome::Unknown if atomic.halfmoves() >= 100 => State::Failed,
                            Outcome::Unknown => State::Open,
                        };
                        assert!(state(&pos, attacker) == expected, "{start}");
                    }
                    let moves = legal_moves(&pos);
                    if atomic.outcome().is_known() || moves.is_empty() {
                        break;
                    }
                    seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    pos = pos.play(moves[(seed >> 33) as usize % moves.len()]);
                }
            }
        }
    }

    /// Short wins, worked out as `short` does, against the plain search of
    /// every line in `plainly`, over random games, for either side as the
    /// attacker, without the tables.
    #[test]
    fn works_out_short_wins_as_plainly() {
        let (positions, wins) = short_as_plainly(None, &STARTS, 3);
        assert!(positions > 1000 && wins > 100, "{positions} {wins}");
    }

    /// `works_out_short_wins_as_plainly` with the tables, from positions of
    /// few pieces too, where they decide; with the tables up to five pieces
    /// in `PROVARI_TABLEBASES`:
    /// `cargo test --release -- --ignored works_out_short_wins_with_tables`.
    #[test]
    #[ignore = "needs the tablebases"]
    fn works_out_short_wins_with_tables() {
        let path = std::env::var_os("PROVARI_TABLEBASES").expect("PROVARI_TABLEBASES");
        let tables = Tablebase::open(std::path::Path::new(&path)).unwrap();
        let few = [
            "4k3/8/8/2r5/8/3P4/1N6/R3K3 w - - 0 1",
            "1n2k3/8/8/8/8/8/5PP1/K2Q4 w - - 0 1",
            "8/8/3kb3/8/2n5/8/2P1Q3/4K1N1 w - - 0 1",
            "r3k3/1pp5/8/8/8/8/2PPB3/R3K2N w Qq - 0 1",
            "2kr4/ppp5/8/8/8/8/5PPP/2R3K1 b - - 0 1",
        ];
        // A win in two whose last move is a capture into the tables, and
        // then every defence: from 16 pieces with five-piece tables.
        let reach: Position = from_fen("kN6/BR6/K7/3p4/5Q2/3NQR1p/3BRB1R/6N1 b - - 0 1").unwrap();
        for attacker in [Color::White, Color::Black] {
            assert_eq!(
                short(&reach, attacker, Some(&tables)),
                plainly::short(&reach, attacker, Some(&tables))
            );
        }
        assert_eq!(short(&reach, Color::White, Some(&tables)), Some((1, None)));
        let starts: Vec<&str> = STARTS.iter().chain(&few).copied().collect();
        let (positions, wins) = short_as_plainly(Some(&tables), &starts, 10);
        assert!(positions > 1000 && wins > 100, "{positions} {wins}");
    }

    const STARTS: [&str; 8] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn2k2r/pp3ppp/2p5/3pP3/8/2N5/PPP2PPP/R3K2R w KQkq d6 0 1",
        "r3k2r/1P4pp/8/2pP4/8/8/p5PP/R3K2R w KQkq c6 0 1",
        "rnbqkbnr/ppppp1p1/5p2/7p/8/4PN2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "r3k2r/8/8/3qQ3/8/8/8/R3K2R b KQkq - 0 1",
        "8/2P5/8/3k4/8/3K4/5p2/8 w - - 0 1",
        "rnb1kb1r/p1qppp1p/1pp5/6p1/P2P1QnP/N1P2P2/1P2P1P1/R1B1KBNR b KQkq - 1 10",
        "r3kb1r/1p3Npp/3ppn2/p4p2/8/8/PPPPPPPP/R1BQKB1R w KQkq - 0 1",
    ];

    /// Random games from each start, `games` of them, comparing `short` with
    /// `plainly::short` at every position for either attacker: the positions
    /// compared, and the wins found.
    fn short_as_plainly(
        tables: Option<&Tablebase>,
        starts: &[&str],
        games: usize,
    ) -> (usize, usize) {
        let (mut positions, mut wins) = (0, 0);
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        for start in starts {
            for _ in 0..games {
                let mut pos: Position = from_fen(start).unwrap();
                for _ in 0..60 {
                    for attacker in [Color::White, Color::Black] {
                        let found = short(&pos, attacker, tables);
                        assert_eq!(found, plainly::short(&pos, attacker, tables), "{start}");
                        positions += 1;
                        wins += usize::from(found.is_some());
                    }
                    let moves = legal_moves(&pos);
                    if moves.is_empty() || state(&pos, Color::White) != State::Open {
                        break;
                    }
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    pos = pos.play(moves[(seed >> 33) as usize % moves.len()]);
                }
            }
        }
        (positions, wins)
    }

    /// Short wins as they were first worked out, every line played out in
    /// full: what `short` must agree with.
    mod plainly {
        use shakmaty::Color;

        use super::super::{
            SHORT, State, Verdict, held_verdict, is_capture, judge, legal_moves, won_at_once,
        };
        use crate::board::{Move, Position};
        use crate::tablebase::Tablebase;

        pub fn short(
            pos: &Position,
            attacker: Color,
            tablebase: Option<&Tablebase>,
        ) -> Option<(u32, Option<Move>)> {
            if pos.turn() != attacker {
                return (0..=SHORT)
                    .find(|&moves| lost_within(pos, attacker, tablebase, moves))
                    .map(|moves| (moves, None));
            }
            let legal = legal_moves(pos);
            if judge(pos, attacker, &legal) != State::Open {
                return None;
            }
            if held_verdict(pos, tablebase) == Some(Verdict::Win) {
                return Some((0, None));
            }
            (1..=SHORT).find_map(|moves| {
                win_within(pos, attacker, tablebase, moves, &legal).map(|m| (moves, Some(m)))
            })
        }

        fn win_within(
            pos: &Position,
            attacker: Color,
            tablebase: Option<&Tablebase>,
            moves: u32,
            legal: &[Move],
        ) -> Option<Move> {
            legal.iter().copied().find(|&m| {
                let next = pos.play(m);
                won_at_once(&next, attacker, is_capture(pos, m), next.in_check())
                    || lost_within(&next, attacker, tablebase, moves - 1)
            })
        }

        fn lost_within(
            pos: &Position,
            attacker: Color,
            tablebase: Option<&Tablebase>,
            moves: u32,
        ) -> bool {
            if moves == 0
                && tablebase.is_none_or(|tables| pos.occupied().count() > tables.max_pieces() + 10)
            {
                return false;
            }
            let replies = legal_moves(pos);
            if judge(pos, attacker, &replies) != State::Open {
                return false;
            }
            if held_verdict(pos, tablebase) == Some(Verdict::Loss) {
                return true;
            }
            replies.into_iter().all(|reply| {
                let next = pos.play(reply);
                let legal = legal_moves(&next);
                judge(&next, attacker, &legal) == State::Open
                    && (held_verdict(&next, tablebase) == Some(Verdict::Win)
                        || (moves > 0
                            && win_within(&next, attacker, tablebase, moves, &legal).is_some()))
            })
        }
    }

    /// The format's order as shakmaty sees it: checks first, then by the
    /// standard UCI squares and promotion.
    fn shakmaty_order(pos: &Atomic) -> Vec<String> {
        let mut moves: Vec<_> = pos
            .legal_moves()
            .into_iter()
            .map(|m| {
                let mut next = pos.clone();
                next.play_unchecked(m);
                let uci = UciMove::from_standard(m);
                let UciMove::Normal {
                    from,
                    to,
                    promotion,
                } = uci
                else {
                    unreachable!()
                };
                let promotion = promotion.map_or(0, |role| role as u32);
                (
                    (!next.is_check(), from.to_u32(), to.to_u32(), promotion),
                    uci.to_string(),
                )
            })
            .collect();
        moves.sort();
        moves.into_iter().map(|(_, uci)| uci).collect()
    }
}
