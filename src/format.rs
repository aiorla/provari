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
use crate::moves;
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
    pos.pseudo_legal()
        .as_slice()
        .iter()
        .copied()
        .filter(|&m| pos.legal(m))
        .collect()
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

struct Walk<'a, C> {
    attacker: Color,
    tablebase: Option<&'a Tablebase>,
    pieces: usize,
    chooser: C,
    /// Each position proven, by hash, at the highest counter yet, with its
    /// length then.
    proven: HashMap<u64, (u32, u32)>,
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
        if let Some(&(highest, length)) = self.proven.get(&hash)
            && counter <= highest
        {
            return Ok(length);
        }
        if !self.line.insert(hash) {
            return Err(format!("a line repeats {}", fen(pos)));
        }
        let (length, m) = self.prove(pos, legal)?;
        self.line.remove(&hash);
        if self
            .proven
            .get(&hash)
            .is_none_or(|&(highest, _)| counter > highest)
        {
            self.proven.insert(hash, (counter, length));
            if let Some(strategy) = &mut self.strategy {
                strategy.insert(hash, m.map(|m| pos.to_shakmaty(m)));
            }
        }
        Ok(length)
    }

    /// Proves an attacker position: its length, and its move, none for a
    /// tablebase win.
    fn prove(&mut self, pos: &Position, legal: Vec<Move>) -> Result<(u32, Option<Move>), String> {
        if self.verdict(pos)? == Some(Verdict::Win) {
            return Ok((0, None));
        }
        let mut moves: Vec<(Move, Position, bool)> = Vec::with_capacity(legal.len());
        for m in legal {
            let next = pos.play(m);
            let check = next.in_check();
            if won_at_once(&next, self.attacker, is_capture(pos, m), check) {
                return Ok((1, Some(m)));
            }
            moves.push((m, next, check));
        }
        moves.sort_by_key(|(m, _, check)| (!*check, number(pos, *m)));
        for (m, next, _) in moves.iter().take_while(|(_, _, check)| *check) {
            let decided = self.decided.get();
            if let Some(length) = self.wins_in_two(next)? {
                return Ok((length, Some(*m)));
            }
            // A check that does not win in two would not with fewer tables
            // either, so the verdicts met in its lines are not the proof's.
            self.decided.set(decided);
        }
        let ordered: Vec<Move> = moves.iter().map(|(m, ..)| *m).collect();
        let index = self.chooser.choose(pos, &ordered)?;
        let (m, next, _) = &moves[index];
        let replies = legal_moves(next);
        Ok((1 + self.defender_to_move(next, replies)?, Some(*m)))
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
        replies.sort_by_key(|&m| number(pos, m));
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
        strategy: None,
        decided: Cell::new(0),
        line: HashSet::new(),
    };
    let length = walk.root(&board)?;
    let indices = walk.chooser.indices;
    // Tables the proof never needed are not asked of its checkers.
    let header = Header {
        root: Fen::from_position(root, EnPassantMode::Legal).to_string(),
        attacker: (attacker != board.turn()).then_some(attacker),
        tablebases: walk.decided.get(),
        moves: length,
        positions: indices.len() as u64,
    };
    Ok((header, indices))
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
        SMALL, State, decode, encode_jsonl, is_capture, legal_moves, number, small_defended, state,
        strategy, won_at_once,
    };
    use crate::board::{Position, from_fen};
    use crate::search::hash;

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
