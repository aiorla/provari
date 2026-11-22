//! Antichess's compact proof format (`.apr`): a proof written from the
//! search's proofs and tolius's tables, and read back into the White
//! positions it proves, walking each as the format lays out and rejecting one
//! that fails. A file is a header line of JSON, then White's moves in a zstd
//! frame, each its index among the legal moves in square order (`ordered`).
//! The header holds the format's version, under `format`, and the proof's
//! `variant`, `root`, `moves` and `positions`; proofs written before
//! 2026-11-21 keep the version under another name, which is read the same.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};
use shakmaty::{
    CastlingMode, EnPassantMode, FromSetup, Move, MoveList, Position, variant::Antichess,
};

use crate::antichess::rules::{self, LAST_COUNTER, Status};
use crate::antichess::search::{Dfpn, MOST_NEED};
use crate::antichess::tablebases::{Entry, MOST_PIECES, Tablebases};

/// The format's version.
const VERSION: u64 = 1;
/// The zstd level proofs are written at, as provari writes its own.
const LEVEL: i32 = 19;
/// The file extension of a proof in this format.
pub const EXTENSION: &str = "apr";

/// A proof's header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    /// The root, at the highest 50-move counter the proof holds at.
    pub root: String,
    /// The proof's length: the most White moves any line takes.
    pub moves: u32,
    /// The moves written.
    pub positions: u64,
}

/// A proof in this format.
pub struct Written {
    pub header: Header,
    /// The White positions its walk proves, forced ones included.
    pub walked: u64,
    /// The file's bytes.
    pub bytes: Vec<u8>,
}

/// A White position a proof proves: the position without the move counters,
/// White's move in UCI, its length then in White moves, and the 50-move
/// counter it was proven at, which it holds at and below.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub fen: String,
    pub m: String,
    pub length: u32,
    pub counter: u32,
}

/// A White position a proof proves, as `Step` gives it, in the few bytes a
/// table of millions keeps: its 128-bit key and White's packed move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Proven {
    pub key: u128,
    pub m: rules::Packed,
    pub length: u32,
    pub counter: u32,
}

/// A move's number in square order: the square it leaves, the square it
/// reaches, its promotion (none, knight, bishop, rook, queen, king).
fn number(m: Move) -> u32 {
    let from = m.from().map_or(0, |square| square as u32);
    let promotion = m.promotion().map_or(0, |role| role as u32 - 1);
    (from * 64 + m.to() as u32) * 6 + promotion
}

/// Legal moves in the format's order, square order.
pub fn ordered(legal: &MoveList) -> Vec<Move> {
    let mut moves: Vec<Move> = legal.iter().copied().collect();
    moves.sort_unstable_by_key(|&m| number(m));
    moves
}

/// Where a walk's moves come from: the search's or old proofs when writing,
/// the file's when reading. `choose` returns White's move in a position with
/// several legal moves, which come in the format's order, and `budget` White
/// moves left.
trait Chooser {
    fn choose(
        &mut self,
        pos: &Antichess,
        key: u64,
        budget: u32,
        ordered: &[Move],
    ) -> Result<usize, String>;
}

struct Walk<C> {
    chooser: C,
    /// Each position proven, by key: the counters and lengths it was proven
    /// at, none at a counter as high with a length as short as another's.
    proven: HashMap<u64, Vec<(u32, u32)>>,
    /// The positions of the line being walked.
    line: HashSet<u64>,
    walked: u64,
    /// The White positions walked, if kept, as steps or in short.
    steps: Option<Vec<Step>>,
    short: Option<Vec<Proven>>,
}

impl<C: Chooser> Walk<C> {
    fn new(chooser: C, steps: bool) -> Walk<C> {
        Walk {
            chooser,
            proven: HashMap::new(),
            line: HashSet::new(),
            walked: 0,
            steps: steps.then(Vec::new),
            short: None,
        }
    }

    /// A White position's length, within `budget` White moves.
    fn white(&mut self, pos: &Antichess, budget: u32) -> Result<u32, String> {
        let legal = pos.legal_moves();
        match rules::status(pos, &legal) {
            Status::WhiteWon => return Ok(0),
            Status::Open if budget > 0 => {}
            Status::Open => {
                return Err(format!(
                    "{}: its line has no White move left",
                    rules::fen(pos)
                ));
            }
            other => return Err(format!("{}: {other:?} on a line", rules::fen(pos))),
        }
        let key = rules::key(pos);
        let counter = pos.halfmoves();
        if let Some(length) = self.known(key, counter, budget) {
            return Ok(length);
        }
        if !self.line.insert(key) {
            return Err(format!("{}: a line repeats it", rules::fen(pos)));
        }
        let m = if legal.len() == 1 {
            legal[0]
        } else {
            let moves = ordered(&legal);
            moves[self.chooser.choose(pos, key, budget, &moves)?]
        };
        // Kept in the order the walk meets them, the root first.
        let place = self.steps.as_mut().map(|steps| {
            steps.push(Step {
                fen: rules::short_fen(pos),
                m: rules::uci(m),
                length: 0,
                counter,
            });
            steps.len() - 1
        });
        let short = self.short.as_mut().map(|short| {
            short.push(Proven {
                key: rules::key128(pos),
                m: rules::Packed::new(m),
                length: 0,
                counter,
            });
            short.len() - 1
        });
        let mut after = pos.clone();
        after.play_unchecked(m);
        let length = 1 + self.black(&after, budget)?;
        self.line.remove(&key);
        self.walked += 1;
        if let (Some(steps), Some(place)) = (&mut self.steps, place) {
            steps[place].length = length;
        }
        if let (Some(short), Some(place)) = (&mut self.short, short) {
            short[place].length = length;
        }
        let known = self.proven.entry(key).or_default();
        known.retain(|&(at, shortest)| at > counter || shortest < length);
        known.push((counter, length));
        Ok(length)
    }

    /// The least length a position was proven with at a counter at least
    /// `counter`, if within `budget`.
    fn known(&self, key: u64, counter: u32, budget: u32) -> Option<u32> {
        self.proven
            .get(&key)?
            .iter()
            .filter(|&&(at, length)| at >= counter && length <= budget)
            .map(|&(_, length)| length)
            .min()
    }

    /// The longest length of a White position after a Black reply.
    fn black(&mut self, pos: &Antichess, budget: u32) -> Result<u32, String> {
        let replies = pos.legal_moves();
        let status = rules::status(pos, &replies);
        if status != Status::Open {
            return Err(format!(
                "{}: {status:?} after White's move",
                rules::fen(pos)
            ));
        }
        let key = rules::key(pos);
        if !self.line.insert(key) {
            return Err(format!("{}: a line repeats it", rules::fen(pos)));
        }
        let mut longest = 0;
        for reply in ordered(&replies) {
            let mut next = pos.clone();
            next.play_unchecked(reply);
            longest = longest.max(self.white(&next, budget - 1)?);
        }
        self.line.remove(&key);
        Ok(longest)
    }
}

/// White's moves from a source, written as their indices.
struct Writing<F> {
    source: F,
    indices: Vec<u32>,
}

impl<F: FnMut(&Antichess, u64, u32, &[Move]) -> Option<Move>> Chooser for Writing<F> {
    fn choose(
        &mut self,
        pos: &Antichess,
        key: u64,
        budget: u32,
        ordered: &[Move],
    ) -> Result<usize, String> {
        let m = (self.source)(pos, key, budget, ordered)
            .ok_or_else(|| format!("{}: no proof holds at its counter", rules::fen(pos)))?;
        let index = ordered
            .iter()
            .position(|&legal| legal == m)
            .ok_or_else(|| format!("{}: the proof's move is not legal", rules::fen(pos)))?;
        self.indices.push(index as u32);
        Ok(index)
    }
}

/// A file's moves, read in order.
struct Reading<'a> {
    indices: &'a [u32],
    next: usize,
}

impl Chooser for Reading<'_> {
    fn choose(
        &mut self,
        pos: &Antichess,
        _: u64,
        _: u32,
        ordered: &[Move],
    ) -> Result<usize, String> {
        let index = *self
            .indices
            .get(self.next)
            .ok_or("the proof's moves run out")? as usize;
        self.next += 1;
        if index >= ordered.len() {
            return Err(format!(
                "{}: move {index} of {}",
                rules::fen(pos),
                ordered.len()
            ));
        }
        Ok(index)
    }
}

/// Writes a proof of `root` within `most` White moves, at `root`'s counter,
/// with the move `source` gives in each White position with several, from its
/// key, the White moves left and the legal moves in order.
fn write(
    root: &Antichess,
    most: u32,
    source: impl FnMut(&Antichess, u64, u32, &[Move]) -> Option<Move>,
) -> Result<Written, String> {
    let mut walk = Walk::new(
        Writing {
            source,
            indices: Vec::new(),
        },
        false,
    );
    // The root's length, within the bound it is written within.
    let moves = walk.white(root, most)?;
    let header = Header {
        root: rules::fen(root),
        moves,
        positions: walk.chooser.indices.len() as u64,
    };
    let bytes = file(&header, &walk.chooser.indices)?;
    Ok(Written {
        header,
        walked: walk.walked,
        bytes,
    })
}

/// The proof that White, to move in `root`, wins within `most` moves, as the
/// search left it in its proofs and with its tables: from the highest
/// counter the root's proof holds at, the tables' quickest win where they
/// give one (`quickest`), otherwise each position's shortest proof at its
/// counter.
pub fn from_search(search: &Dfpn, root: &Antichess, most: u32) -> Result<Written, String> {
    let tables = search.tablebases();
    let counter = root.halfmoves();
    let (top, need) = match search.proof(rules::key(root), counter, most) {
        Some(proof) if proof.limit > counter && proof.limit <= LAST_COUNTER => {
            (at_counter(root, proof.limit)?, proof.length)
        }
        Some(proof) => (root.clone(), proof.length),
        // A root already won needs no proof, and the search keeps none.
        None => (root.clone(), 0),
    };
    let mut answers = HashMap::new();
    write(&top, need, |pos, key, budget, ordered| {
        if let Some(tables) = tables
            && let Some(m) = quickest(tables, &mut answers, pos, budget, ordered)
        {
            return Some(m);
        }
        let m = search.proof(key, pos.halfmoves(), MOST_NEED)?.m?;
        ordered
            .iter()
            .copied()
            .find(|&legal| crate::antichess::packed(legal) == m)
    })
}

/// White's quickest win by the tables, in a position they hold: of the moves
/// after which Black loses within `budget` White moves, before the 50-move
/// rule could draw, the one after which Black loses soonest, the first in
/// square order of those. Where the search settled a position from the
/// tables, its quickest win is one of them, and the tables' depth falls along
/// every line of it, so that each position after it is one they settle too.
fn quickest(
    tables: Tablebases,
    answers: &mut HashMap<u64, Option<Entry>>,
    pos: &Antichess,
    budget: u32,
    ordered: &[Move],
) -> Option<Move> {
    if pos.board().occupied().count() > MOST_PIECES {
        return None;
    }
    let mut best: Option<(u32, Move)> = None;
    for &m in ordered {
        let mut after = pos.clone();
        after.play_unchecked(m);
        if rules::status_counted(&after, rules::count_moves(&after)) != Status::Open {
            continue;
        }
        let entry = *answers
            .entry(rules::key(&after))
            .or_insert_with(|| tables.probe(&after));
        if let Some(Entry::Loss(plies)) = entry
            && plies / 2 < budget
            && after.halfmoves() + plies <= LAST_COUNTER + 1
            && best.is_none_or(|(soonest, _)| plies < soonest)
        {
            best = Some((plies, m));
        }
    }
    best.map(|(_, m)| m)
}

/// The same position at another 50-move counter.
fn at_counter(pos: &Antichess, counter: u32) -> Result<Antichess, String> {
    let mut setup = pos.to_setup(EnPassantMode::Legal);
    setup.halfmoves = counter;
    Antichess::from_setup(setup, CastlingMode::Standard)
        .map_err(|error| format!("{}: {error}", rules::fen(pos)))
}

/// A proof's file: its header line, then its moves in a zstd frame.
fn file(header: &Header, indices: &[u32]) -> Result<Vec<u8>, String> {
    let mut body = Vec::with_capacity(indices.len());
    for &index in indices {
        let mut value = index;
        while value >= 128 {
            body.push((value & 127) as u8 | 128);
            value >>= 7;
        }
        body.push(value as u8);
    }
    // With zstd's checksum, so that a damaged file shows as damaged.
    let body = zstd::bulk::Compressor::new(LEVEL)
        .and_then(|mut compressor| {
            compressor.include_checksum(true)?;
            compressor.compress(&body)
        })
        .map_err(|error| format!("cannot compress: {error}"))?;
    let mut bytes = format!(
        r#"{{"format":{VERSION},"variant":"antichess","root":{},"moves":{},"positions":{}}}"#,
        json!(header.root),
        header.moves,
        header.positions
    )
    .into_bytes();
    bytes.push(b'\n');
    bytes.extend(body);
    Ok(bytes)
}

/// Writes a proof's file through a temporary one, so that a proof there is
/// always whole.
pub fn save(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let failed = |error: std::io::Error| format!("cannot write {}: {error}", path.display());
    let partial = path.with_extension("partial");
    let mut out = File::create(&partial).map_err(failed)?;
    out.write_all(bytes).map_err(failed)?;
    out.sync_all().map_err(failed)?;
    fs::rename(&partial, path).map_err(failed)
}

/// Writes a file of a directory that several builds may write at once,
/// through a temporary file of this process's and this call's own, so that no
/// other's bytes take its name.
pub fn save_shared(path: &Path, bytes: &[u8]) -> Result<(), String> {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let failed = |error: std::io::Error| format!("cannot write {}: {error}", path.display());
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(
        ".{}.{}.partial",
        std::process::id(),
        CALLS.fetch_add(1, Ordering::Relaxed)
    ));
    let partial = path.with_file_name(name);
    let written = File::create(&partial)
        .and_then(|mut out| out.write_all(bytes).and_then(|()| out.sync_all()))
        .and_then(|()| fs::rename(&partial, path));
    if written.is_err() {
        let _ = fs::remove_file(&partial);
    }
    written.map_err(failed)
}

/// A proof's header from the start of its file, and where its header line
/// ends.
fn header(bytes: &[u8]) -> Result<(Header, usize), String> {
    let end = bytes
        .iter()
        .take(4096)
        .position(|&byte| byte == b'\n')
        .ok_or("not a proof: no header line")?;
    let head: Value =
        serde_json::from_slice(&bytes[..end]).map_err(|error| format!("not a proof: {error}"))?;
    if head.as_object().map(serde_json::Map::len) != Some(5) {
        return Err(format!(
            "not a proof: a header without exactly its five fields: {head}"
        ));
    }
    // The version's key is `format`, or another name in older proofs: the
    // one key that is none of the others.
    const FIELDS: [&str; 4] = ["variant", "root", "moves", "positions"];
    let version = head
        .as_object()
        .and_then(|fields| {
            fields
                .iter()
                .find(|(key, _)| !FIELDS.contains(&key.as_str()))
        })
        .map(|(_, value)| value);
    if version != Some(&json!(VERSION)) || head["variant"] != "antichess" {
        return Err(format!("not a proof this version reads: {head}"));
    }
    let number = |key: &str| head[key].as_u64().ok_or(format!("a bad {key}"));
    let header = Header {
        root: head["root"].as_str().ok_or("a bad root")?.to_owned(),
        moves: u32::try_from(number("moves")?).map_err(|_| "a bad moves")?,
        positions: number("positions")?,
    };
    Ok((header, end))
}

/// A proof file's header, read from its first line alone.
pub fn read_header(path: &Path) -> Result<Header, String> {
    let mut start = Vec::with_capacity(4096);
    File::open(path)
        .and_then(|file| file.take(4096).read_to_end(&mut start))
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    header(&start).map(|(header, _)| header)
}

/// A proof's header and moves from the bytes of its file.
pub fn decode(bytes: &[u8]) -> Result<(Header, Vec<u32>), String> {
    let (header, end) = self::header(bytes)?;
    let positions = usize::try_from(header.positions).map_err(|_| "a bad positions")?;
    // Read as a stream, so that memory follows the data rather than the
    // header: a move takes at most five bytes, and anything longer is not a
    // proof.
    let damaged = |error: std::io::Error| format!("a damaged proof: {error}");
    let mut frame = zstd::stream::read::Decoder::with_buffer(&bytes[end + 1..])
        .map_err(damaged)?
        .single_frame();
    let mut body = Vec::new();
    (&mut frame)
        .take(header.positions.saturating_mul(5).saturating_add(1))
        .read_to_end(&mut body)
        .map_err(damaged)?;
    if !frame.finish().is_empty() {
        return Err("a damaged proof: not one whole zstd frame".into());
    }
    let mut indices = Vec::with_capacity(positions.min(body.len()));
    let mut value = 0u64;
    let mut shift = 0;
    for byte in body {
        if byte == 0 && shift > 0 {
            return Err("a damaged proof: a move not in its shortest form".into());
        }
        value |= u64::from(byte & 127) << shift;
        if byte < 128 {
            indices.push(u32::try_from(value).map_err(|_| "a damaged proof: a move too large")?);
            (value, shift) = (0, 0);
        } else {
            shift += 7;
            if shift > 28 {
                return Err("a damaged proof: a move too large".into());
            }
        }
    }
    if shift > 0 || indices.len() != positions {
        return Err("a damaged proof: its moves do not match its header".into());
    }
    Ok((header, indices))
}

/// Walks a proof back into the White positions it proves, each with its move,
/// length and counter.
pub fn steps(header: &Header, indices: &[u32]) -> Result<Vec<Step>, String> {
    let mut walk = Walk::new(Reading { indices, next: 0 }, true);
    read(&mut walk, header, indices)?;
    Ok(walk.steps.unwrap_or_default())
}

/// `steps` in short.
pub fn proven(header: &Header, indices: &[u32]) -> Result<Vec<Proven>, String> {
    let mut walk = Walk::new(Reading { indices, next: 0 }, false);
    walk.short = Some(Vec::new());
    read(&mut walk, header, indices)?;
    Ok(walk.short.unwrap_or_default())
}

fn read(walk: &mut Walk<Reading>, header: &Header, indices: &[u32]) -> Result<(), String> {
    let root = rules::parse_fen(&header.root)?;
    if root.turn() != shakmaty::Color::White {
        return Err(format!("{}: the root is not White to move", header.root));
    }
    let length = walk.white(&root, header.moves)?;
    if walk.chooser.next != indices.len() || length != header.moves {
        return Err("the proof does not match its header".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use shakmaty::Position;

    use super::{Header, Step, Written, decode, file, ordered, save_shared, steps, write};
    use crate::antichess::rules;

    #[test]
    fn files_of_a_shared_directory_are_written_whole_whoever_else_writes() {
        // A chunk file and its cuts differ only in their extension, and two
        // builds write the same names: each write must leave its own bytes.
        let dir =
            std::env::temp_dir().join(format!("provari-antichess-shared-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let writers: Vec<_> = (0..8u8)
            .map(|writer| {
                let path = dir.join(["file.chunks", "file.cuts"][usize::from(writer % 2)]);
                std::thread::spawn(move || {
                    let bytes = vec![writer % 2; 1 << 16];
                    for _ in 0..50 {
                        save_shared(&path, &bytes).unwrap();
                        assert_eq!(std::fs::read(&path).unwrap(), bytes);
                    }
                })
            })
            .collect();
        for writer in writers {
            writer.join().unwrap();
        }
        // Nothing temporary stays, and a write that fails leaves none either.
        let names = |dir: &std::path::Path| {
            let mut names: Vec<_> = std::fs::read_dir(dir)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().into_string().unwrap())
                .collect();
            names.sort();
            names
        };
        assert_eq!(names(&dir), ["file.chunks", "file.cuts"]);
        std::fs::create_dir(dir.join("taken")).unwrap();
        assert!(save_shared(&dir.join("taken"), b"bytes").is_err());
        assert_eq!(names(&dir), ["file.chunks", "file.cuts", "taken"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    const TWO: &str = "7r/8/8/8/8/8/8/RN6 w - - 0 1";

    /// White gives the rook away, then the knight: `Ra8 Rxa8 Na3 Rxa3`.
    fn two() -> Written {
        let moves = [
            ("7r/8/8/8/8/8/8/RN6 w - -", "a1a8"),
            ("r7/8/8/8/8/8/8/1N6 w - -", "b1a3"),
        ];
        let root = rules::parse_fen(TWO).unwrap();
        write(&root, 2, |pos, _, _, ordered| {
            let fen = rules::short_fen(pos);
            let (_, m) = moves.iter().find(|(at, _)| *at == fen)?;
            let m = rules::parse_uci(pos, m).ok()?;
            ordered.contains(&m).then_some(m)
        })
        .unwrap()
    }

    /// A proof's bytes with `from` replaced by `to` in its header line.
    fn edited(bytes: &[u8], from: &str, to: &str) -> Vec<u8> {
        let end = bytes.iter().position(|&byte| byte == b'\n').unwrap();
        let line = std::str::from_utf8(&bytes[..end]).unwrap();
        assert!(line.contains(from));
        [line.replacen(from, to, 1).as_bytes(), &bytes[end..]].concat()
    }

    fn header(root: &str, moves: u32, positions: u64) -> Header {
        Header {
            root: root.into(),
            moves,
            positions,
        }
    }

    fn step(fen: &str, m: &str, length: u32, counter: u32) -> Step {
        Step {
            fen: fen.into(),
            m: m.into(),
            length,
            counter,
        }
    }

    #[test]
    fn writes_and_reads_back() {
        let written = two();
        assert_eq!(written.header, header(TWO, 2, 2));
        assert_eq!(written.walked, 2);
        let line = format!(
            r#"{{"format":1,"variant":"antichess","root":"{TWO}","moves":2,"positions":2}}"#
        );
        assert!(written.bytes.starts_with(format!("{line}\n").as_bytes()));
        let (read, indices) = decode(&written.bytes).unwrap();
        assert_eq!(read, written.header);
        // Ra8 is the seventh of White's ten moves, Na3 the second of three.
        assert_eq!(indices, [6, 1]);
        assert_eq!(
            steps(&read, &indices).unwrap(),
            [
                step("7r/8/8/8/8/8/8/RN6 w - -", "a1a8", 2, 0),
                step("r7/8/8/8/8/8/8/1N6 w - -", "b1a3", 1, 0),
            ]
        );
    }

    #[test]
    fn moves_go_in_square_order() {
        let pos = rules::parse_fen("7r/8/8/8/8/8/8/RN6 w - - 0 1").unwrap();
        let order: Vec<String> = ordered(&pos.legal_moves())
            .into_iter()
            .map(rules::uci)
            .collect();
        let expected = [
            "a1a2", "a1a3", "a1a4", "a1a5", "a1a6", "a1a7", "a1a8", "b1d2", "b1a3", "b1c3",
        ];
        assert_eq!(order, expected);
    }

    #[test]
    fn damaged_or_wrong_proofs_are_refused() {
        let written = two();
        let bytes = &written.bytes;
        assert!(decode(&bytes[..bytes.len() - 1]).is_err());
        let mut flipped = bytes.clone();
        *flipped.last_mut().unwrap() ^= 1;
        assert!(decode(&flipped).is_err());
        assert!(decode(&edited(bytes, "\"format\":1", "\"format\":2")).is_err());
        // An older proof's version, under another name, reads the same.
        let older = edited(bytes, "\"format\":1", "\"writer\":1");
        assert_eq!(decode(&older).unwrap(), decode(bytes).unwrap());
        assert!(decode(&edited(bytes, "\"format\":1", "\"writer\":2")).is_err());
        let extra = edited(bytes, "\"moves\"", "\"extra\":0,\"moves\"");
        assert!(decode(&extra).unwrap_err().contains("five fields"));
        // A header claiming more moves than memory holds is refused, not a crash.
        for positions in [10u64.pow(15), u64::MAX] {
            let positions = format!("\"positions\":{positions}");
            let huge = edited(bytes, "\"positions\":2", &positions);
            assert!(decode(&huge).unwrap_err().contains("do not match"));
        }
        // Anything after the frame, even another one, is refused.
        let frame = bytes.iter().position(|&byte| byte == b'\n').unwrap() + 1;
        let twice = [bytes.as_slice(), &bytes[frame..]].concat();
        assert!(decode(&twice).unwrap_err().contains("one whole"));

        let check = |moves: u32, indices: &[u32]| {
            let header = header(TWO, moves, indices.len() as u64);
            let (header, indices) = decode(&file(&header, indices).unwrap()).unwrap();
            steps(&header, &indices).map(|_| ())
        };
        assert_eq!(check(2, &[6, 1]), Ok(()));
        let error = |moves, indices: &[u32]| check(moves, indices).unwrap_err();
        assert!(error(2, &[6, 3]).contains("move 3 of 3"));
        assert!(error(2, &[6]).contains("run out"));
        assert!(error(2, &[6, 1, 0]).contains("does not match"));
        // Longer than its length, or shorter: Na3 is not won within one move.
        assert!(error(3, &[6, 1]).contains("does not match"));
        assert!(error(1, &[6, 1]).contains("no White move left"));
        // Ra2 leaves Black quiet moves, after which no White move wins at once.
        assert!(error(2, &[0, 0]).contains("no White move left"));
    }

    /// Black's first reply moves the rook between a1 and b1, so a knight
    /// that goes to g8 and back repeats the root.
    #[test]
    fn a_line_may_not_repeat_a_position() {
        let root = header("8/8/7N/8/8/8/8/r7 w - - 0 1", 5, 2);
        let (root, indices) = decode(&file(&root, &[3, 1]).unwrap()).unwrap();
        assert!(
            steps(&root, &indices)
                .unwrap_err()
                .contains("a line repeats")
        );
    }
}
