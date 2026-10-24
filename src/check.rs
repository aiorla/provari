//! Checks win proofs without the prover's code: the rules come from shakmaty
//! alone and the tablebase verdicts from shakmaty-syzygy.
//!
//! This module shares no code with the rest of provari, in either direction,
//! so that a mistake in the prover's board, search or proofs cannot also hide
//! in the check. It uses nothing from the rest of the crate (a test reads
//! this file to make sure), and only its entry points are visible outside
//! it, `run` and `check_proof`, so the prover can run the checker but not
//! borrow its parts. Keep it that way: a helper both sides need is written
//! twice.
//!
//! Proofs come in the compact format of `FORMAT.md`, walked as it describes
//! (`Walker`), or in JSON lines, with a position and its move a line
//! (`Checker`). Either way, from the root the attacker, the side to move
//! unless a compact header names the other side, plays the proof's move in
//! each of its positions and every legal defence is followed. Every line must
//! end in a win: the defender's king exploded or mated, or a tablebase win
//! right after a capture or pawn move. The 50-move limit, a draw, or any other
//! position the proof has no move for makes the proof invalid, and so does any
//! line that can repeat a position: the positions the proof reaches, without
//! their 50-move counters, must never lead back to one another. JSON lines
//! leave out the positions where a capture wins at once; the compact format
//! also those it settles by a mate or a check that wins in two.
//!
//! `provari check PROOF... [--tablebases DIR] [--workers N] [--seconds S]
//! [--compact-only]` prints a JSON line for each proof, in order, and exits
//! with 1 if any is not shown valid. A valid proof's line gives the attacker
//! positions it proves, its length and the 50-move counter up to which its
//! root may go, worked out as `FORMAT.md` describes (for JSON lines, the
//! root's own). `valid` is false for a proof found wrong, and null for one it
//! cannot check: the tables it needs missing, or beyond its limits
//! (`MOST_BYTES` a file, `MOST_POSITIONS` moves written, `MOST_PLACES`
//! positions walked, `MOST_DEPTH` attacker moves deep, S seconds a proof).
//! With `--compact-only` it refuses JSON lines, which a server need not read.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use shakmaty::{
    Bitboard, Board, CastlingMode, Color, EnPassantMode, KnownOutcome, Move, Outcome, Position,
    PositionError, Square, fen::Fen, uci::UciMove, variant::Atomic,
};
use shakmaty_syzygy::{Tablebase, Wdl};

/// The stack of a checking thread, for proofs with long lines.
const STACK: usize = 64 << 20;

/// The compact format's version.
const VERSION: u64 = 1;
/// The compact header's fields, all required, and `attacker` besides where
/// the side to move is not the attacker.
const FIELDS: [&str; 6] = [
    "provari",
    "variant",
    "root",
    "tablebases",
    "moves",
    "positions",
];
/// What a proof may ask of the checker: the file's size, the header's, the
/// moves a compact proof writes, and the attacker moves along a line.
const MOST_BYTES: u64 = 1 << 30;
const MOST_HEADER: usize = 4096;
const MOST_POSITIONS: u64 = 20_000_000;
const MOST_DEPTH: usize = 10_000;
/// The positions a check may walk, some 1 GB for every 3 million.
const MOST_PLACES: usize = 20_000_000;
/// The most pieces of Atomic's tables, which JSON lines do not name.
const TABLE_PIECES: usize = 6;
/// The highest 50-move counter of a position that is not over: one at 100
/// must already be.
const LAST: u32 = 99;

/// What the command line limits a check to.
#[derive(Clone, Copy, Default)]
struct Limits {
    /// The seconds a proof's check may take.
    seconds: Option<f64>,
    /// Whether proofs in JSON lines are refused.
    compact_only: bool,
}

impl Limits {
    fn deadline(self) -> Option<Instant> {
        self.seconds
            .map(|seconds| Instant::now() + Duration::from_secs_f64(seconds))
    }
}

/// Stops a walk at its limits, looked at on its first position and every
/// 1,024th after.
fn within(visits: &mut u64, deadline: Option<Instant>, places: usize) -> Result<(), String> {
    *visits += 1;
    if *visits % 1024 != 1 {
        return Ok(());
    }
    if deadline.is_some_and(|deadline| Instant::now() > deadline) {
        return Err("cannot check: it takes longer than the time allowed".into());
    }
    if places > MOST_PLACES {
        return Err(format!("cannot check: more than {MOST_PLACES} positions"));
    }
    Ok(())
}

/// Checks every proof given, and returns whether all are valid.
pub fn run(args: &[String]) -> Result<bool, String> {
    let mut proofs = Vec::new();
    let mut directory = None;
    let mut workers = 1;
    let mut limits = Limits::default();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--tablebases" => directory = Some(PathBuf::from(value()?)),
            "--workers" => workers = value()?.parse().map_err(|_| "bad --workers")?,
            "--seconds" => limits.seconds = Some(value()?.parse().map_err(|_| "bad --seconds")?),
            "--compact-only" => limits.compact_only = true,
            unknown if unknown.starts_with("--") => {
                return Err(format!("unknown option {unknown}: see provari --help"));
            }
            _ => proofs.push(PathBuf::from(arg)),
        }
    }
    if proofs.is_empty() {
        return Err(
            "usage: provari check PROOF... [--tablebases DIR] [--workers N] \
                    [--seconds S] [--compact-only]"
                .into(),
        );
    }
    let tablebase = directory.as_deref().map(open_tables).transpose()?;
    let next = AtomicUsize::new(0);
    let results = Mutex::new(vec![None; proofs.len()]);
    std::thread::scope(|scope| {
        for _ in 0..workers.max(1) {
            std::thread::Builder::new()
                .stack_size(STACK)
                .spawn_scoped(scope, || {
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(path) = proofs.get(index) else {
                            break;
                        };
                        let result = check_file(path, tablebase.as_ref(), limits);
                        results.lock().unwrap()[index] = Some(result);
                    }
                })
                .expect("cannot start a checking thread");
        }
    });
    let mut all_valid = true;
    for (path, result) in proofs.iter().zip(results.into_inner().unwrap()) {
        let line = verdict(
            &path.display().to_string(),
            result.expect("every proof is checked"),
        );
        all_valid &= line["valid"] == true;
        println!("{line}");
    }
    Ok(all_valid)
}

/// The tables in a directory, or in several listed as `PATH` lists them,
/// which only native builds can read. A directory without any is an error.
#[cfg(any(unix, windows))]
fn open_tables(directories: &Path) -> Result<Tablebase<Atomic>, String> {
    let mut tables = Tablebase::new();
    for directory in std::env::split_paths(directories) {
        let added = tables
            .add_directory(&directory)
            .map_err(|error| format!("cannot open {}: {error}", directory.display()))?;
        if added == 0 {
            return Err(format!("no Atomic tables in {}", directory.display()));
        }
    }
    Ok(tables)
}

#[cfg(not(any(unix, windows)))]
fn open_tables(directories: &Path) -> Result<Tablebase<Atomic>, String> {
    Err(format!(
        "cannot open {}: this build reads no directories",
        directories.display()
    ))
}

/// Checks one proof from its bytes, as `run` checks a file, without its
/// limits on time: the JSON line it would print, named `name`. The browser
/// build's entry point.
pub fn check_proof(name: &str, bytes: &[u8], tablebase: Option<&Tablebase<Atomic>>) -> Value {
    verdict(name, check_bytes(bytes, tablebase, Limits::default()))
}

/// What a valid proof shows: the attacker positions it proves, its length,
/// and the 50-move counter up to which its root may go.
type Shown = (usize, Value, u32);

/// A proof's JSON line: `valid` true with what it proves, or false with
/// why, or null where it could not be checked.
fn verdict(name: &str, result: Result<Shown, String>) -> Value {
    match result {
        Ok((won, moves, counter)) => json!({
            "proof": name,
            "valid": true,
            "won_positions": won,
            "moves": moves,
            "highest_counter": counter,
        }),
        Err(error) => {
            let unchecked = error.starts_with("cannot check");
            json!({
                "proof": name,
                "valid": if unchecked { Value::Null } else { json!(false) },
                "error": error,
            })
        }
    }
}

fn check_file(
    path: &Path,
    tablebase: Option<&Tablebase<Atomic>>,
    limits: Limits,
) -> Result<Shown, String> {
    let size = fs::metadata(path)
        .map_err(|error| format!("cannot read: {error}"))?
        .len();
    if size > MOST_BYTES {
        return Err(format!("cannot check: larger than {MOST_BYTES} bytes"));
    }
    let bytes = fs::read(path).map_err(|error| format!("cannot read: {error}"))?;
    check_bytes(&bytes, tablebase, limits)
}

/// Checks a proof in either format.
fn check_bytes(
    bytes: &[u8],
    tablebase: Option<&Tablebase<Atomic>>,
    limits: Limits,
) -> Result<Shown, String> {
    let deadline = limits.deadline();
    if let Some((header, body)) = compact_header(bytes) {
        return check_compact(&header, body, tablebase, deadline);
    }
    if limits.compact_only {
        return Err("not in the compact format".into());
    }
    let text = std::str::from_utf8(bytes).map_err(|_| "neither format")?;
    check(text, tablebase, deadline)
}

/// The header of a proof in the compact format, its first line, and the rest
/// of the file; none for JSON lines, whose header has no `provari` field.
fn compact_header(bytes: &[u8]) -> Option<(Value, &[u8])> {
    let end = bytes
        .iter()
        .take(MOST_HEADER)
        .position(|&byte| byte == b'\n')?;
    let header: Value = serde_json::from_slice(&bytes[..end]).ok()?;
    header.get("provari")?;
    Some((header, &bytes[end + 1..]))
}

/// Checks a proof in the compact format: how many attacker positions it
/// proves, its length, and the counter up to which its root may go.
fn check_compact(
    header: &Value,
    body: &[u8],
    tables: Option<&Tablebase<Atomic>>,
    deadline: Option<Instant>,
) -> Result<Shown, String> {
    let fields = header.as_object().ok_or("a bad header")?;
    let named = usize::from(fields.contains_key("attacker"));
    if fields.len() != FIELDS.len() + named || !FIELDS.iter().all(|key| fields.contains_key(*key)) {
        return Err(format!(
            "the header's fields must be {}, and attacker where the side to move is not",
            FIELDS.join(", ")
        ));
    }
    let number = |key: &str| {
        fields[key]
            .as_u64()
            .ok_or(format!("a bad {key} in the header"))
    };
    if number("provari")? != VERSION {
        return Err(format!("format version {} is not known", fields["provari"]));
    }
    if fields["variant"] != "atomic" {
        return Err(format!("variant {} is not known", fields["variant"]));
    }
    let root = parse(fields["root"].as_str().ok_or("a bad root in the header")?)?;
    // The side to move attacks, unless the header names the other side.
    let attacker = match fields.get("attacker") {
        None => root.turn(),
        Some(named) => match named.as_str().map(|name| name.parse::<Color>()) {
            Some(Ok(attacker)) if attacker != root.turn() => attacker,
            Some(Ok(_)) => return Err("the header names the side to move as the attacker".into()),
            _ => return Err(format!("a bad attacker in the header: {named}")),
        },
    };
    let (pieces, claimed, positions) = (
        number("tablebases")?,
        number("moves")?,
        number("positions")?,
    );
    if positions > MOST_POSITIONS {
        return Err(format!(
            "cannot check: more than {MOST_POSITIONS} moves written"
        ));
    }
    let tablebase = match pieces {
        0 => None,
        _ => Some(
            tables
                .filter(|tables| tables.max_pieces() as u64 >= pieces)
                .ok_or(format!(
                    "cannot check: needs the tablebases up to {pieces} pieces"
                ))?,
        ),
    };
    let moves = read_moves(body, positions)?;
    let mut walker = Walker {
        attacker,
        tablebase,
        pieces: pieces as usize,
        moves: &moves,
        next: 0,
        proven: HashMap::new(),
        ids: HashMap::new(),
        places: Vec::new(),
        edges: Vec::new(),
        line: HashSet::new(),
        depth: 0,
        won: 0,
        deadline,
        visits: 0,
    };
    let (length, counter) = walker.root(&root)?;
    if walker.next != moves.len() {
        return Err(format!(
            "{} of the proof's moves are left over",
            moves.len() - walker.next
        ));
    }
    cycle(&walker.places, &walker.edges)?;
    if u64::from(length) != claimed {
        return Err(format!("the proof takes {length} moves, not {claimed}"));
    }
    Ok((walker.won, json!(length), counter))
}

/// The moves of a compact proof: one zstd frame holding exactly `positions`
/// LEB128 numbers, each in one or two bytes, as no position has 16,384 legal
/// moves. The frame is read only as far as that many moves could reach, so
/// that memory follows the data, not the header's claim.
fn read_moves(body: &[u8], positions: u64) -> Result<Vec<u32>, String> {
    let frame = zstd::zstd_safe::find_frame_compressed_size(body)
        .map_err(|_| "a damaged zstd frame".to_string())?;
    if frame != body.len() {
        return Err("data after the zstd frame".into());
    }
    let most = positions * 2;
    let mut bytes = Vec::new();
    zstd::stream::read::Decoder::with_buffer(body)
        .map_err(|error| format!("a damaged zstd frame: {error}"))?
        .single_frame()
        .take(most + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("a damaged zstd frame: {error}"))?;
    if bytes.len() as u64 > most {
        return Err(format!("more moves than the header's {positions}"));
    }
    let mut moves = Vec::with_capacity(bytes.len());
    let mut bytes = bytes.iter();
    while let Some(&first) = bytes.next() {
        let value = if first < 128 {
            u32::from(first)
        } else {
            match bytes.next() {
                Some(&second) if second != 0 && second < 128 => {
                    u32::from(first & 127) | u32::from(second) << 7
                }
                Some(_) => return Err("a move not in its shortest form, or too large".into()),
                None => return Err("the last move is cut short".into()),
            }
        };
        moves.push(value);
    }
    if moves.len() as u64 != positions {
        return Err(format!(
            "{} moves where the header says {positions}",
            moves.len()
        ));
    }
    Ok(moves)
}

/// A move's number in the format's order: the square it leaves, the square it
/// reaches (the king's, castling), its promotion.
fn number(m: Move) -> u32 {
    match UciMove::from_standard(m) {
        UciMove::Normal {
            from,
            to,
            promotion,
        } => {
            let promotion = match promotion {
                None => 0,
                Some(shakmaty::Role::Knight) => 1,
                Some(shakmaty::Role::Bishop) => 2,
                Some(shakmaty::Role::Rook) => 3,
                Some(_) => 4,
            };
            (from.to_u32() * 64 + to.to_u32()) * 5 + promotion
        }
        _ => unreachable!("no drops or null moves in Atomic"),
    }
}

/// Checks a proof: how many attacker positions it wins, its length, and the
/// root's own counter, as its moves are given by position and counter.
fn check(
    text: &str,
    tablebase: Option<&Tablebase<Atomic>>,
    deadline: Option<Instant>,
) -> Result<Shown, String> {
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header: Value = serde_json::from_str(lines.next().ok_or("an empty proof")?)
        .map_err(|error| format!("a bad header: {error}"))?;
    let root = parse(header["root"].as_str().ok_or("no root")?)?;
    let mut checker = Checker {
        attacker: root.turn(),
        tablebase,
        ids: HashMap::new(),
        places: Vec::new(),
        moves: HashMap::new(),
        won: HashSet::new(),
        edges: Vec::new(),
        deadline,
        visits: 0,
        depth: 0,
    };
    for line in lines {
        let entry: Value =
            serde_json::from_str(line).map_err(|error| format!("a bad line: {error}"))?;
        let Some(uci) = entry["move"].as_str() else {
            continue;
        };
        let pos = parse(entry["fen"].as_str().ok_or("a line without a position")?)?;
        let m: UciMove = uci.parse().map_err(|_| format!("a bad move {uci}"))?;
        let id = checker.id(&pos);
        checker.moves.insert((id, pos.halfmoves()), m);
    }
    let id = checker.id(&root);
    if !checker.ends(&root)? {
        checker.attacker_to_move(&root, id)?;
    }
    checker.check_repetitions()?;
    Ok((checker.won.len(), header["moves"].clone(), root.halfmoves()))
}

/// A position from its FEN, read as the prover reads one: as Lichess does,
/// castling rights and an en passant square the position cannot have are
/// dropped, and more material than a game can reach is let be.
fn parse(fen: &str) -> Result<Atomic, String> {
    let setup: Fen = fen
        .parse()
        .map_err(|error| format!("a bad FEN {fen}: {error}"))?;
    setup
        .into_position(CastlingMode::Standard)
        .or_else(PositionError::ignore_invalid_castling_rights)
        .or_else(PositionError::ignore_invalid_ep_square)
        .or_else(PositionError::ignore_too_much_material)
        .map_err(|error| format!("an illegal position {fen}: {error}"))
}

fn fen(pos: &Atomic) -> String {
    Fen::from_position(pos, EnPassantMode::Legal).to_string()
}

/// The side to move's legal moves, on the heap. Generated here, out of
/// line, the walks' frames do not hold shakmaty's lists of 256 moves.
#[inline(never)]
fn legal(pos: &Atomic) -> Vec<Move> {
    pos.legal_moves().into_iter().collect()
}

fn place(pos: &Atomic) -> Place {
    Place {
        board: pos.board().clone(),
        turn: pos.turn(),
        castling: pos.castles().castling_rights(),
        ep: pos.legal_ep_square(),
    }
}

/// A position without its 50-move counter, which is what repeats: the en
/// passant square counts only when taking there is legal.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Place {
    board: Board,
    turn: Color,
    castling: Bitboard,
    ep: Option<Square>,
}

struct Checker<'a> {
    attacker: Color,
    tablebase: Option<&'a Tablebase<Atomic>>,
    /// Every place met, numbered.
    ids: HashMap<Place, u32>,
    places: Vec<Place>,
    /// The proof's move in each attacker position, by place and counter.
    moves: HashMap<(u32, u32), UciMove>,
    /// The attacker positions shown won, by place and counter.
    won: HashSet<(u32, u32)>,
    /// Every move the proof allows, between places; a line can repeat a
    /// position only through a cycle here.
    edges: Vec<Vec<u32>>,
    deadline: Option<Instant>,
    visits: u64,
    /// The attacker moves of the line being walked.
    depth: usize,
}

impl Checker<'_> {
    fn id(&mut self, pos: &Atomic) -> u32 {
        let place = place(pos);
        if let Some(&id) = self.ids.get(&place) {
            return id;
        }
        let id = self.places.len() as u32;
        self.ids.insert(place.clone(), id);
        self.places.push(place);
        self.edges.push(Vec::new());
        id
    }

    /// The tablebase verdict for the side to move, where the prover trusts
    /// one: right after a capture or pawn move, without castling rights.
    #[inline(never)]
    fn verdict(&self, pos: &Atomic) -> Option<Wdl> {
        let tables = self.tablebase?;
        if pos.halfmoves() != 0 || pos.castles().any() {
            return None;
        }
        tables.probe_wdl_after_zeroing(pos).ok()
    }

    /// Why the check cannot tell whether the tables settle a position, if it
    /// cannot: one they would judge, with no more pieces than they hold,
    /// while it has no tables or theirs give no verdict.
    #[inline(never)]
    fn unjudged(&self, pos: &Atomic) -> Option<String> {
        if pos.halfmoves() != 0
            || pos.castles().any()
            || pos.board().occupied().count() > TABLE_PIECES
        {
            return None;
        }
        match self.tablebase {
            None => Some(format!(
                "cannot check: needs the tablebases for {}",
                fen(pos)
            )),
            Some(tables) => tables.probe_wdl_after_zeroing(pos).err().map(|error| {
                format!(
                    "cannot check: no tablebase verdict for {}: {error}",
                    fen(pos)
                )
            }),
        }
    }

    /// A failure at or past a position, which is the tables' to judge where
    /// they could have settled the position before it: JSON lines do not
    /// say which positions they settle, nor write anything past them.
    fn unsettled(&self, pos: &Atomic, error: String) -> String {
        if error.starts_with("cannot check") {
            return error;
        }
        self.unjudged(pos).unwrap_or(error)
    }

    // The 50-move counter grows along a line until a capture or pawn move,
    // after which no earlier position returns, so keyed with the counter the
    // proof cannot loop and every position is checked once.
    fn attacker_to_move(&mut self, pos: &Atomic, id: u32) -> Result<(), String> {
        let key = (id, pos.halfmoves());
        if self.won.contains(&key) {
            return Ok(());
        }
        within(&mut self.visits, self.deadline, self.places.len())?;
        if self.verdict(pos) == Some(Wdl::Win) {
            self.won.insert(key);
            return Ok(());
        }
        let Some(&uci) = self.moves.get(&key) else {
            if self.wins_by_capture(pos) {
                self.won.insert(key);
                return Ok(());
            }
            let error = format!("the proof has no move for {}", fen(pos));
            return Err(self.unsettled(pos, error));
        };
        let m = uci
            .to_move(pos)
            .map_err(|_| self.unsettled(pos, format!("{uci} is not legal in {}", fen(pos))))?;
        let mut next = pos.clone();
        next.play_unchecked(m);
        self.depth += 1;
        if self.depth > MOST_DEPTH {
            return Err(format!(
                "cannot check: a line longer than {MOST_DEPTH} attacker moves"
            ));
        }
        self.defender_to_move(&next, id)
            .map_err(|error| self.unsettled(pos, error))?;
        self.depth -= 1;
        self.won.insert(key);
        Ok(())
    }

    fn defender_to_move(&mut self, pos: &Atomic, before: u32) -> Result<(), String> {
        if self.ends(pos)? || self.verdict(pos) == Some(Wdl::Loss) {
            return Ok(());
        }
        let id = self.id(pos);
        self.edges[before as usize].push(id);
        for reply in legal(pos) {
            let mut next = pos.clone();
            next.play_unchecked(reply);
            if !self.ends(&next)? {
                let next_id = self.id(&next);
                self.edges[id as usize].push(next_id);
                self.attacker_to_move(&next, next_id)
                    .map_err(|error| self.unsettled(pos, error))?;
            }
        }
        Ok(())
    }

    /// Whether a capture wins at once, as one exploding the king does.
    #[inline(never)]
    fn wins_by_capture(&self, pos: &Atomic) -> bool {
        pos.legal_moves().iter().any(|m| {
            let mut next = pos.clone();
            next.play_unchecked(*m);
            m.is_capture()
                && next.outcome()
                    == Outcome::Known(KnownOutcome::Decisive {
                        winner: self.attacker,
                    })
        })
    }

    /// Whether the game is over, which must be with the attacker winning.
    #[inline(never)]
    fn ends(&self, pos: &Atomic) -> Result<bool, String> {
        match pos.outcome() {
            Outcome::Known(KnownOutcome::Decisive { winner }) if winner == self.attacker => {
                Ok(true)
            }
            Outcome::Known(_) => Err(format!(
                "the game ends without the attacker winning: {}",
                fen(pos)
            )),
            Outcome::Unknown if pos.halfmoves() >= 100 => {
                Err(format!("the 50-move limit is reached: {}", fen(pos)))
            }
            Outcome::Unknown => Ok(false),
        }
    }

    /// Rejects a cycle among the places the proof reaches.
    fn check_repetitions(&self) -> Result<(), String> {
        cycle(&self.places, &self.edges)
    }
}

/// Rejects a cycle among places linked by the moves a proof allows.
fn cycle(places: &[Place], edges: &[Vec<u32>]) -> Result<(), String> {
    // Unvisited, on the walk, or done.
    let mut state = vec![0u8; places.len()];
    for start in 0..places.len() {
        if state[start] != 0 {
            continue;
        }
        state[start] = 1;
        let mut walk = vec![(start, 0)];
        while let Some((place, next)) = walk.last_mut() {
            let Some(&successor) = edges[*place].get(*next) else {
                state[*place] = 2;
                walk.pop();
                continue;
            };
            *next += 1;
            let successor = successor as usize;
            match state[successor] {
                0 => {
                    state[successor] = 1;
                    walk.push((successor, 0));
                }
                1 => {
                    let place = &places[successor];
                    return Err(format!(
                        "a line can repeat the position {} {}",
                        place.board,
                        if place.turn == Color::White { 'w' } else { 'b' },
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// How a game stands for the attacker.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Won,
    /// Over otherwise, or at the 50-move limit.
    Failed,
    Open,
}

/// The walk of a compact proof (`FORMAT.md`): moves read where it
/// needs one, the defences generated, positions it can settle itself
/// settled, and each position's length worked out, with the counter up to
/// which its proof holds.
struct Walker<'a> {
    attacker: Color,
    /// The tables, where the header allows them, and the most pieces it does.
    tablebase: Option<&'a Tablebase<Atomic>>,
    pieces: usize,
    moves: &'a [u32],
    next: usize,
    /// Each place proven, at the highest counter yet, with its length then
    /// and the counter up to which that proof holds.
    proven: HashMap<u32, (u32, u32, u32)>,
    /// Every place met, numbered, and the links of the proof's moves.
    ids: HashMap<Place, u32>,
    places: Vec<Place>,
    edges: Vec<Vec<u32>>,
    /// The places of the line being walked, and its attacker moves.
    line: HashSet<u32>,
    depth: usize,
    /// The attacker positions proven.
    won: usize,
    deadline: Option<Instant>,
    visits: u64,
}

impl Walker<'_> {
    fn id(&mut self, pos: &Atomic) -> u32 {
        let place = place(pos);
        if let Some(&id) = self.ids.get(&place) {
            return id;
        }
        let id = self.places.len() as u32;
        self.ids.insert(place.clone(), id);
        self.places.push(place);
        self.edges.push(Vec::new());
        id
    }

    /// How the game stands, as shakmaty's `outcome` has it but from the side
    /// to move's legal moves, generated once, and the 50-move limit.
    fn judge(&self, pos: &Atomic, legal: &[Move]) -> State {
        match pos.variant_outcome() {
            Outcome::Known(KnownOutcome::Decisive { winner }) => {
                return if winner == self.attacker {
                    State::Won
                } else {
                    State::Failed
                };
            }
            Outcome::Known(_) => return State::Failed,
            Outcome::Unknown => {}
        }
        if legal.is_empty() {
            return if pos.is_check() && pos.turn() != self.attacker {
                State::Won
            } else {
                State::Failed
            };
        }
        if pos.is_insufficient_material() || pos.halfmoves() >= 100 {
            return State::Failed;
        }
        State::Open
    }

    /// Whether the attacker's move just played won at once: the defender's
    /// king exploded, which only a capture does, or the defender mated,
    /// which only a check does.
    #[inline(never)]
    fn won_at_once(next: &Atomic, capture: bool, check: bool) -> bool {
        (capture && next.is_variant_end()) || (check && next.legal_moves().is_empty())
    }

    /// The tablebase verdict for the side to move, where the proof allows
    /// one: at most the header's pieces, the counter at 0, no castling
    /// rights. A verdict the tables cannot give stops the check.
    #[inline(never)]
    fn verdict(&self, pos: &Atomic) -> Result<Option<Wdl>, String> {
        let Some(tables) = self.tablebase else {
            return Ok(None);
        };
        if pos.halfmoves() != 0
            || pos.castles().any()
            || pos.board().occupied().count() > self.pieces
        {
            return Ok(None);
        }
        tables
            .probe_wdl_after_zeroing(pos)
            .map(Some)
            .map_err(|error| {
                format!(
                    "cannot check: no tablebase verdict for {}: {error}",
                    fen(pos)
                )
            })
    }

    /// The root's length, and the counter up to which it may go with the
    /// proof still holding.
    fn root(&mut self, root: &Atomic) -> Result<(u32, u32), String> {
        let legal = legal(root);
        match self.judge(root, &legal) {
            State::Won => Ok((0, LAST)),
            State::Failed => Err(format!(
                "the root is over or at the 50-move limit: {}",
                fen(root)
            )),
            State::Open => {
                let id = self.id(root);
                if root.turn() == self.attacker {
                    self.attacker_to_move(root, id, legal)
                } else {
                    self.defender_to_move(root, id)
                }
            }
        }
    }

    /// Proves an attacker position, or finds it proven at a counter at
    /// least its own, and returns its length and the counter up to which
    /// its proof holds.
    fn attacker_to_move(
        &mut self,
        pos: &Atomic,
        id: u32,
        legal: Vec<Move>,
    ) -> Result<(u32, u32), String> {
        let counter = pos.halfmoves();
        if let Some(&(highest, length, holds)) = self.proven.get(&id)
            && counter <= highest
        {
            return Ok((length, holds));
        }
        within(&mut self.visits, self.deadline, self.places.len())?;
        if !self.line.insert(id) {
            return Err(format!("a line repeats {}", fen(pos)));
        }
        self.depth += 1;
        if self.depth > MOST_DEPTH {
            return Err(format!(
                "cannot check: a line longer than {MOST_DEPTH} attacker moves"
            ));
        }
        let (length, holds) = self.prove(pos, id, legal)?;
        self.depth -= 1;
        self.line.remove(&id);
        self.won += 1;
        let proven = self.proven.entry(id).or_insert((counter, length, holds));
        if counter > proven.0 {
            *proven = (counter, length, holds);
        }
        Ok((length, holds))
    }

    fn prove(&mut self, pos: &Atomic, id: u32, legal: Vec<Move>) -> Result<(u32, u32), String> {
        // The tables' verdicts count only at a counter of 0.
        if self.verdict(pos)? == Some(Wdl::Win) {
            return Ok((0, 0));
        }
        let mut moves: Vec<(Atomic, u32, bool)> = Vec::with_capacity(legal.len());
        for m in legal {
            let mut next = pos.clone();
            next.play_unchecked(m);
            let check = next.is_check();
            if Self::won_at_once(&next, m.is_capture(), check) {
                return Ok((1, LAST));
            }
            moves.push((next, number(m), check));
        }
        moves.sort_by_key(|&(_, number, check)| (!check, number));
        for (next, _, check) in &moves {
            if !check {
                break;
            }
            if let Some((length, holds)) = self.wins_in_two(next)? {
                return Ok((length, through(next, holds)));
            }
        }
        let index = *self
            .moves
            .get(self.next)
            .ok_or("the proof's moves run out")? as usize;
        self.next += 1;
        let Some((next, ..)) = moves.get(index) else {
            return Err(format!("move {index} of {} in {}", moves.len(), fen(pos)));
        };
        let next = next.clone();
        let next_id = self.id(&next);
        self.edges[id as usize].push(next_id);
        let (length, holds) = self.defender_to_move(&next, next_id)?;
        Ok((1 + length, through(&next, holds)))
    }

    /// Whether a check leaves the defender lost at once whatever it does:
    /// lost by the tablebases, or every defence letting a capture win at
    /// once or reaching a tablebase win; and the length then, with the
    /// counter up to which that holds.
    #[inline(never)]
    fn wins_in_two(&self, pos: &Atomic) -> Result<Option<(u32, u32)>, String> {
        let replies = pos.legal_moves();
        if self.judge(pos, &replies) != State::Open {
            return Ok(None);
        }
        if self.verdict(pos)? == Some(Wdl::Loss) {
            return Ok(Some((1, 0)));
        }
        let (mut longest, mut holds) = (0, LAST);
        for reply in replies {
            let mut next = pos.clone();
            next.play_unchecked(reply);
            let legal = next.legal_moves();
            if self.judge(&next, &legal) != State::Open {
                return Ok(None);
            }
            if self.verdict(&next)? == Some(Wdl::Win) {
                continue;
            }
            if !Self::capture_wins(&next, &legal) {
                return Ok(None);
            }
            longest = 1;
            holds = holds.min(through(&next, LAST));
        }
        Ok(Some((1 + longest, holds)))
    }

    fn capture_wins(pos: &Atomic, legal: &[Move]) -> bool {
        legal.iter().any(|m| {
            m.is_capture() && {
                let mut next = pos.clone();
                next.play_unchecked(*m);
                let check = next.is_check();
                Self::won_at_once(&next, true, check)
            }
        })
    }

    /// Follows every defence, in square order, and returns the longest,
    /// with the counter up to which every defence still loses.
    fn defender_to_move(&mut self, pos: &Atomic, id: u32) -> Result<(u32, u32), String> {
        let replies = legal(pos);
        match self.judge(pos, &replies) {
            State::Won => return Ok((0, LAST)),
            State::Failed => {
                return Err(format!(
                    "the line ends without the attacker winning: {}",
                    fen(pos)
                ));
            }
            State::Open => {}
        }
        if self.verdict(pos)? == Some(Wdl::Loss) {
            return Ok((0, 0));
        }
        if !self.line.insert(id) {
            return Err(format!("a line repeats {}", fen(pos)));
        }
        let mut replies = replies;
        replies.sort_by_key(|&m| number(m));
        let (mut longest, mut holds) = (0, LAST);
        for reply in replies {
            let mut next = pos.clone();
            next.play_unchecked(reply);
            let legal = legal(&next);
            if self.judge(&next, &legal) != State::Open {
                return Err(format!(
                    "the defence {} ends the line without the attacker winning: {}",
                    UciMove::from_standard(reply),
                    fen(pos)
                ));
            }
            let next_id = self.id(&next);
            self.edges[id as usize].push(next_id);
            let (length, after) = self.attacker_to_move(&next, next_id, legal)?;
            longest = longest.max(length);
            holds = holds.min(through(&next, after));
        }
        self.line.remove(&id);
        Ok((longest, holds))
    }
}

/// The counter up to which a position's proof holds through its move to
/// `next`, whose own proof holds up to `holds`: `LAST` after a capture or
/// pawn move, which resets the counter, and otherwise one below `holds`, as
/// `next`'s counter is one above.
fn through(next: &Atomic, holds: u32) -> u32 {
    if next.halfmoves() == 0 {
        LAST
    } else {
        holds.saturating_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::{Checker, Limits, check, check_bytes, parse};
    use serde_json::{Value, json};
    use shakmaty::variant::Atomic;
    use shakmaty_syzygy::Tablebase;
    use std::collections::{HashMap, HashSet};

    /// `format::SMALL` in the compact format, as the prover writes it.
    fn compact() -> Vec<u8> {
        crate::format::encode_jsonl(crate::format::SMALL, None)
            .unwrap()
            .1
    }

    fn compact_verdict(bytes: &[u8]) -> Result<(usize, u64), String> {
        check_bytes(bytes, None, Limits::default())
            .map(|(won, length, _)| (won, length.as_u64().unwrap()))
    }

    /// A compact proof's header and moves, to tamper with, and back.
    fn parts(bytes: &[u8]) -> (Value, Vec<u8>) {
        let end = bytes.iter().position(|&byte| byte == b'\n').unwrap();
        let moves = zstd::bulk::decompress(&bytes[end + 1..], 1 << 20).unwrap();
        (serde_json::from_slice(&bytes[..end]).unwrap(), moves)
    }

    fn assemble(header: &Value, moves: &[u8]) -> Vec<u8> {
        let mut bytes = header.to_string().into_bytes();
        bytes.push(b'\n');
        bytes.extend(zstd::bulk::compress(moves, 3).unwrap());
        bytes
    }

    #[test]
    fn accepts_a_compact_proof() {
        assert_eq!(compact_verdict(&compact()).map(|(_, length)| length), Ok(4));
        // The same proof in JSON lines.
        assert_eq!(
            check(crate::format::SMALL, None, None).map(|(_, length, _)| length),
            Ok(json!(4))
        );
    }

    /// A win for the side not to move: the header names the attacker, and
    /// the walk starts with the defences.
    #[test]
    fn accepts_a_proof_for_the_side_not_to_move() {
        let (_, bytes) = crate::format::small_defended();
        let (won, _) = compact_verdict(&compact()).unwrap();
        assert_eq!(compact_verdict(&bytes), Ok((won - 1, 3)));
        // The check of `settles_a_check_that_wins_in_two` played: both king
        // moves let the bishop explode the king, so nothing is written.
        let root = "rnbq1b1r/pp1pk2p/2pBppp1/8/3PP3/2N5/PPP2P1P/R2QKB1R b KQ - 1 1";
        let header = json!({
            "provari": 1, "variant": "atomic", "root": root, "attacker": "white",
            "tablebases": 0, "moves": 1, "positions": 0,
        });
        assert_eq!(compact_verdict(&assemble(&header, &[])), Ok((2, 1)));
        let (header, moves) = parts(&bytes);
        for (attacker, start) in [
            (
                json!("black"),
                "the header names the side to move as the attacker",
            ),
            (json!("red"), "a bad attacker in the header"),
            (json!(1), "a bad attacker in the header"),
        ] {
            let mut changed = header.clone();
            changed["attacker"] = attacker;
            let error = compact_verdict(&assemble(&changed, &moves)).unwrap_err();
            assert!(error.starts_with(start), "{error}");
        }
        // Without it, the side to move attacks, and the moves prove nothing.
        let mut unnamed = header.clone();
        unnamed.as_object_mut().unwrap().remove("attacker");
        assert!(compact_verdict(&assemble(&unnamed, &moves)).is_err());
    }

    /// A compact proof with its root's 50-move counter set to `counter`.
    fn at_counter(bytes: &[u8], counter: u32) -> Vec<u8> {
        let (mut header, moves) = parts(bytes);
        let root = header["root"].as_str().unwrap().to_owned();
        let mut fields: Vec<String> = root.split(' ').map(str::to_owned).collect();
        fields[4] = counter.to_string();
        header["root"] = json!(fields.join(" "));
        assemble(&header, &moves)
    }

    /// The counter a proof reports for its root is, for these proofs, the
    /// highest the root may have: the same proof checks there, as it does at
    /// its own counter, and not one above. JSON lines give their moves by
    /// position and counter, so only the root's own counter.
    #[test]
    fn reports_the_highest_counter_its_root_may_have() {
        let shown = |bytes: &[u8]| check_bytes(bytes, None, Limits::default());
        let root = "rnbq1b1r/pp1pk2p/2p1ppp1/8/3PPB2/2N5/PPP2P1P/R2QKB1R w KQ - 0 1";
        let header = json!({
            "provari": 1, "variant": "atomic", "root": root,
            "tablebases": 0, "moves": 2, "positions": 0,
        });
        // The furthest any line of each goes before its first capture or pawn
        // move, to a position that must not be over: 6 plies from SMALL's
        // root, at 0, 5 from the position after its first move, at 1, and 2
        // from the check that wins in two.
        for (bytes, highest) in [
            (compact(), 93),
            (crate::format::small_defended().1, 94),
            (assemble(&header, &[]), 97),
        ] {
            let (won, length, counter) = shown(&bytes).unwrap();
            assert_eq!(counter, highest);
            assert_eq!(
                shown(&at_counter(&bytes, highest)),
                Ok((won, length, highest))
            );
            let error = shown(&at_counter(&bytes, highest + 1)).unwrap_err();
            assert!(!error.starts_with("cannot check"), "{error}");
        }
        for clock in [0, 97] {
            let text = proof(clock).join("\n");
            assert_eq!(
                check(&text, None, None).map(|(.., counter)| counter),
                Ok(clock)
            );
        }
    }

    #[test]
    fn settles_a_check_that_wins_in_two() {
        let root = "rnbq1b1r/pp1pk2p/2p1ppp1/8/3PPB2/2N5/PPP2P1P/R2QKB1R w KQ - 0 1";
        let header = json!({
            "provari": 1, "variant": "atomic", "root": root,
            "tablebases": 0, "moves": 2, "positions": 0,
        });
        assert_eq!(compact_verdict(&assemble(&header, &[])), Ok((1, 2)));
    }

    #[test]
    fn rejects_damaged_compact_proofs() {
        let good = compact();
        let (header, moves) = parts(&good);
        assert_eq!(moves.len(), 6);
        let rejected = |bytes: &[u8], start: &str| {
            let error = compact_verdict(bytes).unwrap_err();
            assert!(error.starts_with(start), "{error}");
        };
        rejected(&good[..good.len() - 1], "a damaged zstd frame");
        let mut longer = good.clone();
        longer.push(0);
        rejected(&longer, "data after the zstd frame");
        // A bit flipped in zstd's checksum, which the prover writes.
        let mut flipped = good.clone();
        let last = flipped.len() - 2;
        flipped[last] ^= 1;
        rejected(&flipped, "a damaged zstd frame");
        for (key, value, start) in [
            ("positions", json!(5), "6 moves where the header says 5"),
            ("positions", json!(7), "6 moves where the header says 7"),
            ("positions", json!(200_000_000), "cannot check: more than"),
            ("moves", json!(5), "the proof takes 4 moves, not 5"),
            ("provari", json!(2), "format version 2"),
            ("variant", json!("chess"), "variant"),
            (
                "root",
                json!("8/8/8/8/8/8/8/8 w - - 0 1"),
                "an illegal position",
            ),
            ("tablebases", json!(5), "cannot check"),
        ] {
            let mut changed = header.clone();
            changed[key] = value;
            rejected(&assemble(&changed, &moves), start);
        }
        let mut extra = header.clone();
        extra["note"] = json!(1);
        rejected(&assemble(&extra, &moves), "the header's fields");
        let mut missing = header.clone();
        missing.as_object_mut().unwrap().remove("moves");
        rejected(&assemble(&missing, &moves), "the header's fields");
        let mut far = moves.clone();
        far[0] = 120;
        rejected(&assemble(&header, &far), "move 120 of");
        let mut long = vec![moves[0] | 128, 0];
        long.extend(&moves[1..]);
        rejected(&assemble(&header, &long), "a move not in its shortest form");
        let mut more = moves.clone();
        more.push(0);
        let mut counted = header.clone();
        counted["positions"] = json!(7);
        rejected(
            &assemble(&counted, &more),
            "1 of the proof's moves are left over",
        );
    }

    #[test]
    fn keeps_to_its_limits() {
        let at_once = Limits {
            seconds: Some(0.0),
            compact_only: false,
        };
        let error = check_bytes(&compact(), None, at_once).unwrap_err();
        assert!(
            error.starts_with("cannot check: it takes longer"),
            "{error}"
        );
        let compact_only = Limits {
            seconds: None,
            compact_only: true,
        };
        let error = check_bytes(crate::format::SMALL.as_bytes(), None, compact_only).unwrap_err();
        assert_eq!(error, "not in the compact format");
        // A small frame that would fill far more than its header's moves
        // can: the checker reads only as far as they could reach.
        let (mut header, _) = parts(&compact());
        header["positions"] = json!(1_000_000);
        let bomb = assemble(&header, &vec![0; 10_000_000]);
        assert!(bomb.len() < 2_000);
        let error = compact_verdict(&bomb).unwrap_err();
        assert!(error.starts_with("more moves than the header's"), "{error}");
    }

    /// Damaged versions of real proofs: the checker answers each without
    /// panicking, and accepts one only where the prover's own reader, which
    /// walks the format with other code, walks it too; where only the reader
    /// does, the checker must have found what only it looks for.
    fn fuzz(
        proofs: &[Vec<u8>],
        tries: usize,
        tables: Option<&Tablebase<Atomic>>,
        reader_tables: Option<&crate::tablebase::Tablebase>,
    ) -> (usize, usize) {
        let mut seed = 0x853c_49e6_748f_ea9b_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as usize
        };
        let only_the_checker = [
            "a line repeats",
            "a line can repeat",
            "cannot check",
            "data after the zstd frame",
        ];
        let mut accepted = 0;
        for done in 0..tries {
            if tries >= 2_000 && done % 2_000 == 0 {
                eprintln!("{done} of {tries}, {accepted} accepted");
            }
            let base = &proofs[next() % proofs.len()];
            let (header, moves) = parts(base);
            let counted = |moves: &[u8]| {
                let mut changed = header.clone();
                changed["positions"] = json!(moves.len());
                assemble(&changed, moves)
            };
            let bytes = match next() % 9 {
                0 => {
                    let mut bytes = base.clone();
                    let at = next() % bytes.len();
                    bytes[at] ^= 1 << (next() % 8);
                    bytes
                }
                1 => base[..next() % base.len()].to_vec(),
                2 if !moves.is_empty() => {
                    let mut changed = moves.clone();
                    let at = next() % changed.len();
                    changed[at] = (next() % 48) as u8;
                    assemble(&header, &changed)
                }
                3 if !moves.is_empty() => {
                    let mut fewer = moves.clone();
                    fewer.remove(next() % moves.len());
                    counted(&fewer)
                }
                4 => {
                    let mut more = moves.clone();
                    more.insert(next() % (moves.len() + 1), (next() % 8) as u8);
                    counted(&more)
                }
                5 => {
                    let mut changed = header.clone();
                    let claimed = changed["moves"].as_u64().unwrap();
                    changed["moves"] = json!(claimed.saturating_sub(1));
                    assemble(&changed, &moves)
                }
                6 if moves.len() > 1 => {
                    let mut swapped = moves.clone();
                    swapped.swap(next() % moves.len(), next() % moves.len());
                    assemble(&header, &swapped)
                }
                7 => {
                    let mut changed = header.clone();
                    match next() % 3 {
                        0 => changed["attacker"] = json!("white"),
                        1 => changed["attacker"] = json!("black"),
                        _ => {
                            changed.as_object_mut().unwrap().remove("attacker");
                        }
                    }
                    assemble(&changed, &moves)
                }
                _ => {
                    let mut changed = header.clone();
                    let root = changed["root"].as_str().unwrap();
                    let root = if root.contains(" w ") {
                        root.replacen(" w ", " b ", 1)
                    } else {
                        root.replacen(" b ", " w ", 1)
                    };
                    changed["root"] = json!(root);
                    assemble(&changed, &moves)
                }
            };
            let limits = Limits {
                seconds: Some(10.0),
                compact_only: true,
            };
            // Each walk in a thread with the stack it has in use.
            let (checked, read) = std::thread::scope(|scope| {
                let checking = std::thread::Builder::new()
                    .stack_size(super::STACK)
                    .spawn_scoped(scope, || check_bytes(&bytes, tables, limits))
                    .unwrap();
                let reading = std::thread::Builder::new()
                    .stack_size(crate::STACK)
                    .spawn_scoped(scope, || match crate::format::decode(&bytes) {
                        Ok(Some((header, moves))) => {
                            crate::format::strategy(&header, &moves, reader_tables).is_ok()
                        }
                        _ => false,
                    })
                    .unwrap();
                (checking.join().unwrap(), reading.join().unwrap())
            });
            match (&checked, read) {
                (Ok(_), true) => accepted += 1,
                (Ok(_), false) => panic!("the checker accepts what the reader cannot walk"),
                (Err(error), true) => assert!(
                    only_the_checker
                        .iter()
                        .any(|start| error.starts_with(start)),
                    "only the reader accepts: {error}"
                ),
                (Err(_), false) => {}
            }
        }
        (tries, accepted)
    }

    #[test]
    fn answers_damaged_proofs_as_the_reader_does() {
        let proofs = [compact(), crate::format::small_defended().1];
        let (tries, accepted) = fuzz(&proofs, 400, None, None);
        // Some changed moves still prove the win, if otherwise.
        assert!(
            accepted > 0 && accepted < tries / 2,
            "{accepted} of {tries}"
        );
    }

    /// The same on stored proofs of up to 20 KB, at length, from the
    /// directories of `.pvr` proofs in `PROVARI_PROOFS` (a list as `PATH` is)
    /// with the tablebases they need in `PROVARI_TABLEBASES` (another):
    /// `cargo test --release -- --ignored answers_damaged_stored_proofs`.
    #[test]
    #[ignore = "minutes long"]
    fn answers_damaged_stored_proofs() {
        let sets = std::env::var_os("PROVARI_PROOFS").expect("PROVARI_PROOFS");
        let tables = std::path::PathBuf::from(
            std::env::var_os("PROVARI_TABLEBASES").expect("PROVARI_TABLEBASES"),
        );
        let checker_tables = super::open_tables(&tables).unwrap();
        let reader_tables = crate::tablebase::Tablebase::open(&tables).unwrap();
        let mut proofs = Vec::new();
        for set in std::env::split_paths(&sets) {
            for entry in std::fs::read_dir(set).unwrap() {
                let path = entry.unwrap().path();
                if path.extension().is_some_and(|e| e == "pvr")
                    && std::fs::metadata(&path).unwrap().len() <= 20_000
                {
                    proofs.push(std::fs::read(&path).unwrap());
                }
            }
        }
        let (tries, accepted) = fuzz(&proofs, 20_000, Some(&checker_tables), Some(&reader_tables));
        eprintln!(
            "{} proofs, {tries} damaged versions, {accepted} accepted",
            proofs.len()
        );
    }

    /// A check after which both king moves let the bishop explode the king.
    fn proof(clock: u32) -> Vec<String> {
        let after = clock + 2;
        vec![
            format!(
                r#"{{"moves":2,"root":"rnbq1b1r/pp1pk2p/2p1ppp1/8/3PPB2/2N5/PPP2P1P/R2QKB1R w KQ - {clock} 1"}}"#
            ),
            format!(
                r#"{{"fen":"rnbq1b1r/pp1pk2p/2p1ppp1/8/3PPB2/2N5/PPP2P1P/R2QKB1R w KQ - {clock} 1","move":"f4d6"}}"#
            ),
            format!(
                r#"{{"fen":"rnbqkb1r/pp1p3p/2pBppp1/8/3PP3/2N5/PPP2P1P/R2QKB1R w KQ - {after} 2","move":"d6f8"}}"#
            ),
            format!(
                r#"{{"fen":"rnbq1b1r/pp1p1k1p/2pBppp1/8/3PP3/2N5/PPP2P1P/R2QKB1R w KQ - {after} 2","move":"d6f8"}}"#
            ),
        ]
    }

    fn verdict(lines: &[String]) -> Result<usize, String> {
        check(&lines.join("\n"), None, None).map(|(won, ..)| won)
    }

    #[test]
    fn accepts_a_win() {
        assert_eq!(verdict(&proof(0)), Ok(3));
        // The defences reach counter 99, where the game still goes on.
        assert_eq!(verdict(&proof(97)), Ok(3));
    }

    #[test]
    fn finds_the_explosions_left_out() {
        assert_eq!(verdict(&proof(0)[..2]), Ok(3));
        assert_eq!(verdict(&proof(97)[..3]), Ok(3));
    }

    #[test]
    fn rejects_a_missing_move() {
        let lines = [proof(0)[0].clone(), proof(0)[2].clone()];
        let error = verdict(&lines).unwrap_err();
        assert!(error.starts_with("the proof has no move for"), "{error}");
    }

    #[test]
    fn rejects_an_illegal_move() {
        let mut lines = proof(0);
        lines[1] = lines[1].replace("f4d6", "f4d7");
        let error = verdict(&lines).unwrap_err();
        assert!(error.starts_with("f4d7 is not legal"), "{error}");
    }

    #[test]
    fn cannot_check_what_the_tables_settle_without_them() {
        // A win the tables give at the root, and a loss they give after a
        // pawn move, below which JSON lines write nothing.
        let won = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1";
        let before = "4k3/8/8/8/8/8/P7/3QK3 w - - 5 1";
        let lost = "4k3/8/8/8/8/P7/8/3QK3 b - - 0 1";
        for (lines, settled) in [
            (
                [
                    json!({"root": won, "moves": 0}).to_string(),
                    json!({"fen": won, "tablebase": "win"}).to_string(),
                ],
                won,
            ),
            (
                [
                    json!({"root": before, "moves": 1}).to_string(),
                    json!({"fen": before, "move": "a2a3"}).to_string(),
                ],
                lost,
            ),
        ] {
            assert_eq!(
                verdict(&lines),
                Err(format!("cannot check: needs the tablebases for {settled}"))
            );
            let error = check(&lines.join("\n"), Some(&Tablebase::new()), None).unwrap_err();
            let start = format!("cannot check: no tablebase verdict for {settled}: ");
            assert!(error.starts_with(&start), "{error}");
        }
    }

    #[test]
    fn rejects_the_50_move_limit() {
        for lines in [&proof(98)[..], &proof(98)[..2]] {
            let error = verdict(lines).unwrap_err();
            assert!(error.starts_with("the 50-move limit is reached"), "{error}");
        }
    }

    #[test]
    fn checks_a_root_draw_before_a_zeroing_capture() {
        for clock in [99, 100] {
            let root =
                format!("rnbqkb1r/pp1p3p/2pBppp1/8/3PP3/2N5/PPP2P1P/R2QKB1R w KQ - {clock} 2");
            let lines = [
                serde_json::json!({"root":root, "moves":1}).to_string(),
                serde_json::json!({"fen":root, "move":"d6f8"}).to_string(),
            ];
            // With the explosion written out, and left for the checker.
            for lines in [&lines[..], &lines[..1]] {
                let result = verdict(lines);
                if clock == 99 {
                    assert_eq!(result, Ok(1));
                } else {
                    assert!(
                        result
                            .unwrap_err()
                            .starts_with("the 50-move limit is reached")
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_a_line_that_can_repeat() {
        let fens = [
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/3K4 b - - 1 1",
            "3k4/8/8/8/8/8/8/3K4 w - - 2 2",
        ];
        let mut checker = Checker {
            attacker: shakmaty::Color::White,
            tablebase: None,
            ids: HashMap::new(),
            places: Vec::new(),
            moves: HashMap::new(),
            won: HashSet::new(),
            edges: Vec::new(),
            deadline: None,
            visits: 0,
            depth: 0,
        };
        let ids: Vec<u32> = fens
            .iter()
            .map(|fen| checker.id(&parse(fen).unwrap()))
            .collect();
        checker.edges[ids[0] as usize].push(ids[1]);
        checker.edges[ids[1] as usize].push(ids[2]);
        assert!(checker.check_repetitions().is_ok());
        checker.edges[ids[2] as usize].push(ids[0]);
        let error = checker.check_repetitions().unwrap_err();
        assert!(error.starts_with("a line can repeat"), "{error}");
    }

    /// Tables from several directories, listed as `PATH` lists them, as the
    /// search reads them; a listed directory without any is an error that
    /// names it.
    #[test]
    fn opens_tables_from_several_directories() {
        let base = std::env::temp_dir().join(format!("provari-check-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let [small, large, empty] = ["small", "large", "empty"].map(|name| base.join(name));
        for directory in [&small, &large, &empty] {
            std::fs::create_dir_all(directory).unwrap();
        }
        // Named as tables, of a size tables have: they open only when probed.
        std::fs::write(small.join("KQvK.atbw"), [0; 16]).unwrap();
        std::fs::write(large.join("KQRvKQR.atbw"), [0; 16]).unwrap();
        let open = |directories: &[&std::path::PathBuf]| {
            super::open_tables(std::path::Path::new(
                &std::env::join_paths(directories).unwrap(),
            ))
        };
        assert_eq!(open(&[&small]).unwrap().max_pieces(), 3);
        assert_eq!(open(&[&small, &large]).unwrap().max_pieces(), 6);
        let error = open(&[&small, &empty]).err().unwrap();
        assert_eq!(error, format!("no Atomic tables in {}", empty.display()));
        std::fs::remove_dir_all(base).unwrap();
    }

    /// A misspelled option is refused, rather than checked as a proof.
    #[test]
    fn refuses_unknown_options() {
        let args = ["proof.pvr", "--tablebase", "tables"].map(String::from);
        assert_eq!(
            super::run(&args),
            Err("unknown option --tablebase: see provari --help".to_owned())
        );
    }

    /// The module's code, before its tests, names no path into the rest of
    /// the crate.
    #[test]
    fn uses_nothing_from_the_prover() {
        let source = include_str!("check.rs");
        let code = source.split("#[cfg(test)]").next().unwrap();
        for path in ["crate::", "super::"] {
            assert!(!code.contains(path), "the checker uses {path}");
        }
    }
}
