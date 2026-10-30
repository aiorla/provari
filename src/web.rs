//! The browser build's entry points, for WebAssembly: plain exported
//! functions that a page's worker calls, with bytes passed through this
//! module's memory (`alloc` for the page to write into; replies packed as
//! the pointer in the low 32 bits and the length in the high ones, for the
//! page to read and `release`), JSON requests and replies, and the page's
//! clock and handlers imported (`provari.now_ms`; `provari.progress` for a
//! search's progress lines, `provari.found` for each shorter win while it
//! shortens).
//!
//! A worker loads what it needs once: the network, the model and the
//! tables to prove positions; the tables alone to show positions, open
//! proofs and check them. A search's other threads each run in a worker of
//! their own on this module's memory, which the page's worker starts before
//! the search spawns them (`web/worker.js`). That worker is busy until the
//! search ends, so a page that browses while it proves runs two, and stops a
//! search through that worker's clock: the search ends, as at its deadline,
//! once the flag at `stop_flag` is set, which the worker sets as the page
//! asks it to when the search looks at the clock. A proof's bytes stay here
//! after `prove` until `take_proof` fetches them.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use serde_json::{Value, json};
use shakmaty::variant::Atomic;

use shakmaty::uci::UciMove;
use shakmaty::{Color, EnPassantMode, Position as _, fen::Fen};

use crate::clock::Instant;
use crate::dfpn::{self, Dfpn, Guide};
use crate::model::Model;
use crate::nnue::Network;
use crate::tablebase::{Memory, Tablebase, Verdict};
use crate::{check, format, lines, position};

#[link(wasm_import_module = "provari")]
unsafe extern "C" {
    /// Hands the page a progress line, UTF-8 JSON of `len` bytes at `ptr`.
    fn progress(ptr: *const u8, len: usize);
    /// Hands the page a shorter win found while shortening: its proof in the
    /// compact format, `len` bytes at `ptr`.
    fn found(ptr: *const u8, len: usize);
}

/// What a worker loads once, the last proof written, the proof open for
/// browsing with its root, and the last PGN tree written.
#[derive(Default)]
struct State {
    network: Option<Network>,
    model: Option<Model>,
    files: Memory,
    tables: Option<Tablebase>,
    checker_tables: Option<shakmaty_syzygy::Tablebase<Atomic>>,
    proof: Vec<u8>,
    open: Option<(Atomic, lines::Proof)>,
    pgn: Vec<u8>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// Stops the running search, as its deadline does.
static STOP: AtomicBool = AtomicBool::new(false);

/// Where the stop flag is in this module's memory: a byte that the page's
/// worker sets to 1 to stop the running search, and to 0 for the next.
#[unsafe(no_mangle)]
pub extern "C" fn stop_flag() -> *const AtomicBool {
    &STOP
}

/// Memory for the page to write `len` bytes into, which the call it is
/// passed to takes over.
#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    Box::into_raw(vec![0u8; len].into_boxed_slice()).cast()
}

/// Frees a reply once the page has read it.
///
/// # Safety
///
/// `ptr` and `len` must be a reply this module gave, released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn release(ptr: *mut u8, len: usize) {
    // SAFETY: the caller passes back a boxed slice of this length.
    drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
}

/// Takes over what the page wrote where `alloc` said.
///
/// # Safety
///
/// `ptr` and `len` must come from `alloc`, passed once.
unsafe fn input(ptr: *mut u8, len: usize) -> Vec<u8> {
    // SAFETY: as the caller promises.
    unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) }.into_vec()
}

/// Hands bytes to the page: the pointer in the low 32 bits, the length in the
/// high ones.
fn reply(bytes: Vec<u8>) -> u64 {
    let bytes = bytes.into_boxed_slice();
    let len = bytes.len() as u64;
    let ptr = Box::into_raw(bytes).cast::<u8>() as usize as u64;
    ptr | len << 32
}

/// A JSON reply: the value, or `{"error": ...}`.
fn answer(result: Result<Value, String>) -> u64 {
    let value = result.unwrap_or_else(|error| json!({ "error": error }));
    reply(value.to_string().into_bytes())
}

fn text(bytes: Vec<u8>) -> Result<String, String> {
    String::from_utf8(bytes).map_err(|_| "not UTF-8".to_string())
}

/// Loads the network from its file's bytes.
///
/// # Safety
///
/// As `input`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn load_network(ptr: *mut u8, len: usize) -> u64 {
    // SAFETY: as the caller promises.
    let bytes = unsafe { input(ptr, len) };
    answer(Network::from_bytes(&bytes).map(|network| {
        STATE.with_borrow_mut(|state| state.network = Some(network));
        json!({ "ok": true })
    }))
}

/// Loads the provability model from its file's text.
///
/// # Safety
///
/// As `input`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn load_model(ptr: *mut u8, len: usize) -> u64 {
    // SAFETY: as the caller promises.
    let bytes = unsafe { input(ptr, len) };
    answer(
        text(bytes)
            .and_then(|text| Model::from_text(&text))
            .map(|model| {
                STATE.with_borrow_mut(|state| state.model = Some(model));
                json!({ "ok": true })
            }),
    )
}

/// Adds a tablebase file, named as on disk; `open_tables` opens them all.
///
/// # Safety
///
/// As `input`, for both the name and the bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn add_table(
    name_ptr: *mut u8,
    name_len: usize,
    ptr: *mut u8,
    len: usize,
) -> u64 {
    // SAFETY: as the caller promises.
    let (name, bytes) = unsafe { (input(name_ptr, name_len), input(ptr, len)) };
    answer(text(name).map(|name| {
        STATE.with_borrow_mut(|state| state.files.add(&name, bytes));
        json!({ "ok": true })
    }))
}

/// Opens the tables added, for the search and, apart, for the checker.
#[unsafe(no_mangle)]
pub extern "C" fn open_tables() -> u64 {
    answer(STATE.with_borrow_mut(|state| {
        let files = Arc::new(std::mem::take(&mut state.files));
        let tables = Tablebase::from_memory(files.clone()).map_err(|e| e.to_string())?;
        let pieces = tables.max_pieces();
        let mut checker_tables = shakmaty_syzygy::Tablebase::with_filesystem(files);
        checker_tables
            .add_directory(Memory::ROOT)
            .map_err(|e| e.to_string())?;
        state.tables = Some(tables);
        state.checker_tables = Some(checker_tables);
        Ok(json!({ "pieces": pieces }))
    }))
}

/// Proves a position: `{"fen", "attacker": "white" or "black", "seconds",
/// "nodes", "memory_mb", "most": attacker moves, "progress": seconds,
/// "shorten": true, "threads"}`, all but the FEN optional (the attacker by
/// default the side to move, one thread by default), guided by the strongest
/// guide loaded. With `shorten`, after a win it looks for shorter ones within
/// the same budget, handing the page the proof of each win shorter as written
/// (`provari.found`), the first win's included, so that a page can show it at
/// once and keeps the best if it stops the search. The search stops at its
/// deadline, its budget or the stop flag (`stop_flag`). Hands the page its
/// progress lines as `provari prove --json --progress` prints them, and
/// replies as `provari prove --json` prints, the proof kept for
/// `take_proof`, or with `{"error"}` alone for a request it refuses, such as
/// one for more memory than the browser gives.
///
/// # Safety
///
/// As `input`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn prove(ptr: *mut u8, len: usize) -> u64 {
    // SAFETY: as the caller promises.
    let bytes = unsafe { input(ptr, len) };
    answer(
        text(bytes)
            .and_then(|text| serde_json::from_str(&text).map_err(|e| e.to_string()))
            .and_then(|request: Value| STATE.with_borrow_mut(|state| search(state, &request))),
    )
}

fn search(state: &mut State, request: &Value) -> Result<Value, String> {
    let root = position(request["fen"].as_str().ok_or("no fen")?)?;
    // A game already over has nothing to prove: the reply says how it ended.
    if let Some((winner, reason)) = crate::ending(&root) {
        return Ok(json!({
            "found": false,
            "over": reason,
            "winner": winner.map(side),
            "nodes": 0,
            "ms": 0,
        }));
    }
    let attacker = match &request["attacker"] {
        Value::Null => root.turn(),
        named => named
            .as_str()
            .and_then(|name| name.parse().ok())
            .ok_or("bad attacker")?,
    };
    let nodes = request["nodes"].as_u64();
    let seconds = request["seconds"]
        .as_f64()
        .unwrap_or(if nodes.is_some() { 3.2e7 } else { 10.0 });
    let memory = usize::try_from(request["memory_mb"].as_u64().unwrap_or(256))
        .map_err(|_| "bad memory_mb")?;
    let most = match request["most"].as_u64() {
        Some(most) => Some(u32::try_from(most).map_err(|_| "bad most")?),
        None => None,
    };
    let shorten = request["shorten"].as_bool().unwrap_or(false);
    let threads = match request["threads"].as_u64() {
        Some(threads) => usize::try_from(threads)
            .ok()
            .filter(|&threads| threads >= 1)
            .ok_or("bad threads")?,
        None => 1,
    };
    let guide = match (&state.model, &state.network) {
        (Some(model), network) => Guide::Model(model, network.as_ref()),
        (None, Some(network)) => Guide::Network(network),
        (None, None) => Guide::Hand,
    };
    let tables = state.tables.as_ref();
    let started = Instant::now();
    let deadline = started + Duration::from_secs_f64(seconds);
    // As `provari prove --threads`: threads search as many times the nodes in
    // the time, and share the memory.
    let needed = dfpn::proofs_needed(Some(seconds * threads as f64), nodes);
    let split = dfpn::split(memory, needed);
    let mut search = Dfpn::try_with_threads(&root, tables, guide, deadline, split, threads)
        .map_err(|_| format!("This browser cannot give a search {memory} MB."))?;
    search.set_attacker(attacker);
    search.set_stop(&STOP);
    if let Some(nodes) = nodes {
        search.set_limit(nodes);
    }
    if let Some(most) = most {
        search.set_most(most);
    }
    if let Some(every) = request["progress"].as_f64() {
        search.set_progress_sink(
            Duration::from_secs_f64(every),
            Box::new(|line| {
                let line = line.to_string();
                // SAFETY: the page reads the bytes before this returns.
                unsafe { progress(line.as_ptr(), line.len()) };
            }),
        );
    }
    let mut found = search.solve(&root);
    let mut shortened = None;
    if shorten && let Some(length) = found {
        let first_ms = started.elapsed().as_millis() as u64;
        let (first, file) = proof_file(&search, &root, attacker, tables)?;
        hand(&file);
        // Only wins shorter as written, which can be shorter than the
        // search's own length.
        let mut handed = first.moves;
        let shortening = search.shorten(&root, length, |search, _| {
            if let Ok((header, file)) = proof_file(search, &root, attacker, tables)
                && header.moves < handed
            {
                handed = header.moves;
                hand(&file);
            }
        });
        found = Some(shortening.length);
        shortened = Some((first.moves, first_ms, shortening));
    }
    let mut result = json!({
        "found": found.is_some(),
        "nodes": search.nodes,
        "memory_mb": memory,
    });
    if threads > 1 {
        result["threads"] = json!(threads);
    }
    if attacker != root.turn() {
        result["attacker"] = json!(side(attacker));
    }
    if let Some(most) = most {
        result["most"] = json!(most);
    }
    if search.proofs_full() {
        result["error"] = json!("the proofs filled their memory: give more memory");
    }
    if let Some((proof, disproof)) = search.root_numbers
        && found.is_none()
    {
        result["root"] = json!([proof, disproof]);
        if disproof == 0 {
            result["refuted"] = json!(true);
        }
    }
    // Cut short, the line the search stopped on, the hardest it met, with
    // the moves still open along it, where it kept it, while it reported
    // (`Dfpn::line`).
    let (line, open) = search.line();
    if found.is_none() && result.get("refuted").is_none() && !line.is_empty() {
        result["line"] = json!(line);
        result["open"] = json!(open);
    }
    if let Some((first, first_ms, shortening)) = shortened {
        result["first_mate"] = json!(first);
        result["first_mate_ms"] = json!(first_ms);
        result["passes"] = json!(shortening.passes);
        result["shortest"] = json!(shortening.shortest);
    }
    if found.is_some() {
        let (header, file) = proof_file(&search, &root, attacker, tables)?;
        result["mate"] = json!(header.moves);
        result["proof_moves"] = json!(header.positions);
        state.proof = file;
    }
    result["ms"] = json!(started.elapsed().as_millis() as u64);
    Ok(result)
}

/// The proof a search found, in the compact format: its header and its file.
fn proof_file(
    search: &Dfpn,
    root: &Atomic,
    attacker: Color,
    tables: Option<&Tablebase>,
) -> Result<(format::Header, Vec<u8>), String> {
    let (header, indices) = format::walk(root, attacker, tables, |pos| {
        search
            .winning_move(&pos.to_atomic(), &(pos.hash(), pos.halfmoves()))
            .and_then(|m| format::board_move(pos, &m))
    })?;
    let file = format::file(&header, &indices)?;
    Ok((header, file))
}

/// Hands the page a shorter win's proof.
fn hand(file: &[u8]) {
    // SAFETY: the page reads the bytes before this returns.
    unsafe { found(file.as_ptr(), file.len()) };
}

/// The last proof written's bytes, in the compact format.
#[unsafe(no_mangle)]
pub extern "C" fn take_proof() -> u64 {
    reply(STATE.with_borrow_mut(|state| std::mem::take(&mut state.proof)))
}

/// Checks a proof from its bytes, as `provari check` does.
///
/// # Safety
///
/// As `input`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn check(ptr: *mut u8, len: usize) -> u64 {
    // SAFETY: as the caller promises.
    let bytes = unsafe { input(ptr, len) };
    STATE.with_borrow(|state| {
        reply(
            check::check_proof("proof", &bytes, state.checker_tables.as_ref())
                .to_string()
                .into_bytes(),
        )
    })
}

/// Opens a proof in the compact format for browsing, in place of the one
/// open before: replies with its root, its attacker, the attacker moves it
/// takes (`mate`), the moves it writes, the tablebases it declares, a
/// longest line in UCI, and how many moves and comments its PGN tree writes
/// (`pgn`).
///
/// # Safety
///
/// As `input`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn open_proof(ptr: *mut u8, len: usize) -> u64 {
    // SAFETY: as the caller promises.
    let bytes = unsafe { input(ptr, len) };
    answer(STATE.with_borrow_mut(|state| {
        state.open = None;
        let (header, indices) =
            format::decode(&bytes)?.ok_or("not a proof in the compact format")?;
        let tables = state.tables.as_ref();
        let (root, attacker, moves) = format::strategy(&header, &indices, tables)?;
        let mut proof = lines::Proof::new(attacker, moves);
        let (longest, _) = proof.longest_line(&root, tables)?;
        let (moves, comments) = proof.pgn_size(&root, tables)?;
        let reply = json!({
            "root": header.root,
            "attacker": side(attacker),
            "mate": header.moves,
            "positions": header.positions,
            "tablebases": header.tablebases,
            "longest": longest.iter().map(|&m| uci(m)).collect::<Vec<_>>(),
            "pgn": { "moves": moves, "comments": comments },
        });
        state.open = Some((root, proof));
        Ok(reply)
    }))
}

/// Closes the proof open for browsing.
#[unsafe(no_mangle)]
pub extern "C" fn close_proof() {
    STATE.with_borrow_mut(|state| state.open = None);
}

/// A position for the page's board: `{"fen"}` in, and out its FEN, the side
/// to move, how the game ended if it has, its legal moves in UCI with the
/// FEN each leads to and whether it ends the game (`ends`), and, where the
/// open proof covers them, the attacker moves it takes from the position and
/// after each move (`value`), and its move here (`proof_move`); `tablebase`
/// where the proof ends at a tablebase win.
///
/// # Safety
///
/// As `input`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn show_position(ptr: *mut u8, len: usize) -> u64 {
    // SAFETY: as the caller promises.
    let bytes = unsafe { input(ptr, len) };
    answer(STATE.with_borrow_mut(|state| {
        let request: Value = serde_json::from_str(&text(bytes)?).map_err(|e| e.to_string())?;
        let pos = position(request["fen"].as_str().ok_or("no fen")?)?;
        let tables = state.tables.as_ref();
        let mut proof = state.open.as_mut().map(|(_, proof)| proof);
        let mut value = |pos: &Atomic| proof.as_mut().and_then(|proof| proof.value(pos, tables));
        let here = value(&pos);
        let (outcome, termination) = ending(&pos);
        let mut moves = Vec::new();
        if outcome.is_none() {
            for m in pos.legal_moves() {
                let mut next = pos.clone();
                next.play_unchecked(m);
                moves.push(json!({
                    "uci": uci(m),
                    "fen": fen(&next),
                    "value": value(&next),
                    "ends": ending(&next).0.is_some(),
                }));
            }
        }
        let mut reply = json!({
            "fen": fen(&pos),
            "turn": side(pos.turn()),
            "outcome": outcome,
            "termination": termination,
            "value": here,
            "moves": moves,
        });
        if let (Some(_), Some((_, proof))) = (here, &state.open)
            && pos.turn() == proof.attacker()
            && outcome.is_none()
        {
            match proof.proof_move(&pos, tables)? {
                Some(m) => reply["proof_move"] = json!(uci(m)),
                None => reply["tablebase"] = json!(true),
            }
        }
        if let (Some(_), Some((_, proof))) = (here, &state.open)
            && pos.turn() != proof.attacker()
            && outcome.is_none()
            && tables.and_then(|tables| tables.probe(&pos)) == Some(Verdict::Loss)
        {
            reply["tablebase"] = json!(true);
        }
        Ok(reply)
    }))
}

/// Writes the open proof's whole tree as PGN, as `provari line --pgn` writes
/// it from the root, handing the page `{"written": N}` progress lines as it
/// writes its moves and comments; replies with its length in bytes, and
/// keeps the text for `take_pgn`.
#[unsafe(no_mangle)]
pub extern "C" fn proof_pgn() -> u64 {
    answer(STATE.with_borrow_mut(|state| {
        let tables = state.tables.as_ref();
        let (root, proof) = state.open.as_mut().ok_or("no proof is open")?;
        let pgn = proof.pgn(root, tables, &mut |written| {
            let line = json!({ "written": written }).to_string();
            // SAFETY: the page reads the bytes before this returns.
            unsafe { progress(line.as_ptr(), line.len()) };
        })?;
        state.pgn = pgn.into_bytes();
        Ok(json!({ "bytes": state.pgn.len() }))
    }))
}

/// The last PGN tree written, as UTF-8 text.
#[unsafe(no_mangle)]
pub extern "C" fn take_pgn() -> u64 {
    reply(STATE.with_borrow_mut(|state| std::mem::take(&mut state.pgn)))
}

fn fen(pos: &Atomic) -> String {
    Fen::from_position(pos, EnPassantMode::Legal).to_string()
}

fn uci(m: shakmaty::Move) -> String {
    UciMove::from_standard(m).to_string()
}

fn side(color: Color) -> &'static str {
    if color.is_white() { "white" } else { "black" }
}

/// How the game ended at `pos`, if it has, as the site's explorer names it:
/// the outcome, and why (`crate::ending`).
fn ending(pos: &Atomic) -> (Option<&'static str>, Option<&'static str>) {
    match crate::ending(pos) {
        Some((Some(winner), reason)) => {
            let outcome = if winner.is_white() {
                "white_win"
            } else {
                "black_win"
            };
            (Some(outcome), Some(reason))
        }
        Some((None, reason)) => (Some("draw"), Some(reason)),
        None => (None, None),
    }
}
