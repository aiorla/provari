//! Prove forced Atomic wins, by default for the side to move.
//!
//! `provari prove FEN [options]` searches one position;
//! `provari bench FILE --out FILE [options]` searches each position of a JSONL
//! benchmark, one thread per position, and appends a result line for each.
//! `--attacker white|black`, or a benchmark position's `attacker`, searches
//! for that side's win instead, whoever is to move. `--search dfpn` (the
//! default) runs proof-number search; `--search exhaustive` is the win-in-N
//! baseline, for the side to move only. `--most N` accepts only wins in at
//! most N attacker moves (unlimited for df-pn by default, 60 for the
//! baseline, which searches every length up to it). `--seconds S` and
//! `--nodes N` budget each search, alone or together, whichever comes first:
//! a node budget alone has no time limit and repeats exactly whatever the
//! machine's speed; with neither, a search has 10 s. `--memory MB` gives each
//! search its memory, its table and its proofs, by default in proportion to
//! its budget, as engines size their tables to the time control; the searches
//! together may take at most `--max-memory MB`, by default half the machine's
//! memory, or the prover refuses to start. `--threads N` searches each
//! position with N threads that share its memory, its default sized for N
//! times the nodes in the time, and spread out over the tree so that each
//! takes a line of its own (`dfpn`). `provari prove` prints for people
//! (`text`): a status line while it searches, every second where standard
//! error is a terminal, and its result in words; `--json` prints its result
//! as a line of JSON for programs, as `bench` always does. `--progress
//! SECONDS` reports how a search goes that often, 0 for never, in JSON lines
//! for `--json` and `bench`. Results report the budgets a search had. The
//! strongest evaluation there is guides the search: a provability model
//! (`--model FILE`), else Atomic-Stockfish's network (`--network FILE`), else
//! a hand-crafted one; release builds carry a network and a model (`BUNDLED`)
//! for searches whose command line names none, and `--guide` names the guide
//! instead. `provari eval NETWORK` prints the network's evaluation of each FEN
//! read from standard input. Proofs are written in the compact format
//! (`format`), or in JSON lines for a file name ending in `.jsonl`, wins for
//! the side to move only, which `provari convert` writes in the compact
//! format. `provari check` checks proofs with none of the prover's code
//! (`check`), and `provari line` shows its lines. `provari uci` runs as an
//! engine of the Universal Chess Interface, for chess GUIs and tools (`uci`).

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use shakmaty::{Color, EnPassantMode, Position as _, fen::Fen, uci::UciMove, variant::Atomic};

// `check` shares no code with the rest, in either direction: see its
// documentation.
use provari::{
    STACK, board, check, dfpn, ending, format, lines, model, nnue, position, search, tablebase,
    walk,
};

use dfpn::{Dfpn, Guide};
use search::{Search, read_proof};
use tablebase::Tablebase;

mod text;
mod uci;

/// The time a search has when given neither `--seconds` nor `--nodes`.
const DEFAULT_SECONDS: f64 = 10.0;

#[derive(Clone)]
struct Options {
    search: String,
    /// The side whose win to search for, if given; by default the side to
    /// move.
    attacker: Option<Color>,
    /// The time limit given, if any (`time_limit`).
    seconds: Option<f64>,
    /// A node budget for each df-pn search, alone or with a time limit
    /// (`time_limit`).
    nodes: Option<u64>,
    /// Perturbs each df-pn search's steps by up to 1%; 0 leaves them alone.
    seed: u64,
    /// The threads each df-pn search takes, which share its memory.
    threads: usize,
    /// The most attacker moves a win may take.
    most: Option<u32>,
    /// Each search's memory in MB, if not the default for its budget.
    memory: Option<usize>,
    /// The most memory the searches together may take, in MB, if not half
    /// the machine's.
    max_memory: Option<usize>,
    /// Seconds between progress lines, for a df-pn search, 0 for none; by
    /// default none, but for `prove`'s status line on a terminal.
    progress: Option<f64>,
    /// `prove` prints its result as JSON, for programs.
    json: bool,
    /// Where `prove`'s searches print their status lines, if they do: for
    /// people, where `--json` is not given. Not an option.
    status: Option<text::Status>,
    /// After a win, looks for shorter ones within the same budget.
    shorten: bool,
    /// The share of their room at which the proofs drop their mates in one,
    /// if before they fill: for tests of the search after, so not in `help`.
    drop_mates_at: Option<f64>,
    tablebases: Option<PathBuf>,
    network: Option<PathBuf>,
    model: Option<PathBuf>,
    /// The guide asked for: `model`, `network`, `hand` or `none`; by
    /// default the strongest there is.
    guide: Option<String>,
    proofs: Option<PathBuf>,
    workers: usize,
    out: Option<PathBuf>,
    /// Where `provari line` writes the proof's tree as PGN.
    pgn: Option<PathBuf>,
    /// Where `provari line --pgn` stops the tree: at positions the proof wins
    /// within this many attacker moves.
    until: Option<u32>,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print!("{}", help());
        return;
    }
    let (command, rest) = args
        .split_first()
        .map(|(c, r)| (c.as_str(), r))
        .unwrap_or(("", &[]));
    let result = match command {
        // Its line says what went wrong, and the exit code that something did.
        "prove" => match prove(rest) {
            Ok(true) => Ok(()),
            Ok(false) => std::process::exit(2),
            Err(error) => Err(error),
        },
        "bench" => bench(rest),
        "eval" => evaluate(rest),
        "line" => line(rest),
        "convert" => match convert(rest) {
            Ok(true) => Ok(()),
            Ok(false) => std::process::exit(1),
            Err(error) => Err(error),
        },
        "uci" => uci::run(rest),
        "version" | "--version" => {
            println!("{}", version());
            Ok(())
        }
        "help" => {
            print!("{}", help());
            Ok(())
        }
        "check" => match check::run(rest) {
            Ok(true) => Ok(()),
            Ok(false) => std::process::exit(1),
            Err(error) => Err(error),
        },
        "" => {
            eprint!("{}", help());
            std::process::exit(2);
        }
        _ => Err(format!("unknown command {command}: see provari --help")),
    };
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

/// The version, and whether the network and the model are built in.
fn version() -> String {
    let bundled = if BUNDLED.is_some() {
        ", with its network and model built in"
    } else {
        ""
    };
    format!("provari {}{bundled}", env!("CARGO_PKG_VERSION"))
}

/// What `provari --help` prints: the commands and the main options.
fn help() -> String {
    format!(
        "{}\nProves forced wins in Atomic chess.\n\n{HELP}",
        version()
    )
}

const HELP: &str = "\
Usage: provari COMMAND [OPTIONS]

Commands:
  prove FEN                searches one position for a win
  bench POSITIONS.jsonl    searches each position of a file, several at a time
  check PROOF...           verifies proofs, independently of the search
  line PROOF ['MOVES']     shows a proof's lines (--pgn FILE writes its tree)
  convert PROOF.jsonl...   converts JSON-lines proofs to the compact format
  eval NETWORK < FENS      prints a network's evaluations
  uci                      runs as a UCI engine, for chess GUIs and tools
  version                  prints the version

Main options:
  --seconds S, --nodes N   the budget, whichever comes first (10 s by default)
  --memory MB              each search's memory (by default, from its budget)
  --threads N              each search's threads, which share its memory
  --attacker white|black   whose win to prove (by default, the side to move's)
  --most N                 only wins in at most N attacker moves
  --tablebases DIR[:DIR]   Lichess's Atomic Syzygy tables (.atbw), for endings
  --shorten                after a win, looks for shorter ones in the budget
  --proofs PATH            where prove writes its proof (bench: a folder)
  --out PATH               where bench and convert write
  --workers N              searches or checks at a time
  --progress S             reports a search's progress every S seconds
  --json                   prove's result as JSON, for programs

README.md explains every command and option:
https://github.com/aiorla/provari
";

fn parse(args: &[String]) -> Result<(Vec<String>, Options), String> {
    let mut options = Options {
        search: "dfpn".to_owned(),
        attacker: None,
        seconds: None,
        nodes: None,
        seed: 0,
        threads: 1,
        most: None,
        memory: None,
        max_memory: None,
        progress: None,
        json: false,
        status: None,
        shorten: false,
        drop_mates_at: None,
        tablebases: None,
        network: None,
        model: None,
        guide: None,
        proofs: None,
        workers: 1,
        out: None,
        pgn: None,
        until: None,
    };
    let mut free = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = || iter.next().cloned().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--search" => options.search = value()?,
            "--attacker" => options.attacker = Some(side(&value()?)?),
            "--seconds" => options.seconds = Some(value()?.parse().map_err(|_| "bad --seconds")?),
            "--nodes" => options.nodes = Some(value()?.parse().map_err(|_| "bad --nodes")?),
            "--seed" => options.seed = value()?.parse().map_err(|_| "bad --seed")?,
            "--threads" => match value()?.parse() {
                Ok(threads) if threads >= 1 => options.threads = threads,
                _ => return Err("--threads takes a number from 1".into()),
            },
            "--most" => options.most = Some(value()?.parse().map_err(|_| "bad --most")?),
            "--memory" => options.memory = Some(value()?.parse().map_err(|_| "bad --memory")?),
            "--max-memory" => {
                options.max_memory = Some(value()?.parse().map_err(|_| "bad --max-memory")?)
            }
            // At most a billion seconds, which a clock can add to its time.
            "--progress" => match value()?.parse() {
                Ok(seconds) if (0.0..=1e9).contains(&seconds) => options.progress = Some(seconds),
                _ => return Err("--progress takes seconds, 0 for none".into()),
            },
            "--json" => options.json = true,
            "--workers" => options.workers = value()?.parse().map_err(|_| "bad --workers")?,
            "--tablebases" => options.tablebases = Some(value()?.into()),
            "--network" => options.network = Some(value()?.into()),
            "--model" => options.model = Some(value()?.into()),
            "--guide" => match value()?.as_str() {
                guide @ ("model" | "network" | "hand" | "none") => {
                    options.guide = Some(guide.to_owned())
                }
                _ => return Err("--guide is model, network, hand or none".into()),
            },
            "--no-eval" => options.guide = Some("none".to_owned()),
            "--shorten" => options.shorten = true,
            "--drop-mates-at" => {
                options.drop_mates_at = Some(value()?.parse().map_err(|_| "bad --drop-mates-at")?)
            }
            "--proofs" => options.proofs = Some(value()?.into()),
            "--out" => options.out = Some(value()?.into()),
            "--pgn" => options.pgn = Some(value()?.into()),
            "--until" => options.until = Some(value()?.parse().map_err(|_| "bad --until")?),
            // A misspelled option would otherwise pass for a FEN's field, a
            // move or a file, and a run go on without it.
            unknown if unknown.starts_with("--") => {
                return Err(format!("unknown option {unknown}: see provari --help"));
            }
            _ => free.push(arg.clone()),
        }
    }
    if options.shorten && options.search == "exhaustive" {
        return Err("--shorten is for df-pn: the baseline finds the shortest win itself".into());
    }
    if options.threads > 1 && options.search == "exhaustive" {
        return Err("--threads is for df-pn: the baseline searches with one".into());
    }
    Ok((free, options))
}

/// The side named, as `--attacker` and a benchmark's `attacker` name it.
fn side(name: &str) -> Result<Color, String> {
    name.parse()
        .map_err(|_| format!("the attacker is white or black, not {name}"))
}

impl Options {
    /// Each search's table and proofs in MB: `--memory` divided by
    /// `dfpn::split`, which gives the proofs what the budget needs where it
    /// can, or by default a table of about what a search fills in the time or
    /// nodes it has, some 10 MB a second or 20 bytes a node, as a power of two
    /// between 16 MB and 4 GB, and half that again for its proofs, or more
    /// where its budget needs it (`dfpn::proofs_needed`): a search stops
    /// when its proofs fill, once they have dropped their mates in one, while
    /// a full table costs little. So an hour has a 4 GB table and 5.4 GB of
    /// proofs. Its threads search as many times the nodes in its time, and
    /// share its memory.
    fn memory(&self) -> (usize, usize) {
        let threads = self.threads as f64;
        let seconds = self.time_limit().map(|seconds| seconds * threads);
        let needed = dfpn::proofs_needed(seconds, self.nodes);
        if let Some(memory) = self.memory {
            return dfpn::split(memory, needed);
        }
        let filled = match self.nodes {
            Some(nodes) => nodes.div_ceil(50_000) as usize,
            None => (self.seconds.unwrap_or(DEFAULT_SECONDS) * 10.0 * threads).ceil() as usize,
        };
        let table = filled.next_power_of_two().clamp(16, 4096);
        (table, (table / 2).max(needed))
    }

    /// Each search's memory in all, table and proofs.
    fn memory_mb(&self) -> usize {
        let (table, proofs) = self.memory();
        table + proofs
    }

    /// A search's time limit: `--seconds` when given, else none under a node
    /// budget, else the default.
    fn time_limit(&self) -> Option<f64> {
        match (self.seconds, self.nodes) {
            (Some(seconds), _) => Some(seconds),
            (None, Some(_)) => None,
            (None, None) => Some(DEFAULT_SECONDS),
        }
    }

    /// When a search must stop, from now: its time limit, or a year for none.
    fn deadline(&self) -> Instant {
        Instant::now() + Duration::from_secs_f64(self.time_limit().unwrap_or(3.2e7))
    }

    /// The budgets a search has, for its result: its time limit, node budget,
    /// memory and longest win, where it has them.
    fn budgets(&self, result: &mut Value) {
        if let Some(seconds) = self.time_limit() {
            result["seconds"] = json!(seconds);
        }
        if let Some(nodes) = self.nodes {
            result["node_limit"] = json!(nodes);
        }
        result["memory_mb"] = json!(self.memory_mb());
        if self.threads > 1 {
            result["threads"] = json!(self.threads);
        }
        if let Some(most) = self.most {
            result["most"] = json!(most);
        }
        if self.shorten {
            result["shorten"] = json!(true);
        }
        if let Some(share) = self.drop_mates_at {
            result["drop_mates_at"] = json!(share);
        }
    }

    /// Refuses searches that together would take more than `--max-memory`,
    /// rather than shrink them, which would change their results. By default
    /// that is half the machine's memory: with the usual programs open, one
    /// search of 16 GB on a 24 GB Mac had part of its memory compressed and ran
    /// at a fifth of its speed. A server running nothing else can allow more.
    fn check_memory(&self, searches: usize) -> Result<(), String> {
        let needed = self.memory_mb() * searches;
        let Some(most) = self.max_memory.or_else(|| physical_mb().map(|mb| mb / 2)) else {
            return Ok(());
        };
        if needed > most {
            return Err(format!(
                "{searches} × {} MB of searches exceed the {most} MB allowed: give fewer \
                 --workers, less --memory, or more --max-memory",
                self.memory_mb()
            ));
        }
        Ok(())
    }
}

/// This machine's memory in MB, if it can tell.
#[cfg(windows)]
fn physical_mb() -> Option<usize> {
    /// MEMORYSTATUSEX, of which only the physical memory is read here: the
    /// call reads its length and writes the rest.
    #[allow(dead_code)]
    #[repr(C)]
    struct MemoryStatus {
        length: u32,
        load: u32,
        physical: u64,
        rest: [u64; 6],
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatus) -> i32;
    }
    let mut status = MemoryStatus {
        length: size_of::<MemoryStatus>() as u32,
        load: 0,
        physical: 0,
        rest: [0; 6],
    };
    // SAFETY: the structure has MEMORYSTATUSEX's layout and gives its length.
    let filled = unsafe { GlobalMemoryStatusEx(&mut status) } != 0;
    filled.then_some((status.physical >> 20) as usize)
}

/// This machine's memory in MB, if it can tell.
#[cfg(not(windows))]
fn physical_mb() -> Option<usize> {
    if cfg!(target_os = "macos") {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        let bytes: u64 = String::from_utf8(output.stdout).ok()?.trim().parse().ok()?;
        Some((bytes >> 20) as usize)
    } else {
        let info = fs::read_to_string("/proc/meminfo").ok()?;
        let line = info
            .lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))?;
        let kilobytes: u64 = line.trim().strip_suffix("kB")?.trim().parse().ok()?;
        Some((kilobytes >> 10) as usize)
    }
}

/// The network and the provability model that release binaries carry, built
/// with the `bundled` feature from `data/network.nnue` and `data/model.json`
/// (which `data/fetch.sh` fetches), for searches whose command line names
/// none; other builds carry none.
#[cfg(feature = "bundled")]
const BUNDLED: Option<(&[u8], &str)> = Some((
    include_bytes!("../data/network.nnue"),
    include_str!("../data/model.json"),
));
#[cfg(not(feature = "bundled"))]
const BUNDLED: Option<(&[u8], &str)> = None;

/// What every search shares: the tablebases, the network, and the model.
struct Inputs {
    tablebase: Option<Tablebase>,
    network: Option<nnue::Network>,
    model: Option<model::Model>,
}

impl Inputs {
    /// The inputs the options name; for a command that searches, the built-in
    /// network where they name none, and the built-in model with it, the
    /// network it was fitted on, where they name neither.
    fn open(options: &Options, searches: bool) -> Result<Inputs, String> {
        let guide = options.guide.as_deref();
        let bundled = BUNDLED.filter(|_| searches && !matches!(guide, Some("hand" | "none")));
        let network = match (&options.network, bundled) {
            (Some(path), _) => Some(nnue::Network::load(path)?),
            (None, Some((network, _))) => Some(nnue::Network::from_bytes(network)?),
            (None, None) => None,
        };
        let model = match (&options.model, bundled) {
            (Some(path), _) => Some(model::Model::load(path)?),
            (None, Some((_, model))) if options.network.is_none() && guide != Some("network") => {
                Some(model::Model::from_text(model)?)
            }
            _ => None,
        };
        let inputs = Inputs {
            tablebase: options
                .tablebases
                .as_deref()
                .map(|dir| Tablebase::open(dir).map_err(|e| format!("tablebases: {e}")))
                .transpose()?,
            network,
            model,
        };
        if inputs.network.is_none() && inputs.model.as_ref().is_some_and(|m| m.reads_layer()) {
            return Err("the model reads the network's first layer: give --network".into());
        }
        match guide {
            Some("model") if inputs.model.is_none() => Err("--guide model needs --model".into()),
            Some("network") if inputs.network.is_none() => {
                Err("--guide network needs --network".into())
            }
            _ => Ok(inputs),
        }
    }

    /// What guides the search: `--guide`'s choice, else the strongest there
    /// is: the model, else the network, else the hand-crafted evaluation.
    fn guide(&self, options: &Options) -> Guide<'_> {
        match options.guide.as_deref() {
            Some("none") => Guide::None,
            Some("hand") => Guide::Hand,
            guide => match (&self.model, &self.network) {
                (Some(model), network) if guide != Some("network") => {
                    Guide::Model(model, network.as_ref())
                }
                (_, Some(network)) => Guide::Network(network),
                _ => Guide::Hand,
            },
        }
    }

    fn search(&self, root: &Atomic, attacker: Color, options: &Options) -> Dfpn<'_> {
        self.try_search(root, attacker, options)
            .unwrap_or_else(|error| panic!("the search's memory: {error}"))
    }

    /// As `search`, unless the memory cannot be had, which `provari uci` says
    /// before it goes on.
    fn try_search(
        &self,
        root: &Atomic,
        attacker: Color,
        options: &Options,
    ) -> Result<Dfpn<'_>, std::collections::TryReserveError> {
        let mut search = Dfpn::try_with_threads(
            root,
            self.tablebase.as_ref(),
            self.guide(options),
            options.deadline(),
            options.memory(),
            options.threads,
        )?;
        search.set_attacker(attacker);
        if let Some(nodes) = options.nodes {
            search.set_limit(nodes);
        }
        if let Some(most) = options.most {
            search.set_most(most);
        }
        // People are shown the line a search cut short stopped on, which a
        // search keeps only when asked, as it costs time.
        match &options.status {
            Some(status) => match status.every {
                Some(every) => search.set_progress_sink(every, status.sink(root.clone(), attacker)),
                None => search.keep_line(),
            },
            None => {
                if let Some(seconds) = options.progress.filter(|&seconds| seconds > 0.0) {
                    search.set_progress(Duration::from_secs_f64(seconds));
                }
            }
        }
        search.set_seed(options.seed);
        if let Some(share) = options.drop_mates_at {
            search.set_drop_mates_at(share);
        }
        Ok(search)
    }
}

/// Searches one position for `attacker`'s win, by default the side to move's,
/// and writes its proof when it finds one.
fn run(
    fen: &str,
    attacker: Option<Color>,
    options: &Options,
    inputs: &Inputs,
    proof: Option<&Path>,
) -> Value {
    let root = match position(fen) {
        Ok(root) => root,
        Err(error) => return json!({ "error": error }),
    };
    // A game already over has nothing to prove: the result says how it ended.
    if let Some((winner, reason)) = ending(&root) {
        return json!({
            "found": false,
            "mate": null,
            "over": reason,
            "winner": winner.map(|color| if color.is_white() { "white" } else { "black" }),
            "nodes": 0,
            "ms": 0,
        });
    }
    let attacker = attacker.unwrap_or(root.turn());
    if attacker != root.turn() {
        if options.search == "exhaustive" {
            return json!({ "error": "the baseline searches wins for the side to move only" });
        }
        if proof.is_some_and(is_jsonl) {
            return json!({ "error": "JSON lines prove wins for the side to move only" });
        }
    }
    let tablebase = inputs.tablebase.as_ref();
    // People are shown a win's longest line.
    let longest = options.status.is_some();
    let mut started = Instant::now();
    let mut shortened = None;
    let (moves, first_ms, nodes, (full, dropped), root_numbers, written, hardest) =
        if options.search == "exhaustive" {
            // The baseline has no node budget: only a time limit.
            let seconds = options.seconds.unwrap_or(DEFAULT_SECONDS);
            let deadline = started + Duration::from_secs_f64(seconds);
            // The baseline keeps no proofs apart: its table takes the table's
            // share.
            let entries = options.memory().0 << 20 >> 5;
            let mut search = Search::new(&root, tablebase, deadline, entries);
            let moves = search.solve(&root, options.most.unwrap_or(60));
            let first_ms = started.elapsed().as_millis() as u64;
            let written = moves.map(|moves| {
                write_proof(
                    &root,
                    attacker,
                    fen,
                    moves,
                    tablebase,
                    proof,
                    longest,
                    |_, key| search.winning_move(key),
                )
            });
            (
                moves,
                first_ms,
                search.nodes,
                (false, None),
                None,
                written,
                None,
            )
        } else {
            let mut search = inputs.search(&root, attacker, options);
            // The clock and the time limit start once the search's memory is set
            // up, as an engine's do at `go`, after its hash is ready.
            started = Instant::now();
            search.set_deadline(options.deadline());
            let mut moves = search.solve(&root);
            let first_ms = started.elapsed().as_millis() as u64;
            if options.shorten
                && let Some(length) = moves
            {
                // The first win as written, which can already be shorter than
                // the search's own length, as the walk takes each position's
                // shortest proof.
                let first = write_proof(
                    &root,
                    attacker,
                    fen,
                    length,
                    tablebase,
                    None,
                    false,
                    |pos, key| search.winning_move(pos, key),
                )
                .map_or(length, |written| written.length);
                let mut done = Shortened {
                    first,
                    first_ms,
                    first_nodes: search.nodes,
                    passes: 0,
                    ms: first_ms,
                    nodes: search.nodes,
                    shortest: false,
                };
                let shortening = search.shorten(&root, length, |search, _| {
                    done.ms = started.elapsed().as_millis() as u64;
                    done.nodes = search.nodes;
                });
                done.passes = shortening.passes;
                done.shortest = shortening.shortest;
                moves = Some(shortening.length);
                shortened = Some(done);
            }
            let written = moves.map(|moves| {
                write_proof(
                    &root,
                    attacker,
                    fen,
                    moves,
                    tablebase,
                    proof,
                    longest,
                    |pos, key| search.winning_move(pos, key),
                )
            });
            let full = (search.proofs_full(), search.mates_dropped);
            let hardest = moves.is_none().then(|| search.line());
            (
                moves,
                first_ms,
                search.nodes,
                full,
                search.root_numbers,
                written,
                hardest,
            )
        };
    let elapsed = started.elapsed().as_millis() as u64;
    // The length of the proof as written: the compact format settles some
    // positions by their quickest win, so it can be shorter than the
    // search's own.
    let mate = match &written {
        Some(Ok(written)) => Some(written.length),
        _ => moves,
    };
    let mut result = json!({
        "found": moves.is_some(),
        "mate": mate,
        "first_mate_ms": moves.map(|_| first_ms),
        "nodes": nodes,
        "ms": elapsed,
    });
    if attacker != root.turn() {
        result["attacker"] = json!(attacker.to_string());
    }
    if options.search == "exhaustive" {
        result["seconds"] = json!(options.seconds.unwrap_or(DEFAULT_SECONDS));
        result["memory_mb"] = json!(options.memory_mb());
        result["most"] = json!(options.most.unwrap_or(60));
    } else {
        options.budgets(&mut result);
    }
    if let Some(shortened) = shortened {
        result["first_mate"] = json!(shortened.first);
        result["first_mate_ms"] = json!(shortened.first_ms);
        result["first_mate_nodes"] = json!(shortened.first_nodes);
        result["passes"] = json!(shortened.passes);
        result["mate_ms"] = json!(shortened.ms);
        result["mate_nodes"] = json!(shortened.nodes);
        result["shortest"] = json!(shortened.shortest);
    }
    // When the proofs dropped their mates in one, as they do when they first
    // fill.
    if let Some(nodes) = dropped {
        result["mates_dropped"] = json!(nodes);
    }
    if full {
        // A win found before the proofs filled, while shortening it, stands.
        if moves.is_some() {
            result["proofs_full"] = json!(true);
        } else {
            result["error"] = json!("the proofs filled their memory: give more --memory");
        }
    }
    // An unproven root's numbers where the search stopped, which tell the
    // roots a longer search may prove; a disproof number of 0 settles that
    // no win exists (under a limit, none in so many moves).
    if let Some((proof, disproof)) = root_numbers
        && moves.is_none()
    {
        result["root"] = json!([proof, disproof]);
        if disproof == 0 {
            result["refuted"] = json!(true);
        }
    }
    // Cut short, the line the search stopped on, the hardest it met, with
    // the moves still open along it, where the search kept it: while it
    // reported, or for people (`Dfpn::line`).
    if let Some((line, open)) = hardest
        && !line.is_empty()
        && result.get("refuted").is_none()
    {
        result["line"] = json!(line);
        result["open"] = json!(open);
    }
    match written {
        Some(Ok(written)) => {
            if proof.is_some() {
                result[written.field] = json!(written.count);
            }
            // For people: a longest line, and whether lines may end at
            // tablebase wins, which makes the length a bound.
            if let Some((line, at_table)) = written.longest {
                let line: Vec<String> = line
                    .into_iter()
                    .map(|m| UciMove::from_standard(m).to_string())
                    .collect();
                result["line"] = json!(line);
                if at_table {
                    result["line_tablebase"] = json!(true);
                }
                if written.tablebases {
                    result["tablebases"] = json!(true);
                }
            }
        }
        Some(Err(error)) => result["error"] = json!(error),
        None => {}
    }
    result
}

/// What `--shorten` did after the first win: its length as written, when and
/// after how many nodes it was found, the passes run, when and after how many
/// nodes the best win was found, and whether the last pass settled that no
/// shorter win exists.
struct Shortened {
    first: u32,
    first_ms: u64,
    first_nodes: u64,
    passes: u32,
    ms: u64,
    nodes: u64,
    shortest: bool,
}

/// A proof read back: its length as written, the result field for its size,
/// with the size, whether lines may end at tablebase wins, and a longest
/// line, with whether it ends at one, if asked for.
struct Written {
    length: u32,
    field: &'static str,
    count: usize,
    tablebases: bool,
    longest: Option<(Vec<shakmaty::Move>, bool)>,
}

/// Whether a proof file is in JSON lines, by its name; any other is compact.
fn is_jsonl(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension == "jsonl")
}

/// Reads the proof of `attacker`'s win from a root back, walking it as the
/// compact format does (`format`), and writes it into `path` if given; in
/// JSON lines for a name ending in `.jsonl`, a win for the side to move, a
/// line for the root and one for each attacker position, through a temporary
/// file so that a proof there is always whole. Returns its length as written
/// and its size: the moves a compact proof writes, the positions JSON lines
/// list; with `longest`, a longest line of it too (`lines`).
#[allow(clippy::too_many_arguments)]
fn write_proof(
    root: &Atomic,
    attacker: Color,
    fen: &str,
    moves: u32,
    tablebase: Option<&Tablebase>,
    path: Option<&Path>,
    longest: bool,
    mut winning_move: impl FnMut(&Atomic, &search::Key) -> Option<shakmaty::Move>,
) -> Result<Written, String> {
    if path.is_none_or(|path| !is_jsonl(path)) {
        let (header, indices) = format::walk(root, attacker, tablebase, |pos| {
            winning_move(&pos.to_atomic(), &(pos.hash(), pos.halfmoves()))
                .and_then(|m| format::board_move(pos, &m))
        })?;
        if let Some(path) = path {
            format::save(path, &format::file(&header, &indices)?)?;
        }
        let longest = match longest {
            true => {
                let (_, _, moves) = format::strategy(&header, &indices, tablebase)?;
                Some(lines::Proof::new(attacker, moves).longest_line(root, tablebase)?)
            }
            false => None,
        };
        return Ok(Written {
            length: header.moves,
            field: "proof_moves",
            count: header.positions as usize,
            tablebases: header.tablebases > 0,
            longest,
        });
    }
    let error = |error: std::io::Error| format!("cannot write the proof: {error}");
    let partial = path.map(|path| path.with_extension("partial"));
    let mut out = match &partial {
        Some(partial) => Some(BufWriter::new(File::create(partial).map_err(error)?)),
        None => None,
    };
    if let Some(out) = &mut out {
        writeln!(out, "{}", json!({ "root": fen, "moves": moves })).map_err(error)?;
    }
    let (mut strategy, mut tablebases) = (walk::Moves::new(), false);
    let positions = read_proof(root, tablebase, winning_move, |pos, m| {
        tablebases |= m.is_none();
        if longest {
            strategy.entry(search::hash(pos)).or_insert(m);
        }
        let Some(out) = &mut out else {
            return Ok(());
        };
        let fen = Fen::from_position(pos, EnPassantMode::Legal).to_string();
        let line = match m {
            Some(m) => json!({ "fen": fen, "move": UciMove::from_standard(m).to_string() }),
            None => json!({ "fen": fen, "tablebase": "win" }),
        };
        writeln!(out, "{line}").map_err(error)
    })?;
    if let (Some(out), Some(partial), Some(path)) = (out, partial, path) {
        out.into_inner()
            .map_err(|failed| error(failed.into_error()))?;
        fs::rename(partial, path).map_err(error)?;
    }
    let longest = match longest {
        true => Some(lines::Proof::new(attacker, strategy).longest_line(root, tablebase)?),
        false => None,
    };
    Ok(Written {
        length: moves,
        field: "proof_positions",
        count: positions,
        tablebases,
        longest,
    })
}

/// Writes proofs in JSON lines again in the compact format: `provari convert
/// PROOF.jsonl... --out DIR [--tablebases DIR] [--workers N]` writes
/// DIR/NAME.pvr for each, with the tablebases the proof was made with, and
/// prints a JSON line for each, in the order they finish. Returns whether
/// every proof converted.
fn convert(args: &[String]) -> Result<bool, String> {
    let (free, options) = parse(args)?;
    let out = options.out.clone().ok_or("needs --out")?;
    if free.is_empty() {
        return Err("usage: provari convert PROOF.jsonl... --out DIR".into());
    }
    fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let inputs = Inputs::open(&options, false)?;
    let queue = Mutex::new(free.iter());
    // Held to print a line whole: whether every proof so far converted.
    let converted = Mutex::new(true);
    std::thread::scope(|scope| {
        for _ in 0..options.workers.max(1) {
            let worker = std::thread::Builder::new().stack_size(STACK);
            let spawned = worker.spawn_scoped(scope, || {
                loop {
                    let Some(input) = queue.lock().unwrap().next() else {
                        break;
                    };
                    let path = Path::new(input);
                    let name = path.with_extension(format::EXTENSION);
                    let target = out.join(name.file_name().unwrap_or_default());
                    let size = |path: &Path| fs::metadata(path).map_or(0, |meta| meta.len());
                    let line = match convert_proof(path, &target, inputs.tablebase.as_ref()) {
                        Ok(header) => json!({
                            "proof": input,
                            "out": target.display().to_string(),
                            "moves": header.moves,
                            "positions": header.positions,
                            "bytes": size(&target),
                            "jsonl_bytes": size(path),
                        }),
                        Err(error) => json!({ "proof": input, "error": error }),
                    };
                    let mut converted = converted.lock().unwrap();
                    *converted &= line.get("error").is_none();
                    println!("{line}");
                }
            });
            spawned.expect("cannot start a worker");
        }
    });
    Ok(converted.into_inner().unwrap())
}

/// Writes a proof in JSON lines in the compact format, through a temporary
/// file so that a proof there is always whole.
fn convert_proof(
    path: &Path,
    target: &Path,
    tablebase: Option<&Tablebase>,
) -> Result<format::Header, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (header, bytes) = format::encode_jsonl(&text, tablebase)?;
    format::save(target, &bytes)?;
    Ok(header)
}

/// Prints the network's evaluation of each position read from standard
/// input, one FEN a line: `provari eval NETWORK`.
fn evaluate(args: &[String]) -> Result<(), String> {
    let [path] = args else {
        return Err("usage: provari eval NETWORK < FENS".into());
    };
    let network = nnue::Network::load(Path::new(path))?;
    for line in std::io::stdin().lines() {
        let line = line.map_err(|error| error.to_string())?;
        let fen = line.trim();
        if fen.is_empty() {
            continue;
        }
        let pos = board::Position::from_atomic(&position(fen)?);
        println!("{}", network.evaluate(&pos));
    }
    Ok(())
}

/// Searches one position and prints its result, for people (`text`), or with
/// `--json` as JSON. Returns whether the result holds no error, such as a bad
/// FEN or a proof that could not be written, which people read on standard
/// error.
fn prove(args: &[String]) -> Result<bool, String> {
    let (free, mut options) = parse(args)?;
    options.check_memory(1)?;
    let fen = free.join(" ");
    if !options.json {
        options.status = Some(text::Status::new(options.progress));
    }
    let inputs = Inputs::open(&options, true)?;
    let proof = options.proofs.clone();
    let result = std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(STACK)
            .spawn_scoped(scope, || {
                run(&fen, options.attacker, &options, &inputs, proof.as_deref())
            })
            .expect("cannot start the search")
            .join()
            .expect("the search failed")
    });
    let Some(status) = &options.status else {
        println!("{result}");
        return Ok(result.get("error").is_none());
    };
    status.clear();
    if let Ok(root) = position(&fen) {
        let attacker = options.attacker.unwrap_or(root.turn());
        let path = proof.as_ref().map(|path| path.display().to_string());
        let said = text::result(&result, &root, attacker, path.as_deref());
        // An error before the search, such as an attacker it refuses, says
        // nothing else.
        if !said.is_empty() {
            println!("{said}");
        }
    }
    match result["error"].as_str() {
        Some(error) => {
            eprintln!("error: {error}");
            Ok(false)
        }
        None => Ok(true),
    }
}

/// Searches each position of a JSONL benchmark and appends a result for each.
fn bench(args: &[String]) -> Result<(), String> {
    let (free, options) = parse(args)?;
    options.check_memory(options.workers)?;
    if let Some(dir) = &options.proofs {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let inputs = Inputs::open(&options, true)?;
    each(&free, &options, "id", |position| {
        let id = position["id"].as_str().unwrap_or_default();
        let fen = position["fen"].as_str().unwrap_or_default();
        let proof = options
            .proofs
            .as_ref()
            .map(|dir| dir.join(format!("{id}.{}", format::EXTENSION)));
        // A position's own attacker, else `--attacker`'s.
        let attacker = match &position["attacker"] {
            Value::Null => Ok(options.attacker),
            Value::String(name) => side(name).map(Some),
            other => Err(format!("the attacker is white or black, not {other}")),
        };
        let mut result = match attacker {
            Ok(attacker) => run(fen, attacker, &options, &inputs, proof.as_deref()),
            Err(error) => json!({ "error": error }),
        };
        result["id"] = json!(id);
        result["engine"] = json!(format!("prover-{}", options.search));
        if options.seed != 0 {
            result["seed"] = json!(options.seed);
        }
        vec![result]
    })
}

/// A proof file's root, its attacker, and its moves: one a board, the first
/// listed in JSON lines, which prove wins for the side to move, as the
/// prover keeps one proof; in the compact format, those of the positions the
/// walk proves at their highest counter, with the moves of the positions it
/// settles.
fn read_moves(
    input: &str,
    tablebase: Option<&Tablebase>,
) -> Result<(Atomic, Color, walk::Moves), String> {
    if let Some((header, indices)) = format::read(Path::new(input))? {
        return format::strategy(&header, &indices, tablebase);
    }
    let text = fs::read_to_string(input).map_err(|e| format!("{input}: {e}"))?;
    let mut lines = text.lines();
    let head: Value =
        serde_json::from_str(lines.next().ok_or("an empty proof")?).map_err(|e| e.to_string())?;
    let root = position(head["root"].as_str().ok_or("a proof without its root")?)?;
    let attacker = root.turn();
    let mut moves = walk::Moves::new();
    for line in lines {
        let value: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let pos = position(value["fen"].as_str().ok_or("a line without its position")?)?;
        let m = match value["move"].as_str() {
            Some(uci) => Some(
                uci.parse::<UciMove>()
                    .map_err(|e| e.to_string())?
                    .to_move(&pos)
                    .map_err(|e| e.to_string())?,
            ),
            None => None,
        };
        moves.entry(search::hash(&pos)).or_insert(m);
    }
    Ok((root, attacker, moves))
}

/// Shows a checked proof's lines: `provari line PROOF [MOVES]
/// [--tablebases DIR] [--pgn FILE [--until N]]` plays MOVES from its root
/// (SAN or UCI, move numbers allowed) and prints, against the proof's moves,
/// the most attacker moves the defender can last from there, the proof's
/// move, each defence with the attacker moves it lasts, longest first, and a
/// longest line. A tablebase win counts as won where it is reached, as in the
/// proof's length. `--pgn` writes the proof's tree from there as well, and
/// `--until N` stops it where the proof wins within N attacker moves.
fn line(args: &[String]) -> Result<(), String> {
    let (free, options) = parse(args)?;
    let (input, played) = free
        .split_first()
        .ok_or("usage: provari line PROOF [MOVES]")?;
    let inputs = Inputs::open(&options, false)?;
    let pgn = options.pgn.as_deref();
    let result = std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(STACK)
            .spawn_scoped(scope, || {
                show_line(input, &played.join(" "), &inputs, pgn, options.until)
            })
            .expect("cannot start the walk")
            .join()
            .expect("the walk failed")
    })?;
    println!("{result}");
    Ok(())
}

fn show_line(
    input: &str,
    played: &str,
    inputs: &Inputs,
    pgn: Option<&Path>,
    until: Option<u32>,
) -> Result<Value, String> {
    let tablebase = inputs.tablebase.as_ref();
    let (root, attacker, moves) = read_moves(input, tablebase)?;
    let (result, text) = lines::show(
        root,
        attacker,
        moves,
        tablebase,
        played,
        pgn.is_some(),
        until,
    )?;
    if let (Some(path), Some(text)) = (pgn, text) {
        fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(result)
}

/// Runs `job` on each JSONL line of the input with `--workers` threads,
/// appending what it returns to `--out` and skipping inputs whose `id`
/// already appears there under `key`. A line that is not JSON with a string
/// `id` is refused before any job starts; blank lines are skipped.
fn each(
    free: &[String],
    options: &Options,
    key: &str,
    job: impl Fn(&Value) -> Vec<Value> + Sync,
) -> Result<(), String> {
    let input = free.first().ok_or("needs an input file")?;
    let out = options.out.clone().ok_or("needs --out")?;
    let done: Vec<String> = match File::open(&out) {
        Ok(file) => BufReader::new(file)
            .lines()
            .map_while(Result::ok)
            .filter_map(|line| serde_json::from_str::<Value>(&line).ok())
            .filter_map(|v| v[key].as_str().map(str::to_owned))
            .collect(),
        Err(_) => Vec::new(),
    };
    let mut items = Vec::new();
    let file = File::open(input).map_err(|e| format!("{input}: {e}"))?;
    for (number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{input}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let at = |problem: String| format!("{input}, line {}: {problem}", number + 1);
        let item: Value = serde_json::from_str(&line).map_err(|e| at(e.to_string()))?;
        let id = item["id"]
            .as_str()
            .ok_or_else(|| at("no id, as a string".into()))?;
        if !done.iter().any(|d| d == id) {
            items.push(item);
        }
    }
    let queue = Mutex::new(items.iter());
    let output = Mutex::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&out)
            .map_err(|e| format!("{}: {e}", out.display()))?,
    );
    let total = items.len();
    let finished = Mutex::new(0usize);
    std::thread::scope(|scope| {
        for _ in 0..options.workers {
            // Deep lines recurse deeply.
            let worker = std::thread::Builder::new().stack_size(STACK);
            let spawned = worker.spawn_scoped(scope, || {
                loop {
                    let Some(item) = queue.lock().unwrap().next() else {
                        break;
                    };
                    let lines = job(item);
                    let mut file = output.lock().unwrap();
                    for line in lines {
                        writeln!(file, "{line}").ok();
                    }
                    let mut count = finished.lock().unwrap();
                    *count += 1;
                    eprint!("\r{}/{total}", *count);
                }
            });
            spawned.expect("cannot start a worker");
        }
    });
    eprintln!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::{Value, json};
    use shakmaty::Color;

    use super::{Inputs, parse, run};

    /// A game already over is not searched: the result says how it ended.
    #[test]
    fn says_how_a_game_ended() {
        let (_, options) = parse(&[]).unwrap();
        let inputs = Inputs {
            tablebase: None,
            network: None,
            model: None,
        };
        let exploded = "rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQ - 0 1";
        let result = run(exploded, None, &options, &inputs, None);
        assert_eq!(result["found"], false);
        assert_eq!(result["over"], "variant_loss");
        assert_eq!(result["winner"], "white");
        assert_eq!(result["nodes"], 0);
        let stalemate = run(
            "k7/8/1Q6/8/8/8/8/6K1 b - - 0 1",
            None,
            &options,
            &inputs,
            None,
        );
        assert_eq!(stalemate["over"], "stalemate");
        assert!(stalemate["winner"].is_null());
    }

    /// `--attacker` searches for the side not to move's win, and the result
    /// says whose; naming the side to move changes nothing. The baseline and
    /// JSON lines refuse the other side.
    #[test]
    fn proves_a_win_for_either_side() {
        let args =
            |args: &[&str]| -> Vec<String> { args.iter().map(|arg| (*arg).to_owned()).collect() };
        let inputs = Inputs {
            tablebase: None,
            network: None,
            model: None,
        };
        // White's bishop has just checked; both king moves let it explode
        // the king.
        let fen = "rnbq1b1r/pp1pk2p/2pBppp1/8/3PP3/2N5/PPP2P1P/R2QKB1R b KQ - 1 1";
        let (_, options) = parse(&args(&["--nodes", "1000", "--attacker", "white"])).unwrap();
        let result = run(fen, options.attacker, &options, &inputs, None);
        assert_eq!(
            (&result["found"], &result["mate"], &result["attacker"]),
            (&json!(true), &json!(1), &json!("white"))
        );
        let (_, options) = parse(&args(&["--nodes", "1000"])).unwrap();
        let named = run(fen, Some(Color::Black), &options, &inputs, None);
        let unnamed = run(fen, None, &options, &inputs, None);
        assert!(named.get("attacker").is_none());
        assert_eq!(
            (&named["found"], &named["nodes"]),
            (&unnamed["found"], &unnamed["nodes"])
        );
        let (_, options) = parse(&args(&["--search", "exhaustive"])).unwrap();
        let baseline = run(fen, Some(Color::White), &options, &inputs, None);
        assert_eq!(
            baseline["error"],
            "the baseline searches wins for the side to move only"
        );
        let (_, options) = parse(&args(&["--nodes", "1000"])).unwrap();
        let jsonl = run(
            fen,
            Some(Color::White),
            &options,
            &inputs,
            Some(Path::new("p.jsonl")),
        );
        assert_eq!(
            jsonl["error"],
            "JSON lines prove wins for the side to move only"
        );
        assert!(parse(&args(&["--attacker", "red"])).is_err());
    }

    /// `--threads` searches with threads that share the memory: they find
    /// the win one thread finds, and the result says how many. A search
    /// takes at least one thread, the baseline only one.
    #[test]
    fn searches_with_threads() {
        let args =
            |args: &[&str]| -> Vec<String> { args.iter().map(|arg| (*arg).to_owned()).collect() };
        let inputs = Inputs {
            tablebase: None,
            network: None,
            model: None,
        };
        let fen = "r1b1kb1r/pp1p1qp1/2n1p3/1B3P1p/3PP3/2P5/PP4PP/R2QK2R w KQkq - 0 1";
        let given = ["--nodes", "1000000", "--memory", "16", "--threads", "3"];
        let (_, options) = parse(&args(&given)).unwrap();
        let result = run(fen, None, &options, &inputs, None);
        assert_eq!(result["found"], true, "{result}");
        assert_eq!(result["threads"], 3);
        assert!(parse(&args(&["--threads", "0"])).is_err());
        assert!(parse(&args(&["--threads", "2", "--search", "exhaustive"])).is_err());
    }

    /// The baseline finds the shortest win itself, so it refuses `--shorten`.
    #[test]
    fn shortens_only_proof_number_search() {
        let args =
            |args: &[&str]| -> Vec<String> { args.iter().map(|arg| (*arg).to_owned()).collect() };
        assert!(parse(&args(&["--shorten"])).unwrap().1.shorten);
        assert!(parse(&args(&["--shorten", "--search", "exhaustive"])).is_err());
    }

    /// A misspelled option is refused, rather than taken for a FEN's field, a
    /// move or a file; a FEN's `-` fields still pass.
    #[test]
    fn refuses_unknown_options() {
        let args =
            |args: &[&str]| -> Vec<String> { args.iter().map(|arg| (*arg).to_owned()).collect() };
        assert_eq!(
            parse(&args(&["in.jsonl", "--tablebase", "tables"])).err(),
            Some("unknown option --tablebase: see provari --help".to_owned())
        );
        let fen = "8/8/8/8/8/8/8/K6k w - - 0 1";
        let (free, _) = parse(&args(&fen.split(' ').collect::<Vec<_>>())).unwrap();
        assert_eq!(free.join(" "), fen);
    }

    /// `prove` and `convert` tell whether they met an error, for their exit
    /// codes: a bad FEN, or a proof that cannot be written or read. A game
    /// already over is no error.
    #[test]
    fn prove_and_convert_tell_their_errors() {
        let base = std::env::temp_dir().join(format!("provari-errors-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let prove = |fen: &str, proof: &Path| {
            let mut args: Vec<String> = fen.split(' ').map(str::to_owned).collect();
            args.extend(["--nodes", "100", "--proofs"].map(str::to_owned));
            args.push(proof.display().to_string());
            super::prove(&args)
        };
        let mate = "4k3/3p4/8/8/8/8/8/3RK3 w - - 0 1";
        let proof = base.join("win.jsonl");
        assert_eq!(prove(mate, &proof), Ok(true));
        assert_eq!(prove("7k/8/8/8/8/8/8/K7 w - - 0 1", &proof), Ok(true));
        assert_eq!(prove("8/8/8/8/8/8/8/K7 w - - 0 1", &proof), Ok(false));
        assert_eq!(prove(mate, &base.join("none").join("win.pvr")), Ok(false));
        let convert = |proofs: &[&Path]| {
            let mut args = vec!["--out".to_owned(), base.join("out").display().to_string()];
            args.extend(proofs.iter().map(|path| path.display().to_string()));
            super::convert(&args)
        };
        assert_eq!(convert(&[&proof]), Ok(true));
        assert_eq!(convert(&[&proof, &base.join("missing.jsonl")]), Ok(false));
        std::fs::remove_dir_all(base).unwrap();
    }

    /// Every system the tests run on tells its memory, which `--max-memory`
    /// takes half of by default.
    #[test]
    fn finds_the_machines_memory() {
        assert!(super::physical_mb().is_some_and(|mb| mb > 0));
    }

    /// `bench`'s input refuses a line without a string id, naming it, before
    /// any search starts; blank lines pass.
    #[test]
    fn bench_input_needs_string_ids() {
        let base = std::env::temp_dir().join(format!("provari-ids-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let input = base.join("in.jsonl");
        let out = base.join("out.jsonl");
        let (_, options) = parse(&["--out".to_owned(), out.display().to_string()]).unwrap();
        let free = [input.display().to_string()];
        let echo = |item: &Value| vec![json!({ "id": item["id"] })];
        let results = || std::fs::read_to_string(&out).unwrap().lines().count();
        std::fs::write(&input, "{\"id\": \"a\"}\n\n{\"id\": \"b\"}\n").unwrap();
        assert_eq!(super::each(&free, &options, "id", echo), Ok(()));
        assert_eq!(results(), 2);
        std::fs::write(&input, "{\"id\": \"c\"}\n{\"id\": 4}\n").unwrap();
        let error = super::each(&free, &options, "id", echo).unwrap_err();
        assert!(error.ends_with("line 2: no id, as a string"), "{error}");
        assert_eq!(results(), 2);
        std::fs::remove_dir_all(base).unwrap();
    }

    /// A search cut short tells the line it stopped on, with the moves open
    /// along it, where it kept it: while it reported, or for people; a
    /// refuted one does not. Shown to people, a win gives a longest line, one
    /// move for a mate in one.
    #[test]
    fn tells_the_line_a_search_stopped_on() {
        let args =
            |args: &[&str]| -> Vec<String> { args.iter().map(|arg| (*arg).to_owned()).collect() };
        let inputs = Inputs {
            tablebase: None,
            network: None,
            model: None,
        };
        let people = Some(super::text::Status {
            every: None,
            terminal: false,
        });
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let budget = ["--nodes", "5000", "--memory", "16"];
        let (_, options) = parse(&args(&budget)).unwrap();
        let quiet = run(start, None, &options, &inputs, None);
        assert!(quiet.get("line").is_none(), "{quiet}");
        // Reporting, though never in so few nodes, or for people.
        let (_, reporting) =
            parse(&args(&[&budget[..], &["--progress", "1000"]].concat())).unwrap();
        let mut shown = options.clone();
        shown.status = people.clone();
        for options in [reporting, shown] {
            let cut = run(start, None, &options, &inputs, None);
            let (line, open) = (
                cut["line"].as_array().unwrap(),
                cut["open"].as_array().unwrap(),
            );
            assert!(!line.is_empty() && line.len() == open.len(), "{cut}");
            assert_eq!(cut["nodes"], quiet["nodes"]);
        }
        let (_, mut options) = parse(&args(&["--nodes", "5000", "--most", "1"])).unwrap();
        options.status = people.clone();
        let refuted = run(start, None, &options, &inputs, None);
        assert_eq!(refuted["refuted"], true);
        assert!(refuted.get("line").is_none(), "{refuted}");
        let mate = "4k3/3p4/8/8/8/8/8/3QK3 w - - 0 1";
        let (_, mut options) = parse(&args(&["--nodes", "1000"])).unwrap();
        assert!(
            run(mate, None, &options, &inputs, None)
                .get("line")
                .is_none()
        );
        options.status = people;
        let shown = run(mate, None, &options, &inputs, None);
        assert_eq!(shown["line"], json!(["d1d7"]), "{shown}");
    }

    /// `--progress` takes seconds, 0 for none, and `--json` asks for JSON.
    #[test]
    fn reads_progress_and_json() {
        let parsed = |args: &[&str]| {
            let args: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
            parse(&args).map(|(_, options)| (options.progress, options.json))
        };
        assert_eq!(parsed(&[]), Ok((None, false)));
        assert_eq!(
            parsed(&["--progress", "0", "--json"]),
            Ok((Some(0.0), true))
        );
        assert_eq!(parsed(&["--progress", "0.5"]), Ok((Some(0.5), false)));
        for bad in ["-1", "NaN", "inf", "soon"] {
            assert!(parsed(&["--progress", bad]).is_err(), "{bad}");
        }
    }

    /// The help lists every command, in lines of at most 80 characters.
    #[test]
    fn help_lists_every_command() {
        let help = super::help();
        let commands = ["prove", "bench", "check", "line", "convert", "eval", "uci"];
        for command in commands.iter().chain(&["version"]) {
            assert!(help.contains(&format!("\n  {command} ")), "{command}");
        }
        assert!(
            help.lines().all(|line| line.chars().count() <= 80),
            "{help}"
        );
    }

    /// `--seconds` and `--nodes` both count when given; a node budget alone
    /// has no time limit; with neither, a search has the default 10 s.
    #[test]
    fn budgets_combine_as_given() {
        let limit = |args: &[&str]| {
            let args: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
            let (_, options) = parse(&args).unwrap();
            (options.time_limit(), options.nodes)
        };
        assert_eq!(limit(&[]), (Some(10.0), None));
        assert_eq!(limit(&["--seconds", "60"]), (Some(60.0), None));
        assert_eq!(limit(&["--nodes", "1000"]), (None, Some(1000)));
        assert_eq!(
            limit(&["--nodes", "1000", "--seconds", "5"]),
            (Some(5.0), Some(1000))
        );
    }

    /// The default memory follows the budget, and the proofs grow past half
    /// the table only where a long budget needs them; `--memory` is divided
    /// as given.
    #[test]
    fn memory_follows_the_budget() {
        let memory = |args: &[&str]| {
            let args: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
            let (_, options) = parse(&args).unwrap();
            options.memory()
        };
        assert_eq!(memory(&[]), (128, 64));
        assert_eq!(memory(&["--seconds", "60"]), (1024, 512));
        assert_eq!(memory(&["--seconds", "1200"]), (4096, 2048));
        assert_eq!(memory(&["--seconds", "3600"]), (4096, 5400));
        assert_eq!(memory(&["--nodes", "32000000"]), (1024, 512));
        assert_eq!(memory(&["--nodes", "1000000000"]), (4096, 2862));
        assert_eq!(
            memory(&["--seconds", "3600", "--nodes", "32000000"]),
            (1024, 512)
        );
        assert_eq!(memory(&["--memory", "12000"]), (4096, 7904));
        // Given memory, the proofs take what the budget needs, up to three
        // quarters, and at least what the table's power of two leaves.
        assert_eq!(memory(&["--memory", "1024", "--seconds", "60"]), (512, 512));
        assert_eq!(
            memory(&["--memory", "4096", "--seconds", "600"]),
            (2048, 2048)
        );
        assert_eq!(
            memory(&["--memory", "2048", "--seconds", "3600"]),
            (512, 1536)
        );
        assert_eq!(
            memory(&["--memory", "8000", "--seconds", "7200"]),
            (2000, 6000)
        );
        assert_eq!(
            memory(&["--memory", "8000", "--nodes", "1000000000"]),
            (4096, 3904)
        );
        // Threads search as many times the nodes in the time, and share the
        // memory.
        assert_eq!(memory(&["--threads", "4"]), (512, 256));
        assert_eq!(
            memory(&["--threads", "8", "--seconds", "600"]),
            (4096, 7200)
        );
        assert_eq!(
            memory(&["--threads", "8", "--nodes", "32000000"]),
            (1024, 512)
        );
    }
}
