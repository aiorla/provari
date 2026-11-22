//! `--variant antichess`: provari's commands for Antichess under Lichess's
//! rules, White's wins within a number of White moves
//! (`provari::antichess`).
//!
//! `prove FEN` searches one position and prints its result as a line of
//! JSON; `bench POSITIONS.jsonl --out RESULTS.jsonl` searches each position
//! of a JSON-lines file (its `id`, `fen` and optional `most`), `--workers` at
//! a time, and appends a result line for each, skipping ids already there.
//! Each worker keeps its memory, `--memory MB` (256 by default), of which
//! `--table MB` is the table (a quarter by default) and the rest the proofs,
//! and forgets what a search learned before the next, unless `--share` keeps
//! it until the proofs fill half their room. `--threads N` searches each
//! position with N threads that share that memory: they find the same wins
//! and refutations as one thread, by other trees, so their nodes, the win's
//! length within `--most` and its proof may differ from run to run.
//! `--most N` (100 by default, a bench line's own `most` first), `--nodes N`
//! and `--seconds S` bound each search; with neither budget a search runs
//! until it ends or its memory fills. `--network FILE` guides the search by
//! Fairy-Stockfish's Antichess network, and `--tablebases DIR` settles
//! positions of four pieces or fewer from tolius's tables, the whole set of
//! `.an0` files in `DIR`, whose unpacked blocks take at most
//! `--tablebase-memory MB` (2048 by default). `--proofs PATH` writes proofs
//! in Antichess's compact format (`antichess::format`): a file for
//! `prove`, a folder for `bench`, one `ID.apr` a position. `check FILE...`
//! plays every line of each proof to its end and prints what it proves.
//!
//! `serve` keeps one search's memory and answers each JSON line of its
//! standard input with one on its standard output, so that a driver can
//! choose each position as the last ones come back: a line
//! has the `id`, `fen` and `most` of a bench line, and may give `nodes` and
//! `seconds` for its budget and `proof` for the file of its proof.
//! `unpack-tables PACKED PLAIN` writes the tables of `PACKED` unpacked into
//! the new folder `PLAIN`, which `--tablebases` then reads in place.

use std::fs;
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

use serde_json::{Value, json};

use provari::STACK;
use provari::antichess::format;
use provari::antichess::nnue::Network;
use provari::antichess::search::{
    DEFAULT_MEMORY_MB, DEFAULT_MOST, Dfpn, limits, most, root, search_next,
};
use provari::antichess::tablebases::{self, Tablebases, WHOLE_SET};

use super::Options;

/// What a search takes from the options, once for the whole run.
struct Inputs {
    network: Option<&'static Network>,
    tablebases: Option<Tablebases>,
    /// The table's and the proofs' MB.
    memory: (usize, usize),
}

impl Inputs {
    fn open(options: &Options) -> Result<Inputs, String> {
        refuse_atomic_options(options)?;
        let network = match &options.network {
            // Loaded once, for the whole process.
            Some(path) => Some(&*Box::leak(Box::new(Network::load(path)?))),
            None => None,
        };
        let tablebases = match &options.tablebases {
            Some(dir) => {
                let tables = Tablebases::open(dir)?;
                if tables.count < WHOLE_SET {
                    return Err(format!(
                        "{} holds {} of the {WHOLE_SET} tablebases",
                        dir.display(),
                        tables.count
                    ));
                }
                if let Some(mb) = options.tablebase_memory {
                    tables.keep(mb.saturating_mul(1 << 20));
                }
                Some(tables)
            }
            None if options.tablebase_memory.is_some() => {
                return Err("--tablebase-memory needs --tablebases".into());
            }
            None => None,
        };
        let total = options.memory.unwrap_or(DEFAULT_MEMORY_MB);
        let table = options.table.unwrap_or(total / 4);
        if table == 0 || table >= total {
            return Err(format!(
                "--table {table} needs at least 1 MB, and to leave part of --memory {total} \
                 for proofs"
            ));
        }
        Ok(Inputs {
            network,
            tablebases,
            memory: (table, total - table),
        })
    }

    /// A worker's search.
    fn dfpn(&self, threads: usize) -> Dfpn {
        let (table, proofs) = self.memory;
        let mut dfpn = Dfpn::with_threads(table, proofs, threads);
        if let Some(network) = self.network {
            dfpn.set_network(network);
        }
        if let Some(tables) = self.tablebases {
            dfpn.set_tablebases(tables);
        }
        dfpn
    }

    fn total_mb(&self) -> usize {
        self.memory.0 + self.memory.1
    }
}

/// Refuses the options that only Atomic's search reads, rather than ignore
/// them.
fn refuse_atomic_options(options: &Options) -> Result<(), String> {
    let given = [
        ("--attacker", options.attacker.is_some()),
        ("--search", options.search != "dfpn"),
        ("--seed", options.seed != 0),
        ("--model", options.model.is_some()),
        ("--guide", options.guide.is_some()),
        ("--shorten", options.shorten),
        ("--progress", options.progress.is_some()),
        ("--drop-mates-at", options.drop_mates_at.is_some()),
        ("--pgn", options.pgn.is_some()),
        ("--until", options.until.is_some()),
    ];
    match given.iter().find(|(_, given)| *given) {
        Some((name, _)) => Err(format!("{name} is for Atomic, not --variant antichess")),
        None => Ok(()),
    }
}

/// A position's bound on White's moves: its own, else `--most`'s, else 100.
fn bound(own: &Value, options: &Options) -> Result<u32, String> {
    match own {
        Value::Null => most(options.most.unwrap_or(DEFAULT_MOST)),
        raw => raw
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| format!("most {raw} is not a number of moves"))
            .and_then(most),
    }
}

/// `provari prove FEN --variant antichess`: whether it printed a result.
pub fn prove(fen: &str, options: &Options) -> Result<bool, String> {
    if options.workers > 1 || options.out.is_some() {
        return Err("--workers and --out are for bench".into());
    }
    if options.share {
        return Err("--share is for bench".into());
    }
    let inputs = Inputs::open(options)?;
    options.check_memory_mb(inputs.total_mb(), 1)?;
    let pos = root(fen)?;
    let most = bound(&Value::Null, options)?;
    let result = std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(STACK)
            .spawn_scoped(scope, || -> Result<Value, String> {
                let mut dfpn = inputs.dfpn(options.threads);
                let (result, proof) = search_next(&mut dfpn, &pos, most, false, || {
                    limits(options.nodes, options.seconds)
                });
                if let (Some(path), Some(proof)) = (&options.proofs, &proof) {
                    format::save(path, &proof.bytes)?;
                }
                Ok(result)
            })
            .expect("cannot start the search")
            .join()
            .expect("the search failed")
    })?;
    println!("{result}");
    Ok(result.get("error").is_none())
}

/// `provari bench POSITIONS.jsonl --variant antichess --out RESULTS.jsonl`.
pub fn bench(free: &[String], options: &Options) -> Result<(), String> {
    let inputs = Inputs::open(options)?;
    options.check_memory_mb(inputs.total_mb(), options.workers)?;
    if let Some(dir) = &options.proofs {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    // Each worker's search, its memory had once.
    let start = || None;
    super::each_with(
        free,
        options,
        "id",
        start,
        |kept: &mut Option<Dfpn>, position| {
            let id = position["id"].as_str().unwrap_or_default();
            let searched = (|| -> Result<Value, String> {
                let fen = position["fen"].as_str().ok_or("no fen")?;
                let pos = root(fen)?;
                let most = bound(&position["most"], options)?;
                let dfpn = kept.get_or_insert_with(|| inputs.dfpn(options.threads));
                let (mut result, proof) = search_next(dfpn, &pos, most, options.share, || {
                    limits(options.nodes, options.seconds)
                });
                if let (Some(dir), Some(proof)) = (&options.proofs, &proof) {
                    let path = dir.join(format!("{id}.{}", format::EXTENSION));
                    format::save(&path, &proof.bytes)?;
                    result["proof"] = json!(path.display().to_string());
                }
                Ok(result)
            })();
            let mut result = searched.unwrap_or_else(|error| json!({ "error": error }));
            result["id"] = json!(id);
            result["variant"] = json!("antichess");
            vec![result]
        },
    )
}

/// `provari check --variant antichess FILE...`: plays every line of each
/// proof to its end, printing a line of JSON for each, and returns whether
/// every one holds.
pub fn check(args: &[String]) -> Result<bool, String> {
    if args.is_empty() {
        return Err("usage: provari check --variant antichess PROOF.apr...".into());
    }
    if let Some(option) = args.iter().find(|arg| arg.starts_with("--")) {
        return Err(format!(
            "{option} is not an option of check --variant antichess"
        ));
    }
    let mut every = true;
    for path in args {
        let path = Path::new(path);
        let line = std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(STACK)
                .spawn_scoped(scope, || check_one(path))
                .expect("cannot start the check")
                .join()
                .expect("the check failed")
        });
        every &= line.get("error").is_none();
        println!("{line}");
    }
    Ok(every)
}

/// What one proof proves, or why it fails.
fn check_one(path: &Path) -> Value {
    let name = path.display().to_string();
    let checked = (|| -> Result<Value, String> {
        let bytes = fs::read(path).map_err(|error| format!("cannot read it: {error}"))?;
        let (header, indices) = format::decode(&bytes)?;
        let steps = format::steps(&header, &indices)?;
        let pieces = |fen: &str| {
            let board = fen.split(' ').next().unwrap_or_default();
            board.bytes().filter(u8::is_ascii_alphabetic).count()
        };
        let mut by_pieces = vec![0u64; pieces(&header.root) + 1];
        for step in &steps {
            by_pieces[pieces(&step.fen)] += 1;
        }
        Ok(json!({
            "root": header.root,
            "moves": header.moves,
            "white_positions": steps.len(),
            "white_positions_by_pieces": by_pieces,
        }))
    })();
    let mut line = checked.unwrap_or_else(|error| json!({ "error": error }));
    line["proof"] = json!(name);
    line
}

/// `provari serve --variant antichess`: answers each line of the standard
/// input (see the module's notes) until it ends.
pub fn serve(free: &[String], options: &Options) -> Result<(), String> {
    if !free.is_empty() || options.workers > 1 || options.out.is_some() || options.proofs.is_some()
    {
        return Err("serve takes its positions and files from its input".into());
    }
    let inputs = Inputs::open(options)?;
    options.check_memory_mb(inputs.total_mb(), 1)?;
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(STACK)
            .spawn_scoped(scope, || -> Result<(), String> {
                let mut dfpn = inputs.dfpn(options.threads);
                let mut out = std::io::stdout().lock();
                for line in std::io::stdin().lock().lines() {
                    let line = line.map_err(|error| format!("cannot read a request: {error}"))?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let answer = serde_json::from_str::<Value>(&line)
                        .map_err(|error| format!("not a request: {error}"))
                        .and_then(|request| answer(&mut dfpn, &request, options))
                        .unwrap_or_else(|error| json!({ "error": error }));
                    writeln!(out, "{answer}")
                        .and_then(|()| out.flush())
                        .map_err(|error| format!("cannot answer: {error}"))?;
                }
                Ok(())
            })
            .expect("cannot start the search")
            .join()
            .expect("the search failed")
    })
}

/// What `serve` answers to one request.
fn answer(dfpn: &mut Dfpn, request: &Value, options: &Options) -> Result<Value, String> {
    let id = request["id"].as_str().ok_or("a search needs an id")?;
    let fen = request["fen"].as_str().ok_or("a search needs a fen")?;
    let pos = root(fen)?;
    let most = bound(&request["most"], options)?;
    let nodes = match &request["nodes"] {
        Value::Null => options.nodes,
        raw => Some(
            raw.as_u64()
                .ok_or_else(|| format!("nodes {raw} is not a number of nodes"))?,
        ),
    };
    let seconds = match &request["seconds"] {
        Value::Null => options.seconds,
        raw => Some(
            raw.as_f64()
                .filter(|seconds| *seconds >= 0.0)
                .ok_or_else(|| format!("seconds {raw} is not a number of seconds"))?,
        ),
    };
    let (mut result, proof) =
        search_next(dfpn, &pos, most, options.share, || limits(nodes, seconds));
    result["id"] = json!(id);
    if let (Some(path), Some(proof)) = (request["proof"].as_str(), &proof) {
        format::save(Path::new(path), &proof.bytes)?;
        result["proof"] = json!(path);
    }
    Ok(result)
}

/// `provari unpack-tables --variant antichess PACKED PLAIN`: unpacks the
/// tables of `PACKED` into `PLAIN`, then opens them there, so that the reader
/// has taken every one.
pub fn unpack_tables(free: &[String]) -> Result<(), String> {
    let [packed, plain] = free else {
        return Err("usage: provari unpack-tables --variant antichess PACKED PLAIN".into());
    };
    let (packed, plain) = (Path::new(packed), Path::new(plain));
    let started = Instant::now();
    let unpacked = tablebases::unpack(packed, plain)?;
    let tables = Tablebases::open(plain)?;
    if tables.count != unpacked.tables {
        return Err(format!(
            "{} holds {} tables, not the {} unpacked",
            plain.display(),
            tables.count,
            unpacked.tables
        ));
    }
    println!(
        "{}",
        json!({
            "tables": unpacked.tables,
            "bytes": unpacked.bytes,
            "ms": started.elapsed().as_millis() as u64,
        })
    );
    Ok(())
}
