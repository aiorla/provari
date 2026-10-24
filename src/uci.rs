//! `provari uci`: provari as an engine of the Universal Chess Interface, for
//! chess GUIs and tools. It reads UCI's commands from standard input and
//! answers on standard output: `uci`, `isready`, `setoption`, `ucinewgame`,
//! `position`, `go`, `stop`, `ponderhit` and `quit`. Each `go` runs one search
//! as `provari prove` does, in a thread of its own so that `isready` and
//! `stop` are answered meanwhile: `go mate N` is `--most N`, `nodes` and
//! `movetime` or the clock budget it, and `infinite`, or no limit, searches
//! until it settles the position or is stopped. A win is reported as engines
//! report mates, with a longest line of its proof as the principal variation
//! and its first move as the best move; without one the best move is the null
//! move, 0000, and an `info string` says why.

use std::io::Write;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::Value;
use shakmaty::{Color, Move, Position as _, uci::UciMove, variant::Atomic};

use provari::{
    STACK, board, dfpn::Dfpn, ending, format, lines, position, tablebase::Tablebase, walk,
};

use super::{Inputs, Options, parse, physical_mb};

/// Each search's memory in MB, its table and its proofs, unless `--memory` or
/// `Hash` gives another.
const HASH: usize = 1024;
/// The most memory `Hash` may give, in MB.
const MOST_HASH: usize = 1 << 20;
/// The time limit of a search without one: a year, as `provari prove` gives
/// a search under a node budget alone. It also gives the proofs the most
/// memory they may take (`dfpn::split`).
const NO_LIMIT: f64 = 3.2e7;
/// The moves a clock is shared over when `go` gives no `movestogo`.
const MOVES_TO_GO: u64 = 30;
/// A tablebase win's score, less the plies to it, as Stockfish reports one.
const TABLEBASE_WIN: i64 = 20000;
/// UCI's commands: unknown words before one are skipped.
const COMMANDS: [&str; 11] = [
    "uci",
    "debug",
    "isready",
    "setoption",
    "register",
    "ucinewgame",
    "position",
    "go",
    "stop",
    "ponderhit",
    "quit",
];

/// Where the engine's lines go: standard output, or a test's record.
type Out = Arc<dyn Fn(&str) + Send + Sync>;

/// `provari uci [OPTIONS]`: the options choose the guide, and give `Hash`
/// (`--memory`), `SyzygyPath` (`--tablebases`), `Attacker` (`--attacker`) and
/// `Shorten` (`--shorten`) their defaults; `--progress S` is how often a
/// search prints its progress, every second by default. Each search's budget
/// comes from its `go`.
pub fn run(args: &[String]) -> Result<(), String> {
    let (free, options) = parse(args)?;
    if !free.is_empty() {
        return Err("usage: provari uci [OPTIONS]".into());
    }
    if options.seconds.is_some() || options.nodes.is_some() || options.most.is_some() {
        return Err("provari uci takes each search's budget from go".into());
    }
    if options.search != "dfpn" || options.proofs.is_some() || options.out.is_some() {
        return Err("provari uci searches with df-pn and writes no files".into());
    }
    let inputs = Inputs::open(&options, true)?;
    let out: Out = Arc::new(|line: &str| {
        let mut stdout = std::io::stdout().lock();
        // A GUI that has gone reads nothing more, and its commands end.
        let _ = writeln!(stdout, "{line}").and_then(|()| stdout.flush());
    });
    Engine::new(options, inputs, out).serve(std::io::stdin().lines().map_while(Result::ok));
    Ok(())
}

/// The engine between commands: its options, the position set up, and the
/// search running, if any.
struct Engine {
    /// The command line's options, with `setoption`'s, which each `go`
    /// completes with its budget.
    options: Options,
    inputs: Arc<Inputs>,
    /// The tablebases' folders, as `SyzygyPath` gives them.
    syzygy: String,
    /// The position to search, or why the last `position` set up none.
    root: Result<Atomic, String>,
    running: Option<Running>,
    out: Out,
}

/// A search in its thread: what stops it, and what holds its best move back.
struct Running {
    thread: JoinHandle<()>,
    stop: Arc<AtomicBool>,
    hold: Arc<Hold>,
}

/// Holds a search's best move back while set, as UCI asks after `go infinite`
/// and `go ponder`, until `stop` or `ponderhit`.
struct Hold {
    held: Mutex<bool>,
    released: Condvar,
}

impl Hold {
    fn new(held: bool) -> Hold {
        Hold {
            held: Mutex::new(held),
            released: Condvar::new(),
        }
    }

    fn release(&self) {
        *self.held.lock().unwrap() = false;
        self.released.notify_all();
    }

    fn is_held(&self) -> bool {
        *self.held.lock().unwrap()
    }

    fn wait(&self) {
        let held = self.held.lock().unwrap();
        drop(self.released.wait_while(held, |held| *held).unwrap());
    }
}

impl Engine {
    fn new(mut options: Options, inputs: Inputs, out: Out) -> Engine {
        options.memory = Some(options.memory.unwrap_or(HASH));
        let syzygy = options
            .tablebases
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default();
        Engine {
            options,
            inputs: Arc::new(inputs),
            syzygy,
            root: Ok(Atomic::default()),
            running: None,
            out,
        }
    }

    /// Answers each command until `quit`, or until the commands end, as
    /// piped ones do, and the search running with them.
    fn serve(mut self, commands: impl Iterator<Item = String>) {
        for line in commands {
            if !self.command(&line) {
                self.stop();
                return;
            }
        }
        self.finish();
    }

    /// Answers a command; false for `quit`.
    fn command(&mut self, line: &str) -> bool {
        let words: Vec<&str> = line.split_whitespace().collect();
        let Some(at) = words.iter().position(|word| COMMANDS.contains(word)) else {
            return true;
        };
        let args = &words[at + 1..];
        match words[at] {
            "uci" => self.identify(),
            "isready" => (self.out)("readyok"),
            "setoption" => self.set_option(args),
            "position" => self.root = set_up(args),
            "go" => self.go(args),
            "stop" => self.stop(),
            "ponderhit" => {
                if let Some(running) = &self.running {
                    running.hold.release();
                }
            }
            "quit" => return false,
            // `debug`, `register`, and `ucinewgame`: every search starts from
            // nothing.
            _ => {}
        }
        true
    }

    fn say(&self, text: &str) {
        (self.out)(&format!("info string {text}"));
    }

    /// The engine's name and options, as `uci` asks.
    fn identify(&self) {
        let attacker = match self.options.attacker {
            None => "Mover",
            Some(Color::White) => "White",
            Some(Color::Black) => "Black",
        };
        let syzygy = if self.syzygy.is_empty() {
            "<empty>"
        } else {
            &self.syzygy
        };
        let hash = self.options.memory.unwrap_or(HASH);
        for line in [
            format!("id name provari {}", env!("CARGO_PKG_VERSION")),
            "id author aiorla".to_owned(),
            format!("option name Hash type spin default {hash} min 1 max {MOST_HASH}"),
            format!("option name SyzygyPath type string default {syzygy}"),
            format!(
                "option name Attacker type combo default {attacker} var Mover var White var Black"
            ),
            format!(
                "option name Shorten type check default {}",
                self.options.shorten
            ),
            "option name UCI_Variant type combo default atomic var atomic".to_owned(),
            "uciok".to_owned(),
        ] {
            (self.out)(&line);
        }
    }

    /// `setoption name NAME [value VALUE]`, the name in any case.
    fn set_option(&mut self, args: &[&str]) {
        let Some((&"name", rest)) = args.split_first() else {
            return;
        };
        let at = rest
            .iter()
            .position(|word| *word == "value")
            .unwrap_or(rest.len());
        let name = rest[..at].join(" ");
        let value = rest.get(at + 1..).unwrap_or_default().join(" ");
        match name.to_ascii_lowercase().as_str() {
            "hash" => match value.parse() {
                Ok(mb) if (1..=MOST_HASH).contains(&mb) => self.options.memory = Some(mb),
                _ => self.say(&format!("Hash is from 1 to {MOST_HASH} MB, not {value}")),
            },
            "syzygypath" => self.open_tables(&value),
            "attacker" => match value.to_ascii_lowercase().as_str() {
                "mover" => self.options.attacker = None,
                "white" => self.options.attacker = Some(Color::White),
                "black" => self.options.attacker = Some(Color::Black),
                _ => self.say(&format!(
                    "the attacker is Mover, White or Black, not {value}"
                )),
            },
            "shorten" => match value.to_ascii_lowercase().as_str() {
                "true" => self.options.shorten = true,
                "false" => self.options.shorten = false,
                _ => self.say(&format!("Shorten is true or false, not {value}")),
            },
            "uci_variant" if !value.eq_ignore_ascii_case("atomic") => {
                self.say("provari proves Atomic wins only")
            }
            "uci_variant" => {}
            _ => self.say(&format!("provari has no option {name}")),
        }
    }

    /// Opens the tables `SyzygyPath` names, folders separated as `PATH`
    /// separates them, or none for `<empty>`.
    fn open_tables(&mut self, value: &str) {
        let path = if value == "<empty>" { "" } else { value };
        let tables = match path {
            "" => Ok(None),
            path => Tablebase::open(Path::new(path)).map(Some),
        };
        self.finish();
        // A search lets go of the inputs before its best move.
        let Some(inputs) = Arc::get_mut(&mut self.inputs) else {
            self.say("the tablebases cannot change during a search");
            return;
        };
        match tables {
            Ok(tables) => {
                let pieces = tables.as_ref().map(Tablebase::max_pieces);
                inputs.tablebase = tables;
                self.syzygy = path.to_owned();
                if let Some(pieces) = pieces {
                    self.say(&format!("tablebases of up to {pieces} pieces"));
                }
            }
            Err(error) => {
                inputs.tablebase = None;
                self.syzygy.clear();
                self.say(&format!("tablebases: {error}"));
            }
        }
    }

    /// Starts a search of the position set up, with `go`'s limits.
    fn go(&mut self, args: &[&str]) {
        let started = Instant::now();
        self.finish();
        let limits = Limits::read(args);
        let turn = self.root.as_ref().map_or(Color::White, |root| root.turn());
        let seconds = limits.seconds(turn).unwrap_or(NO_LIMIT);
        let mut options = self.options.clone();
        options.seconds = Some(seconds);
        options.nodes = limits.nodes;
        options.most = limits.mate;
        options.progress = None;
        let job = Job {
            root: self.root.clone(),
            options,
            started,
            deadline: started + Duration::from_secs_f64(seconds),
            exact: limits.mate.is_some(),
            every: Duration::from_secs_f64(self.options.progress.unwrap_or(1.0)),
        };
        let stop = Arc::new(AtomicBool::new(false));
        let hold = Arc::new(Hold::new(limits.infinite || limits.ponder));
        let (inputs, out) = (self.inputs.clone(), self.out.clone());
        let (stopped, held) = (stop.clone(), hold.clone());
        let thread = std::thread::Builder::new()
            .stack_size(STACK)
            .spawn(move || {
                let best =
                    panic::catch_unwind(AssertUnwindSafe(|| job.run(&inputs, &stopped, &out)))
                        .unwrap_or_else(|_| {
                            out("info string the search failed");
                            None
                        });
                // The inputs are free again before the GUI hears the search
                // has ended.
                drop(inputs);
                held.wait();
                out(&format!("bestmove {}", best.map_or("0000".to_owned(), uci)));
            });
        match thread {
            Ok(thread) => self.running = Some(Running { thread, stop, hold }),
            Err(error) => {
                self.say(&format!("cannot start the search: {error}"));
                (self.out)("bestmove 0000");
            }
        }
    }

    /// Waits for the search running to end, as a GUI does before its next
    /// search, but stops one held for `stop`, which would not end.
    fn finish(&mut self) {
        if self
            .running
            .as_ref()
            .is_some_and(|running| running.hold.is_held())
        {
            self.stop();
        } else if let Some(running) = self.running.take() {
            let _ = running.thread.join();
        }
    }

    /// Stops the search running, if any, which then gives its best move.
    fn stop(&mut self) {
        if let Some(running) = self.running.take() {
            running.stop.store(true, Ordering::Relaxed);
            running.hold.release();
            let _ = running.thread.join();
        }
    }
}

/// What `go` asks for: times in milliseconds.
#[derive(Default)]
struct Limits {
    nodes: Option<u64>,
    mate: Option<u32>,
    movetime: Option<u64>,
    wtime: Option<i64>,
    btime: Option<i64>,
    winc: Option<i64>,
    binc: Option<i64>,
    movestogo: Option<u64>,
    infinite: bool,
    ponder: bool,
}

impl Limits {
    /// `go`'s words; `depth` and `searchmoves`, which a proof-number search
    /// has no use for, and unknown words are skipped.
    fn read(args: &[&str]) -> Limits {
        let mut limits = Limits::default();
        let mut words = args.iter();
        while let Some(&word) = words.next() {
            let mut number = || words.next().and_then(|word| word.parse::<i64>().ok());
            match word {
                "nodes" => limits.nodes = number().map(|n| n.max(0) as u64),
                "mate" => limits.mate = number().map(|n| n.clamp(0, u32::MAX.into()) as u32),
                "movetime" => limits.movetime = number().map(|n| n.max(0) as u64),
                "wtime" => limits.wtime = number(),
                "btime" => limits.btime = number(),
                "winc" => limits.winc = number(),
                "binc" => limits.binc = number(),
                "movestogo" => limits.movestogo = number().map(|n| n.max(1) as u64),
                "infinite" => limits.infinite = true,
                "ponder" => limits.ponder = true,
                _ => {}
            }
        }
        limits
    }

    /// The seconds a search may take: `movetime`, else a share of the clock
    /// of `turn`, the side to move: its time over the moves to go and three
    /// quarters of its increment, at most nine tenths of its time. None
    /// without either.
    fn seconds(&self, turn: Color) -> Option<f64> {
        if let Some(ms) = self.movetime {
            return Some(ms as f64 / 1000.0);
        }
        let (time, increment) = match turn {
            Color::White => (self.wtime, self.winc),
            Color::Black => (self.btime, self.binc),
        };
        let time = time?.max(0) as f64;
        let increment = increment.unwrap_or(0).max(0) as f64;
        let moves = self.movestogo.unwrap_or(MOVES_TO_GO) as f64;
        Some((time / moves + increment * 0.75).min(time * 0.9) / 1000.0)
    }
}

/// The position a `position` command sets up: `startpos` or `fen FEN`, then
/// any `moves`, in UCI's notation.
fn set_up(args: &[&str]) -> Result<Atomic, String> {
    let at = args
        .iter()
        .position(|word| *word == "moves")
        .unwrap_or(args.len());
    let mut pos = match &args[..at] {
        ["startpos", ..] => Atomic::default(),
        ["fen", fen @ ..] => position(&fen.join(" "))?,
        _ => return Err("position takes startpos or fen FEN".into()),
    };
    for word in args.get(at + 1..).unwrap_or_default() {
        let m = word
            .parse::<UciMove>()
            .ok()
            .and_then(|m| m.to_move(&pos).ok())
            .ok_or(format!("{word} is not a legal move here"))?;
        pos.play_unchecked(m);
    }
    Ok(pos)
}

fn uci(m: Move) -> String {
    UciMove::from_standard(m).to_string()
}

/// A `go`: the position, the options with its budget, when it started and
/// must stop, and how it reports.
struct Job {
    root: Result<Atomic, String>,
    options: Options,
    started: Instant,
    deadline: Instant,
    /// Under `go mate`, where the search plays tablebase wins out, its proofs
    /// need no tables, and their lengths are exact.
    exact: bool,
    /// How often the search prints its progress.
    every: Duration,
}

impl Job {
    /// Searches and prints what it finds; returns the best move, if any.
    fn run(&self, inputs: &Inputs, stop: &AtomicBool, out: &Out) -> Option<Move> {
        let say = |text: &str| out(&format!("info string {text}"));
        let root = match &self.root {
            Ok(root) => root,
            Err(error) => {
                say(error);
                return None;
            }
        };
        if let Some((winner, reason)) = ending(root) {
            // As Stockfish reports a game already over: lost, or drawn.
            let score = if winner.is_some() { "mate 0" } else { "cp 0" };
            out(&format!("info depth 0 score {score}"));
            say(&format!("the game is over: {reason}"));
            return None;
        }
        let memory = self.options.memory_mb();
        let allowed = self
            .options
            .max_memory
            .or_else(|| physical_mb().map(|mb| mb / 2));
        if let Some(allowed) = allowed
            && memory > allowed
        {
            say(&format!(
                "a search of {memory} MB exceeds the {allowed} MB allowed: give less Hash, \
                 or more --max-memory"
            ));
            return None;
        }
        let attacker = self.options.attacker.unwrap_or(root.turn());
        let mut search = match inputs.try_search(root, attacker, &self.options) {
            Ok(search) => search,
            Err(error) => {
                say(&format!("the search's memory: {error}"));
                return None;
            }
        };
        // The GUI's clock runs from `go`, before the memory is set up.
        search.set_deadline(self.deadline);
        search.set_stop(stop);
        let (progress_out, started) = (out.clone(), self.started);
        search.set_progress_sink(
            self.every,
            Box::new(move |line| progress_out(&progress(line, started))),
        );
        let report = Report {
            root,
            attacker,
            tables: inputs.tablebase.as_ref(),
            exact: self.exact,
            started,
            out,
        };
        let found = search.solve(root);
        let Some(length) = found else {
            report.end(&search);
            say(&report.no_win(&search, self.options.most));
            return None;
        };
        let mut best = report.win(&search);
        if self.options.shorten {
            let shortening = search.shorten(root, length, |search, _| {
                if let Ok(win) = report.describe(search)
                    && best.as_ref().is_none_or(|best| win.moves < best.moves)
                {
                    report.print(&win, search);
                    best = Some(win);
                }
            });
            report.end(&search);
            if shortening.shortest {
                say("no shorter win exists");
            }
        }
        best.map(|win| win.line[0])
    }
}

/// A progress line, as UCI's info: the time since `go`, the nodes and their
/// rate, and how full the proofs are, in thousandths.
fn progress(line: &Value, started: Instant) -> String {
    let ms = (started.elapsed().as_millis() as u64).max(1);
    let nodes = line["nodes"].as_u64().unwrap_or_default();
    let full = (line["proofs"].as_f64().unwrap_or_default() * 1000.0).round();
    format!(
        "info time {ms} nodes {nodes} nps {} hashfull {}",
        nodes * 1000 / ms,
        full.min(1000.0) as u64
    )
}

/// How a search's wins are reported.
struct Report<'a> {
    root: &'a Atomic,
    attacker: Color,
    tables: Option<&'a Tablebase>,
    exact: bool,
    started: Instant,
    out: &'a Out,
}

/// A win as reported: its length in attacker moves, whether lines end at
/// tablebase verdicts, and a longest line, never empty.
struct Win {
    moves: u32,
    tablebases: bool,
    line: Vec<Move>,
}

impl Report<'_> {
    /// The win of a search that has found one, as its proof is written: a
    /// longest line against its moves.
    fn describe(&self, search: &Dfpn) -> Result<Win, String> {
        let tables = if self.exact { None } else { self.tables };
        let (header, indices) = format::walk(self.root, self.attacker, tables, |pos| {
            search
                .winning_move(&pos.to_atomic(), &(pos.hash(), pos.halfmoves()))
                .and_then(|m| format::board_move(pos, &m))
        })?;
        let (_, _, moves) = format::strategy(&header, &indices, tables)?;
        let (mut line, _) =
            lines::Proof::new(self.attacker, moves).longest_line(self.root, tables)?;
        if line.is_empty() {
            // The tables decide the root itself: the search's move, or for
            // the defender, who loses whatever it plays, any.
            line.extend(if self.root.turn() == self.attacker {
                let pos = board::Position::from_atomic(self.root);
                walk::winning_capture(self.root, self.attacker)
                    .or_else(|| search.winning_move(self.root, &(pos.hash(), pos.halfmoves())))
            } else {
                self.root.legal_moves().first().copied()
            });
        }
        if line.is_empty() {
            return Err("the proof gives no move at its root".into());
        }
        Ok(Win {
            moves: header.moves,
            tablebases: header.tablebases > 0,
            line,
        })
    }

    /// Reports the first win a search found.
    fn win(&self, search: &Dfpn) -> Option<Win> {
        match self.describe(search) {
            Ok(win) => {
                self.print(&win, search);
                Some(win)
            }
            Err(error) => {
                (self.out)(&format!("info string {error}"));
                None
            }
        }
    }

    /// A win's line: its score from the side to move's view, as a mate in
    /// moves, or with lines ending at tablebase verdicts, which give no exact
    /// length, as a tablebase win.
    fn print(&self, win: &Win, search: &Dfpn) {
        let plies = win.line.len();
        let sign = if self.root.turn() == self.attacker {
            1
        } else {
            -1
        };
        let score = if win.tablebases {
            format!("cp {}", sign * (TABLEBASE_WIN - plies as i64))
        } else {
            format!("mate {}", sign * i64::from(win.moves))
        };
        let ms = (self.started.elapsed().as_millis() as u64).max(1);
        let pv: Vec<String> = win.line.iter().map(|&m| uci(m)).collect();
        (self.out)(&format!(
            "info depth {plies} score {score} nodes {} nps {} time {ms} pv {}",
            search.nodes,
            search.nodes * 1000 / ms,
            pv.join(" ")
        ));
        if win.tablebases {
            let within = match win.moves {
                0 => "the position is a tablebase win".to_owned(),
                1 => "within 1 move every line wins or reaches a tablebase win".to_owned(),
                moves => format!("within {moves} moves every line wins or reaches a tablebase win"),
            };
            (self.out)(&format!("info string {within}"));
        }
    }

    /// A search's last line, when it ends without a win or after shortening
    /// one.
    fn end(&self, search: &Dfpn) {
        let ms = (self.started.elapsed().as_millis() as u64).max(1);
        (self.out)(&format!(
            "info time {ms} nodes {} nps {}",
            search.nodes,
            search.nodes * 1000 / ms
        ));
    }

    /// Why a search ended without a win.
    fn no_win(&self, search: &Dfpn, most: Option<u32>) -> String {
        let side = if self.attacker.is_white() {
            "White"
        } else {
            "Black"
        };
        if search.proofs_full() {
            "the proofs filled their memory: give more Hash".to_owned()
        } else if search
            .root_numbers
            .is_some_and(|(_, disproof)| disproof == 0)
        {
            match most {
                Some(1) => format!("refuted: {side} has no win in 1 move"),
                Some(most) => format!("refuted: {side} has no win in at most {most} moves"),
                None => format!("refuted: {side} has no win"),
            }
        } else {
            format!("no win found for {side}")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, mpsc};
    use std::thread;
    use std::time::{Duration, Instant};

    use provari::dfpn::{Dfpn, Guide, split};
    use serde_json::json;
    use shakmaty::{Color, Move, Position as _, uci::UciMove};

    use super::{Engine, Limits, Out, Report, Win};
    use crate::{Inputs, parse};

    /// A position whose first win, in 8 moves after 1,243 nodes with the
    /// hand-crafted guide, shortens to 7 after 3,304.
    const SHORTENS: &str = "r1b1kb1r/pp1p1qp1/2n1p3/1B3P1p/3PP3/2P5/PP4PP/R2QK2R w KQkq - 0 1";

    fn inputs() -> Inputs {
        Inputs {
            tablebase: None,
            network: None,
            model: None,
        }
    }

    /// An engine guided by the hand-crafted evaluation, 16 MB a search, and
    /// the lines it writes.
    fn engine() -> (Engine, Arc<Mutex<Vec<String>>>) {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let record = lines.clone();
        let out: Out = Arc::new(move |line: &str| record.lock().unwrap().push(line.to_owned()));
        let (_, options) = parse(&["--memory".to_owned(), "16".to_owned()]).unwrap();
        (Engine::new(options, inputs(), out), lines)
    }

    /// What the engine writes for these commands, once they end, without
    /// the times and speeds, which vary.
    fn session(commands: &[&str]) -> Vec<String> {
        let (engine, lines) = engine();
        engine.serve(commands.iter().map(|command| (*command).to_owned()));
        let lines = lines.lock().unwrap();
        lines.iter().map(|line| steady(line)).collect()
    }

    fn steady(line: &str) -> String {
        let mut words = line.split(' ');
        let mut kept = Vec::new();
        while let Some(word) = words.next() {
            if line.starts_with("info ") && matches!(word, "time" | "nps") {
                words.next();
            } else {
                kept.push(word);
            }
        }
        kept.join(" ")
    }

    #[test]
    fn names_itself_and_its_options() {
        let lines = session(&["uci", "setoption name hash value 32", "uci", "isready"]);
        let version = env!("CARGO_PKG_VERSION");
        assert_eq!(lines[0], format!("id name provari {version}"));
        for option in [
            "option name Hash type spin default 16 min 1 max 1048576",
            "option name Hash type spin default 32 min 1 max 1048576",
            "option name SyzygyPath type string default <empty>",
            "option name Attacker type combo default Mover var Mover var White var Black",
            "option name Shorten type check default false",
            "option name UCI_Variant type combo default atomic var atomic",
        ] {
            assert!(lines.iter().any(|line| line == option), "{option}");
        }
        assert_eq!(lines[lines.len() - 2..], ["uciok", "readyok"]);
    }

    /// A win's score counts moves from the side to move's view, negative for
    /// the defender, and its line is the proof's longest; unknown words
    /// before a command are skipped.
    #[test]
    fn reports_mates_from_the_side_to_move() {
        let lines = session(&[
            "position fen 4k3/3p4/8/8/8/8/8/3QK3 w - - 0 1",
            "go",
            "position fen rnbq1b1r/pp1pk2p/2pBppp1/8/3PP3/2N5/PPP2P1P/R2QKB1R b KQ - 1 1",
            "setoption name Attacker value white",
            "joho go nodes 1000",
        ]);
        assert_eq!(
            lines,
            [
                "info depth 1 score mate 1 nodes 1 pv d1d7",
                "bestmove d1d7",
                "info depth 2 score mate -1 nodes 1 pv e7f7 d6f8",
                "bestmove e7f7",
            ]
        );
    }

    /// A win whose lines may end at tablebase verdicts scores as Stockfish
    /// scores tablebase wins, from the side to move's view, and an `info
    /// string` gives the moves within which every line wins or reaches one.
    #[test]
    fn reports_tablebase_wins() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let record = lines.clone();
        let out: Out = Arc::new(move |line: &str| record.lock().unwrap().push(line.to_owned()));
        let print = |fen: &str, moves: u32, line: &str| {
            let root = provari::position(fen).unwrap();
            let mut pos = root.clone();
            let line: Vec<Move> = line
                .split(' ')
                .map(|word| {
                    let m = word.parse::<UciMove>().unwrap().to_move(&pos).unwrap();
                    pos.play_unchecked(m);
                    m
                })
                .collect();
            let report = Report {
                root: &root,
                attacker: Color::White,
                tables: None,
                exact: false,
                started: Instant::now(),
                out: &out,
            };
            let search = Dfpn::new(&root, None, Guide::Hand, Instant::now(), split(1, 0));
            let win = Win {
                moves,
                tablebases: true,
                line,
            };
            report.print(&win, &search);
        };
        print(
            "4k3/7p/8/8/8/8/2QQ4/4K3 b - - 0 1",
            2,
            "h7h5 d2d7 e8f8 c2c8",
        );
        print("4k3/7p/8/8/8/8/2QQ4/4K3 w - - 0 1", 1, "c2h7");
        print("4k3/8/8/8/8/8/2QQ4/4K3 w - - 0 1", 0, "d2d7");
        let lines: Vec<String> = lines
            .lock()
            .unwrap()
            .iter()
            .map(|line| steady(line))
            .collect();
        assert_eq!(
            lines,
            [
                "info depth 4 score cp -19996 nodes 0 pv h7h5 d2d7 e8f8 c2c8",
                "info string within 2 moves every line wins or reaches a tablebase win",
                "info depth 1 score cp 19999 nodes 0 pv c2h7",
                "info string within 1 move every line wins or reaches a tablebase win",
                "info depth 1 score cp 19999 nodes 0 pv d2d7",
                "info string the position is a tablebase win",
            ]
        );
    }

    /// Without a win the best move is the null move, and an `info string`
    /// says why; a position over or not set up is not searched.
    #[test]
    fn says_why_it_has_no_move() {
        let lines = session(&[
            "position startpos",
            "go mate 1",
            "go nodes 2048",
            "position fen rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQ - 0 1",
            "go",
            "position fen k7/8/1Q6/8/8/8/8/6K1 b - - 0 1",
            "go",
            "position startpos moves e2e4 e7e5 e1e3",
            "go",
            "position fen 4k3/8/8/8/8/8/8/4K3 x - - 0 1",
            "go",
        ]);
        assert_eq!(
            lines,
            [
                "info nodes 1",
                "info string refuted: White has no win in 1 move",
                "bestmove 0000",
                "info nodes 2048",
                "info string no win found for White",
                "bestmove 0000",
                "info depth 0 score mate 0",
                "info string the game is over: variant_loss",
                "bestmove 0000",
                "info depth 0 score cp 0",
                "info string the game is over: stalemate",
                "bestmove 0000",
                "info string e1e3 is not a legal move here",
                "bestmove 0000",
                "info string bad FEN: invalid turn part in fen",
                "bestmove 0000",
            ]
        );
    }

    /// `go nodes N` searches as `provari prove --nodes N` does, with the
    /// memory `Hash` gives; with `Shorten`, each shorter win is reported as
    /// found, and the best move is the shortest's.
    #[test]
    fn searches_as_prove_does() {
        let lines = session(&[
            "setoption name Hash value 16",
            &format!("position fen {SHORTENS}"),
            "go nodes 3000",
            "setoption name Shorten value true",
            "go nodes 4000",
        ]);
        let prove = |args: &[&str]| {
            let args: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
            let (_, options) = parse(&args).unwrap();
            crate::run(SHORTENS, None, &options, &inputs(), None)
        };
        let proved = prove(&["--nodes", "3000", "--memory", "16"]);
        assert_eq!(
            (&proved["nodes"], &proved["mate"]),
            (&json!(1243), &json!(8))
        );
        let first = "info depth 15 score mate 8 nodes 1243 pv ";
        assert!(lines[0].starts_with(first), "{}", lines[0]);
        assert_eq!(
            lines[1],
            format!("bestmove {}", &lines[0][first.len()..][..4])
        );
        let shortened = prove(&["--nodes", "4000", "--memory", "16", "--shorten"]);
        assert_eq!(
            (&shortened["mate"], &shortened["mate_nodes"]),
            (&json!(7), &json!(3304))
        );
        assert!(lines[2].starts_with(first), "{}", lines[2]);
        let shorter = "info depth 13 score mate 7 nodes 3304 pv ";
        assert!(lines[3].starts_with(shorter), "{}", lines[3]);
        assert_eq!(lines[4], format!("info nodes {}", shortened["nodes"]));
        assert_eq!(
            lines[5],
            format!("bestmove {}", &lines[3][shorter.len()..][..4])
        );
        assert_eq!(lines.len(), 6);
    }

    /// After `go infinite` the best move waits for `stop`, and `isready` is
    /// answered meanwhile, as it is during any search.
    #[test]
    fn waits_for_stop_after_go_infinite() {
        let (engine, lines) = engine();
        let (commands, received) = mpsc::channel::<String>();
        let serving = thread::spawn(move || engine.serve(received.into_iter()));
        let send = |command: &str| commands.send(command.to_owned()).unwrap();
        let wait_for = |count: usize, start: &str| {
            let deadline = Instant::now() + Duration::from_secs(20);
            while lines
                .lock()
                .unwrap()
                .iter()
                .filter(|line| line.starts_with(start))
                .count()
                < count
            {
                assert!(Instant::now() < deadline, "no {start}");
                thread::sleep(Duration::from_millis(2));
            }
        };
        send("position fen 4k3/3p4/8/8/8/8/8/3QK3 w - - 0 1");
        send("go infinite");
        wait_for(1, "info depth 1 score mate 1");
        send("isready");
        wait_for(1, "readyok");
        let ended = |lines: &[String]| lines.iter().any(|line| line.starts_with("bestmove"));
        assert!(!ended(&lines.lock().unwrap()));
        send("stop");
        wait_for(1, "bestmove d1d7");
        send("position startpos");
        send("go infinite");
        send("isready");
        wait_for(2, "readyok");
        send("stop");
        wait_for(1, "bestmove 0000");
        // Commands that end stop a search held for `stop`.
        send("go ponder");
        drop(commands);
        serving.join().unwrap();
        let lines = lines.lock().unwrap();
        assert_eq!(lines.last().unwrap(), "bestmove 0000");
        assert_eq!(
            lines
                .iter()
                .filter(|line| line.starts_with("bestmove"))
                .count(),
            3
        );
    }

    /// Options it cannot take are refused with the reason, and the rest of
    /// the line is read as UCI reads it.
    #[test]
    fn refuses_options_it_cannot_take() {
        let lines = session(&[
            "setoption name Hash value 0",
            "setoption name Attacker value red",
            "setoption name Shorten value maybe",
            "setoption name UCI_Variant value chess",
            "setoption name UCI_Variant value atomic",
            "setoption name Clear Hash",
            "setoption name SyzygyPath value <empty>",
        ]);
        assert_eq!(
            lines,
            [
                "info string Hash is from 1 to 1048576 MB, not 0",
                "info string the attacker is Mover, White or Black, not red",
                "info string Shorten is true or false, not maybe",
                "info string provari proves Atomic wins only",
                "info string provari has no option Clear Hash",
            ]
        );
        let lines = session(&["setoption name SyzygyPath value /no/such/folder"]);
        assert!(
            lines[0].starts_with("info string tablebases: /no/such/folder"),
            "{lines:?}"
        );
    }

    /// `movetime` budgets a search, else the side to move's clock does: its
    /// time over the moves to go and three quarters of its increment, at most
    /// nine tenths of its time.
    #[test]
    fn budgets_from_the_clock() {
        let limits = |words: &str| Limits::read(&words.split(' ').collect::<Vec<_>>());
        let clock = limits("wtime 60000 btime 30000 winc 1000 movestogo 10");
        assert_eq!(clock.seconds(Color::White), Some(6.75));
        assert_eq!(clock.seconds(Color::Black), Some(3.0));
        assert_eq!(limits("btime 90000").seconds(Color::Black), Some(3.0));
        let short = limits("wtime 1000 winc 2000")
            .seconds(Color::White)
            .unwrap();
        assert!((short - 0.9).abs() < 1e-9, "{short}");
        let all = limits("wtime 60000 movetime 500 nodes 7 mate 3 depth 9 infinite");
        assert_eq!(all.seconds(Color::White), Some(0.5));
        assert_eq!(
            (all.nodes, all.mate, all.infinite),
            (Some(7), Some(3), true)
        );
        assert_eq!(limits("infinite").seconds(Color::White), None);
    }
}
