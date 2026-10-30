//! What `provari prove` prints for people, where `--json` prints JSON: while
//! it searches, a status line on standard error, rewritten in place on a
//! terminal, and at the end its result in a sentence or two, with a line.
//! `provari uci` sends the status line's numbers and line as an `info
//! string`.
//!
//! The status line gives the seconds, the nodes and their rate, the root's
//! proof number rebuilt from the search's line with its peak, or once it is
//! below half its peak, "closing in", and the line itself in SAN, with the
//! defences not yet beaten after each attacker move:
//!
//! ```text
//!  88 s  16.0M nodes   181k/s  pn 16.0k (peak 17.2k)  1. Nc7+ (1 left) Qxc7 2. c3 (7 left) b5
//! ```

use std::io::{IsTerminal, Write};
use std::time::Duration;

use serde_json::Value;
use shakmaty::{Color, Position as _, uci::UciMove, variant::Atomic};

use provari::{dfpn::ProgressSink, lines};

/// The columns of a terminal that cannot tell its width.
const COLUMNS: usize = 80;

/// The width the root's numbers take in a status line, so that its line
/// stays in place: "pn 16.0k (peak 17.2k)" at most.
const NUMBERS: usize = 21;

/// A count in five characters at most: 159, 1.7k, 16.0k, 181k, 29.8M.
pub fn short(n: u64) -> String {
    if n < 1000 {
        return n.to_string();
    }
    let mut value = n as f64;
    for unit in ["k", "M", "G", "T"] {
        value /= 1000.0;
        if value < 99.95 {
            return format!("{value:.1}{unit}");
        }
        if value < 999.5 {
            return format!("{value:.0}{unit}");
        }
    }
    format!("{value:.0}P")
}

/// A count of nodes: 1 node, 169k nodes.
fn nodes(n: u64) -> String {
    match n {
        1 => "1 node".to_owned(),
        _ => format!("{} nodes", short(n)),
    }
}

/// Milliseconds as people read them: 40 ms, 2.5 s, 165 s.
fn duration(ms: u64) -> String {
    match ms {
        0..1000 => format!("{ms} ms"),
        1000..99_950 => format!("{:.1} s", ms as f64 / 1000.0),
        _ => format!("{} s", (ms + 500) / 1000),
    }
}

/// The side's name, capitalised.
fn name(side: Color) -> &'static str {
    match side {
        Color::White => "White",
        Color::Black => "Black",
    }
}

/// The root's proof number as the status line gives it: with its peak, or
/// once below half of it, "closing in", a sign that a proof may be near.
pub fn closeness(proof: u64, peak: u64) -> String {
    if 2 * proof < peak {
        format!("pn {}, closing in", short(proof))
    } else {
        format!("pn {} (peak {})", short(proof), short(peak))
    }
}

/// A line of UCI moves from `root` in SAN, a string for each ply, numbered as
/// PGN numbers moves, each attacker move with the replies still open after
/// it, `open[i]` being the moves still open where the `i`th was played:
/// "1. Nc7+ (1 left)", "Qxc7", "2. c3 (7 left)". A move that is not legal
/// ends the line.
pub fn plies(root: &Atomic, attacker: Color, moves: &[Value], open: &[Value]) -> Vec<String> {
    let mut pos = root.clone();
    let mut plies = Vec::with_capacity(moves.len());
    for (i, uci) in moves.iter().enumerate() {
        let Some(m) = uci
            .as_str()
            .and_then(|uci| uci.parse::<UciMove>().ok())
            .and_then(|uci| uci.to_move(&pos).ok())
        else {
            break;
        };
        let attacking = pos.turn() == attacker;
        let mut ply = lines::play(&mut pos, m, i == 0);
        if attacking && let Some(left) = open.get(i + 1).and_then(Value::as_u64) {
            ply.push_str(&format!(" ({left} left)"));
        }
        plies.push(ply);
    }
    plies
}

/// A progress record's root numbers and line: what follows its counts in a
/// status line, and what `provari uci` sends as an `info string`.
pub fn progress(record: &Value, root: &Atomic, attacker: Color) -> (String, Vec<String>) {
    let number = |key: &str| record[key].as_u64().unwrap_or_default();
    let empty = Vec::new();
    let list = |key: &str| record[key].as_array().unwrap_or(&empty);
    (
        closeness(number("proof"), number("peak")),
        plies(root, attacker, list("line"), list("open")),
    )
}

/// A progress record as a status line, in `width` columns less one when
/// given, the line's last plies dropped to fit and spaces after it to cover
/// a longer one before.
pub fn status(record: &Value, root: &Atomic, attacker: Color, width: Option<usize>) -> String {
    let number = |key: &str| record[key].as_u64().unwrap_or_default();
    let (numbers, plies) = progress(record, root, attacker);
    let mut text = format!(
        "{:>3} s  {:>5} nodes  {:>5}/s  {numbers:<NUMBERS$}",
        number("seconds"),
        short(number("nodes")),
        short(number("rate")),
    );
    let room = width.map_or(usize::MAX, |width| width.saturating_sub(1));
    for (i, ply) in plies.iter().enumerate() {
        let gap = if i == 0 { "  " } else { " " };
        if text.len() + gap.len() + ply.len() > room {
            break;
        }
        text.push_str(gap);
        text.push_str(ply);
    }
    match width {
        Some(_) => {
            text.truncate(room);
            format!("{text:<room$}")
        }
        None => text.trim_end().to_owned(),
    }
}

/// Where status lines go, if anywhere, for `--progress` (`seconds`): every
/// so many seconds, 0 for never, by default every second where standard
/// error is a terminal, rewritten in place there, a line each elsewhere.
#[derive(Clone)]
pub struct Status {
    pub every: Option<Duration>,
    /// Whether standard error is a terminal, whose width is read again for
    /// each line, as it may change.
    pub terminal: bool,
}

impl Status {
    pub fn new(seconds: Option<f64>) -> Status {
        let terminal = std::io::stderr().is_terminal();
        let every = match seconds {
            Some(seconds) => (seconds > 0.0).then(|| Duration::from_secs_f64(seconds)),
            None => terminal.then_some(Duration::from_secs(1)),
        };
        Status { every, terminal }
    }

    /// Prints each progress record of a search of `root` for `attacker` as a
    /// status line.
    pub fn sink(&self, root: Atomic, attacker: Color) -> ProgressSink {
        let terminal = self.terminal;
        Box::new(move |record| {
            let width = terminal.then(columns);
            let line = status(record, &root, attacker, width);
            let mut stderr = std::io::stderr().lock();
            // Standard error that has gone takes nothing more.
            let _ = match width {
                Some(_) => write!(stderr, "\r{line}").and_then(|()| stderr.flush()),
                None => writeln!(stderr, "{line}"),
            };
        })
    }

    /// Clears the status line from a terminal, before the result.
    pub fn clear(&self) {
        if self.every.is_some() && self.terminal {
            let blank = " ".repeat(columns().saturating_sub(1));
            eprint!("\r{blank}\r");
        }
    }
}

/// Standard error's terminal width: as the terminal gives it, else
/// `COLUMNS`, else 80.
fn columns() -> usize {
    terminal_columns().or_else(env_columns).unwrap_or(COLUMNS)
}

/// `COLUMNS`, as shells set it.
fn env_columns() -> Option<usize> {
    std::env::var("COLUMNS")
        .ok()?
        .trim()
        .parse()
        .ok()
        .filter(|&columns| columns > 0)
}

/// Standard error's terminal width, from the terminal, on Linux and macOS.
#[cfg(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    target_os = "macos"
))]
fn terminal_columns() -> Option<usize> {
    use std::ffi::{c_int, c_ulong};
    /// A terminal's size, as `TIOCGWINSZ` writes it: only its columns are
    /// read here.
    #[allow(dead_code)]
    #[repr(C)]
    struct WinSize {
        rows: u16,
        columns: u16,
        width: u16,
        height: u16,
    }
    #[cfg(target_os = "linux")]
    const TIOCGWINSZ: c_ulong = 0x5413;
    #[cfg(target_os = "macos")]
    const TIOCGWINSZ: c_ulong = 0x4008_7468;
    unsafe extern "C" {
        fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    }
    let mut size = WinSize {
        rows: 0,
        columns: 0,
        width: 0,
        height: 0,
    };
    // SAFETY: TIOCGWINSZ writes a `struct winsize`, whose layout `WinSize`
    // has, and nothing else.
    let read = unsafe { ioctl(2, TIOCGWINSZ, &raw mut size) } == 0;
    (read && size.columns > 0).then_some(size.columns.into())
}

/// Elsewhere, the width is not read from the terminal.
#[cfg(not(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    target_os = "macos"
)))]
fn terminal_columns() -> Option<usize> {
    None
}

/// How a game already over ended, won by `winner` if by anyone, with `turn`
/// to move.
fn over(reason: &str, winner: Option<Color>, turn: Color) -> String {
    let how = match (reason, winner) {
        ("variant_loss", Some(winner)) => format!(
            "{}'s king has exploded, and {} has won",
            name(!winner),
            name(winner)
        ),
        ("checkmate", Some(winner)) => format!(
            "{} is checkmated, and {} has won",
            name(!winner),
            name(winner)
        ),
        (_, Some(winner)) => format!("{} has won", name(winner)),
        ("stalemate", None) => format!("{} is stalemated, a draw", name(turn)),
        ("insufficient_material", None) => "neither side can win, a draw".to_owned(),
        ("fifty_moves", None) => "fifty moves without a capture or a pawn move, a draw".to_owned(),
        (_, None) => "a draw".to_owned(),
    };
    format!("The game is over: {how}.")
}

/// A `provari prove` result for people, from its JSON, for `attacker`'s win
/// from `root`: what it found, how long that took, and a line, the longest
/// of a win, else the one the search stopped on, with the replies left after
/// each attacker move; and where it wrote the proof. Its error, if any, is
/// for standard error.
pub fn result(result: &Value, root: &Atomic, attacker: Color, proof: Option<&str>) -> String {
    if let Some(reason) = result["over"].as_str() {
        let winner = match result["winner"].as_str() {
            Some("white") => Some(Color::White),
            Some("black") => Some(Color::Black),
            _ => None,
        };
        return over(reason, winner, root.turn());
    }
    let side = name(attacker);
    let number = |key: &str| result[key].as_u64();
    let searched = nodes(number("nodes").unwrap_or_default());
    let ms = duration(number("ms").unwrap_or_default());
    let empty = Vec::new();
    let list = |key: &str| result[key].as_array().unwrap_or(&empty);
    let mut text = String::new();
    if let Some(mate) = number("mate").filter(|_| result["found"] == true) {
        text = match result["tablebases"] == true {
            true => format!(
                "{side} wins within {mate} moves, where every line mates or reaches a tablebase win"
            ),
            false => format!("{side} mates in {mate}"),
        };
        if let Some(first) = number("first_mate").filter(|&first| first != mate) {
            text.push_str(&format!(", shortened from {first}"));
        }
        // When the win given was found, which shortening may be later than
        // the first.
        let (found_ms, found_nodes) = match (number("mate_ms"), number("mate_nodes")) {
            (Some(ms), Some(nodes)) => (ms, nodes),
            _ => (
                number("first_mate_ms").unwrap_or_default(),
                number("nodes").unwrap_or_default(),
            ),
        };
        text.push_str(&format!(
            ", found in {} and {}.",
            duration(found_ms),
            nodes(found_nodes)
        ));
        if result["shortest"] == true && number("first_mate").is_some() {
            text.push_str(" No shorter win exists.");
        }
        let line = plies(root, attacker, list("line"), &[]);
        if !line.is_empty() {
            let end = if result["line_tablebase"] == true {
                " (tablebase win)"
            } else {
                ""
            };
            text.push_str(&format!("\nLongest line: {}{end}", line.join(" ")));
        }
        if let (Some(path), Some(moves)) =
            (proof, number("proof_moves").or(number("proof_positions")))
        {
            let what = if result.get("proof_moves").is_some() {
                "moves"
            } else {
                "positions"
            };
            text.push_str(&format!(
                "\nProof written to {path} ({} {what}).",
                short(moves)
            ));
        }
    } else if result["refuted"] == true {
        text = match number("most") {
            Some(1) => format!("{side} has no win in 1 move"),
            Some(most) => format!("{side} has no win in at most {most} moves"),
            None => format!("{side} has no win"),
        };
        text.push_str(&format!(", settled in {ms} and {searched}."));
    } else if result.get("found").is_some() {
        text = format!("No win found for {side} in {ms} and {searched}.");
        let line = plies(root, attacker, list("line"), list("open"));
        if !line.is_empty() {
            text.push_str(&format!("\nHardest line: {}", line.join(" ")));
        }
        if let Some([proof, disproof]) = result["root"].as_array().map(Vec::as_slice) {
            let number = |value: &Value| short(value.as_u64().unwrap_or_default());
            text.push_str(&format!(
                "\nThe root's numbers: pn {}, dn {}.",
                number(proof),
                number(disproof)
            ));
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use shakmaty::Color;

    use super::{closeness, duration, result, short, status};

    #[test]
    fn counts_fit_five_characters() {
        let cases = [
            (0, "0"),
            (999, "999"),
            (1000, "1.0k"),
            (1749, "1.7k"),
            (16_049, "16.0k"),
            (99_949, "99.9k"),
            (99_950, "100k"),
            (181_400, "181k"),
            (999_499, "999k"),
            (999_500, "1.0M"),
            (29_800_000, "29.8M"),
            (1_073_741_823, "1.1G"),
        ];
        for (n, text) in cases {
            assert_eq!(short(n), text, "{n}");
        }
        assert_eq!(duration(40), "40 ms");
        assert_eq!(duration(2_450), "2.5 s");
        assert_eq!(duration(99_949), "99.9 s");
        assert_eq!(duration(165_400), "165 s");
    }

    /// Below half its peak, the proof number is closing in.
    #[test]
    fn closes_in_below_half_the_peak() {
        assert_eq!(closeness(17_246, 17_246), "pn 17.2k (peak 17.2k)");
        assert_eq!(closeness(8_623, 17_246), "pn 8.6k (peak 17.2k)");
        assert_eq!(closeness(8_622, 17_246), "pn 8.6k, closing in");
        assert_eq!(closeness(1_733, 17_246), "pn 1.7k, closing in");
    }

    /// A status line counts, gives the root's numbers and plays its line in
    /// SAN from the root, with the defences left after each attacker move;
    /// on a terminal it fits its width, whole plies only, padded to cover a
    /// longer line.
    #[test]
    fn shows_a_status_line() {
        let root =
            provari::position("r1b1kbnr/ppp1pppp/2n5/3q4/3P4/2N5/PPP2PPP/R1BQKBNR b KQkq - 1 4")
                .unwrap();
        let record = json!({
            "seconds": 88, "nodes": 16_012_345, "rate": 181_234, "proof": 16_012,
            "disproof": 321, "peak": 17_246,
            "line": ["d5a5", "c1d2", "a5b6", "c3d5"], "open": [30, 2, 38, 7],
        });
        let line = status(&record, &root, Color::White, None);
        assert_eq!(
            line,
            " 88 s  16.0M nodes   181k/s  pn 16.0k (peak 17.2k)  4... Qa5 5. Bd2 (38 left) Qb6 \
             6. Nd5"
        );
        let fitted = status(&record, &root, Color::White, Some(80));
        assert_eq!(fitted.len(), 79);
        assert_eq!(
            fitted.trim_end(),
            " 88 s  16.0M nodes   181k/s  pn 16.0k (peak 17.2k)  4... Qa5 5. Bd2 (38 left)"
        );
        let narrow = status(&record, &root, Color::White, Some(20));
        assert_eq!(narrow, " 88 s  16.0M nodes ");
        // A line that turns illegal stops there.
        let wrong = json!({ "line": ["d5a5", "e1e8"], "open": [1, 1] });
        assert!(status(&wrong, &root, Color::White, None).ends_with("4... Qa5"));
    }

    /// Results read as sentences: a win with its longest line and its
    /// proof, a refutation, and a search cut short with its hardest line.
    #[test]
    fn says_what_a_search_found() {
        let root =
            provari::position("r1b1kb1r/pp1p1qp1/2n1p3/1B3P1p/3PP3/2P5/PP4PP/R2QK2R w KQkq - 0 1")
                .unwrap();
        let win = json!({
            "found": true, "mate": 8, "first_mate_ms": 41, "nodes": 1243, "ms": 50,
            "line": ["b5c6", "a7a6"], "proof_moves": 34,
        });
        assert_eq!(
            result(&win, &root, Color::White, Some("win.pvr")),
            "White mates in 8, found in 41 ms and 1.2k nodes.\n\
             Longest line: 1. Bxc6 a6\n\
             Proof written to win.pvr (34 moves)."
        );
        let shortened = json!({
            "found": true, "mate": 7, "first_mate": 8, "first_mate_ms": 41, "nodes": 4000,
            "ms": 120, "mate_ms": 97, "mate_nodes": 3304, "shortest": true,
        });
        assert_eq!(
            result(&shortened, &root, Color::White, None),
            "White mates in 7, shortened from 8, found in 97 ms and 3.3k nodes. \
             No shorter win exists."
        );
        let at_once = json!({
            "found": true, "mate": 1, "first_mate_ms": 0, "nodes": 1, "ms": 0, "line": ["d1d7"],
        });
        let exposed = provari::position("4k3/3p4/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        assert_eq!(
            result(&at_once, &exposed, Color::White, None),
            "White mates in 1, found in 0 ms and 1 node.\nLongest line: 1. Qxd7#"
        );
        let refuted =
            json!({ "found": false, "nodes": 2048, "ms": 1500, "most": 3, "refuted": true });
        assert_eq!(
            result(&refuted, &root, Color::Black, None),
            "Black has no win in at most 3 moves, settled in 1.5 s and 2.0k nodes."
        );
        let unproven = json!({
            "found": false, "nodes": 2_000_000, "ms": 12_000, "root": [16_012, 3_201],
            "line": ["b5c6", "a7a6", "d1b3"], "open": [24, 3, 31],
        });
        assert_eq!(
            result(&unproven, &root, Color::White, None),
            "No win found for White in 12.0 s and 2.0M nodes.\n\
             Hardest line: 1. Bxc6 (3 left) a6 2. Qb3\n\
             The root's numbers: pn 16.0k, dn 3.2k."
        );
        let over = json!({ "found": false, "over": "variant_loss", "winner": "white" });
        let exploded =
            provari::position("rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQ - 0 1").unwrap();
        assert_eq!(
            result(&over, &exploded, Color::Black, None),
            "The game is over: Black's king has exploded, and White has won."
        );
    }
}
