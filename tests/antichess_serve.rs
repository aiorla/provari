//! `provari serve --variant antichess` answers each request line with one line, and searches
//! each position from nothing, or with `--share` from what earlier searches
//! learned.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

/// After `1. e3 d6`: White wins in 14 more moves.
const D6: &str = "rnbqkbnr/ppp1pppp/3p4/8/8/4P3/PPPP1PPP/RNBQKBNR w - - 0 2";
/// After `1. e3 a6`: White wins in 27 more moves, far beyond a small budget.
const A6: &str = "rnbqkbnr/1ppppppp/p7/8/8/4P3/PPPP1PPP/RNBQKBNR w - - 0 2";

struct Server {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl Server {
    fn start() -> Server {
        Server::with(&["serve", "--variant", "antichess", "--memory", "64"])
    }

    fn with(args: &[&str]) -> Server {
        let mut child = Command::new(env!("CARGO_BIN_EXE_provari"))
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("the prover starts");
        let input = child.stdin.take().expect("its input");
        let output = BufReader::new(child.stdout.take().expect("its output"));
        Server {
            child,
            input,
            output,
        }
    }

    fn ask(&mut self, line: &str) -> Value {
        writeln!(self.input, "{line}").expect("a request goes");
        self.input.flush().expect("a request goes");
        let mut answer = String::new();
        self.output.read_line(&mut answer).expect("an answer comes");
        serde_json::from_str(&answer).expect("an answer is JSON")
    }

    fn finish(self) {
        drop(self.input);
        let status = self
            .child
            .wait_with_output()
            .expect("the prover ends")
            .status;
        assert!(status.success());
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "provari-antichess-serve-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

#[test]
fn answers_each_search_from_nothing() {
    let dir = scratch("proofs");
    let proof = dir.join("d6.apr");
    let mut server = Server::start();

    let won = server.ask(&json!({"id": "d6", "fen": D6, "most": 14, "proof": proof}).to_string());
    assert_eq!(won["id"], "d6");
    assert_eq!(won["found"], true);
    assert_eq!(won["need"], 14);
    assert!(won["limit"].is_null() || won["limit"].as_u64().is_some_and(|limit| limit <= 99));
    assert_eq!(won["proof"], json!(proof));
    // Each position searched is counted by its pieces, of which none has more
    // than the root's 32.
    let by_pieces = won["nodes_by_pieces"].as_array().expect("a census");
    assert_eq!(by_pieces.len(), 33);
    assert_eq!(
        by_pieces.iter().filter_map(Value::as_u64).sum::<u64>(),
        won["nodes"].as_u64().expect("nodes")
    );
    let bytes = std::fs::read(&proof).expect("the proof file");
    let (header, _) = provari::antichess::format::decode(&bytes).expect("a proof");
    assert_eq!(header.moves, 14);
    assert!(
        header
            .root
            .starts_with("rnbqkbnr/ppp1pppp/3p4/8/8/4P3/PPPP1PPP/RNBQKBNR w - - ")
    );

    // check plays every line of it to its end, and refuses a damaged one.
    let check = |path: &PathBuf| {
        Command::new(env!("CARGO_BIN_EXE_provari"))
            .args(["check", "--variant", "antichess"])
            .arg(path)
            .output()
            .expect("the checker runs")
    };
    let checked = check(&proof);
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    let checked: Value = serde_json::from_slice(&checked.stdout).expect("a JSON line");
    assert_eq!(checked["moves"], 14);
    assert_eq!(checked["white_positions"], won["proof_positions"]);
    let by_pieces = checked["white_positions_by_pieces"]
        .as_array()
        .expect("a census");
    assert_eq!(by_pieces.len(), 33);
    assert_eq!(
        by_pieces.iter().filter_map(Value::as_u64).sum::<u64>(),
        checked["white_positions"].as_u64().expect("positions")
    );
    let damaged = dir.join("damaged.apr");
    std::fs::write(&damaged, &bytes[..bytes.len() - 1]).expect("a damaged copy");
    assert!(!check(&damaged).status.success());

    let open =
        server.ask(&json!({"id": "a6", "fen": A6, "most": 27, "nodes": 100_000}).to_string());
    assert_eq!(open["found"], false);
    assert_eq!(open["stopped"], "nodes");

    // What the search of d6 proved is gone, so a small budget cannot repeat it,
    // and the same search again takes the same nodes.
    let small =
        server.ask(&json!({"id": "d6 small", "fen": D6, "most": 14, "nodes": 1}).to_string());
    assert_eq!(small["found"], false);
    assert_eq!(small["stopped"], "nodes");
    let again = server.ask(&json!({"id": "d6 again", "fen": D6, "most": 14}).to_string());
    assert_eq!(again["found"], true);
    assert_eq!(again["nodes"], won["nodes"]);
    assert_eq!(again["nodes_by_pieces"], won["nodes_by_pieces"]);
    assert_eq!(again["proofs_kept"], won["proofs_kept"]);

    assert!(server.ask("not json")["error"].is_string());
    assert!(server.ask("{\"id\": \"no fen\"}")["error"].is_string());
    assert!(server.ask("{\"load\": \"proofs\"}")["error"].is_string());
    let black = "rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b - - 0 1";
    assert!(server.ask(&json!({"id": "black", "fen": black}).to_string())["error"].is_string());
    server.finish();
    std::fs::remove_dir_all(&dir).expect("the scratch directory goes");
}

#[test]
fn shares_what_earlier_searches_proved() {
    let dir = scratch("shared");
    let proof = dir.join("d6.apr");
    let mut server = Server::with(&[
        "serve",
        "--variant",
        "antichess",
        "--memory",
        "64",
        "--share",
    ]);
    let won = server.ask(&json!({"id": "d6", "fen": D6, "most": 14}).to_string());
    assert_eq!(won["found"], true);
    assert_eq!(won["proofs_before"], 0);
    let kept = won["proofs_kept"].as_u64().expect("proofs kept");
    assert!(kept > 0);

    // The proofs of d6 stay, so a budget of one node finds its win again, and
    // its proof file is whole.
    let again = server.ask(
        &json!({"id": "d6 again", "fen": D6, "most": 14, "nodes": 1, "proof": proof}).to_string(),
    );
    assert_eq!(again["found"], true);
    assert_eq!(again["need"], 14);
    assert_eq!(again["proofs_before"], kept);
    assert!(again["nodes"].as_u64().expect("nodes") <= 1);
    assert_eq!(again["proof_positions"], won["proof_positions"]);
    let checked = Command::new(env!("CARGO_BIN_EXE_provari"))
        .args(["check", "--variant", "antichess"])
        .arg(&proof)
        .output()
        .expect("the checker runs");
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    let checked: Value = serde_json::from_slice(&checked.stdout).expect("a JSON line");
    assert_eq!(checked["moves"], 14);
    server.finish();
    std::fs::remove_dir_all(&dir).expect("the scratch directory goes");
}

#[test]
fn takes_its_table_from_its_memory() {
    // Three quarters of the memory for the table leaves the rest for proofs.
    let mut server = Server::with(&[
        "serve",
        "--variant",
        "antichess",
        "--memory",
        "64",
        "--table",
        "48",
    ]);
    let won = server.ask(&json!({"id": "d6", "fen": D6, "most": 14}).to_string());
    assert_eq!(won["found"], true);
    assert_eq!(won["need"], 14);
    assert!(won["proofs_before"].is_null());
    server.finish();

    let refused = |args: &[&str]| {
        let output = Command::new(env!("CARGO_BIN_EXE_provari"))
            .args(args)
            .output()
            .expect("the prover runs");
        !output.status.success() && !output.stderr.is_empty()
    };
    assert!(refused(&[
        "serve",
        "--variant",
        "antichess",
        "--memory",
        "64",
        "--table",
        "64"
    ]));
    assert!(refused(&[
        "serve",
        "--variant",
        "antichess",
        "--memory",
        "64",
        "--table",
        "0"
    ]));
    assert!(refused(&[
        "serve",
        "--variant",
        "antichess",
        "--table",
        "256"
    ]));
    assert!(refused(&[
        "prove",
        "--variant",
        "antichess",
        D6,
        "--most",
        "14",
        "--memory",
        "64",
        "--share"
    ]));
}
