//! `provari prove --variant antichess --tablebases` with tolius's whole set of tables, from
//! `PROVARI_ANTICHESS_TABLEBASES`: without them, or with only a few, it has nothing
//! to read, and the test passes without them.

use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

use provari::antichess::format;
use provari::antichess::tablebases::{Tablebases, WHOLE_SET};

/// A rook and a pawn against a king, which White wins in 50 plies, 25 moves
/// (niklasf's antichess-tb-sys's own test).
const ROOK_AND_PAWN: &str = "7k/8/5K2/8/8/1R6/P7/8 w - - 0 1";

#[test]
fn proves_a_long_win_from_the_tables() {
    let Some(dir) = std::env::var_os("PROVARI_ANTICHESS_TABLEBASES").map(PathBuf::from) else {
        return;
    };
    if Tablebases::open(&dir).expect("the tables").count < WHOLE_SET {
        return;
    }
    let scratch = std::env::temp_dir().join(format!(
        "provari-antichess-tablebases-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&scratch).expect("a scratch directory");
    let proof = scratch.join("rook-and-pawn.apr");
    let output = Command::new(env!("CARGO_BIN_EXE_provari"))
        .args([
            "prove",
            "--variant",
            "antichess",
            ROOK_AND_PAWN,
            "--most",
            "30",
            "--memory",
            "64",
        ])
        .arg("--tablebases")
        .arg(&dir)
        .arg("--proofs")
        .arg(&proof)
        .output()
        .expect("the prover runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).expect("a JSON line");
    assert_eq!(result["found"], true);
    assert_eq!(result["need"], 25);
    // Its 50 plies end before the 50-move rule could draw from counter 50,
    // or from higher if its first move is the pawn's.
    let limit = result["limit"].as_u64().expect("a limit");
    assert!((50..=99).contains(&limit), "{limit}");
    assert!(
        result["tablebase_hits"]
            .as_u64()
            .is_some_and(|hits| hits > 0)
    );
    assert!(result["nodes"].as_u64().is_some_and(|nodes| nodes < 100));

    let bytes = std::fs::read(&proof).expect("the proof file");
    let (header, indices) = format::decode(&bytes).expect("a proof");
    assert_eq!(header.moves, 25);
    assert_eq!(
        header.root,
        format!("7k/8/5K2/8/8/1R6/P7/8 w - - {limit} 1")
    );
    let steps = format::steps(&header, &indices).expect("a proof that checks");
    assert_eq!(
        steps.len() as u64,
        result["proof_positions"].as_u64().unwrap()
    );
    std::fs::remove_dir_all(&scratch).expect("the scratch directory goes");
}
