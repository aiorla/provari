//! Tables unpacked by `provari unpack-tables`, which the reader takes in
//! place: with `PROVARI_ANTICHESS_TABLEBASES`, the two- and three-piece tables of
//! that set are unpacked, and read as their packed files are. The reader's
//! own tests make and unpack tables of their own (`tablebases::file`).

#![cfg(feature = "antichess-tablebases")]

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use provari::antichess::rules;
use provari::antichess::tablebases::{Entry, Tablebases};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("provari-antichess-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// Whether the file at `path` is a table unpacked by `unpack-tables`.
fn is_unpacked(path: &Path) -> bool {
    let mut head = [0; 8];
    fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut head))
        .is_ok_and(|()| &head == b"an0plain")
}

/// The probes of niklasf's antichess-tb-sys's own test that two or three pieces make,
/// and the plies for the side to move, positive when it wins.
const REFERENCE: [(&str, i32); 3] = [
    ("8/8/8/8/8/3k4/8/1R6 b - - 0 1", -35),
    ("8/8/8/8/8/P7/8/7k b - - 0 1", -45),
    ("8/8/K7/8/8/6b1/8/3b4 w - - 0 1", -47),
];

#[test]
fn the_set_s_small_tables_unpack_and_read_alike() {
    let Some(set) = std::env::var_os("PROVARI_ANTICHESS_TABLEBASES").map(PathBuf::from) else {
        return;
    };
    // Two and three pieces: a table's name lists them, as KvBB.
    let mut small: Vec<PathBuf> = fs::read_dir(&set)
        .expect("the tables")
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension().is_some_and(|extension| extension == "an0")
                && path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .is_some_and(|stem| stem.len() <= 4)
        })
        .collect();
    small.sort();
    if small.is_empty() || small.iter().any(|path| is_unpacked(path)) {
        return;
    }
    let dir = scratch("unpacked-set");
    let packed = dir.join("packed");
    fs::create_dir(&packed).unwrap();
    for path in &small {
        fs::copy(path, packed.join(path.file_name().unwrap())).unwrap();
    }
    let plain = dir.join("plain");
    let output = Command::new(env!("CARGO_BIN_EXE_provari"))
        .args(["unpack-tables", "--variant", "antichess"])
        .args([&packed, &plain])
        .output()
        .expect("the prover runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).expect("a JSON line");
    assert_eq!(result["tables"], small.len());
    for path in &small {
        let name = path.file_name().unwrap();
        assert!(is_unpacked(&plain.join(name)));
    }

    // Every position of three pieces on the files a to c, either side to
    // move, reads alike packed and unpacked.
    let (packed_tables, tables) = (
        Tablebases::open(&packed).expect("the packed tables"),
        Tablebases::open(&plain).expect("the unpacked tables"),
    );
    let squares: Vec<usize> = (0..64).filter(|square| square % 8 < 3).collect();
    let mut read = 0;
    for &a in &squares {
        for &b in &squares {
            for &c in &squares {
                if a == b || b == c || a == c {
                    continue;
                }
                let mut board = ['.'; 64];
                (board[a], board[b], board[c]) = ('R', 'n', 'p');
                if c / 8 == 0 || c / 8 == 7 {
                    continue;
                }
                for turn in ["w", "b"] {
                    let fen = format!("{} {turn} - - 0 1", placement(&board));
                    let pos = rules::parse_fen(&fen).unwrap();
                    let entry = tables.probe(&pos);
                    assert_eq!(entry, packed_tables.probe(&pos), "{fen}");
                    read += usize::from(entry.is_some());
                }
            }
        }
    }
    assert!(read > 1000, "{read}");
    assert_eq!(tables.count, small.len());
    for (fen, plies) in REFERENCE {
        let pos = rules::parse_fen(fen).unwrap();
        let Some(entry) = tables.probe(&pos) else {
            continue;
        };
        let expected = match plies {
            0 => Entry::Draw,
            1.. => Entry::Win(plies.unsigned_abs()),
            ..0 => Entry::Loss(plies.unsigned_abs()),
        };
        assert_eq!(entry, expected, "{fen}");
    }
    // Nothing kept: the reader takes their bytes where they are.
    assert_eq!(tables.kept(), 0);
    fs::remove_dir_all(&dir).expect("the scratch directory goes");
}

/// A FEN's pieces from a board of a1 to h8.
fn placement(board: &[char; 64]) -> String {
    let mut rows = Vec::new();
    for rank in (0..8).rev() {
        let mut row = String::new();
        let mut empty = 0;
        for file in 0..8 {
            match board[rank * 8 + file] {
                '.' => empty += 1,
                piece => {
                    if empty > 0 {
                        row.push_str(&empty.to_string());
                        empty = 0;
                    }
                    row.push(piece);
                }
            }
        }
        if empty > 0 {
            row.push_str(&empty.to_string());
        }
        rows.push(row);
    }
    rows.join("/")
}
