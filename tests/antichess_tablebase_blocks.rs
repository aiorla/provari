//! The tables' kept blocks, in a process of their own, as their limit is the
//! process's: a probe answers the same from a kept block as from one unpacked
//! afresh, and the blocks kept stay within the limit. It reads the tables of
//! `PROVARI_ANTICHESS_TABLEBASES`, the whole set or the few of niklasf's
//! antichess-tb-sys, packed or unpacked, and passes without them.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use provari::antichess::rules;
use provari::antichess::tablebases::Tablebases;

/// Materials of three and four pieces, White's then Black's, that the few
/// tables hold too.
const MATERIALS: [(&str, &str); 8] = [
    ("KBN", "k"),
    ("KRP", "k"),
    ("BNP", "k"),
    ("BNP", "p"),
    ("KPP", "k"),
    ("KQR", "k"),
    ("RP", "pp"),
    ("PP", "p"),
];

/// A block's bytes unpacked.
const BLOCK: usize = 58_969;

#[test]
fn kept_blocks_answer_as_unpacked_ones() {
    let Some(dir) = std::env::var_os("PROVARI_ANTICHESS_TABLEBASES").map(PathBuf::from) else {
        return;
    };
    let tables = Tablebases::open(&dir).expect("the tables");
    let positions = positions(1200);

    // Nothing kept: each probe unpacks its block, or finds it among the
    // thread's last eight.
    tables.keep(0);
    let fresh: Vec<_> = positions.iter().map(|pos| tables.probe(pos)).collect();
    assert_eq!(tables.kept(), 0);
    let answered = fresh.iter().filter(|entry| entry.is_some()).count();
    assert!(answered > positions.len() / 2, "{answered}");

    // An unpacked set is read in place, and keeps nothing however much it may.
    if unpacked(&dir) {
        tables.keep(usize::MAX);
        let read: Vec<_> = positions.iter().map(|pos| tables.probe(pos)).collect();
        assert_eq!(read, fresh);
        assert_eq!(tables.kept(), 0);
        return;
    }

    // A few blocks kept, and no more.
    tables.keep(4 * BLOCK);
    let few: Vec<_> = positions.iter().map(|pos| tables.probe(pos)).collect();
    assert_eq!(few, fresh);
    assert!(
        (1..=4 * BLOCK).contains(&tables.kept()),
        "{}",
        tables.kept()
    );

    // Every block kept, then read from where it is kept.
    tables.keep(usize::MAX);
    let filling: Vec<_> = positions.iter().map(|pos| tables.probe(pos)).collect();
    assert_eq!(filling, fresh);
    let kept = tables.kept();
    assert!(kept > 4 * BLOCK, "{kept}");
    let reading: Vec<_> = positions.iter().map(|pos| tables.probe(pos)).collect();
    assert_eq!(reading, fresh);
    assert_eq!(tables.kept(), kept);
}

/// Whether the set in `dir` is unpacked (`provari unpack-tables`): its
/// tables start with their own header rather than gzip's.
fn unpacked(dir: &Path) -> bool {
    let table = fs::read_dir(dir)
        .expect("the tables")
        .map(|entry| entry.expect("an entry").path())
        .find(|path| path.extension().is_some_and(|extension| extension == "an0"))
        .expect("a table");
    let mut head = [0; 8];
    fs::File::open(table)
        .and_then(|mut file| file.read_exact(&mut head))
        .is_ok_and(|()| &head == b"an0plain")
}

/// `count` legal positions of `MATERIALS`, each side to move, from a fixed
/// seed.
fn positions(count: usize) -> Vec<shakmaty::variant::Antichess> {
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = move |below: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % below as u64) as usize
    };
    let mut positions = Vec::new();
    while positions.len() < count {
        let (white, black) = MATERIALS[positions.len() % MATERIALS.len()];
        let mut board = ['.'; 64];
        let pieces = white
            .chars()
            .chain(black.chars().map(|c| c.to_ascii_lowercase()));
        for piece in pieces {
            let pawn = piece.eq_ignore_ascii_case(&'p');
            loop {
                let square = next(64);
                let rank = square / 8;
                if board[square] == '.' && !(pawn && (rank == 0 || rank == 7)) {
                    board[square] = piece;
                    break;
                }
            }
        }
        let turn = if next(2) == 0 { "w" } else { "b" };
        let fen = format!("{} {turn} - - 0 1", placement(&board));
        if let Ok(pos) = rules::parse_fen(&fen) {
            positions.push(pos);
        }
    }
    positions
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
