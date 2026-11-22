//! tolius's depth-to-win tables of Antichess positions with two to four
//! pieces, read by our own translation of his antichess-tb-api's reader
//! (`reader`, `index`), which answers every position as his does.
//!
//! A table gives, for the side to move, the plies to the end with best play:
//! how soon it wins, how long it holds out, or that the game is drawn, all as
//! the game goes without the 50-move rule. The search turns that into a value
//! for White (`dfpn::antichess`), and the proof writer (`format`) plays the
//! quickest wins they give, so that a proof still holds every line to its end.
//!
//! A table's files are mapped into memory (`file`). A packed table is read a
//! block of 58 KB at a time, which costs a hundred times what the rest of a
//! probe does to inflate, so the set keeps the blocks it inflates, for every
//! table and thread, up to a limit (`Tablebases::keep`), and a search
//! inflates each block it reads once; beyond the limit, each thread keeps
//! its last eight. Kept blocks are read without locks, by any number of
//! threads. A set unpacked whole (`unpack`) is read in place instead, and
//! keeps nothing: its 5.3 GiB of files stay in the system's page cache,
//! which every process reading them shares.
//!
//! The tables play on where Lichess draws for insufficient material
//! (`rules::insufficient_material`): with a bishop and a pawn each, the pawns
//! stuck face to face, a side may still win there by walling its bishop in
//! behind its pawn. Positions that are or may become such are left to the
//! search (`may_get_stuck`).

use std::path::Path;

use shakmaty::{Board, Position, variant::Antichess};

pub use reader::unpack;

/// The most pieces a table holds.
pub const MOST_PIECES: usize = 4;
/// The tables of the whole set: 21 with two pieces, 126 with three and 567
/// with four.
pub const WHOLE_SET: usize = 714;
/// The most bytes of unpacked blocks kept until `Tablebases::keep` says
/// otherwise.
pub const DEFAULT_KEPT: usize = 2 << 30;

/// What `unpack` wrote: how many tables, and their files' bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unpacked {
    pub tables: usize,
    pub bytes: u64,
}

/// What a table says of a position, for the side to move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entry {
    /// The side to move wins in so many plies.
    Win(u32),
    /// The side to move loses in so many plies.
    Loss(u32),
    Draw,
}

/// The tables, open for the whole process, which every thread reads at once.
#[derive(Clone, Copy, Debug)]
pub struct Tablebases {
    /// How many tables the set holds.
    pub count: usize,
    set: &'static reader::Set,
}

impl Tablebases {
    /// Opens the tables in `dir`, the files of tolius's set (`*.an0`), to
    /// keep up to `DEFAULT_KEPT` bytes of unpacked blocks. A set opens once
    /// for the process: opening the same folder again is free.
    pub fn open(dir: &Path) -> Result<Tablebases, String> {
        reader::open(dir).map(|set| Tablebases {
            count: set.count(),
            set,
        })
    }

    /// Keeps up to `bytes` of the tables' unpacked blocks from now on, for the
    /// whole process, as its searches first read them. Blocks kept stay kept;
    /// beyond the limit, each thread keeps its last eight for a while.
    pub fn keep(self, bytes: usize) {
        self.set.keep(bytes);
    }

    /// The bytes of unpacked blocks kept.
    pub fn kept(self) -> usize {
        self.set.kept()
    }

    /// What the tables say of `pos`, a position whose game goes on, if they
    /// hold it and it cannot meet Lichess's insufficient material.
    pub fn probe(self, pos: &Antichess) -> Option<Entry> {
        let board = pos.board();
        if board.occupied().count() > MOST_PIECES || may_get_stuck(board) {
            return None;
        }
        self.set.probe(pos)
    }
}

/// Whether a position may come to Lichess's insufficient material where the
/// tables do not draw: a bishop and a pawn each, the pawns face to face. As a
/// pawn leaves its file only by capturing, which leaves too few pieces for
/// that, it takes two pieces a side, all bishops and pawns, with a white pawn
/// below a black one on its file. Bishops alone can never capture or run out
/// of moves, so the tables draw them too.
pub fn may_get_stuck(board: &Board) -> bool {
    let white = board.pawns() & board.white();
    let black = board.pawns() & board.black();
    board.occupied().count() == 4
        && board.occupied() == board.bishops() | board.pawns()
        && board.white().count() == 2
        && white.into_iter().any(|pawn| {
            black
                .into_iter()
                .any(|other| other.file() == pawn.file() && other.rank() > pawn.rank())
        })
}

#[cfg(feature = "antichess-tablebases")]
mod file;
#[cfg(feature = "antichess-tablebases")]
mod index;
#[cfg(feature = "antichess-tablebases")]
mod reader;

#[cfg(not(feature = "antichess-tablebases"))]
mod reader {
    use std::path::Path;

    use shakmaty::variant::Antichess;

    use super::{Entry, Unpacked};

    /// No set opens without the reader.
    #[derive(Debug)]
    pub enum Set {}

    impl Set {
        pub fn count(&self) -> usize {
            match *self {}
        }

        pub fn keep(&self, _: usize) {
            match *self {}
        }

        pub fn kept(&self) -> usize {
            match *self {}
        }

        pub fn probe(&self, _: &Antichess) -> Option<Entry> {
            match *self {}
        }
    }

    pub fn open(_: &Path) -> Result<&'static Set, String> {
        Err("built without the antichess-tablebases feature".into())
    }

    pub fn unpack(_: &Path, _: &Path) -> Result<Unpacked, String> {
        Err("built without the antichess-tablebases feature".into())
    }
}

#[cfg(test)]
pub mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::antichess::rules;

    /// The tables the tests read, from `PROVARI_ANTICHESS_TABLEBASES`:
    /// tolius's whole set, or the few that niklasf's antichess-tb-sys holds
    /// for its own tests. Without it, the tests that need them pass without them.
    pub fn tables() -> Option<Tablebases> {
        let dir = PathBuf::from(std::env::var_os("PROVARI_ANTICHESS_TABLEBASES")?);
        Some(Tablebases::open(&dir).unwrap())
    }

    /// The probes antichess-tb-sys's own test makes, which its few tables
    /// answer: (FEN, the plies for the side to move, positive when it wins,
    /// negative when it loses, 0 when drawn).
    const REFERENCE: [(&str, i32); 21] = [
        ("7k/8/5K2/8/8/1R6/P7/8 w - - 0 1", 50),
        ("8/8/8/8/8/3k4/8/1R6 b - - 0 1", -35),
        ("8/8/8/8/8/P7/8/7k b - - 0 1", -45),
        ("8/8/K7/8/8/6b1/8/3b4 w - - 0 1", -47),
        ("8/4p3/8/8/8/P7/P7/8 w - - 99 105", 68),
        ("2Q2R2/8/8/8/8/8/8/1K5k w - - 0 1", 110),
        ("8/1P6/8/8/8/K2P4/8/2k5 w - - 0 1", 142),
        ("8/5N2/5B2/8/k7/8/8/2K5 w - - 0 1", 148),
        ("8/8/8/8/8/5P2/1B6/2N3k1 w - - 0 1", 174),
        ("8/8/8/8/8/5Pp1/3B4/6N1 w - - 0 1", 174),
        ("8/8/8/5N2/8/B4P2/8/6k1 w - - 0 1", 162),
        ("8/8/8/8/3N4/B4P2/8/6k1 b - - 0 1", -161),
        ("8/5P2/8/8/3N4/B7/8/7k w - - 0 1", 72),
        ("8/2NK4/8/8/8/B2k4/8/8 b - - 0 1", -105),
        ("8/2NK4/8/8/8/B7/2k5/8 w - - 0 1", 104),
        ("8/8/8/8/k7/8/8/K3K1N1 w - - 0 1", 104),
        ("8/8/8/8/k7/8/4N3/K3K3 b - - 1 1", 0),
        ("8/8/8/8/1k6/8/4N3/K3K3 w - - 0 1", 0),
        ("8/p7/8/1P6/7p/7R/8/8 b - - 0 25", 4),
        ("8/8/8/pP6/7p/7R/8/8 w - a6 0 26", -3),
        ("8/8/8/pP6/8/5p2/1P6/8 w - a6 0 26", -11),
    ];

    #[test]
    fn reads_what_antichess_tb_sys_reads() {
        let Some(tables) = tables() else {
            return;
        };
        for (fen, plies) in REFERENCE {
            let expected = match plies {
                0 => Entry::Draw,
                1.. => Entry::Win(plies.unsigned_abs()),
                ..0 => Entry::Loss(plies.unsigned_abs()),
            };
            let pos = rules::parse_fen(fen).unwrap();
            assert_eq!(tables.probe(&pos), Some(expected), "{fen}");
        }
        // Without the en passant capture, the last is drawn.
        let pos = rules::parse_fen("8/8/8/pP6/8/5p2/1P6/8 w - - 0 26").unwrap();
        assert_eq!(tables.probe(&pos), Some(Entry::Draw));
        // Five pieces are beyond them.
        let pos = rules::parse_fen("3N4/8/8/6Br/2pP4/8/8/1Q6 b - - 0 1").unwrap();
        assert_eq!(tables.probe(&pos), None);
    }

    /// A bishop and a pawn a side, or what may become them by promoting, with
    /// a white pawn below a black one on its file, are left to the search;
    /// with pawns on other files or past each other, another piece, three
    /// pieces on one side, a side without a pawn, or three pieces in all, they
    /// are not.
    #[test]
    fn leaves_what_may_get_stuck_to_the_search() {
        let cases = [
            ("8/7b/8/8/5p2/5P2/8/B7 w - - 0 1", true),
            ("8/7b/5p2/8/8/5P2/8/B7 w - - 0 1", true),
            ("8/8/8/8/5p2/5P2/7p/B7 w - - 0 1", true),
            ("8/7p/8/8/5p2/5P2/1P6/8 w - - 0 1", true),
            ("8/7b/8/8/4p3/5P2/8/B7 w - - 0 1", false),
            ("8/7b/8/5P2/5p2/8/8/B7 w - - 0 1", false),
            ("8/7b/8/8/5p2/5P2/8/N7 w - - 0 1", false),
            ("8/7b/8/8/8/5P2/1P6/B7 w - - 0 1", false),
            ("8/8/8/8/5p2/7p/8/BB6 w - - 0 1", false),
            ("8/7b/8/8/5p2/5P2/8/8 w - - 0 1", false),
        ];
        for (fen, stuck) in cases {
            let pos = rules::parse_fen(fen).unwrap();
            assert_eq!(may_get_stuck(pos.board()), stuck, "{fen}");
        }
    }
}
