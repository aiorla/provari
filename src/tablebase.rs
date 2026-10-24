//! Tablebase verdicts for the side to move, trusted only right after a capture
//! or pawn move, where the 50-move counter is zero.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use shakmaty::{Position, variant::Atomic};
use shakmaty_syzygy::filesystem::{Filesystem, RandomAccessFile, ReadHint};
use shakmaty_syzygy::{Tablebase as Tables, Wdl};

pub struct Tablebase {
    tables: Tables<Atomic>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Verdict {
    Win,
    Loss,
    /// A draw, or a result the 50-move rule can spoil.
    Other,
}

impl Tablebase {
    /// The tables in a directory, or in several listed as `PATH` lists them
    /// (separated by `:`, or `;` on Windows), which only native builds can
    /// read. A directory without any is an error, as its tables are likely in
    /// folders below it.
    #[cfg(any(unix, windows))]
    pub fn open(directories: &Path) -> io::Result<Tablebase> {
        let mut tables = Tables::new();
        for directory in std::env::split_paths(directories) {
            let named = |error: io::Error| {
                io::Error::new(error.kind(), format!("{}: {error}", directory.display()))
            };
            if tables.add_directory(&directory).map_err(named)? == 0 {
                let none = io::Error::new(io::ErrorKind::NotFound, "no Atomic tables");
                return Err(named(none));
            }
        }
        Ok(Tablebase { tables })
    }

    /// Tables from files held in memory, as the browser build has them.
    pub fn from_memory(files: Arc<Memory>) -> io::Result<Tablebase> {
        let mut tables = Tables::with_filesystem(files);
        tables.add_directory(Memory::ROOT)?;
        Ok(Tablebase { tables })
    }

    /// The most pieces the tables hold.
    pub fn max_pieces(&self) -> usize {
        self.tables.max_pieces()
    }

    /// The verdict for a position the caller knows the tables should hold:
    /// an error, not a missing verdict, where they cannot give one.
    pub fn probe_held(&self, pos: &Atomic) -> Result<Verdict, String> {
        match self.tables.probe_wdl_after_zeroing(pos) {
            Ok(Wdl::Win) => Ok(Verdict::Win),
            Ok(Wdl::Loss) => Ok(Verdict::Loss),
            Ok(_) => Ok(Verdict::Other),
            Err(error) => Err(format!("no tablebase verdict: {error}")),
        }
    }

    pub fn probe(&self, pos: &Atomic) -> Option<Verdict> {
        let pieces = pos.board().occupied().count();
        if pieces > self.tables.max_pieces() || pos.halfmoves() != 0 || pos.castles().any() {
            return None;
        }
        match self.tables.probe_wdl_after_zeroing(pos).ok()? {
            Wdl::Win => Some(Verdict::Win),
            Wdl::Loss => Some(Verdict::Loss),
            _ => Some(Verdict::Other),
        }
    }
}

/// Table files in memory, by name, for the tables to read as if from a
/// directory: the browser build's, which has no files.
#[derive(Default)]
pub struct Memory {
    files: HashMap<PathBuf, Arc<[u8]>>,
}

impl Memory {
    /// The directory the files appear in.
    pub const ROOT: &'static str = "/tables";

    pub fn add(&mut self, name: &str, bytes: Vec<u8>) {
        self.files
            .insert(Path::new(Memory::ROOT).join(name), bytes.into());
    }

    fn file(&self, path: &Path) -> io::Result<&Arc<[u8]>> {
        self.files
            .get(path)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, path.display().to_string()))
    }
}

impl Filesystem for Memory {
    fn regular_file_size(&self, path: &Path) -> io::Result<u64> {
        Ok(self.file(path)?.len() as u64)
    }

    fn read_dir(&self, _path: &Path) -> io::Result<Vec<PathBuf>> {
        Ok(self.files.keys().cloned().collect())
    }

    fn open(&self, path: &Path) -> io::Result<Box<dyn RandomAccessFile>> {
        Ok(Box::new(MemoryFile(self.file(path)?.clone())))
    }
}

struct MemoryFile(Arc<[u8]>);

impl RandomAccessFile for MemoryFile {
    fn read_at(&self, buf: &mut [u8], offset: u64, _hint: ReadHint) -> io::Result<usize> {
        let start = (offset as usize).min(self.0.len());
        let length = buf.len().min(self.0.len() - start);
        buf[..length].copy_from_slice(&self.0[start..start + length]);
        Ok(length)
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::Tablebase;

    /// A directory for a test's files, empty, under the system's.
    fn directory(name: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!("provari-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    /// A file named as a table, of a size tables have: the tables open their
    /// files only when probed.
    fn table(directory: &Path, name: &str) {
        std::fs::write(directory.join(name), [0; 16]).unwrap();
    }

    /// Tables in several directories, listed as `PATH` lists them, as the
    /// six-piece ones may be kept apart from the rest; a listed directory
    /// without any is an error that names it.
    #[test]
    fn opens_tables_from_several_directories() {
        let small = directory("small");
        let large = directory("large");
        let empty = directory("empty");
        table(&small, "KQvK.atbw");
        table(&small, "KQvK.atbz");
        table(&large, "KQRvKQR.atbw");
        let list = |directories: &[&Path]| std::env::join_paths(directories).unwrap();
        let open = |directories: &[&Path]| Tablebase::open(Path::new(&list(directories)));
        assert_eq!(open(&[&small]).unwrap().max_pieces(), 3);
        assert_eq!(open(&[&small, &large]).unwrap().max_pieces(), 6);
        let error = open(&[&small, &empty]).err().unwrap().to_string();
        assert!(error.contains("no Atomic tables"), "{error}");
        assert!(error.contains(&empty.display().to_string()), "{error}");
        let missing = small.join("missing");
        let error = open(&[&missing]).err().unwrap().to_string();
        assert!(error.contains(&missing.display().to_string()), "{error}");
        for directory in [small, large, empty] {
            std::fs::remove_dir_all(directory).unwrap();
        }
    }
}
