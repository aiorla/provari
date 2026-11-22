//! tolius's reader of his tables (`tb/egtb/tb_reader.cpp` of his
//! antichess-tb-api, GPL version 3 or later), in Rust: the same answers,
//! with every table read in place and without locks.
//!
//! An entry holds the plies to the end for the side to move, positive when
//! it wins, and its plies to a zeroing move. Positions with an en passant
//! capture are not stored: they are worked out from the positions after
//! each capture (`probe_ep`), as his reader does. Where his reader stops on
//! something it does not expect, this one answers nothing.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use shakmaty::{Color, EnPassantMode, Position, Role, variant::Antichess};

use super::file::{Kept, TableFile};
use super::index::{self, Index, Material, Placed};
use super::{DEFAULT_KEPT, Entry, Unpacked};

const COMPRESSED: u16 = 1 << 0;
const HAS_DTZ: u16 = 1 << 1;
const VAL_BIG: u16 = 1 << 2;
const VAL_12BIT: u16 = 1 << 3;
const DTZ101_AS_DRAW: u16 = 1 << 4;
const SKIP_ONLY_MOVES: u16 = 1 << 6;
const VAL_DTZ_SEPARATED: u16 = 1 << 7;

const DRAW: i16 = 0x3333;
const NONE: i16 = 0x5555;
const DTZ_MAX: u8 = 101;
/// The most bytes of a table's head that its index reads.
const MAX_IDX_SIZE: usize = 28;

/// Something the reader did not expect, for which it answers nothing.
struct Unexpected;

type Probed = Result<(i16, u8), Unexpected>;

/// One table: its file, its index, and whether its values take two bytes.
struct Table {
    file: TableFile,
    index: Index,
    big: bool,
}

/// A set of tables, open for the whole process.
pub struct Set {
    dir: PathBuf,
    /// Each table by the material it holds, either way round.
    tables: HashMap<u64, usize>,
    list: Vec<Table>,
    kept: Kept,
    /// The `.an0` files in the folder.
    count: usize,
}

impl fmt::Debug for Set {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Set({}, {} tables)", self.dir.display(), self.count)
    }
}

/// The sets this process opened, each opened once.
static OPENED: Mutex<Vec<&'static Set>> = Mutex::new(Vec::new());

fn key_of(material: &Material) -> u64 {
    material
        .iter()
        .flat_map(|side| side[1..].iter())
        .fold(0, |key, &n| key << 4 | u64::from(n))
}

fn table_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let failed = |error: std::io::Error| format!("cannot read {}: {error}", dir.display());
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir).map_err(failed)? {
        let path = entry.map_err(failed)?.path();
        if path.extension().is_some_and(|extension| extension == "an0") && path.is_file() {
            paths.push(path);
        }
    }
    if paths.is_empty() {
        return Err(format!("{} holds no tablebases", dir.display()));
    }
    paths.sort();
    Ok(paths)
}

/// Opens the tables in `dir`, once for the process.
pub fn open(dir: &Path) -> Result<&'static Set, String> {
    let mut opened = OPENED.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some(set) = opened.iter().find(|set| set.dir == dir) {
        return Ok(set);
    }
    let set: &'static Set = Box::leak(Box::new(load(dir)?));
    opened.push(set);
    Ok(set)
}

fn load(dir: &Path) -> Result<Set, String> {
    let mut set = Set {
        dir: dir.to_path_buf(),
        tables: HashMap::new(),
        list: Vec::new(),
        kept: Kept::default(),
        count: 0,
    };
    for path in table_files(dir)? {
        let name = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| format!("{}: not a table's name", path.display()))?;
        let material = index::material(name, false)
            .filter(|material| index::name(material) == name)
            .ok_or_else(|| format!("{}: not a table's name", path.display()))?;
        let file = TableFile::open(&path)?;
        let table = table(name, file).map_err(|why| format!("{}: {why}", path.display()))?;
        let swapped = index::material(name, true).expect("as above");
        set.tables.insert(key_of(&material), set.list.len());
        set.tables.insert(key_of(&swapped), set.list.len());
        set.list.push(table);
        set.count += 1;
    }
    set.kept.set_limit(DEFAULT_KEPT);
    Ok(set)
}

fn table(name: &str, file: TableFile) -> Result<Table, String> {
    let flags = file.flags();
    let unknown = VAL_12BIT | DTZ101_AS_DRAW | SKIP_ONLY_MOVES | VAL_DTZ_SEPARATED;
    if flags & (COMPRESSED | HAS_DTZ) != COMPRESSED | HAS_DTZ || flags & unknown != 0 {
        return Err(format!("tables of flags {flags:#x} are not read"));
    }
    if file.length == 0 {
        return Err("an empty table".into());
    }
    let kept = Kept::default();
    let mut head = [0; MAX_IDX_SIZE];
    let n = (MAX_IDX_SIZE as u64).min(file.length) as usize;
    file.read(0, &mut head[..n], &kept)?;
    let index = Index::new(name, &head[..n])?;
    if index.header_size == 0 || index.header_size > MAX_IDX_SIZE {
        return Err(format!("wrong idx data size: {}", index.header_size));
    }
    Ok(Table {
        file,
        index,
        big: flags & VAL_BIG != 0,
    })
}

impl Set {
    pub fn count(&self) -> usize {
        self.count
    }

    pub fn keep(&self, bytes: usize) {
        self.kept.set_limit(bytes);
    }

    pub fn kept(&self) -> usize {
        self.kept.bytes()
    }

    /// What the tables say of `pos`, through the answer of his API
    /// (`antichess_tb_probe_dtw`): the plies to the end, or a draw.
    pub fn probe(&self, pos: &Antichess) -> Option<Entry> {
        let (value, _) = self.probe_tb(pos).ok()?;
        if i32::from(value).abs() >= i32::from(DRAW) {
            return Some(Entry::Draw);
        }
        match value {
            1.. => Some(Entry::Win(value.unsigned_abs().into())),
            ..0 => Some(Entry::Loss(value.unsigned_abs().into())),
            0 => None,
        }
    }

    /// A position's value and its plies to a zeroing move (`probe_tb`): 0
    /// and 0 when the side to move has won already.
    fn probe_tb(&self, pos: &Antichess) -> Probed {
        if anti_win(pos) {
            return Ok((0, 0));
        }
        let placed = placed(pos);
        let table = &self.list[*self
            .tables
            .get(&key_of(&placed.material))
            .ok_or(Unexpected)?];
        if pos.ep_square(EnPassantMode::Legal).is_some() {
            self.probe_ep(table, pos)
        } else {
            self.read_one(table, &placed)
        }
    }

    /// A position of `table`'s material whose game goes on, without an en
    /// passant capture (`probe_one`).
    fn probe_one(&self, table: &Table, pos: &Antichess) -> Probed {
        if anti_win(pos) {
            return Ok((0, 0));
        }
        if pos.ep_square(EnPassantMode::Legal).is_some() {
            return Err(Unexpected);
        }
        self.read_one(table, &placed(pos))
    }

    fn read_one(&self, table: &Table, placed: &Placed) -> Probed {
        let key = table.index.index(placed);
        let size = if table.big { 3 } else { 2 };
        let at = (table.index.header_size as u64)
            .checked_add(key.checked_mul(size).ok_or(Unexpected)?)
            .ok_or(Unexpected)?;
        if at + size > table.file.length {
            return Err(Unexpected);
        }
        let mut bytes = [0; 3];
        table
            .file
            .read(at, &mut bytes[..size as usize], &self.kept)
            .map_err(|_| Unexpected)?;
        // The plies to a zeroing move, then the value (`load_one`).
        let dtz = bytes[0];
        let mut value = if table.big {
            i16::from_be_bytes([bytes[1], bytes[2]])
        } else {
            i16::from(bytes[1] as i8)
        };
        if dtz == 0 && value == 0 {
            value = DRAW;
        }
        if value == NONE || i32::from(dtz) > i32::from(value).abs() || dtz > DTZ_MAX {
            return Err(Unexpected);
        }
        Ok((value, dtz))
    }

    /// A position with an en passant capture, from the positions after each
    /// of its moves (`probe_ep`): the best value for the side to move, and
    /// among equal ones the fewest plies to a zeroing move.
    fn probe_ep(&self, table: &Table, pos: &Antichess) -> Probed {
        let moves = pos.legal_moves();
        if moves.is_empty() {
            return Err(Unexpected);
        }
        let mut values = Vec::with_capacity(moves.len());
        let mut score = -2;
        for &m in &moves {
            let capture = m.is_capture();
            let capture_or_promotion = capture || m.is_promotion();
            let zeroing = capture || m.role() == Role::Pawn;
            let mut after = pos.clone();
            after.play_unchecked(m);
            let win = anti_win(&after);
            let loss = !win && after.board().by_color(!after.turn()).is_empty();
            let (value, dtz) = if loss {
                (1, 1)
            } else if win {
                (-1, 1)
            } else {
                if after.ep_square(EnPassantMode::Legal).is_some() {
                    return Err(Unexpected);
                }
                let (value, dtz) = if capture_or_promotion {
                    let (value, _) = self.probe_tb(&after)?;
                    (value, 1)
                } else {
                    let (value, dtz) = self.probe_one(table, &after)?;
                    (value, if zeroing { 1 } else { dtz.wrapping_add(1) })
                };
                if value == NONE {
                    return Err(Unexpected);
                }
                let value = if value == DRAW {
                    value
                } else if value < 0 {
                    (-i32::from(value) + 1) as i16
                } else {
                    (-i32::from(value) - 1) as i16
                };
                (value, dtz)
            };
            let this = match (value, dtz) {
                (..=-1, ..=100) => -2,
                (..=-1, _) => -1,
                (DRAW, _) => 0,
                (1.., 101..) => 1,
                (1.., _) => 2,
                _ => return Err(Unexpected),
            };
            score = score.max(this);
            values.push((value, dtz));
        }
        if score == 0 {
            return Ok((DRAW, 0));
        }
        let (mut best, mut best_dtz) = (if score > 0 { DRAW } else { 0 }, u8::MAX);
        for &(value, dtz) in &values {
            if value == DRAW {
                if score <= 0 {
                    return Err(Unexpected);
                }
                continue;
            }
            if score > 0 {
                if (score == 1 || dtz <= 100)
                    && ((0 < value && value < best) || (value == best && dtz < best_dtz))
                {
                    (best, best_dtz) = (value, dtz);
                    if value == 1 {
                        break;
                    }
                }
            } else if (score == -1 || dtz <= 100)
                && (value < best || (value == best && dtz < best_dtz))
            {
                (best, best_dtz) = (value, dtz);
            }
            if (score >= 0 || value >= 0) && score <= 0 {
                return Err(Unexpected);
            }
        }
        if best == 0 || best == DRAW {
            return Err(Unexpected);
        }
        if (score <= 0 || best <= 0) && (score >= 0 && best >= 0) {
            return Err(Unexpected);
        }
        Ok((best, best_dtz))
    }
}

/// Whether the side to move has won: it has no pieces or no moves.
fn anti_win(pos: &Antichess) -> bool {
    pos.board().by_color(pos.turn()).is_empty() || pos.legal_moves().is_empty()
}

/// A position as the index reads it, its pieces numbered as Stockfish's.
fn placed(pos: &Antichess) -> Placed {
    let board = pos.board();
    let mut placed = Placed {
        board: [0; 64],
        occupied: board.occupied().0,
        turn: usize::from(pos.turn() == Color::Black),
        material: [[0; 7]; 2],
    };
    for (square, piece) in board {
        let color = usize::from(piece.color == Color::Black);
        let role = piece.role as usize;
        placed.board[square as usize] = role as u8 + 8 * color as u8;
        placed.material[color][role] += 1;
    }
    placed
}

/// Writes each table of `packed` unpacked into `plain`, a new folder, each
/// taking its name only once checked.
pub fn unpack(packed: &Path, plain: &Path) -> Result<Unpacked, String> {
    let paths = table_files(packed)?;
    fs::create_dir(plain).map_err(|error| format!("{}: {error}", plain.display()))?;
    let mut bytes = 0;
    for from in &paths {
        let name = from.file_name().expect("a file");
        let to = plain.join(name);
        let partial = to.with_extension("an0.partial");
        bytes += super::file::unpack(from, &partial)
            .map_err(|why| format!("{why}, which leaves {} unfinished", plain.display()))?;
        fs::rename(&partial, &to).map_err(|error| format!("{}: {error}", to.display()))?;
    }
    Ok(Unpacked {
        tables: paths.len(),
        bytes,
    })
}
