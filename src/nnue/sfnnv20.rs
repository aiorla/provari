//! varifish's Atomic networks, pruned for provari: Stockfish's design,
//! SFNNv20, under Atomic's rules. Three sets of inputs feed the first layer's
//! values: K32H3's piece-square inputs (each piece by its square and the
//! perspective's king square, the board mirrored so that the king is on
//! files e to h, in one of three blocks by the heavy pieces on the board),
//! FullThreats (each piece attacking another, by the types, colors and
//! squares of both) and PP_3Wide (each pair of pawns on the same or adjacent
//! files). Pairs of the clipped values are multiplied into the outputs, the
//! side to move's half first, which layer stacks of 1024 → 32 → 32 → 1
//! read: one of 192 by the pieces, the queens, the pawns' halves of the board
//! and the officers, of which a file stores those its table says.
//!
//! varifish's networks have 1024 values a side, 512 outputs. Half of those
//! outputs are almost never above zero, so provari keeps 256 a side, 512
//! values (`prune`, which writes provari's own file, `PRUNED`): the rows the
//! search adds for every move are half as long, and the values are
//! varifish's but for the outputs dropped. The outputs kept sit first in
//! each side's 512 of the dense layers' inputs, the rest zero, so the
//! provability model reads the same 1024 places as with run3b. A pruned file
//! also carries the factor that brings the network's values to run3b's
//! units, which the search's thresholds and the model count in.
//!
//! A position's first layer is kept in two parts, which the transform adds:
//! the piece-square sums, which a move of the side's own king or a change of
//! its block starts again from a cache, and the threats' and pairs' sums,
//! which follow from the parent's unless the king crosses the middle of the
//! board. The threats a move changes come from each square's row of attacked
//! squares: only the rows of the changed squares and of the pieces that
//! attacked them are worked out again.
//!
//! A translation into Rust, modified for provari in 2026, of the evaluation
//! of varifish (<https://github.com/aiorla/varifish>), whose networks are
//! Stockfish's design: Stockfish, a UCI chess playing engine derived from
//! Glaurung 2.1, Copyright (C) 2004-2026 The Stockfish developers (see
//! Stockfish's AUTHORS file), under the GNU General Public License, version
//! 3 or later.

use std::mem::MaybeUninit;

use shakmaty::{Bitboard, Color, Role, Square, attacks};

use super::legacy::{Reader, Transformed, changed_squares};
use crate::board::Position;

/// The version varifish's files start with.
pub(super) const VERSION: u32 = 0x6A44_8AFA;
/// The version provari's pruned files start with: "PRV2" read backwards.
pub(super) const PRUNED: u32 = 0x5052_5632;
/// The first layer's values a side in varifish's networks.
const FILE_HALF: usize = 1024;
/// The first layer's values a side that provari keeps: 256 outputs' pairs.
const HALF: usize = 512;
/// The dense layers' inputs, both sides' 512 places, as `Transformed` holds
/// them; the outputs kept fill the first HALF / 2 of each side's.
const INPUTS: usize = 1024;
/// The threat inputs, then the pawn pairs', whose rows follow theirs.
const THREATS: usize = 59_808;
const PAIRS: usize = 96 * 95 / 2;
/// The piece-square inputs of a block: 32 king buckets of 11 planes.
const BLOCK: usize = 32 * 11 * 64;
const BLOCKS: usize = 3;
/// The layer stacks a position chooses among.
const STACKS: usize = 192;
const FC0: usize = 32;
const FC1: usize = 32;
/// The inputs of the last dense layer: both activations of both layers
/// before it.
const LAST: usize = 2 * FC0 + 2 * FC1;
/// The most squares a move changes: a capture's two and the eight beside it.
const MOVED: usize = 10;
/// The most rows a list holds: 8 threats from each of 64 squares, or every
/// threat of a board with every pair of its pawns.
const LISTED: usize = 640;
const WEIGHT_SCALE_BITS: u32 = 6;
/// The chunks of four of the first layer's outputs.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_feature = "simd128"
))]
const CHUNKS: usize = INPUTS / 4;

/// `change_rows` with a processor's vectors: `$registers` registers of
/// `$lanes` values at a time, every row applied to them before they are
/// stored, so that the values are read and written once however many rows
/// change. `$widen` reads `$lanes` weights as values. Every row must lie
/// inside `$weights`.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_feature = "simd128"
))]
macro_rules! change_rows_in_tiles {
    (
        $lanes:expr, $registers:expr, $load:ident, $store:ident, $add:ident, $sub:ident,
        $widen:ident, $weights:expr, $from:expr, $to:expr, $removed:expr, $added:expr
    ) => {{
        const LANES: usize = $lanes;
        const REGISTERS: usize = $registers;
        const { assert!(HALF.is_multiple_of(LANES * REGISTERS)) };
        let (weights, from, to) = ($weights.as_ptr(), $from.as_ptr(), $to.as_mut_ptr());
        for tile in (0..HALF).step_by(LANES * REGISTERS) {
            let mut sums = [$load(from.add(tile).cast()); REGISTERS];
            for (register, sum) in sums.iter_mut().enumerate().skip(1) {
                *sum = $load(from.add(tile + register * LANES).cast());
            }
            for &row in $removed {
                let row = weights.add(row as usize * HALF + tile);
                for (register, sum) in sums.iter_mut().enumerate() {
                    *sum = $sub(*sum, $widen(row.add(register * LANES)));
                }
            }
            for &row in $added {
                let row = weights.add(row as usize * HALF + tile);
                for (register, sum) in sums.iter_mut().enumerate() {
                    *sum = $add(*sum, $widen(row.add(register * LANES)));
                }
            }
            for (register, sum) in sums.iter().enumerate() {
                $store(to.add(tile + register * LANES).cast(), *sum);
            }
        }
    }};
}

/// What sets the inputs of Atomic's networks apart from chess's: "Atom".
const ATOMIC_HASH: u32 = 0x4174_6F6D;
const TABLE_HASH: u32 = 0x5A7A_B1E5;
/// The hash of the inputs: FullThreats', PP_3Wide's and K32H3's, for a
/// first layer of `half` values a side.
const fn transformer_hash(half: usize) -> u32 {
    let mut hash = 0x2C41_EAEAu32;
    hash = hash.rotate_left(1) ^ 0x5C80_FA69;
    hash = hash.rotate_left(1) ^ 0xAB00_5CEF;
    hash ^ ATOMIC_HASH ^ (2 * half as u32)
}

/// The hash the layer stacks' architecture writes into the file, for a
/// first layer of `half` values a side.
const fn stack_hash(half: usize) -> u32 {
    const fn affine(previous: u32, outputs: u32) -> u32 {
        (0xCC03_DAE4u32.wrapping_add(outputs) ^ (previous >> 1)) ^ (previous << 31)
    }
    const fn relu(previous: u32) -> u32 {
        0x538D_24C7u32.wrapping_add(previous)
    }
    let hash = 0xEC42_E90Du32 ^ (2 * half as u32);
    let hash = relu(affine(hash, FC0 as u32));
    let hash = relu(affine(hash, FC1 as u32));
    affine(hash, 1)
}

const WHITE: usize = Color::White as usize;
const PAWN: usize = Role::Pawn as usize;
const KNIGHT: usize = Role::Knight as usize;
const BISHOP: usize = Role::Bishop as usize;
const ROOK: usize = Role::Rook as usize;
const QUEEN: usize = Role::Queen as usize;
const KING: usize = Role::King as usize;

/// What turns a square into the one a side's inputs see: Black's board is
/// flipped, and a board whose king is on files a to d is mirrored.
const fn orientation(side: usize, king: usize) -> usize {
    (if king & 7 < 4 { 7 } else { 0 }) ^ (if side == WHITE { 0 } else { 56 })
}

/// For each side, square of its king and piece code, what a square is xored
/// with to give the piece's input in its block: the orientation in the low
/// six bits, above them the king's bucket and the piece's plane, the side's
/// own pieces of a role, then the other side's, and both kings in the last.
static PIECE_SQUARES: [[[u16; 16]; 64]; 2] = {
    let mut table = [[[0; 16]; 64]; 2];
    let mut side = 0;
    while side < 2 {
        let mut king = 0;
        while king < 64 {
            let seen = king ^ (if side == WHITE { 0 } else { 56 });
            let (rank, file) = (seen >> 3, seen & 7);
            let bucket = (7 - rank) * 4 + if file < 4 { file } else { 7 - file };
            let mut code = 0;
            while code < 16 {
                let (role, white) = (code & 7, code & 8 != 0);
                if role >= PAWN && role <= KING {
                    let plane = if role == KING {
                        10
                    } else {
                        2 * (role - 1) + (white != (side == WHITE)) as usize
                    };
                    table[side][king][code] =
                        ((bucket * 11 + plane) * 64 + orientation(side, king)) as u16;
                }
                code += 1;
            }
            king += 1;
        }
        side += 1;
    }
    table
};

/// A piece's attacks on an empty board, a pawn's by its color.
const fn reach(role: usize, white: bool, square: usize) -> u64 {
    let square = Square::new(square as u32);
    let color = if white { Color::White } else { Color::Black };
    match role {
        PAWN => attacks::pawn_attacks(color, square).0,
        KNIGHT => attacks::knight_attacks(square).0,
        BISHOP => attacks::bishop_attacks(square, Bitboard(0)).0,
        ROOK => attacks::rook_attacks(square, Bitboard(0)).0,
        QUEEN => attacks::queen_attacks(square, Bitboard(0)).0,
        _ => attacks::king_attacks(square).0,
    }
}

/// A threat's input is the sum of two numbers these tables hold.
struct ThreatTables {
    /// By the attacker's code times 16 plus the attacked piece's, as White
    /// sees them, and by whether the attacker's square is below the attacked
    /// one's: where the pair's inputs start, or THREATS and beyond for a
    /// pair without inputs. Pieces of one type attack each other alike, so
    /// only the attack from the higher square has an input.
    pairs: [[u32; 2]; 256],
    /// By the attacker's role and both squares, oriented: how many attacks
    /// its role makes from the squares before its own, then from its own on
    /// the squares below the attacked one. A pawn's attacks are Black's
    /// where the attacked square is below its own, else White's.
    squares: [[[u16; 64]; 64]; 7],
}

static THREAT_TABLES: ThreatTables = {
    // The place of each attacked role among an attacker's targets, or -1
    // where there is no input: a pawn's are knights and rooks (pawn pairs
    // have their own inputs), and no piece's are kings.
    const TARGETS: [[i32; 6]; 6] = [
        [-1, 0, -1, 1, -1, -1],
        [0, 1, 2, 3, 4, -1],
        [0, 1, 2, 3, -1, -1],
        [0, 1, 2, 3, -1, -1],
        [0, 1, 2, 3, 4, -1],
        [-1, -1, -1, -1, -1, -1],
    ];
    const TARGET_COUNTS: [i32; 7] = [0, 4, 10, 8, 8, 10, 0];
    let mut tables = ThreatTables {
        pairs: [[THREATS as u32; 2]; 256],
        squares: [[[0; 64]; 64]; 7],
    };
    let mut attacks_of = [0i32; 7];
    let mut role = PAWN;
    while role <= KING {
        let mut offset = 0u32;
        let mut from = 0;
        while from < 64 {
            let mut to = 0;
            while to < 64 {
                let reached = reach(role, to >= from, from);
                tables.squares[role][from][to] =
                    (offset + (((1u64 << to) - 1) & reached).count_ones()) as u16;
                to += 1;
            }
            if role != PAWN || (from >= 8 && from < 56) {
                offset += reach(role, true, from).count_ones();
            }
            from += 1;
        }
        attacks_of[role] = offset as i32;
        role += 1;
    }
    let mut start = 0i32;
    // White's pieces by role, then Black's, as the engine numbers them.
    let mut attacker = 0;
    while attacker < 12 {
        let (attacker_white, attacker_role) = (attacker < 6, attacker % 6 + 1);
        let count = TARGET_COUNTS[attacker_role];
        let mut attacked = 0;
        while attacked < 12 {
            let (attacked_white, attacked_role) = (attacked < 6, attacked % 6 + 1);
            let place = TARGETS[attacker_role - 1][attacked_role - 1];
            if place >= 0 {
                let enemy = attacker_white != attacked_white;
                let twin = attacker_role == attacked_role && (enemy || attacker_role != PAWN);
                let color = if attacked_white { 0 } else { count / 2 };
                let input = (start + (color + place) * attacks_of[attacker_role]) as u32;
                let pair = (attacker_role | if attacker_white { 8 } else { 0 }) << 4
                    | (attacked_role | if attacked_white { 8 } else { 0 });
                tables.pairs[pair][0] = input;
                if !twin {
                    tables.pairs[pair][1] = input;
                }
            }
            attacked += 1;
        }
        start += count * attacks_of[attacker_role];
        attacker += 1;
    }
    assert!(start as usize == THREATS);
    tables
};

/// The squares that can hold a pawn making a pair with a pawn on each
/// square: its file and the ones beside it, on ranks 2 to 7, without the
/// square itself.
static PAIRED: [u64; 64] = {
    let mut table = [0; 64];
    let mut square = 0;
    while square < 64 {
        let file = square & 7;
        let mut files = 0x0101_0101_0101_0101u64 << file;
        if file > 0 {
            files |= files >> 1;
        }
        if file < 7 {
            files |= (0x0101_0101_0101_0101u64 << file) << 1;
        }
        table[square] = files & 0x00ff_ffff_ffff_ff00 & !(1 << square);
        square += 1;
    }
    table
};

/// The squares set in a bitboard, from the lowest.
fn squares(mut bits: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        (bits != 0).then(|| {
            let square = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            square
        })
    })
}

/// A threat's input for a side whose orientation is `orient`: THREATS or
/// more where the threat has none.
fn threat(side: usize, orient: usize, attacker: u8, from: usize, to: usize, attacked: u8) -> u32 {
    // Black sees both pieces with their colors swapped.
    let swap = if side == WHITE { 0 } else { 0x88 };
    let pair = (usize::from(attacker) << 4 | usize::from(attacked)) ^ swap;
    let (from, to) = (from ^ orient, to ^ orient);
    THREAT_TABLES.pairs[pair][usize::from(from < to)]
        + u32::from(THREAT_TABLES.squares[usize::from(attacker & 7)][from][to])
}

/// The rows of the pairs that the pawns `updated` make among the pawns
/// `white` and `black`, each pair once, for a side whose orientation is
/// `orient`.
fn pairs(side: usize, orient: usize, updated: u64, white: u64, black: u64, rows: &mut List) {
    // A pawn's number: its square from a2, the other side's pawns after
    // the side's own.
    let id = |square: usize, is_white: bool| {
        let theirs = usize::from(is_white != (side == WHITE));
        (48 * theirs + (square ^ orient)).wrapping_sub(8)
    };
    let unchanged = (white | black) & !updated;
    let mut left = updated;
    while left != 0 {
        let pawn = left.trailing_zeros() as usize;
        left &= left - 1;
        let first = id(pawn, white >> pawn & 1 != 0);
        for partner in squares(PAIRED[pawn] & (unchanged | left)) {
            let second = id(partner, white >> partner & 1 != 0);
            let (low, high) = (first.min(second), first.max(second));
            let row = THREATS + (high.wrapping_mul(high.wrapping_sub(1)) / 2).wrapping_add(low);
            // Pawns on the first or last rank, which no position holds,
            // would fall outside.
            rows.push_if(row as u32, row < THREATS + PAIRS);
        }
    }
}

/// Rows of weights to take away or to add: up to LISTED, unwritten beyond
/// its length.
struct List {
    rows: [MaybeUninit<u32>; LISTED],
    len: usize,
}

impl List {
    /// An empty list. Only its length is written: built as a value, the
    /// whole list would be filled with zeros.
    #[inline(always)]
    fn new() -> List {
        let mut list = MaybeUninit::<List>::uninit();
        // SAFETY: the length is written, and the rows need no value.
        unsafe {
            (&raw mut (*list.as_mut_ptr()).len).write(0);
            list.assume_init()
        }
    }

    /// Adds a row if `keep`, without a branch on it.
    fn push_if(&mut self, row: u32, keep: bool) {
        self.rows[self.len].write(row);
        self.len += usize::from(keep);
    }

    fn as_slice(&self) -> &[u32] {
        // SAFETY: the first `len` rows are written, and a written
        // MaybeUninit<u32> is a u32.
        unsafe { std::slice::from_raw_parts(self.rows.as_ptr().cast(), self.len) }
    }
}

/// The pieces a first layer was computed from: by square, as codes (0 for
/// none, else the role plus 8 for white), and as bitboards by role and of
/// White's.
#[derive(Clone)]
struct Board {
    squares: [u8; 64],
    roles: [u64; 7],
    white: u64,
}

impl Board {
    fn of(pos: &Position) -> Board {
        let mut roles = [0; 7];
        for role in Role::ALL {
            roles[role as usize] = pos.by_role(role).0;
        }
        Board {
            squares: *pos.squares(),
            roles,
            white: pos.by_color(Color::White).0,
        }
    }

    fn occupied(&self) -> u64 {
        self.roles[PAWN]
            | self.roles[KNIGHT]
            | self.roles[BISHOP]
            | self.roles[ROOK]
            | self.roles[QUEEN]
            | self.roles[KING]
    }

    fn of_side(&self, side: usize) -> u64 {
        if side == WHITE {
            self.white
        } else {
            self.occupied() & !self.white
        }
    }

    /// A side's king's square, or a1 without a king: such positions have
    /// ended, and the engine never evaluates them.
    fn king(&self, side: usize) -> usize {
        (self.roles[KING] & self.of_side(side)).trailing_zeros() as usize & 63
    }

    /// The block of a side's piece-square inputs: 2 if the other side has a
    /// queen, else 1 if the board holds a rook or a queen, else 0.
    fn block(&self, side: usize) -> usize {
        if self.roles[QUEEN] & self.of_side(side ^ 1) != 0 {
            2
        } else {
            usize::from(self.roles[ROOK] | self.roles[QUEEN] != 0)
        }
    }

    /// White's pawns, and Black's.
    fn pawns(&self) -> (u64, u64) {
        let pawns = self.roles[PAWN];
        (pawns & self.white, pawns & !self.white)
    }

    /// The squares a square's piece threatens: the pieces it attacks that
    /// its role has inputs with. Kings threaten nothing and nothing
    /// threatens them, a pawn only knights and rooks, and bishops and rooks
    /// no queens.
    fn row(&self, square: usize) -> u64 {
        let code = self.squares[square];
        let at = Square::new(square as u32);
        let occupied = Bitboard(self.occupied());
        let lesser = self.roles[PAWN] | self.roles[KNIGHT] | self.roles[BISHOP] | self.roles[ROOK];
        match usize::from(code & 7) {
            PAWN => {
                let color = Color::from_white(code & 8 != 0);
                attacks::pawn_attacks(color, at).0 & (self.roles[KNIGHT] | self.roles[ROOK])
            }
            KNIGHT => attacks::knight_attacks(at).0 & (lesser | self.roles[QUEEN]),
            BISHOP => attacks::bishop_attacks(at, occupied).0 & lesser,
            ROOK => attacks::rook_attacks(at, occupied).0 & lesser,
            QUEEN => attacks::queen_attacks(at, occupied).0 & (lesser | self.roles[QUEEN]),
            _ => 0,
        }
    }

    /// The pieces attacking any of `targets`, pieces or not, but the kings.
    fn attackers(&self, targets: u64) -> u64 {
        let occupied = Bitboard(self.occupied());
        let (white_pawns, black_pawns) = self.pawns();
        let diagonal = self.roles[BISHOP] | self.roles[QUEEN];
        let straight = self.roles[ROOK] | self.roles[QUEEN];
        let mut attackers = 0;
        for target in squares(targets) {
            let at = Square::new(target as u32);
            attackers |= attacks::knight_attacks(at).0 & self.roles[KNIGHT]
                | attacks::pawn_attacks(Color::White, at).0 & black_pawns
                | attacks::pawn_attacks(Color::Black, at).0 & white_pawns
                | attacks::bishop_attacks(at, occupied).0 & diagonal
                | attacks::rook_attacks(at, occupied).0 & straight;
        }
        attackers
    }

    /// The layer stack of a position with `us` to move, among the 192: by
    /// its number of pieces, by which sides have a queen, by whether its
    /// pawns stand on both halves of the board, one or none, and by whether
    /// a side has two officers, pieces other than pawns and kings.
    fn stack(&self, us: usize) -> usize {
        let occupied = self.occupied();
        let pieces = (occupied.count_ones() as usize).clamp(1, 32);
        let queens = |side: usize| usize::from(self.roles[QUEEN] & self.of_side(side) != 0);
        const QUEEN_SIDE: u64 = 0x0f0f_0f0f_0f0f_0f0f;
        let pawns = self.roles[PAWN];
        let layout = if pawns == 0 {
            2
        } else if pawns & QUEEN_SIDE != 0 && pawns & !QUEEN_SIDE != 0 {
            0
        } else {
            1
        };
        let officers = occupied & !(pawns | self.roles[KING]);
        let most = (officers & self.white)
            .count_ones()
            .max((officers & !self.white).count_ones());
        ((((pieces - 1) / 4 * 4 + 2 * queens(us) + queens(us ^ 1)) * 3) + layout) * 2
            + usize::from(most >= 2)
    }
}

pub struct Network {
    biases: Box<[i16; HALF]>,
    /// The threats' rows, then the pawn pairs', HALF weights each.
    threats: Vec<i8>,
    /// The piece-square rows, block after block.
    piece_squares: Vec<i8>,
    stacks: Vec<Stack>,
    /// Which of the stored stacks each of the 192 uses.
    stored: [u16; STACKS],
    /// The factor, a numerator and a denominator, that brings the values to
    /// run3b's units.
    scale: (i64, i64),
}

struct Stack {
    b0: [i32; FC0],
    /// The first dense layer's weights by chunk of the first layer: for each
    /// four inputs, the four weights of every output in turn.
    w0: Vec<i8>,
    b1: [i32; FC1],
    /// The second dense layer's weights, a row of 64 an output.
    w1: Vec<i8>,
    b2: i32,
    w2: [i8; LAST],
}

impl Network {
    /// A network from a pruned file's bytes (`prune`'s), which start with
    /// `PRUNED`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Network, String> {
        let mut file = Reader::new(bytes);
        let header = read_header(&mut file, PRUNED, HALF)?;
        let biases = file
            .many(HALF, i16::from_le_bytes)?
            .into_boxed_slice()
            .try_into()
            .expect("HALF biases");
        let threats = file.many((THREATS + PAIRS) * HALF, i8::from_le_bytes)?;
        let piece_squares = file.many(BLOCKS * BLOCK * HALF, i8::from_le_bytes)?;
        let mut stacks = Vec::with_capacity(header.count);
        for _ in 0..header.count {
            let layers = read_stack(&mut file, HALF)?;
            // The file has each output's weights in a row, the side to
            // move's HALF / 2 then the other's; the dense layer reads them
            // at the first HALF / 2 places of each side's 512, by chunk of
            // four inputs: for each, the four weights of every output in
            // turn.
            let mut w0 = vec![0; FC0 * INPUTS];
            for (index, &weight) in layers.w0.iter().enumerate() {
                let (output, input) = (index / HALF, index % HALF);
                let input = input / (HALF / 2) * (INPUTS / 2) + input % (HALF / 2);
                w0[(input / 4 * FC0 + output) * 4 + input % 4] = weight;
            }
            stacks.push(Stack { w0, ..layers });
        }
        if !file.is_empty() {
            return Err("the network file has trailing bytes".into());
        }
        Ok(Network {
            biases,
            threats,
            piece_squares,
            stacks,
            stored: header.stored,
            scale: header.scale,
        })
    }

    /// Computes a position's first layer from scratch.
    pub fn refresh(&self, pos: &Position, accumulator: &mut Accumulator) {
        accumulator.board = Board::of(pos);
        self.restart(accumulator, None);
    }

    /// Computes a position's first layer from its parent's: the rows of the
    /// pieces, threats and pawn pairs that changed. A side whose own king
    /// moved, or whose block changed, starts its piece-square sums from the
    /// cache's entry for the king's square, and one whose king crossed the
    /// middle of the board, which mirrors its inputs, sums its threats and
    /// pairs again.
    pub fn update(
        &self,
        parent: &Accumulator,
        pos: &Position,
        accumulator: &mut Accumulator,
        cache: &mut RefreshCache,
    ) {
        accumulator.board = Board::of(pos);
        let changed = changed_squares(&parent.board.squares, &accumulator.board.squares);
        if changed.count_ones() as usize <= MOVED {
            self.follow(parent, accumulator, changed, Some(cache));
        } else {
            // Not a move from its parent.
            self.restart(accumulator, Some(cache));
        }
    }

    /// A quiet move's first layer from its parent's, for a piece other than
    /// a king. The same values as `update`, without building the position.
    pub fn update_quiet(
        &self,
        parent: &Accumulator,
        from: Square,
        to: Square,
        accumulator: &mut Accumulator,
    ) {
        let (from, to) = (from as usize, to as usize);
        let code = parent.board.squares[from];
        debug_assert!(code != 0 && usize::from(code & 7) != KING);
        debug_assert!(parent.board.squares[to] == 0);
        let moved = 1u64 << from | 1 << to;
        let board = &mut accumulator.board;
        board.clone_from(&parent.board);
        board.squares[from] = 0;
        board.squares[to] = code;
        board.roles[usize::from(code & 7)] ^= moved;
        if code & 8 != 0 {
            board.white ^= moved;
        }
        self.follow(parent, accumulator, moved, None);
    }

    /// A first layer whose board is set, from its parent's, whose board
    /// differs on the squares `changed`.
    fn follow(
        &self,
        parent: &Accumulator,
        accumulator: &mut Accumulator,
        changed: u64,
        mut cache: Option<&mut RefreshCache>,
    ) {
        let (before, after) = (&parent.board, &accumulator.board);
        let kings = [after.king(0), after.king(1)];
        let orients = [orientation(0, kings[0]), orientation(1, kings[1])];
        // A side's threats and pairs follow from the parent's while its
        // inputs are oriented alike.
        let follows = [0, 1].map(|side| orients[side] == orientation(side, before.king(side)));
        let mut removed = [List::new(), List::new()];
        let mut added = [List::new(), List::new()];
        // The rows that may differ: those of the changed squares, and of
        // the pieces that attacked one. A piece that attacks one only after
        // the move attacked another before it, the one that stood in its
        // way or the one it left.
        accumulator.rows = parent.rows;
        for from in squares(changed | before.attackers(changed)) {
            let (old, new) = (parent.rows[from], after.row(from));
            accumulator.rows[from] = new;
            // A threat stays if neither of its squares changed.
            let stays = if changed >> from & 1 != 0 {
                0
            } else {
                old & new & !changed
            };
            for (list, board, threats) in [
                (&mut removed, before, old & !stays),
                (&mut added, after, new & !stays),
            ] {
                let attacker = board.squares[from];
                for to in squares(threats) {
                    let attacked = board.squares[to];
                    for side in 0..2 {
                        let row = threat(side, orients[side], attacker, from, to, attacked);
                        list[side].push_if(row, follows[side] && (row as usize) < THREATS);
                    }
                }
            }
        }
        let (pawns_before, pawns_after) = (before.pawns(), after.pawns());
        if pawns_before != pawns_after {
            let gone = (pawns_before.0 & !pawns_after.0) | (pawns_before.1 & !pawns_after.1);
            let come = (pawns_after.0 & !pawns_before.0) | (pawns_after.1 & !pawns_before.1);
            for side in 0..2 {
                if follows[side] {
                    let orient = orients[side];
                    let (white, black) = pawns_before;
                    pairs(side, orient, gone, white, black, &mut removed[side]);
                    let (white, black) = pawns_after;
                    pairs(side, orient, come, white, black, &mut added[side]);
                }
            }
        }
        for side in 0..2 {
            if follows[side] {
                change_rows(
                    &self.threats,
                    &parent.threats[side],
                    &mut accumulator.threats[side],
                    removed[side].as_slice(),
                    added[side].as_slice(),
                );
            } else {
                self.sum_threats(side, accumulator);
            }
            let (king, block) = (kings[side], accumulator.board.block(side));
            if king != parent.board.king(side) || block != parent.board.block(side) {
                self.restart_piece_squares(side, accumulator, cache.as_deref_mut());
                continue;
            }
            let (mut removed, mut added) = (List::new(), List::new());
            let (before, after) = (&parent.board.squares, &accumulator.board.squares);
            let start = (block * BLOCK) as u32;
            for square in squares(changed) {
                let (old, new) = (before[square], after[square]);
                removed.push_if(start + piece_square(side, king, square, old), old != 0);
                added.push_if(start + piece_square(side, king, square, new), new != 0);
            }
            change_rows(
                &self.piece_squares,
                &parent.piece_squares[side],
                &mut accumulator.piece_squares[side],
                removed.as_slice(),
                added.as_slice(),
            );
        }
    }

    /// A first layer whose board is set, without a parent: every row of
    /// attacked squares, the threats and pairs summed, and the piece-square
    /// sums from the cache, if there is one.
    fn restart(&self, accumulator: &mut Accumulator, mut cache: Option<&mut RefreshCache>) {
        for square in 0..64 {
            accumulator.rows[square] = accumulator.board.row(square);
        }
        for side in 0..2 {
            self.sum_threats(side, accumulator);
            self.restart_piece_squares(side, accumulator, cache.as_deref_mut());
        }
    }

    /// A side's threats' and pairs' sums from scratch, from the rows of
    /// attacked squares.
    fn sum_threats(&self, side: usize, accumulator: &mut Accumulator) {
        let board = &accumulator.board;
        let orient = orientation(side, board.king(side));
        let mut rows = List::new();
        for from in squares(board.occupied()) {
            for to in squares(accumulator.rows[from]) {
                let (attacker, attacked) = (board.squares[from], board.squares[to]);
                let row = threat(side, orient, attacker, from, to, attacked);
                rows.push_if(row, (row as usize) < THREATS);
            }
        }
        let (white, black) = board.pawns();
        pairs(side, orient, white | black, white, black, &mut rows);
        change_rows(
            &self.threats,
            &[0; HALF],
            &mut accumulator.threats[side],
            &[],
            rows.as_slice(),
        );
    }

    /// A side's piece-square sums without its parent's: from the cache's
    /// entry for its block and its king's square, brought up to date by the
    /// squares that differ, or without a cache from the biases.
    fn restart_piece_squares(
        &self,
        side: usize,
        accumulator: &mut Accumulator,
        cache: Option<&mut RefreshCache>,
    ) {
        let board = &accumulator.board;
        let (king, block) = (board.king(side), board.block(side));
        let mut entry = cache.map(|cache| &mut cache.entries[(block * 2 + side) * 64 + king]);
        let (from, before): (&[i16; HALF], &[u8; 64]) = match &entry {
            Some(entry) => (&entry.values, &entry.squares),
            None => (&self.biases, &[0; 64]),
        };
        let (mut removed, mut added) = (List::new(), List::new());
        let start = (block * BLOCK) as u32;
        for square in squares(changed_squares(before, &board.squares)) {
            let (old, new) = (before[square], board.squares[square]);
            removed.push_if(start + piece_square(side, king, square, old), old != 0);
            added.push_if(start + piece_square(side, king, square, new), new != 0);
        }
        change_rows(
            &self.piece_squares,
            from,
            &mut accumulator.piece_squares[side],
            removed.as_slice(),
            added.as_slice(),
        );
        if let Some(entry) = &mut entry {
            entry.values = accumulator.piece_squares[side];
            entry.squares = board.squares;
        }
    }

    /// `output_from` for a position with `us` to move, whose pieces the
    /// first layer holds.
    pub fn output_for(
        &self,
        transformed: &Transformed,
        accumulator: &Accumulator,
        us: Color,
    ) -> i32 {
        let stack = &self.stacks[usize::from(self.stored[accumulator.board.stack(us as usize)])];
        let mut sums = stack.b0;
        first_layer(&stack.w0, transformed, &mut sums);
        let total = later_layers(stack, &sums);
        // The layers count 128 * 64 * 2 to one, where the evaluation counts
        // 600 * 16; then its sixteenths, in run3b's units.
        let value = ((i64::from(total) * (600 * 16) / (128 * 64 * 2)) as i32) / 16;
        let (numerator, denominator) = self.scale;
        (i64::from(value) * numerator / denominator) as i32
    }

    /// The first layer as the dense layers read it, into `out`, with `us`
    /// to move: for the side to move, then the other side, the products of
    /// the first HALF / 2 values and the last, each clipped to 0-255, over
    /// 512, at the first HALF / 2 places of the side's 512, the rest zero;
    /// and which chunks of four outputs are not all zero.
    pub fn transform_for(&self, accumulator: &Accumulator, us: Color, out: &mut Transformed) {
        for (half, side) in [us as usize, !us as usize].into_iter().enumerate() {
            let place = &mut out.values[half * (INPUTS / 2)..(half + 1) * (INPUTS / 2)];
            let (output, rest) = place.split_at_mut(HALF / 2);
            products(
                &accumulator.piece_squares[side],
                &accumulator.threats[side],
                output.try_into().expect("a side's outputs"),
            );
            rest.fill(0);
        }
        out.list();
    }
}

/// What a network file says before its first layer.
struct Header {
    /// The layer stacks it stores, and which of them each of the 192 uses.
    count: usize,
    stored: [u16; STACKS],
    /// A pruned file's factor to run3b's units; 1 for varifish's.
    scale: (i64, i64),
    /// The stored stack each of the 192 falls back on, as the file has it.
    fallbacks: Vec<u16>,
}

/// Reads a file's header up to its first layer's biases: varifish's, with
/// `version` `VERSION` and `half` FILE_HALF, or a pruned one, `PRUNED` and
/// HALF, which carries its factor to run3b's units after the description.
fn read_header(file: &mut Reader, version: u32, half: usize) -> Result<Header, String> {
    let found = file.u32()?;
    let hash = file.u32()?;
    let length = file.u32()? as usize;
    if found != version || hash != transformer_hash(half) ^ stack_hash(half) ^ TABLE_HASH {
        return Err(if found == VERSION {
            "a network of varifish's: provari reads it pruned for provari, as its releases carry it"
                .into()
        } else {
            "not an SFNNv20 Atomic network pruned for provari".into()
        });
    }
    file.skip(length)?;
    let scale = if version == PRUNED {
        let (numerator, denominator) = (file.i32()?, file.i32()?);
        if denominator <= 0 {
            return Err("an unexpected factor to run3b's units".into());
        }
        (i64::from(numerator), i64::from(denominator))
    } else {
        (1, 1)
    };
    // The table of layer stacks: how many the file stores, and for each of
    // the 192 the stored one it uses, which uses itself.
    if file.u32()? != TABLE_HASH || file.u32()? as usize != STACKS {
        return Err("an unexpected table of layer stacks".into());
    }
    let count = file.u32()? as usize;
    let fallbacks = file.many(STACKS, u16::from_le_bytes)?;
    let mut stored = [0; STACKS];
    let mut found = 0;
    for (stack, &fallback) in fallbacks.iter().enumerate() {
        if usize::from(fallback) == stack {
            stored[stack] = found;
            found += 1;
        }
    }
    if count == 0 || usize::from(found) != count {
        return Err("an unexpected table of layer stacks".into());
    }
    for (stack, &fallback) in fallbacks.iter().enumerate() {
        let fallback = usize::from(fallback);
        if fallbacks.get(fallback).map(|&own| usize::from(own)) != Some(fallback) {
            return Err("an unexpected table of layer stacks".into());
        }
        stored[stack] = stored[fallback];
    }
    if file.u32()? != transformer_hash(half) {
        return Err("an unexpected feature transformer".into());
    }
    Ok(Header {
        count,
        stored,
        scale,
        fallbacks,
    })
}

/// Reads a layer stack of a first layer of `half` values a side, its first
/// dense layer's weights as the file has them, a row of `half` an output.
fn read_stack(file: &mut Reader, half: usize) -> Result<Stack, String> {
    if file.u32()? != stack_hash(half) {
        return Err("unexpected dense layers".into());
    }
    let mut b0 = [0; FC0];
    for bias in &mut b0 {
        *bias = file.i32()?;
    }
    let w0 = file.many(FC0 * half, i8::from_le_bytes)?;
    let mut b1 = [0; FC1];
    for bias in &mut b1 {
        *bias = file.i32()?;
    }
    let w1 = file.many(FC1 * 2 * FC0, i8::from_le_bytes)?;
    let b2 = file.i32()?;
    let mut w2 = [0; LAST];
    for weight in &mut w2 {
        *weight = file.i8()?;
    }
    Ok(Stack {
        b0,
        w0,
        b1,
        w1,
        b2,
        w2,
    })
}

/// A pruned network for provari from one of varifish's: the outputs `keep`
/// lists, 256 of its 512, in that order, with `scale`, a numerator and a
/// denominator, to bring its values to run3b's units, and `description`.
/// Each output kept takes its two values of each row, its bias and its
/// dense layers' weights; the values of the outputs kept are varifish's.
pub fn prune(
    bytes: &[u8],
    keep: &[usize],
    scale: (i32, i32),
    description: &str,
) -> Result<Vec<u8>, String> {
    let outputs = FILE_HALF / 2;
    if keep.len() != HALF / 2
        || keep.iter().any(|&output| output >= outputs)
        || keep.iter().collect::<std::collections::HashSet<_>>().len() != keep.len()
    {
        return Err(format!("keep {} different outputs of {outputs}", HALF / 2));
    }
    if scale.1 <= 0 {
        return Err("the factor's denominator must be positive".into());
    }
    let mut file = Reader::new(bytes);
    let header = read_header(&mut file, VERSION, FILE_HALF)?;
    // The values kept of a row of FILE_HALF: each output's first, then each
    // one's second.
    let lanes: Vec<usize> = keep
        .iter()
        .copied()
        .chain(keep.iter().map(|&output| output + outputs))
        .collect();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let u32s = |out: &mut Vec<u8>, values: &[u32]| {
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    };
    let description = description.as_bytes();
    u32s(
        &mut out,
        &[
            PRUNED,
            transformer_hash(HALF) ^ stack_hash(HALF) ^ TABLE_HASH,
            description.len() as u32,
        ],
    );
    out.extend_from_slice(description);
    out.extend_from_slice(&scale.0.to_le_bytes());
    out.extend_from_slice(&scale.1.to_le_bytes());
    u32s(&mut out, &[TABLE_HASH, STACKS as u32, header.count as u32]);
    for fallback in &header.fallbacks {
        out.extend_from_slice(&fallback.to_le_bytes());
    }
    u32s(&mut out, &[transformer_hash(HALF)]);
    let biases = file.many(FILE_HALF, i16::from_le_bytes)?;
    for &lane in &lanes {
        out.extend_from_slice(&biases[lane].to_le_bytes());
    }
    for _ in 0..THREATS + PAIRS + BLOCKS * BLOCK {
        let row = file.many(FILE_HALF, u8::from_le_bytes)?;
        out.extend(lanes.iter().map(|&lane| row[lane]));
    }
    for _ in 0..header.count {
        let stack = read_stack(&mut file, FILE_HALF)?;
        u32s(&mut out, &[stack_hash(HALF)]);
        for bias in stack.b0 {
            out.extend_from_slice(&bias.to_le_bytes());
        }
        // The dense layer's inputs kept: the side to move's outputs, then
        // the other side's, 512 on.
        for row in stack.w0.as_chunks::<FILE_HALF>().0 {
            out.extend(keep.iter().map(|&output| row[output] as u8));
            out.extend(keep.iter().map(|&output| row[outputs + output] as u8));
        }
        for bias in stack.b1 {
            out.extend_from_slice(&bias.to_le_bytes());
        }
        out.extend(stack.w1.iter().map(|&weight| weight as u8));
        out.extend_from_slice(&stack.b2.to_le_bytes());
        out.extend(stack.w2.iter().map(|&weight| weight as u8));
    }
    if !file.is_empty() {
        return Err("the network file has trailing bytes".into());
    }
    Ok(out)
}

/// A piece's row in its block, by its square and its code there, for a side
/// whose king is on `king`.
fn piece_square(side: usize, king: usize, square: usize, code: u8) -> u32 {
    (square ^ usize::from(PIECE_SQUARES[side][king][usize::from(code & 15)])) as u32
}

/// The piece-square sums last computed for each block, side and square of
/// its king, with the squares they were computed from, as Stockfish's
/// refresh cache keeps them: a side whose own king moved starts from its new
/// square's entry and changes only the squares that differ. One a search.
pub struct RefreshCache {
    entries: Vec<CacheEntry>,
}

#[derive(Clone)]
struct CacheEntry {
    values: [i16; HALF],
    squares: [u8; 64],
}

impl RefreshCache {
    /// Every entry starts as the empty board's: the biases alone.
    pub fn new(network: &Network) -> RefreshCache {
        let empty = CacheEntry {
            values: *network.biases,
            squares: [0; 64],
        };
        RefreshCache {
            entries: vec![empty; BLOCKS * 2 * 64],
        }
    }
}

/// A position's first layer, for each side's perspective, in its two parts,
/// with the pieces it was computed from and the squares each threatens.
#[derive(Clone)]
pub struct Accumulator {
    /// The biases and the pieces' rows.
    piece_squares: [[i16; HALF]; 2],
    /// The threats' and the pawn pairs' rows.
    threats: [[i16; HALF]; 2],
    board: Board,
    /// For each square, the squares its piece threatens.
    rows: [u64; 64],
}

impl Default for Accumulator {
    fn default() -> Accumulator {
        Accumulator {
            piece_squares: [[0; HALF]; 2],
            threats: [[0; HALF]; 2],
            board: Board {
                squares: [0; 64],
                roles: [0; 7],
                white: 0,
            },
            rows: [0; 64],
        }
    }
}

/// A side's values, `to`: those of `from` less the rows `removed` plus the
/// rows `added`, of HALF weights each. Sums wrap, so the values are those
/// of applying the rows one after the other, in any order.
fn change_rows(
    weights: &[i8],
    from: &[i16; HALF],
    to: &mut [i16; HALF],
    removed: &[u32],
    added: &[u32],
) {
    assert!(
        removed
            .iter()
            .chain(added)
            .all(|&row| (row as usize + 1) * HALF <= weights.len())
    );
    #[cfg(target_arch = "aarch64")]
    // SAFETY: every aarch64 processor has NEON, and every row lies inside
    // `weights`.
    unsafe {
        use std::arch::aarch64::{vaddq_s16, vld1_s8, vld1q_s16, vmovl_s8, vst1q_s16, vsubq_s16};
        let widened = |row: *const i8| vmovl_s8(vld1_s8(row));
        change_rows_in_tiles!(
            8, 16, vld1q_s16, vst1q_s16, vaddq_s16, vsubq_s16, widened, weights, from, to, removed,
            added
        );
    }
    #[cfg(target_feature = "simd128")]
    // SAFETY: every row lies inside `weights`.
    unsafe {
        use std::arch::wasm32::{
            i16x8_add, i16x8_load_extend_i8x8, i16x8_sub, v128_load, v128_store,
        };
        change_rows_in_tiles!(
            8,
            8,
            v128_load,
            v128_store,
            i16x8_add,
            i16x8_sub,
            i16x8_load_extend_i8x8,
            weights,
            from,
            to,
            removed,
            added
        );
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2, and every row lies inside
        // `weights`.
        unsafe { avx2::change_rows(weights, from, to, removed, added) };
        return;
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "simd128")))]
    change_rows_portable(weights, from, to, removed, added);
}

/// `change_rows` one row after the other.
#[cfg_attr(
    any(target_arch = "aarch64", target_feature = "simd128"),
    allow(dead_code)
)]
fn change_rows_portable(
    weights: &[i8],
    from: &[i16; HALF],
    to: &mut [i16; HALF],
    removed: &[u32],
    added: &[u32],
) {
    *to = *from;
    for &row in removed {
        let row = &weights[row as usize * HALF..(row as usize + 1) * HALF];
        for (value, &weight) in to.iter_mut().zip(row) {
            *value = value.wrapping_sub(i16::from(weight));
        }
    }
    for &row in added {
        let row = &weights[row as usize * HALF..(row as usize + 1) * HALF];
        for (value, &weight) in to.iter_mut().zip(row) {
            *value = value.wrapping_add(i16::from(weight));
        }
    }
}

/// A side's outputs: the two parts of its first layer added, and each of
/// the first HALF / 2 sums times the one HALF / 2 on, both clipped to 0-255,
/// over 512.
fn products(piece_squares: &[i16; HALF], threats: &[i16; HALF], output: &mut [u8; HALF / 2]) {
    #[cfg(target_arch = "aarch64")]
    products_neon(piece_squares, threats, output);
    #[cfg(target_feature = "simd128")]
    simd128::products(piece_squares, threats, output);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2.
        unsafe { avx2::products(piece_squares, threats, output) };
        return;
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "simd128")))]
    products_portable(piece_squares, threats, output);
}

/// `products` with NEON, eight outputs at a time: each sum clipped by
/// saturating to an unsigned byte, and the two bytes' product, of 16 bits,
/// shifted down.
#[cfg(target_arch = "aarch64")]
fn products_neon(piece_squares: &[i16; HALF], threats: &[i16; HALF], output: &mut [u8; HALF / 2]) {
    use std::arch::aarch64::{
        vaddq_s16, vld1q_s16, vmull_u8, vqmovun_s16, vshr_n_u8, vshrn_n_u16, vst1_u8,
    };
    // SAFETY: every aarch64 processor has NEON. The loads read eight values
    // inside each part, at an output's index and 512 on, and the store
    // writes eight bytes inside `output`.
    unsafe {
        let (piece_squares, threats) = (piece_squares.as_ptr(), threats.as_ptr());
        let value = |at: usize| {
            vqmovun_s16(vaddq_s16(
                vld1q_s16(piece_squares.add(at)),
                vld1q_s16(threats.add(at)),
            ))
        };
        for at in (0..HALF / 2).step_by(8) {
            // Over 512: over 256 into a byte, then halved.
            let product = vmull_u8(value(at), value(at + HALF / 2));
            let halved = vshr_n_u8::<1>(vshrn_n_u16::<8>(product));
            vst1_u8(output.as_mut_ptr().add(at), halved);
        }
    }
}

#[cfg_attr(
    any(target_arch = "aarch64", target_feature = "simd128"),
    allow(dead_code)
)]
fn products_portable(
    piece_squares: &[i16; HALF],
    threats: &[i16; HALF],
    output: &mut [u8; HALF / 2],
) {
    let value = |index: usize| {
        piece_squares[index]
            .wrapping_add(threats[index])
            .clamp(0, 255) as u32
    };
    for (index, output) in output.iter_mut().enumerate() {
        *output = (value(index) * value(index + HALF / 2) / 512) as u8;
    }
}

/// The first dense layer, over the first layer's nonzero chunks only: a
/// chunk of zeros adds nothing to any output.
fn first_layer(weights: &[i8], transformed: &Transformed, sums: &mut [i32; FC0]) {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        // SAFETY: the processor has the dot-product instructions.
        unsafe { first_layer_dotprod(weights, transformed, sums) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2.
        unsafe { avx2::first_layer(weights, transformed, sums) };
        return;
    }
    #[cfg(target_feature = "simd128")]
    simd128::first_layer(weights, transformed, sums);
    #[cfg(not(target_feature = "simd128"))]
    first_layer_portable(weights, transformed, sums);
}

/// The first dense layer with the dot-product instruction: each nonzero
/// chunk's four inputs against its 32 outputs' weights, the even chunks and
/// the odd ones into separate sums, so that they need not wait on each
/// other. Inputs are at most 127, so they read as signed bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn first_layer_dotprod(weights: &[i8], transformed: &Transformed, sums: &mut [i32; FC0]) {
    use std::arch::aarch64::{
        int32x4_t, vaddq_s32, vdotq_s32, vdupq_n_s32, vld1q_dup_u32, vld1q_s8, vld1q_s32,
        vreinterpretq_s8_u32, vst1q_s32,
    };
    assert_eq!(weights.len(), CHUNKS * FC0 * 4);
    let [even, odd] = transformed.nonzero();
    let both = even.len().min(odd.len());
    let mut first = [vdupq_n_s32(0); FC0 / 4];
    let mut second = [vdupq_n_s32(0); FC0 / 4];
    // Adds a chunk's four inputs times its 32 outputs' weights to `totals`,
    // four outputs each.
    let products = |totals: &mut [int32x4_t; FC0 / 4], chunk: u16| {
        let chunk = usize::from(chunk);
        assert!(chunk < CHUNKS);
        // SAFETY: the chunk is below CHUNKS, so the loads read 4 bytes
        // inside `values`, aligned since `values` is aligned to 64 bytes,
        // and 128 inside `weights`.
        unsafe {
            let inputs = transformed.values.as_ptr().add(chunk * 4).cast();
            let x = vreinterpretq_s8_u32(vld1q_dup_u32(inputs));
            let rows = weights.as_ptr().add(chunk * FC0 * 4);
            for (row, total) in totals.iter_mut().enumerate() {
                *total = vdotq_s32(*total, vld1q_s8(rows.add(row * 16)), x);
            }
        }
    };
    for (&a, &b) in even[..both].iter().zip(&odd[..both]) {
        products(&mut first, a);
        products(&mut second, b);
    }
    for &chunk in even[both..].iter().chain(&odd[both..]) {
        products(&mut first, chunk);
    }
    for (row, (first, second)) in first.into_iter().zip(second).enumerate() {
        // SAFETY: `sums` has 32 values, four per row.
        unsafe {
            let sum = sums.as_mut_ptr().add(row * 4);
            vst1q_s32(sum, vaddq_s32(vld1q_s32(sum), vaddq_s32(first, second)));
        }
    }
}

#[cfg_attr(target_feature = "simd128", allow(dead_code))]
fn first_layer_portable(weights: &[i8], transformed: &Transformed, sums: &mut [i32; FC0]) {
    let [even, odd] = transformed.nonzero();
    for &chunk in even.iter().chain(odd) {
        let chunk = usize::from(chunk);
        let inputs = &transformed.values[chunk * 4..chunk * 4 + 4];
        let (rows, _) = weights[chunk * FC0 * 4..(chunk + 1) * FC0 * 4].as_chunks::<4>();
        for (sum, row) in sums.iter_mut().zip(rows) {
            *sum = sum.wrapping_add(dot(row, inputs));
        }
    }
}

/// The layers after the first dense one, from its sums: both its
/// activations, the second dense layer and both of its, and the output's
/// sum over all four, with the first layer's last two sums added and taken
/// away.
fn later_layers(stack: &Stack, sums: &[i32; FC0]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        // SAFETY: the processor has the dot-product instructions.
        return unsafe { later_layers_dotprod(stack, sums) };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2.
        return unsafe { avx2::later_layers(stack, sums) };
    }
    #[cfg(target_feature = "simd128")]
    return later_layers_with(stack, sums, simd128::dot);
    #[cfg(not(target_feature = "simd128"))]
    later_layers_portable(stack, sums)
}

#[cfg_attr(target_feature = "simd128", allow(dead_code))]
fn later_layers_portable(stack: &Stack, sums: &[i32; FC0]) -> i32 {
    later_layers_with(stack, sums, dot)
}

/// `later_layers` with the dot-product instruction: sixteen inputs in one.
/// Inputs are at most 127, so they read as signed bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn later_layers_dotprod(stack: &Stack, sums: &[i32; FC0]) -> i32 {
    use std::arch::aarch64::{vaddvq_s32, vdotq_s32, vdupq_n_s32, vld1q_s8};
    later_layers_with(stack, sums, |weights, inputs| {
        assert!(weights.len() == inputs.len() && weights.len().is_multiple_of(16));
        let mut total = vdupq_n_s32(0);
        for at in (0..weights.len()).step_by(16) {
            // SAFETY: sixteen bytes from `at` lie inside both.
            total = unsafe {
                vdotq_s32(
                    total,
                    vld1q_s8(weights.as_ptr().add(at)),
                    vld1q_s8(inputs.as_ptr().add(at).cast()),
                )
            };
        }
        vaddvq_s32(total)
    })
}

/// `later_layers` with the sum `dot` gives of weights times inputs, of 64
/// or 128 each.
#[inline(always)]
fn later_layers_with(stack: &Stack, sums: &[i32; FC0], dot: impl Fn(&[i8], &[u8]) -> i32) -> i32 {
    let mut inputs = [0u8; LAST];
    let (first, second) = inputs.split_at_mut(2 * FC0);
    for (index, &sum) in sums.iter().enumerate() {
        first[index] = squared(sum, WEIGHT_SCALE_BITS + 1);
        first[FC0 + index] = clipped(sum, WEIGHT_SCALE_BITS + 1);
    }
    for index in 0..FC1 {
        let row = &stack.w1[index * 2 * FC0..(index + 1) * 2 * FC0];
        let sum = stack.b1[index].wrapping_add(dot(row, first));
        second[index] = squared(sum, WEIGHT_SCALE_BITS);
        second[FC1 + index] = clipped(sum, WEIGHT_SCALE_BITS);
    }
    stack
        .b2
        .wrapping_add(dot(&stack.w2, &inputs))
        .wrapping_add(sums[FC0 - 2])
        .wrapping_sub(sums[FC0 - 1])
}

fn dot(weights: &[i8], inputs: &[u8]) -> i32 {
    weights
        .iter()
        .zip(inputs)
        .map(|(&weight, &input)| i32::from(weight) * i32::from(input))
        .fold(0, i32::wrapping_add)
}

/// A sum as the next layer reads it: over `2^bits`, clipped to 0-127.
fn clipped(sum: i32, bits: u32) -> u8 {
    (sum >> bits).clamp(0, 127) as u8
}

/// A sum's square as the next layer reads it: over `2^(2 bits + 7)`, at
/// most 127.
fn squared(sum: i32, bits: u32) -> u8 {
    ((i64::from(sum) * i64::from(sum)) >> (2 * bits + 7)).min(127) as u8
}

/// The kernels with AVX2, for processors that have it.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadu_si128,
        _mm_shuffle_epi32, _mm_storeu_si128, _mm256_add_epi16, _mm256_add_epi32,
        _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_extracti128_si256, _mm256_hadd_epi32,
        _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_max_epi8,
        _mm256_max_epi16, _mm256_min_epi16, _mm256_mulhi_epi16, _mm256_packs_epi16,
        _mm256_packs_epi32, _mm256_packus_epi16, _mm256_permute4x64_epi64,
        _mm256_permutevar8x32_epi32, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi32,
        _mm256_setzero_si256, _mm256_slli_epi16, _mm256_sra_epi32, _mm256_srl_epi16,
        _mm256_storeu_si256, _mm256_sub_epi16,
    };

    use super::{CHUNKS, FC0, FC1, HALF, LAST, Stack, Transformed, WEIGHT_SCALE_BITS};

    /// Sixteen weights as values.
    ///
    /// # Safety
    ///
    /// The processor must have AVX2, and sixteen bytes at `row` be readable.
    #[target_feature(enable = "avx2")]
    unsafe fn widened(row: *const i8) -> __m256i {
        // SAFETY: the load reads sixteen bytes at `row`.
        unsafe { _mm256_cvtepi8_epi16(_mm_loadu_si128(row.cast())) }
    }

    /// `change_rows`, 128 values at a time in eight registers.
    ///
    /// # Safety
    ///
    /// The processor must have AVX2, and every row lie inside `weights`, as
    /// HALF weights from the row times HALF.
    #[target_feature(enable = "avx2")]
    pub unsafe fn change_rows(
        weights: &[i8],
        from: &[i16; HALF],
        to: &mut [i16; HALF],
        removed: &[u32],
        added: &[u32],
    ) {
        // SAFETY: the caller keeps every row inside `weights`; the loads and
        // stores need no alignment.
        unsafe {
            change_rows_in_tiles!(
                16,
                8,
                _mm256_loadu_si256,
                _mm256_storeu_si256,
                _mm256_add_epi16,
                _mm256_sub_epi16,
                widened,
                weights,
                from,
                to,
                removed,
                added
            );
        }
    }

    /// `products`, 32 outputs at a time: a value times 128 times its pair,
    /// both clipped, has the product over 512 in its high 16 bits.
    ///
    /// # Safety
    ///
    /// The processor must have AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn products(
        piece_squares: &[i16; HALF],
        threats: &[i16; HALF],
        output: &mut [u8; HALF / 2],
    ) {
        let (floor, ceiling) = (_mm256_setzero_si256(), _mm256_set1_epi16(255));
        let (piece_squares, threats) = (piece_squares.as_ptr(), threats.as_ptr());
        // SAFETY: the loads read 16 values inside each part, from an index
        // below 512 or one 512 on, and the store writes 32 bytes inside
        // `output`.
        unsafe {
            let value = |at: usize| {
                let sum = _mm256_add_epi16(
                    _mm256_loadu_si256(piece_squares.add(at).cast()),
                    _mm256_loadu_si256(threats.add(at).cast()),
                );
                _mm256_min_epi16(_mm256_max_epi16(sum, floor), ceiling)
            };
            let product = |at: usize| {
                _mm256_mulhi_epi16(_mm256_slli_epi16::<7>(value(at)), value(at + HALF / 2))
            };
            for at in (0..HALF / 2).step_by(32) {
                // Packing keeps to each 128-bit half, so the bytes come out
                // as the first product's first eight, the second's first
                // eight, then the second eights: the middle two swap back.
                let packed = _mm256_packus_epi16(product(at), product(at + 16));
                let ordered = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);
                _mm256_storeu_si256(output.as_mut_ptr().add(at).cast(), ordered);
            }
        }
    }

    /// `first_layer`: each nonzero chunk's four inputs, repeated, against its
    /// 32 outputs' weights, eight outputs a register, the even chunks and the
    /// odd ones into separate sums, so that they need not wait on each other.
    /// Inputs are at most 127, so the products' pairs fit 16 bits.
    ///
    /// # Safety
    ///
    /// The processor must have AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn first_layer(weights: &[i8], transformed: &Transformed, sums: &mut [i32; FC0]) {
        assert_eq!(weights.len(), CHUNKS * FC0 * 4);
        let [even, odd] = transformed.nonzero();
        let both = even.len().min(odd.len());
        let mut first = [_mm256_setzero_si256(); FC0 / 8];
        let mut second = [_mm256_setzero_si256(); FC0 / 8];
        let ones = _mm256_set1_epi16(1);
        // Adds a chunk's four inputs times its 32 outputs' weights to
        // `totals`, eight outputs each: the products summed in pairs, then
        // the pairs in fours.
        let products = |totals: &mut [__m256i; FC0 / 8], chunk: u16| {
            let chunk = usize::from(chunk);
            assert!(chunk < CHUNKS);
            // SAFETY: the chunk is below CHUNKS, so the loads read 4 bytes
            // inside `values` and 128 inside `weights`.
            unsafe {
                let inputs = transformed.values.as_ptr().add(chunk * 4).cast::<i32>();
                let x = _mm256_set1_epi32(inputs.read_unaligned());
                let rows = weights.as_ptr().add(chunk * FC0 * 4);
                for (eight, total) in totals.iter_mut().enumerate() {
                    let row = _mm256_loadu_si256(rows.add(eight * 32).cast());
                    let pairs = _mm256_maddubs_epi16(x, row);
                    *total = _mm256_add_epi32(*total, _mm256_madd_epi16(pairs, ones));
                }
            }
        };
        for (&a, &b) in even[..both].iter().zip(&odd[..both]) {
            products(&mut first, a);
            products(&mut second, b);
        }
        for &chunk in even[both..].iter().chain(&odd[both..]) {
            products(&mut first, chunk);
        }
        for (eight, (first, second)) in first.into_iter().zip(second).enumerate() {
            // SAFETY: `sums` has 32 values, eight per register.
            unsafe {
                let at = sums.as_mut_ptr().add(eight * 8).cast::<__m256i>();
                let total = _mm256_add_epi32(first, second);
                _mm256_storeu_si256(at, _mm256_add_epi32(_mm256_loadu_si256(at), total));
            }
        }
    }

    /// `later_layers`: the activations of 32 sums at a time, and the dense
    /// layers 32 inputs at a time, their products with the weights summed in
    /// pairs, then the pairs in fours, as `first_layer` takes them; four
    /// outputs' fours are added up across registers.
    ///
    /// # Safety
    ///
    /// The processor must have AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn later_layers(stack: &Stack, sums: &[i32; FC0]) -> i32 {
        assert!(stack.w1.len() == FC1 * 2 * FC0);
        let ones = _mm256_set1_epi16(1);
        let mut inputs = [0u8; LAST];
        let mut second = [0i32; FC1];
        // SAFETY: `sums` and `second` hold 32 values, `inputs` and `w2` 128
        // bytes, of which each layer's activations take 64, and each row of
        // `w1` 64.
        let total = unsafe {
            let fours = |inputs: *const u8, weights: *const i8| {
                let pairs = _mm256_maddubs_epi16(
                    _mm256_loadu_si256(inputs.cast()),
                    _mm256_loadu_si256(weights.cast()),
                );
                _mm256_madd_epi16(pairs, ones)
            };
            activations(sums.as_ptr(), WEIGHT_SCALE_BITS + 1, inputs.as_mut_ptr());
            for four in 0..FC1 / 4 {
                let rows = stack.w1.as_ptr().add(four * 4 * 2 * FC0);
                let row = |index: usize| {
                    let weights = rows.add(index * 2 * FC0);
                    _mm256_add_epi32(
                        fours(inputs.as_ptr(), weights),
                        fours(inputs.as_ptr().add(32), weights.add(32)),
                    )
                };
                let outputs = _mm256_hadd_epi32(
                    _mm256_hadd_epi32(row(0), row(1)),
                    _mm256_hadd_epi32(row(2), row(3)),
                );
                let outputs = _mm_add_epi32(
                    _mm256_castsi256_si128(outputs),
                    _mm256_extracti128_si256::<1>(outputs),
                );
                let biases = _mm_loadu_si128(stack.b1.as_ptr().add(four * 4).cast());
                let at = second.as_mut_ptr().add(four * 4);
                _mm_storeu_si128(at.cast(), _mm_add_epi32(outputs, biases));
            }
            activations(
                second.as_ptr(),
                WEIGHT_SCALE_BITS,
                inputs.as_mut_ptr().add(2 * FC0),
            );
            let mut total = _mm256_setzero_si256();
            for at in (0..LAST).step_by(32) {
                let products = fours(inputs.as_ptr().add(at), stack.w2.as_ptr().add(at));
                total = _mm256_add_epi32(total, products);
            }
            total
        };
        let halves = _mm_add_epi32(
            _mm256_castsi256_si128(total),
            _mm256_extracti128_si256::<1>(total),
        );
        let pairs = _mm_add_epi32(halves, _mm_shuffle_epi32::<0b01_00_11_10>(halves));
        let output = _mm_cvtsi128_si32(_mm_add_epi32(
            pairs,
            _mm_shuffle_epi32::<0b10_11_00_01>(pairs),
        ));
        stack
            .b2
            .wrapping_add(output)
            .wrapping_add(sums[FC0 - 2])
            .wrapping_sub(sums[FC0 - 1])
    }

    /// Both activations of 32 sums, into 64 bytes: `squared`'s, then
    /// `clipped`'s. The sums are first saturated to 16 bits, beyond which a
    /// square is over 127 and a clipped sum 127 or 0 anyway; a square's high
    /// 16 bits are then shifted down, and saturation to a byte clips.
    ///
    /// # Safety
    ///
    /// The processor must have AVX2, 32 values be readable at `sums` and 64
    /// bytes writable at `output`.
    #[target_feature(enable = "avx2")]
    unsafe fn activations(sums: *const i32, bits: u32, output: *mut u8) {
        // Packing keeps to each 128-bit half, so the bytes come out as the
        // four registers' first fours, then their second fours.
        let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let by_bits = _mm_cvtsi32_si128(bits as i32);
        let by_square = _mm_cvtsi32_si128(2 * bits as i32 + 7 - 16);
        // SAFETY: the loads read 32 values at `sums`, and the stores write
        // 64 bytes at `output`.
        unsafe {
            let sum = |eight: usize| _mm256_loadu_si256(sums.add(eight * 8).cast());
            let squares = |first: __m256i, second: __m256i| {
                let words = _mm256_packs_epi32(first, second);
                _mm256_srl_epi16(_mm256_mulhi_epi16(words, words), by_square)
            };
            let squared = _mm256_packs_epi16(squares(sum(0), sum(1)), squares(sum(2), sum(3)));
            let shifted = |eight: usize| _mm256_sra_epi32(sum(eight), by_bits);
            let clipped = _mm256_max_epi8(
                _mm256_packs_epi16(
                    _mm256_packs_epi32(shifted(0), shifted(1)),
                    _mm256_packs_epi32(shifted(2), shifted(3)),
                ),
                _mm256_setzero_si256(),
            );
            _mm256_storeu_si256(output.cast(), _mm256_permutevar8x32_epi32(squared, order));
            _mm256_storeu_si256(
                output.add(32).cast(),
                _mm256_permutevar8x32_epi32(clipped, order),
            );
        }
    }
}

/// The kernels with WebAssembly's 128-bit SIMD, which browsers run on any
/// processor, for the browser build: what NEON does on aarch64. It has no
/// dot product of bytes, so products are taken in 16 bits, of inputs at most
/// 127 and weights that are signed bytes: every product and sum is exact,
/// and the results are the portable code's.
#[cfg(target_feature = "simd128")]
mod simd128 {
    use std::arch::wasm32::{
        i16x8_add, i16x8_max, i16x8_min, i16x8_mul, i16x8_splat, i32x4_add, i32x4_extract_lane,
        i32x4_shuffle, i32x4_splat, u8x16_narrow_i16x8, u16x8_shr, v128, v128_load,
        v128_load32_splat, v128_store,
    };

    use super::{CHUNKS, FC0, HALF, Transformed};

    /// `products`, sixteen outputs at a time: the clipped values' products
    /// fit 16 bits without a sign.
    pub fn products(
        piece_squares: &[i16; HALF],
        threats: &[i16; HALF],
        output: &mut [u8; HALF / 2],
    ) {
        let (floor, ceiling) = (i16x8_splat(0), i16x8_splat(255));
        let (piece_squares, threats) = (piece_squares.as_ptr(), threats.as_ptr());
        // SAFETY: the loads read eight values inside each part, from an
        // index below 512 or one 512 on, and the store writes sixteen bytes
        // inside `output`.
        unsafe {
            let value = |at: usize| {
                let sum = i16x8_add(
                    v128_load(piece_squares.add(at).cast()),
                    v128_load(threats.add(at).cast()),
                );
                i16x8_min(i16x8_max(sum, floor), ceiling)
            };
            let product = |at: usize| u16x8_shr(i16x8_mul(value(at), value(at + HALF / 2)), 9);
            for at in (0..HALF / 2).step_by(16) {
                let bytes = u8x16_narrow_i16x8(product(at), product(at + 8));
                v128_store(output.as_mut_ptr().add(at).cast(), bytes);
            }
        }
    }

    /// `first_layer`: each nonzero chunk's four inputs, widened to 16 bits
    /// and repeated, against its 32 outputs' weights, four outputs in each of
    /// eight registers' pairs; an output's two half sums are added at the
    /// end.
    #[cfg(not(target_feature = "relaxed-simd"))]
    pub fn first_layer(weights: &[i8], transformed: &Transformed, sums: &mut [i32; FC0]) {
        use std::arch::wasm32::{
            i16x8_extend_high_i8x16, i16x8_extend_low_i8x16, i32x4_dot_i16x8,
            u16x8_extend_low_u8x16,
        };
        assert_eq!(weights.len(), CHUNKS * FC0 * 4);
        let [even, odd] = transformed.nonzero();
        let mut totals = [[i32x4_splat(0); 2]; FC0 / 4];
        for &chunk in even.iter().chain(odd) {
            let chunk = usize::from(chunk);
            assert!(chunk < CHUNKS);
            // SAFETY: the chunk is below CHUNKS, so its four inputs lie inside
            // `values` and its 128 weights inside `weights`.
            unsafe {
                let inputs = v128_load32_splat(transformed.values.as_ptr().add(chunk * 4).cast());
                let x = u16x8_extend_low_u8x16(inputs);
                let rows = weights.as_ptr().add(chunk * FC0 * 4);
                for (row, [low, high]) in totals.iter_mut().enumerate() {
                    let w = v128_load(rows.add(row * 16).cast());
                    *low = i32x4_add(*low, i32x4_dot_i16x8(i16x8_extend_low_i8x16(w), x));
                    *high = i32x4_add(*high, i32x4_dot_i16x8(i16x8_extend_high_i8x16(w), x));
                }
            }
        }
        for (row, [low, high]) in totals.into_iter().enumerate() {
            let sum = i32x4_add(
                i32x4_shuffle::<0, 2, 4, 6>(low, high),
                i32x4_shuffle::<1, 3, 5, 7>(low, high),
            );
            // SAFETY: `sums` has 32 values, four per row.
            unsafe {
                let at = sums.as_mut_ptr().add(row * 4).cast::<v128>();
                v128_store(at, i32x4_add(v128_load(at), sum));
            }
        }
    }

    /// `first_layer` with relaxed SIMD's dot product of bytes: each nonzero
    /// chunk's four inputs, repeated, against its 32 outputs' weights, four
    /// outputs a register, the even chunks and the odd ones into separate
    /// sums, as NEON does. The inputs are at most 127, where the product is
    /// exact in every implementation.
    #[cfg(target_feature = "relaxed-simd")]
    pub fn first_layer(weights: &[i8], transformed: &Transformed, sums: &mut [i32; FC0]) {
        use std::arch::wasm32::i32x4_relaxed_dot_i8x16_i7x16_add;
        assert_eq!(weights.len(), CHUNKS * FC0 * 4);
        let [even, odd] = transformed.nonzero();
        let both = even.len().min(odd.len());
        let mut first = [i32x4_splat(0); FC0 / 4];
        let mut second = [i32x4_splat(0); FC0 / 4];
        // Adds a chunk's four inputs times its 32 outputs' weights to
        // `totals`, four outputs each.
        let products = |totals: &mut [v128; FC0 / 4], chunk: u16| {
            let chunk = usize::from(chunk);
            assert!(chunk < CHUNKS);
            // SAFETY: the chunk is below CHUNKS, so its four inputs lie inside
            // `values` and its 128 weights inside `weights`.
            unsafe {
                let x = v128_load32_splat(transformed.values.as_ptr().add(chunk * 4).cast());
                let rows = weights.as_ptr().add(chunk * FC0 * 4);
                for (row, total) in totals.iter_mut().enumerate() {
                    let w = v128_load(rows.add(row * 16).cast());
                    *total = i32x4_relaxed_dot_i8x16_i7x16_add(w, x, *total);
                }
            }
        };
        for (&a, &b) in even[..both].iter().zip(&odd[..both]) {
            products(&mut first, a);
            products(&mut second, b);
        }
        for &chunk in even[both..].iter().chain(&odd[both..]) {
            products(&mut first, chunk);
        }
        for (row, (first, second)) in first.into_iter().zip(second).enumerate() {
            // SAFETY: `sums` has 32 values, four per row.
            unsafe {
                let at = sums.as_mut_ptr().add(row * 4).cast::<v128>();
                v128_store(at, i32x4_add(v128_load(at), i32x4_add(first, second)));
            }
        }
    }

    /// The sum of weights times inputs, sixteen at a time: in two 16-bit dot
    /// products, or in one with relaxed SIMD's dot product of bytes.
    pub fn dot(weights: &[i8], inputs: &[u8]) -> i32 {
        assert!(weights.len() == inputs.len() && weights.len().is_multiple_of(16));
        let mut total = i32x4_splat(0);
        for at in (0..weights.len()).step_by(16) {
            // SAFETY: sixteen bytes from `at` lie inside both.
            let (w, x) = unsafe {
                (
                    v128_load(weights.as_ptr().add(at).cast()),
                    v128_load(inputs.as_ptr().add(at).cast()),
                )
            };
            #[cfg(not(target_feature = "relaxed-simd"))]
            {
                use std::arch::wasm32::{
                    i16x8_extend_high_i8x16, i16x8_extend_low_i8x16, i32x4_dot_i16x8,
                    u16x8_extend_high_u8x16, u16x8_extend_low_u8x16,
                };
                let low = i32x4_dot_i16x8(i16x8_extend_low_i8x16(w), u16x8_extend_low_u8x16(x));
                let high = i32x4_dot_i16x8(i16x8_extend_high_i8x16(w), u16x8_extend_high_u8x16(x));
                total = i32x4_add(total, i32x4_add(low, high));
            }
            #[cfg(target_feature = "relaxed-simd")]
            {
                total = std::arch::wasm32::i32x4_relaxed_dot_i8x16_i7x16_add(w, x, total);
            }
        }
        let pairs = i32x4_add(total, i32x4_shuffle::<2, 3, 0, 1>(total, total));
        i32x4_extract_lane::<0>(i32x4_add(pairs, i32x4_shuffle::<1, 0, 3, 2>(pairs, pairs)))
    }
}

#[cfg(test)]
impl Accumulator {
    /// The number of pieces of the position.
    fn pieces(&self) -> u32 {
        self.board.occupied().count_ones()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::OnceLock;

    use super::*;
    use crate::board::from_fen;
    use crate::nnue::legacy::tests::Random;

    /// A network file of pseudo-random weights, of the sizes a trained one
    /// has: first-layer sums spread around 0-255, three chunks in four of
    /// the first 512 outputs zero, as most are, and the 64 stacks the table
    /// says, every third of the 192.
    pub(crate) fn file() -> Vec<u8> {
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        let mut file = Vec::with_capacity(140_000_000);
        let u32s = |file: &mut Vec<u8>, values: &[u32]| {
            for value in values {
                file.extend_from_slice(&value.to_le_bytes());
            }
        };
        let description = b"provari's test network";
        u32s(
            &mut file,
            &[
                VERSION,
                transformer_hash(FILE_HALF) ^ stack_hash(FILE_HALF) ^ TABLE_HASH,
                description.len() as u32,
            ],
        );
        file.extend_from_slice(description);
        u32s(&mut file, &[TABLE_HASH, STACKS as u32, STACKS as u32 / 3]);
        for stack in 0..STACKS as u16 {
            file.extend_from_slice(&(stack - stack % 3).to_le_bytes());
        }
        u32s(&mut file, &[transformer_hash(FILE_HALF)]);
        for index in 0..FILE_HALF {
            let dead = dead(index);
            let bias = if dead {
                -2000 - (random.next() % 500) as i16
            } else {
                64 + (random.next() % 192) as i16
            };
            file.extend_from_slice(&bias.to_le_bytes());
        }
        // Bytes shifted down, as signed ones, by `shift` bits.
        let mut bytes = |file: &mut Vec<u8>, count: usize, shift: u32| {
            for _ in 0..count.div_ceil(8) {
                file.extend(
                    random
                        .next()
                        .to_le_bytes()
                        .map(|byte| ((byte as i8) >> shift) as u8),
                );
            }
            file.truncate(file.len() - (count.div_ceil(8) * 8 - count));
        };
        bytes(&mut file, (THREATS + PAIRS + BLOCKS * BLOCK) * FILE_HALF, 3);
        for _ in 0..STACKS / 3 {
            file.extend_from_slice(&stack_hash(FILE_HALF).to_le_bytes());
            // Each layer's biases, within 4096 of zero, then its weights.
            for (outputs, inputs, shift) in [(FC0, FILE_HALF, 2), (FC1, 2 * FC0, 3), (1, LAST, 0)] {
                bytes(&mut file, 4 * outputs, 0);
                for bias in file.rchunks_exact_mut(4).take(outputs) {
                    let bits = u32::from_le_bytes([bias[0], bias[1], bias[2], bias[3]]);
                    bias.copy_from_slice(&((bits % 8193) as i32 - 4096).to_le_bytes());
                }
                bytes(&mut file, outputs * inputs, shift);
            }
        }
        file
    }

    /// Whether a value of the test network's first 512 is one of the three
    /// chunks in four whose bias keeps it below zero, so that its output is
    /// always zero.
    fn dead(index: usize) -> bool {
        index < FILE_HALF / 2 && !(index / 4).is_multiple_of(4)
    }

    /// The outputs the test network keeps: the 128 that are not always
    /// zero, then 128 that are.
    fn kept() -> Vec<usize> {
        let (live, dead): (Vec<usize>, Vec<usize>) = (0..FILE_HALF / 2).partition(|&o| !dead(o));
        live.into_iter().chain(dead).take(HALF / 2).collect()
    }

    /// The test network, pruned, read once: its values are those of the
    /// file, which varifish reads, as the outputs dropped are always zero.
    pub(crate) fn network() -> &'static Network {
        static NETWORK: OnceLock<Network> = OnceLock::new();
        NETWORK.get_or_init(|| Network::from_bytes(&pruned()).unwrap())
    }

    /// The test network's pruned file.
    pub(crate) fn pruned() -> Vec<u8> {
        prune(&file(), &kept(), (1, 1), "provari's test network").unwrap()
    }

    /// The first layer in the places of the file's 1024 outputs.
    fn unpruned(values: &[u8; INPUTS]) -> [u8; INPUTS] {
        let mut layer = [0; INPUTS];
        for (place, output) in kept().into_iter().enumerate() {
            layer[output] = values[place];
            layer[INPUTS / 2 + output] = values[INPUTS / 2 + place];
        }
        assert!(values[HALF / 2..INPUTS / 2].iter().all(|&value| value == 0));
        assert!(
            values[INPUTS / 2 + HALF / 2..]
                .iter()
                .all(|&value| value == 0)
        );
        layer
    }

    /// Writes the test network to the file `PROVARI_TEST_NETWORK` names,
    /// for varifish to evaluate: `matches_the_engine`'s values are its own.
    #[test]
    #[ignore]
    fn writes_the_test_network() {
        let path = std::env::var("PROVARI_TEST_NETWORK").unwrap();
        std::fs::write(path, file()).unwrap();
    }

    /// Random games from starts that reach what updates must handle:
    /// captures that blow up many pieces, promotions, en passant, castling,
    /// kings that walk and endings. `seen` gets each position after a move,
    /// with its parent and the move.
    fn games(
        seed: u64,
        rounds: usize,
        mut seen: impl FnMut(&Position, &Position, crate::board::Move),
    ) {
        let starts = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rn2k2r/pp3ppp/2p5/3pP3/8/2N5/PPP2PPP/R3K2R w KQkq d6 0 1",
            "r3k2r/1P4pp/8/2pP4/8/8/p5PP/R3K2R w KQkq c6 0 1",
            "r1bqBb2/pp1pk3/8/4P1pp/8/2P5/PP3P2/R1BK3R b - - 1 11",
            // Pieces side by side, which a capture blows up together.
            "4k3/8/nbrqrbnn/NBRQRBNN/8/8/8/4K3 w - - 0 1",
            "4k3/8/1nqrbn2/1rnbqr2/1NQRBN2/1RNBQR2/8/4K3 b - - 0 1",
            // Kings that walk, among knights, heavy pieces and pawns.
            "8/8/8/8/8/8/NkN5/1K6 w - - 0 1",
            "4K1R1/RR6/8/8/8/8/qqq3p1/7k b - - 0 1",
            "1k6/2qqq3/2rbr3/8/8/3RBR2/3QQQ2/6K1 w - - 0 1",
            "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
        ];
        let mut random = Random(seed);
        for start in starts {
            for _ in 0..rounds {
                let mut pos: Position = from_fen(start).unwrap();
                for _ in 0..120 {
                    let moves: Vec<_> = pos
                        .pseudo_legal()
                        .as_slice()
                        .iter()
                        .copied()
                        .filter(|&m| pos.legal(m))
                        .collect();
                    if moves.is_empty() {
                        break;
                    }
                    let m = moves[(random.next() % moves.len() as u64) as usize];
                    let next = pos.play(m);
                    // A position without a king has ended.
                    if Color::ALL.iter().any(|&side| next.king(side).is_none()) {
                        break;
                    }
                    seen(&pos, &next, m);
                    pos = next;
                }
            }
        }
    }

    /// The network's value and first layer for the side to move, from
    /// scratch.
    fn evaluate(network: &Network, pos: &Position) -> (i32, [u8; INPUTS]) {
        let mut accumulator = Accumulator::default();
        network.refresh(pos, &mut accumulator);
        let mut transformed = Transformed::default();
        network.transform_for(&accumulator, pos.turn(), &mut transformed);
        (
            network.output_for(&transformed, &accumulator, pos.turn()),
            unpruned(&transformed.values),
        )
    }

    /// varifish's own evaluations of the test network: of a few positions,
    /// and of every position of random games, the values and first layers
    /// as one FNV-1a hash of their lines, `VALUE LAYER` with the layer in
    /// hexadecimal. They come from varifish's library reading the file
    /// `writes_the_test_network` writes.
    #[test]
    fn matches_the_engine() {
        let network = network();
        let expected = [
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                1928,
            ),
            (
                "r1bqBb2/pp1pk3/8/4P1pp/8/2P5/PP3P2/R1BK3R b - - 1 11",
                -3551,
            ),
            ("4k3/8/nbrqrbnn/NBRQRBNN/8/8/8/4K3 w - - 0 1", 413),
            ("8/8/8/8/8/8/NkN5/1K6 w - - 0 1", 609),
            ("4K1R1/RR6/8/8/8/8/qqq3p1/7k b - - 0 1", 544),
            ("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 b - - 0 1", 1500),
        ];
        for (fen, value) in expected {
            assert_eq!(evaluate(network, &from_fen(fen).unwrap()).0, value, "{fen}");
        }
        let (mut hash, mut positions) = (0xcbf2_9ce4_8422_2325u64, 0);
        games(0x1234_5678_9abc_def1, 30, |_, pos, _| {
            let (value, layer) = evaluate(network, pos);
            let layer: String = layer.iter().map(|value| format!("{value:02x}")).collect();
            for byte in format!("{value} {layer}\n").bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
            positions += 1;
        });
        assert_eq!((positions, hash), (23_299, 0xfcef_0425_287d_0f40));
    }

    /// A position's first layer by its definition: for each side the biases
    /// and the rows of its inputs, added one at a time and wrapping. The
    /// threats are every attack of a piece on another that has an input.
    fn defined(network: &Network, pos: &Position) -> [[i16; HALF]; 2] {
        let board = Board::of(pos);
        let mut values = [*network.biases; 2];
        for (side, values) in values.iter_mut().enumerate() {
            let mut add = |weights: &[i8], row: usize| {
                for (value, &weight) in values.iter_mut().zip(&weights[row * HALF..]) {
                    *value = value.wrapping_add(i16::from(weight));
                }
            };
            let king = board.king(side);
            let orient = orientation(side, king);
            for from in squares(board.occupied()) {
                let code = board.squares[from];
                let row = piece_square(side, king, from, code) as usize;
                add(&network.piece_squares, board.block(side) * BLOCK + row);
                let (role, at) = (usize::from(code & 7), Square::new(from as u32));
                let reached = match role {
                    PAWN => attacks::pawn_attacks(Color::from_white(code & 8 != 0), at),
                    KNIGHT => attacks::knight_attacks(at),
                    BISHOP => attacks::bishop_attacks(at, pos.occupied()),
                    ROOK => attacks::rook_attacks(at, pos.occupied()),
                    QUEEN => attacks::queen_attacks(at, pos.occupied()),
                    _ => Bitboard(0),
                };
                for to in squares(reached.0 & board.occupied()) {
                    let row = threat(side, orient, code, from, to, board.squares[to]) as usize;
                    if row < THREATS {
                        add(&network.threats, row);
                    }
                }
            }
            // Every two pawns on the same file or on files side by side.
            let pawns: Vec<usize> = squares(board.roles[PAWN]).collect();
            for (index, &first) in pawns.iter().enumerate() {
                for &second in &pawns[index + 1..] {
                    if (first & 7).abs_diff(second & 7) > 1 {
                        continue;
                    }
                    let id = |square: usize| {
                        let white = board.white >> square & 1 != 0;
                        48 * usize::from(white != (side == WHITE)) + (square ^ orient) - 8
                    };
                    let (low, high) = (id(first).min(id(second)), id(first).max(id(second)));
                    add(&network.threats, THREATS + high * (high - 1) / 2 + low);
                }
            }
        }
        values
    }

    /// A first layer's two parts, added.
    fn sums(accumulator: &Accumulator) -> [[i16; HALF]; 2] {
        [0, 1].map(|side| {
            std::array::from_fn(|index| {
                accumulator.piece_squares[side][index]
                    .wrapping_add(accumulator.threats[side][index])
            })
        })
    }

    /// The first layers a line updates, and those computed from scratch,
    /// match the definition over random games: after plain moves, captures
    /// that blow up many pieces, moves of a king within its half of the
    /// board and across it, and changes of a side's block.
    #[test]
    fn updates_match_refreshes() {
        let network = network();
        let mut cache = RefreshCache::new(network);
        let mut parent = Accumulator::default();
        // Positions, and those among them after a capture that changed eight
        // squares or more, after a king's move, after one that crossed the
        // middle, after a change of block, and plain moves without their
        // positions.
        let mut seen = [0; 6];
        let mut most = 0;
        let mut last: Option<Position> = None;
        let mut check = |pos: &Position, next: &Position, m: crate::board::Move| {
            if last.as_ref().is_none_or(|last| last.hash() != pos.hash()) {
                network.refresh(pos, &mut parent);
            }
            let expected = defined(network, next);
            let mut updated = Accumulator::default();
            network.update(&parent, next, &mut updated, &mut cache);
            assert_eq!(sums(&updated), expected);
            let mut refreshed = Accumulator::default();
            network.refresh(next, &mut refreshed);
            assert_eq!(sums(&refreshed), expected);
            // Each part, too, and the rows of attacked squares.
            assert_eq!(updated.piece_squares, refreshed.piece_squares);
            assert_eq!(updated.threats, refreshed.threats);
            assert_eq!(updated.rows, refreshed.rows);
            let changed = changed_squares(pos.squares(), next.squares()).count_ones();
            let (before, after) = (Board::of(pos), Board::of(next));
            let mover = pos.turn() as usize;
            seen[0] += 1;
            most = most.max(changed);
            seen[1] += usize::from(changed >= 8);
            seen[2] += usize::from(pos.role_at(m.from()) == Some(Role::King));
            seen[3] += usize::from(
                orientation(mover, before.king(mover)) != orientation(mover, after.king(mover)),
            );
            seen[4] += usize::from((0..2).any(|side| before.block(side) != after.block(side)));
            // A plain move's first layer without the position.
            for &other in pos.pseudo_legal().as_slice().iter().take(8) {
                let (from, to) = (other.from(), other.to());
                let plain = pos.legal(other)
                    && pos.role_at(from) != Some(Role::King)
                    && pos.role_at(to).is_none()
                    && other.promotion().is_none()
                    && !other.is_en_passant()
                    && !other.is_castle();
                if plain {
                    let mut quiet = Accumulator::default();
                    network.update_quiet(&parent, from, to, &mut quiet);
                    assert_eq!(sums(&quiet), defined(network, &pos.play(other)));
                    let them = !pos.turn();
                    let mut transformed = Transformed::default();
                    network.transform_for(&quiet, them, &mut transformed);
                    assert_eq!(
                        (
                            network.output_for(&transformed, &quiet, them),
                            unpruned(&transformed.values)
                        ),
                        evaluate(network, &pos.play(other))
                    );
                    seen[5] += 1;
                }
            }
            last = Some(next.clone());
            parent = updated;
        };
        games(0x9e37_79b9_7f4a_7c15, 12, &mut check);
        // The most squares a move changes: a knight from afar captures a
        // piece with eight around it.
        let pos = from_fen("4k3/8/1nqrbn2/1rnbqr2/1NQRBN2/1RNBQR2/8/4K3 b - - 0 1").unwrap();
        let capture = *pos
            .pseudo_legal()
            .as_slice()
            .iter()
            .find(|m| (m.from(), m.to()) == (Square::B6, Square::C4))
            .unwrap();
        check(&pos, &pos.play(capture), capture);
        assert!(
            seen[0] > 5000
                && seen[1] >= 5
                && seen[2] > 500
                && seen[3] > 50
                && seen[4] > 50
                && seen[5] > 5000,
            "{seen:?}"
        );
        assert_eq!(most as usize, MOVED);
    }

    /// A position that is not a move from its parent, as a search's root
    /// after another's, starts over.
    #[test]
    fn updates_from_any_position() {
        let network = network();
        let mut cache = RefreshCache::new(network);
        let mut accumulator = Accumulator::default();
        let first = from_fen("4k3/8/nbrqrbnn/NBRQRBNN/8/8/8/4K3 w - - 0 1").unwrap();
        network.refresh(&first, &mut accumulator);
        let mut positions = 0;
        games(0x2545_f491_4f6c_dd1d, 1, |_, pos, _| {
            if positions % 7 == 0 {
                let mut updated = Accumulator::default();
                network.update(&accumulator, pos, &mut updated, &mut cache);
                assert_eq!(sums(&updated), defined(network, pos));
                assert_eq!(updated.pieces(), pos.occupied().count() as u32);
                accumulator = updated;
            }
            positions += 1;
        });
        assert!(positions > 300);
    }

    /// Every attack of a piece on another has an input of its own or none,
    /// 59,808 in all with those left out as twins, and every two pawns have
    /// a pair.
    #[test]
    fn inputs_are_distinct() {
        let code = |white: bool, role: usize| (role | if white { 8 } else { 0 }) as u8;
        let mut seen = vec![0u8; THREATS];
        for (attacker_white, attacked_white) in
            [(true, true), (true, false), (false, true), (false, false)]
        {
            for attacker in PAWN..=KING {
                for attacked in PAWN..=KING {
                    for from in 0..64 {
                        if attacker == PAWN && !(8..56).contains(&from) {
                            continue;
                        }
                        for to in squares(reach(attacker, attacker_white, from)) {
                            let (attacker, attacked) = (
                                code(attacker_white, attacker),
                                code(attacked_white, attacked),
                            );
                            // White's side with its king on e1.
                            let row = threat(WHITE, 0, attacker, from, to, attacked) as usize;
                            if row < THREATS {
                                seen[row] += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&count| count <= 1));
        // Pieces of one type attack each other alike, so only attacks from
        // the higher square have inputs: half of the attacks of knights,
        // bishops, rooks and queens on their like, of either color.
        let twins: u32 = [KNIGHT, BISHOP, ROOK, QUEEN]
            .iter()
            .map(|&role| {
                (0..64)
                    .map(|from| reach(role, true, from).count_ones())
                    .sum::<u32>()
            })
            .sum();
        let used = seen.iter().filter(|&&count| count == 1).count();
        assert_eq!(used + 2 * twins as usize, THREATS);

        let mut seen = vec![false; PAIRS];
        for (first_white, second_white) in [(true, true), (true, false), (false, false)] {
            for first in 8..56usize {
                for second in 8..56usize {
                    if first_white == second_white && second <= first {
                        continue;
                    }
                    if (first & 7).abs_diff(second & 7) > 1 {
                        continue;
                    }
                    let (mut white, mut black) = (0u64, 0u64);
                    for (square, is_white) in [(first, first_white), (second, second_white)] {
                        *(if is_white { &mut white } else { &mut black }) |= 1 << square;
                    }
                    if white & black != 0 {
                        continue;
                    }
                    let mut rows = List::new();
                    pairs(WHITE, 0, white | black, white, black, &mut rows);
                    let &[row] = rows.as_slice() else {
                        panic!("{first} {second}: {:?}", rows.as_slice());
                    };
                    let row = row as usize - THREATS;
                    assert!(!seen[row], "{first} {second}");
                    seen[row] = true;
                    // The same pair from Black's side with the colors
                    // swapped and the board flipped.
                    let flip = |bits: u64| bits.swap_bytes();
                    let mut rows = List::new();
                    pairs(
                        0,
                        56,
                        flip(white | black),
                        flip(black),
                        flip(white),
                        &mut rows,
                    );
                    assert_eq!(rows.as_slice(), [(row + THREATS) as u32]);
                }
            }
        }
    }

    /// A file is read whole or refused: cut short, with bytes left over,
    /// of another architecture, or with a table that names a stack the file
    /// does not store.
    #[test]
    fn files_are_read_whole_or_refused() {
        // varifish's own file is refused, with the way to prune it.
        assert!(Network::from_bytes(&file()).is_err_and(|error| error.contains("prune")));
        let file = pruned();
        let refused = |bytes: &[u8]| Network::from_bytes(bytes).is_err();
        assert!(refused(&file[..file.len() - 1]));
        assert!(refused(&[&file[..], &[0]].concat()));
        assert!(refused(&file[..2_000_000]));
        let mut other = file[..2_000_000].to_vec();
        other[4] ^= 1;
        assert!(
            Network::from_bytes(&other).is_err_and(|error| error.starts_with("not an SFNNv20"))
        );
        // The table follows the description and the factor: its hash, the
        // 192, the stored count, then each stack's.
        let table = 12 + 22 + 8;
        let mut other = file.clone();
        other[table + 12 + 2 * 50] = 49;
        assert!(Network::from_bytes(&other).is_err_and(|error| error.contains("table")));
        let mut other = file.clone();
        other[table + 8] += 1;
        assert!(Network::from_bytes(&other).is_err_and(|error| error.contains("table")));
        // The stacks a position chooses among: every third is stored.
        let network = network();
        assert_eq!(network.stacks.len(), STACKS / 3);
        assert!((0..STACKS).all(|stack| usize::from(network.stored[stack]) == stack / 3));
    }

    /// Each kernel gives, with the processor's vectors, what the plain code
    /// gives: rows of weights taken away and added, the products of the
    /// first layer's values in and out of their clipped range, and the dense
    /// layers over inputs of any sparsity and sums of any size.
    #[test]
    fn kernels_as_portable() {
        let mut random = Random(0x2545_f491_4f6c_dd1d);
        let rows = 96;
        let weights: Vec<i8> = (0..rows * HALF).map(|_| random.next() as i8).collect();
        for trial in 0..300 {
            let from: [i16; HALF] = std::array::from_fn(|_| random.next() as i16);
            let mut list = |most: u64| -> Vec<u32> {
                let count = random.next() % (most + 1);
                (0..count)
                    .map(|_| (random.next() % rows as u64) as u32)
                    .collect()
            };
            let (removed, added) = (list(trial % 50), list(trial % 70));
            let (mut vector, mut plain) = ([0; HALF], [0; HALF]);
            change_rows(&weights, &from, &mut vector, &removed, &added);
            change_rows_portable(&weights, &from, &mut plain, &removed, &added);
            assert_eq!(vector, plain, "trial {trial}");
        }
        for trial in 0..2000 {
            // Sums around the clipped range, far from it, and wrapping.
            let spread = [600, 3000, 70_000][trial % 3];
            let mut part = || -> [i16; HALF] {
                std::array::from_fn(|_| {
                    ((random.next() % spread) as i64 - spread as i64 / 2) as i16
                })
            };
            let (piece_squares, threats) = (part(), part());
            let (mut vector, mut plain) = ([0; HALF / 2], [0; HALF / 2]);
            products(&piece_squares, &threats, &mut vector);
            products_portable(&piece_squares, &threats, &mut plain);
            assert_eq!(vector, plain, "trial {trial}");
        }
        let stack = Stack {
            b0: [0; FC0],
            w0: (0..FC0 * INPUTS).map(|_| random.next() as i8).collect(),
            b1: std::array::from_fn(|_| (random.next() % 20_001) as i32 - 10_000),
            w1: (0..FC1 * 2 * FC0).map(|_| random.next() as i8).collect(),
            b2: -1234,
            w2: std::array::from_fn(|_| random.next() as i8),
        };
        for trial in 0..3000 {
            // Outputs of the first layer, at most 127, a chunk of four in
            // `trial % 9 + 1` not zero.
            let mut transformed = Transformed::default();
            for chunk in transformed.values.as_chunks_mut::<4>().0 {
                if random.next().is_multiple_of(trial % 9 + 1) {
                    *chunk = (random.next() as u32 & 0x7f7f_7f7f).to_le_bytes();
                }
            }
            transformed.list();
            let biases: [i32; FC0] = std::array::from_fn(|_| random.next() as i32 >> 12);
            let (mut vector, mut plain) = (biases, biases);
            first_layer(&stack.w0, &transformed, &mut vector);
            first_layer_portable(&stack.w0, &transformed, &mut plain);
            assert_eq!(vector, plain, "trial {trial}");
            // Sums as the first layer gives them, small ones, and any.
            let sums = match trial % 3 {
                0 => plain,
                1 => plain.map(|sum| sum >> 9),
                _ => std::array::from_fn(|_| random.next() as i32),
            };
            assert_eq!(
                later_layers(&stack, &sums),
                later_layers_portable(&stack, &sums),
                "trial {trial}"
            );
        }
    }

    /// The stack of a position, by its pieces, queens, pawns and officers.
    #[test]
    fn stacks_as_defined() {
        for (fen, stack) in [
            // 32 pieces, both queens, pawns on both halves, officers.
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                ((28 + 3) * 3) * 2 + 1,
            ),
            // Two kings: no queens, no pawns, no officers.
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 4),
            // Five pieces, the side to move's queen, pawns on one half.
            ("4k3/8/8/8/8/8/5PP1/3QK3 w - - 0 1", ((4 + 2) * 3 + 1) * 2),
            ("4k3/8/8/8/8/8/5PP1/3QK3 b - - 0 1", ((4 + 1) * 3 + 1) * 2),
            // Four pieces, the other side's queen and its two officers.
            ("4k3/8/8/8/8/8/8/2NQK3 b - - 0 1", (3 + 2) * 2 + 1),
        ] {
            let pos = from_fen(fen).unwrap();
            assert_eq!(Board::of(&pos).stack(pos.turn() as usize), stack, "{fen}");
        }
    }
}
