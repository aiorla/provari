//! Where a table keeps a position: tolius's indexing of his Antichess tables,
//! translated from his antichess-tb-api (`tb/egtb/tb_idx.cpp`), which took it
//! from Stockfish's Syzygy code (Ronald de Man, 2013; Marco Costalba and
//! Lucas Braesch, 2016-2018) and changed it for Antichess, all under the
//! GPL, version 3 or later. It follows his code step for step, quirks
//! included, so that every position lands where his writer put it.
//!
//! Its loops index as his do, so that each line can be set beside his.
//!
//! Pieces are numbered as in Stockfish: a type from 1 (pawn) to 6 (king), and
//! 8 more for Black's. Squares run from a1 (0) to h8 (63).

#![allow(clippy::needless_range_loop)]

use std::sync::LazyLock;

/// The most pieces of a table that the index knows.
const MOST: usize = 6;

/// A side's pieces: how many of each type, by colour (White's first) and
/// type (1 to 6).
pub type Material = [[u8; 7]; 2];

/// A position as the index reads it.
pub struct Placed {
    /// The pieces on each square, 0 for none.
    pub board: [u8; 64],
    /// Squares holding a piece, the lowest first.
    pub occupied: u64,
    /// 0 for White to move, 1 for Black.
    pub turn: usize,
    pub material: Material,
}

impl Placed {
    fn pawns(&self, color: usize) -> u64 {
        let pawn = 1 + 8 * color as u8;
        let mut bits = 0;
        for square in each_square(self.occupied) {
            if self.board[square as usize] == pawn {
                bits |= 1 << square;
            }
        }
        bits
    }
}

fn each_square(mut bits: u64) -> impl Iterator<Item = i32> {
    std::iter::from_fn(move || {
        (bits != 0).then(|| {
            let square = bits.trailing_zeros() as i32;
            bits &= bits - 1;
            square
        })
    })
}

fn file(square: i32) -> i32 {
    square & 7
}

fn rank(square: i32) -> i32 {
    square >> 3
}

fn off_a1h8(square: i32) -> i32 {
    rank(square) - file(square)
}

fn flip_diagonal(square: i32) -> i32 {
    ((square >> 3) | (square << 3)) & 63
}

const TRIANGLE: [i32; 64] = [
    6, 0, 1, 2, 2, 1, 0, 6, //
    0, 7, 3, 4, 4, 3, 7, 0, //
    1, 3, 8, 5, 5, 8, 3, 1, //
    2, 4, 5, 9, 9, 5, 4, 2, //
    2, 4, 5, 9, 9, 5, 4, 2, //
    1, 3, 8, 5, 5, 8, 3, 1, //
    0, 7, 3, 4, 4, 3, 7, 0, //
    6, 0, 1, 2, 2, 1, 0, 6, //
];

#[rustfmt::skip]
const MAP_PP: [[i32; 64]; 10] = [
    [  0, -1,  1,  2,  3,  4,  5,  6,
       7,  8,  9, 10, 11, 12, 13, 14,
      15, 16, 17, 18, 19, 20, 21, 22,
      23, 24, 25, 26, 27, 28, 29, 30,
      31, 32, 33, 34, 35, 36, 37, 38,
      39, 40, 41, 42, 43, 44, 45, 46,
      -1, 47, 48, 49, 50, 51, 52, 53,
      54, 55, 56, 57, 58, 59, 60, 61 ],
    [ 62, -1, -1, 63, 64, 65, -1, 66,
      -1, 67, 68, 69, 70, 71, 72, -1,
      73, 74, 75, 76, 77, 78, 79, 80,
      81, 82, 83, 84, 85, 86, 87, 88,
      89, 90, 91, 92, 93, 94, 95, 96,
      -1, 97, 98, 99,100,101,102,103,
      -1,104,105,106,107,108,109, -1,
     110, -1,111,112,113,114, -1,115 ],
    [116, -1, -1, -1,117, -1, -1,118,
      -1,119,120,121,122,123,124, -1,
      -1,125,126,127,128,129,130, -1,
     131,132,133,134,135,136,137,138,
      -1,139,140,141,142,143,144,145,
      -1,146,147,148,149,150,151, -1,
      -1,152,153,154,155,156,157, -1,
     158, -1, -1,159,160, -1, -1,161 ],
    [162, -1, -1, -1, -1, -1, -1,163,
      -1,164, -1,165,166,167,168, -1,
      -1,169,170,171,172,173,174, -1,
      -1,175,176,177,178,179,180, -1,
      -1,181,182,183,184,185,186, -1,
      -1, -1,187,188,189,190,191, -1,
      -1,192,193,194,195,196,197, -1,
     198, -1, -1, -1, -1, -1, -1,199 ],
    [200, -1, -1, -1, -1, -1, -1,201,
      -1,202, -1, -1,203, -1,204, -1,
      -1, -1,205,206,207,208, -1, -1,
      -1,209,210,211,212,213,214, -1,
      -1, -1,215,216,217,218,219, -1,
      -1, -1,220,221,222,223, -1, -1,
      -1,224, -1,225,226, -1,227, -1,
     228, -1, -1, -1, -1, -1, -1,229 ],
    [230, -1, -1, -1, -1, -1, -1,231,
      -1,232, -1, -1, -1, -1,233, -1,
      -1, -1,234, -1,235,236, -1, -1,
      -1, -1,237,238,239,240, -1, -1,
      -1, -1, -1,241,242,243, -1, -1,
      -1, -1,244,245,246,247, -1, -1,
      -1,248, -1, -1, -1, -1,249, -1,
     250, -1, -1, -1, -1, -1, -1,251 ],
    [ -1, -1, -1, -1, -1, -1, -1,259,
      -1,252, -1, -1, -1, -1,260, -1,
      -1, -1,253, -1, -1,261, -1, -1,
      -1, -1, -1,254,262, -1, -1, -1,
      -1, -1, -1, -1,255, -1, -1, -1,
      -1, -1, -1, -1, -1,256, -1, -1,
      -1, -1, -1, -1, -1, -1,257, -1,
      -1, -1, -1, -1, -1, -1, -1,258 ],
    [ -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1,268, -1,
      -1, -1,263, -1, -1,269, -1, -1,
      -1, -1, -1,264,270, -1, -1, -1,
      -1, -1, -1, -1,265, -1, -1, -1,
      -1, -1, -1, -1, -1,266, -1, -1,
      -1, -1, -1, -1, -1, -1,267, -1,
      -1, -1, -1, -1, -1, -1, -1, -1 ],
    [ -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,274, -1, -1,
      -1, -1, -1,271,275, -1, -1, -1,
      -1, -1, -1, -1,272, -1, -1, -1,
      -1, -1, -1, -1, -1,273, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1 ],
    [ -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1,277, -1, -1, -1,
      -1, -1, -1, -1,276, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1 ],
];

const MULT_TWIST: [i32; 64] = [
    15, 63, 55, 47, 40, 48, 56, 12, //
    62, 11, 39, 31, 24, 32, 8, 57, //
    54, 38, 7, 23, 16, 4, 33, 49, //
    46, 30, 22, 3, 0, 17, 25, 41, //
    45, 29, 21, 2, 1, 18, 26, 42, //
    53, 37, 6, 20, 19, 5, 34, 50, //
    61, 10, 36, 28, 27, 35, 9, 58, //
    14, 60, 52, 44, 43, 51, 59, 13, //
];

/// The a5-c5-a7 triangle.
const TEST45: u64 = 0x1_0307_0000_0000;
const INV_TRIANGLE: [i32; 10] = [1, 2, 3, 10, 11, 19, 0, 9, 18, 27];

/// The tables every index reads, made once (`Tablebases_init`).
struct Maps {
    pawns: [i32; 64],
    b1h1h7: [i32; 64],
    a1d1d4: [i32; 64],
    /// Ways to choose k of n: `binomial[k][n]`.
    binomial: [[i32; 64]; 6],
    lead_pawn_idx: [[i32; 64]; 5],
    lead_pawns_size: [[i32; 4]; 5],
    mult_idx: [[i32; 10]; 5],
    mult_factor: [i32; 5],
}

static MAPS: LazyLock<Maps> = LazyLock::new(|| {
    let mut maps = Maps {
        pawns: [0; 64],
        b1h1h7: [0; 64],
        a1d1d4: [0; 64],
        binomial: [[0; 64]; 6],
        lead_pawn_idx: [[0; 64]; 5],
        lead_pawns_size: [[0; 4]; 5],
        mult_idx: [[0; 10]; 5],
        mult_factor: [0; 5],
    };
    let mut code = 0;
    for square in 0..64 {
        if off_a1h8(square) < 0 {
            maps.b1h1h7[square as usize] = code;
            code += 1;
        }
    }
    let mut diagonal = Vec::new();
    code = 0;
    for square in 0..=27 {
        if off_a1h8(square) < 0 && file(square) <= 3 {
            maps.a1d1d4[square as usize] = code;
            code += 1;
        } else if off_a1h8(square) == 0 && file(square) <= 3 {
            diagonal.push(square);
        }
    }
    for square in diagonal {
        maps.a1d1d4[square as usize] = code;
        code += 1;
    }
    maps.binomial[0][0] = 1;
    for n in 1..64 {
        for k in 0..6usize.min(n + 1) {
            maps.binomial[k][n] = if k > 0 {
                maps.binomial[k - 1][n - 1]
            } else {
                0
            } + if k < n { maps.binomial[k][n - 1] } else { 0 };
        }
    }
    for i in 0..5 {
        let mut sum = 0;
        for j in 0..10 {
            maps.mult_idx[i][j] = sum;
            sum += if i == 0 {
                1
            } else {
                maps.binomial[i][MULT_TWIST[INV_TRIANGLE[j] as usize] as usize]
            };
        }
        maps.mult_factor[i] = sum;
    }
    let mut available = 47;
    for lead in 1..=4 {
        for f in 0..4 {
            let mut idx = 0;
            for r in 1..=6 {
                let square = (f + 8 * r) as usize;
                if lead == 1 {
                    maps.pawns[square] = available;
                    maps.pawns[square ^ 7] = available - 1;
                    available -= 2;
                }
                maps.lead_pawn_idx[lead][square] = idx;
                idx += maps.binomial[lead - 1][maps.pawns[square] as usize];
            }
            maps.lead_pawns_size[lead][f as usize] = idx;
        }
    }
    maps
});

/// How the pieces of one side to move and one file of the leading pawn are
/// grouped and numbered (`PairsData`).
#[derive(Clone, Copy, Default)]
struct Pairs {
    pieces: [u8; MOST],
    group_idx: [u64; MOST + 1],
    group_len: [i32; MOST + 1],
}

impl Pairs {
    /// The positions these groups number.
    fn size(&self) -> u64 {
        let end = self.group_len.iter().position(|&len| len == 0).unwrap_or(7);
        self.group_idx[end]
    }
}

/// A table's index (`TBTable`).
pub struct Index {
    key: Material,
    key2: Material,
    piece_count: usize,
    has_pawns: bool,
    unique: i32,
    min_like_man: i32,
    pawn_count: [u8; 2],
    items: [[Pairs; 4]; 2],
    /// 0 for White, 1 for Black.
    lead_color: usize,
    /// The bytes of the table's head that the index takes.
    pub header_size: usize,
}

/// The material a table's name gives, `KRvK` say, its first side White's
/// (`Position::set` with White).
pub fn material(name: &str, swapped: bool) -> Option<Material> {
    let (strong, weak) = name.split_once('v')?;
    let mut material = [[0; 7]; 2];
    for (side, pieces) in [(0, strong), (1, weak)] {
        let color = side ^ usize::from(swapped);
        for piece in pieces.chars() {
            let role = " PNBRQK".find(piece).filter(|&role| role > 0)?;
            material[color][role] += 1;
        }
    }
    Some(material)
}

/// The name of a table that holds `material`, as the reader looks one up
/// (`board_to_name`): the side with more pieces first, or with more of the
/// first type, kings to pawns, where they differ; Black first when the two
/// are the same.
pub fn name(material: &Material) -> String {
    let side = |color: usize| -> String {
        (1..7)
            .rev()
            .flat_map(|role| {
                std::iter::repeat_n(
                    " PNBRQK".as_bytes()[role] as char,
                    material[color][role] as usize,
                )
            })
            .collect()
    };
    let (white, black) = (side(0), side(1));
    let mut lead = None;
    for role in (1..7).rev() {
        if lead.is_none() {
            match material[0][role].cmp(&material[1][role]) {
                std::cmp::Ordering::Greater => lead = Some(0),
                std::cmp::Ordering::Less => lead = Some(1),
                std::cmp::Ordering::Equal => {}
            }
        }
    }
    match white.len().cmp(&black.len()) {
        std::cmp::Ordering::Greater => lead = Some(0),
        std::cmp::Ordering::Less => lead = Some(1),
        std::cmp::Ordering::Equal => {}
    }
    if lead == Some(0) {
        format!("{white}v{black}")
    } else {
        format!("{black}v{white}")
    }
}

impl Index {
    /// The index of the table `name`, from the first bytes of its data,
    /// which say how its pieces are grouped.
    pub fn new(name: &str, data: &[u8]) -> Result<Index, String> {
        let maps = &*MAPS;
        let key = material(name, false).ok_or_else(|| format!("{name}: not a table's name"))?;
        let key2 = material(name, true).expect("as above");
        let count = |color: usize, role: usize| i32::from(key[color][role]);
        let piece_count: usize = key.iter().flatten().map(|&n| usize::from(n)).sum();
        if !(2..=MOST).contains(&piece_count) || key[0].iter().sum::<u8>() == 0 {
            return Err(format!("{name}: not a table's name"));
        }
        let has_pawns = count(0, 1) + count(1, 1) > 0;
        let mut unique = 0;
        let mut min_like_man = 0;
        for color in 0..2 {
            for role in 1..7 {
                let n = count(color, role);
                if n == 1 {
                    unique += 1;
                }
                if 2 <= n && (n < min_like_man || min_like_man == 0) {
                    min_like_man = n;
                }
            }
        }
        // The side with fewer pawns leads, as it packs better.
        let white_leads = count(1, 1) == 0 || (count(0, 1) > 0 && count(1, 1) >= count(0, 1));
        let pawn_count = if white_leads {
            [key[0][1], key[1][1]]
        } else {
            [key[1][1], key[0][1]]
        };
        let mut index = Index {
            key,
            key2,
            piece_count,
            has_pawns,
            unique,
            min_like_man,
            pawn_count,
            items: [[Pairs::default(); 4]; 2],
            lead_color: 0,
            header_size: 0,
        };

        // How the pieces are grouped, from the data (`set`).
        let sides = if key != key2 { 2 } else { 1 };
        let both = has_pawns && pawn_count[1] > 0;
        let mut at = 0;
        let byte = |at: &mut usize| -> Result<u8, String> {
            let byte = *data
                .get(*at)
                .ok_or_else(|| format!("{name}: a head too short"))?;
            *at += 1;
            Ok(byte)
        };
        for f in 0..if has_pawns { 4 } else { 1 } {
            let first = byte(&mut at)?;
            let second = if both { byte(&mut at)? } else { 0xFF };
            let order = [
                [i32::from(first & 0xF), i32::from(second & 0xF)],
                [i32::from(first >> 4), i32::from(second >> 4)],
            ];
            for k in 0..piece_count {
                let byte = byte(&mut at)?;
                for side in 0..sides {
                    index.items[side][f].pieces[k] = if side == 1 { byte >> 4 } else { byte & 0xF };
                }
            }
            for side in 0..sides {
                index.set_groups(side, f, order[side], maps);
            }
        }
        index.header_size = at;

        // Which colour the pieces of a table without pawns are stored for
        // (`TBTable_init`).
        if !has_pawns {
            let mut stored = [[0; 7]; 2];
            for &piece in &index.items[0][0].pieces[..piece_count] {
                let color = usize::from(piece >> 3 == 1);
                let role = usize::from(piece & 7);
                if !(1..7).contains(&role) {
                    return Err(format!("{name}: a piece of the head unknown"));
                }
                stored[color][role] += 1;
            }
            if stored != index.key {
                std::mem::swap(&mut index.key, &mut index.key2);
                index.lead_color = 1;
            }
            if stored != index.key {
                return Err(format!("{name}: its head holds other pieces"));
            }
        }
        Ok(index)
    }

    fn get(&self, side: usize, f: usize) -> &Pairs {
        &self.items[side % 2][if self.has_pawns { f } else { 0 }]
    }

    /// Groups the pieces of `side` and the file `f` for numbering
    /// (`set_groups`), the groups numbered in `order`.
    fn set_groups(&mut self, side: usize, f: usize, order: [i32; 2], maps: &Maps) {
        let (has_pawns, unique, min_like_man) = (self.has_pawns, self.unique, self.min_like_man);
        let piece_count = self.piece_count;
        let both = has_pawns && self.pawn_count[1] > 0;
        let d = &mut self.items[side][f];
        let mut n = 0;
        let mut first_len = if has_pawns {
            0
        } else if unique >= 3 {
            3
        } else {
            2
        };
        d.group_len[n] = 1;
        for i in 1..piece_count {
            first_len -= 1;
            if first_len > 0 || d.pieces[i] == d.pieces[i - 1] {
                d.group_len[n] += 1;
            } else {
                n += 1;
                d.group_len[n] = 1;
            }
        }
        n += 1;
        d.group_len[n] = 0;

        let mut next = if both { 2 } else { 1 };
        let mut free = 64 - d.group_len[0] - if both { d.group_len[1] } else { 0 };
        let mut idx: u64 = 1;
        let mut k = 0;
        while next < n || k == order[0] || k == order[1] {
            if k == order[0] {
                d.group_idx[0] = idx;
                idx *= if has_pawns {
                    maps.lead_pawns_size[d.group_len[0] as usize][f]
                } else if unique >= 3 {
                    31332
                } else if unique == 2 {
                    518
                } else if min_like_man == 2 {
                    278
                } else {
                    maps.mult_factor[(min_like_man - 1) as usize]
                } as u64;
            } else if k == order[1] {
                d.group_idx[1] = idx;
                idx *=
                    maps.binomial[d.group_len[1] as usize][(48 - d.group_len[0]) as usize] as u64;
            } else {
                d.group_idx[next] = idx;
                idx *= maps.binomial[d.group_len[next] as usize][free as usize] as u64;
                free -= d.group_len[next];
                next += 1;
            }
            k += 1;
        }
        d.group_idx[n] = idx;
    }

    /// The positions of one side to move.
    pub fn size(&self) -> u64 {
        (0..if self.has_pawns { 4 } else { 1 })
            .map(|f| self.items[0][f].size())
            .sum()
    }

    /// Where the table keeps `pos`, whose material it holds (`get_idx`).
    pub fn index(&self, pos: &Placed) -> u64 {
        let maps = &*MAPS;
        let mut squares = [0i32; MOST];
        let mut pieces = [0u8; MOST];
        let mut size = 0;
        let mut lead_pawns_count = 0;
        let mut lead_pawns = 0;
        let mut tb_file = 0;

        // Tables are stored for the side of the key, and those of the same
        // pieces on each side for White to move: else colours and squares
        // flip.
        let symmetric_black = self.key == self.key2 && pos.turn == 1;
        let black_stronger = pos.material != self.key;
        let flip = symmetric_black || black_stronger;
        let flip_color = if flip { 8 } else { 0 };
        let flip_squares = if flip { 0o70 } else { 0 };
        let stm = usize::from(flip) ^ pos.turn;

        // With pawns, a table for each file of the leading pawn, the one
        // nearest the edge and lowest.
        if self.has_pawns {
            let pawn = self.get(0, 0).pieces[0] ^ flip_color;
            lead_pawns = pos.pawns(usize::from(pawn >> 3));
            for square in each_square(lead_pawns) {
                squares[size] = square ^ flip_squares;
                size += 1;
            }
            lead_pawns_count = size;
            // The first of the highest, as `std::max_element` finds it.
            let mut best = 0;
            for i in 1..lead_pawns_count {
                if maps.pawns[squares[i] as usize] > maps.pawns[squares[best] as usize] {
                    best = i;
                }
            }
            squares.swap(0, best);
            tb_file = file(squares[0]);
            if tb_file > 3 {
                tb_file = file(squares[0] ^ 7);
            }
        }
        for square in each_square(pos.occupied ^ lead_pawns) {
            squares[size] = square ^ flip_squares;
            pieces[size] = pos.board[square as usize] ^ flip_color;
            size += 1;
        }
        let d = self.get(stm, tb_file as usize);

        // The pieces in the order the table stores them.
        for i in lead_pawns_count..size.saturating_sub(1) {
            for j in i + 1..size {
                if d.pieces[i] == pieces[j] {
                    pieces.swap(i, j);
                    squares.swap(i, j);
                    break;
                }
            }
        }
        // The leading piece into the a1-d1-d4 triangle.
        if file(squares[0]) > 3 {
            for square in &mut squares[..size] {
                *square ^= 7;
            }
        }

        let mut idx: u64;
        if self.has_pawns {
            idx = maps.lead_pawn_idx[lead_pawns_count][squares[0] as usize] as u64;
            squares[1..lead_pawns_count].sort_by_key(|&square| maps.pawns[square as usize]);
            for i in 1..lead_pawns_count {
                idx += maps.binomial[i][maps.pawns[squares[i] as usize] as usize] as u64;
            }
        } else {
            if rank(squares[0]) > 3 {
                for square in &mut squares[..size] {
                    *square ^= 0o70;
                }
            }
            let lead_len = d.group_len[0] as usize;
            for i in 0..lead_len {
                if off_a1h8(squares[i]) == 0 {
                    continue;
                }
                if off_a1h8(squares[i]) > 0 {
                    for square in &mut squares[i..size] {
                        *square = flip_diagonal(*square);
                    }
                }
                break;
            }
            idx = if self.unique >= 3 {
                let adjust1 = i32::from(squares[1] > squares[0]);
                let adjust2 =
                    i32::from(squares[2] > squares[0]) + i32::from(squares[2] > squares[1]);
                if off_a1h8(squares[0]) != 0 {
                    (maps.a1d1d4[squares[0] as usize] * 63 + (squares[1] - adjust1)) * 62
                        + squares[2]
                        - adjust2
                } else if off_a1h8(squares[1]) != 0 {
                    (6 * 63 + rank(squares[0]) * 28 + maps.b1h1h7[squares[1] as usize]) * 62
                        + squares[2]
                        - adjust2
                } else if off_a1h8(squares[2]) != 0 {
                    6 * 63 * 62
                        + 4 * 28 * 62
                        + rank(squares[0]) * 7 * 28
                        + (rank(squares[1]) - adjust1) * 28
                        + maps.b1h1h7[squares[2] as usize]
                } else {
                    6 * 63 * 62
                        + 4 * 28 * 62
                        + 4 * 7 * 28
                        + rank(squares[0]) * 7 * 6
                        + (rank(squares[1]) - adjust1) * 6
                        + (rank(squares[2]) - adjust2)
                }
            } else if self.unique == 2 {
                let adjust = i32::from(squares[1] > squares[0]);
                if off_a1h8(squares[0]) != 0 {
                    maps.a1d1d4[squares[0] as usize] * 63 + (squares[1] - adjust)
                } else if off_a1h8(squares[1]) != 0 {
                    6 * 63 + rank(squares[0]) * 28 + maps.b1h1h7[squares[1] as usize]
                } else {
                    6 * 63 + 4 * 28 + rank(squares[0]) * 7 + (rank(squares[1]) - adjust)
                }
            } else if self.min_like_man == 2 {
                if TRIANGLE[squares[0] as usize] > TRIANGLE[squares[1] as usize] {
                    squares.swap(0, 1);
                }
                if file(squares[0]) > 3 {
                    for square in &mut squares[..size] {
                        *square ^= 7;
                    }
                }
                if rank(squares[0]) > 3 {
                    for square in &mut squares[..size] {
                        *square ^= 0o70;
                    }
                }
                if off_a1h8(squares[0]) > 0
                    || (off_a1h8(squares[0]) == 0 && off_a1h8(squares[1]) > 0)
                {
                    for square in &mut squares[..size] {
                        *square = flip_diagonal(*square);
                    }
                }
                if TEST45 & (1 << squares[1]) != 0
                    && TRIANGLE[squares[0] as usize] == TRIANGLE[squares[1] as usize]
                {
                    squares.swap(0, 1);
                    for square in &mut squares[..size] {
                        *square = flip_diagonal(*square ^ 0o70);
                    }
                }
                MAP_PP[TRIANGLE[squares[0] as usize] as usize][squares[1] as usize]
            } else {
                for i in 1..lead_len {
                    if TRIANGLE[squares[0] as usize] > TRIANGLE[squares[i] as usize] {
                        squares.swap(0, i);
                    }
                }
                if file(squares[0]) > 3 {
                    for square in &mut squares[..size] {
                        *square ^= 7;
                    }
                }
                if rank(squares[0]) > 3 {
                    for square in &mut squares[..size] {
                        *square ^= 0o70;
                    }
                }
                if off_a1h8(squares[0]) > 0 {
                    for square in &mut squares[..size] {
                        *square = flip_diagonal(*square);
                    }
                }
                for i in 1..lead_len {
                    for j in i + 1..lead_len {
                        if MULT_TWIST[squares[i] as usize] > MULT_TWIST[squares[j] as usize] {
                            squares.swap(i, j);
                        }
                    }
                }
                let mut idx = maps.mult_idx[lead_len - 1][TRIANGLE[squares[0] as usize] as usize];
                for i in 1..lead_len {
                    idx += maps.binomial[i][MULT_TWIST[squares[i] as usize] as usize];
                }
                idx
            } as u64;
        }

        // The other groups, each by its squares in ascending order, a square
        // counting one lower for each square of an earlier group below it.
        idx *= d.group_idx[0];
        let mut start = d.group_len[0] as usize;
        let mut remaining_pawns = self.has_pawns && self.pawn_count[1] > 0;
        let mut next = 0;
        loop {
            next += 1;
            let len = d.group_len[next] as usize;
            if len == 0 {
                break;
            }
            squares[start..start + len].sort_unstable();
            let mut n: u64 = 0;
            for i in 0..len {
                let square = squares[start + i];
                let adjust = squares[..start].iter().filter(|&&s| square > s).count() as i32;
                let at = square - adjust - if remaining_pawns { 8 } else { 0 };
                n += maps.binomial[i + 1][at as usize] as u64;
            }
            remaining_pawns = false;
            idx += n * d.group_idx[next];
            start += len;
        }

        for f in 0..tb_file as usize {
            idx += self.get(stm, f).size();
        }
        if pos.turn == usize::from(usize::from(black_stronger) == self.lead_color)
            && self.key != self.key2
        {
            idx += self.size();
        }
        idx
    }
}
