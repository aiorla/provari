//! Fairy-Stockfish's Antichess network (`antichess-dd3cbe53cd4e.nnue`, from
//! its list of variant networks), which the search reads to guess which
//! replies are Black's best (`dfpn::antichess`).
//!
//! Its layout is Stockfish 14's for variants: a first layer of 512 values per
//! side from 768 features, one for each kind of piece (six roles, ours or
//! theirs) on each square, as Antichess has no king to anchor them; then one
//! of eight stacks of dense layers (16, 32 and 1 outputs) and a material term
//! from as many buckets, picked by the number of pieces. Each side's first
//! layer is kept along the search's line and updated by the squares a move
//! changes.

use std::path::Path;

use shakmaty::{Board, Color, Piece, Position, Role, Square, variant::Antichess};

const VERSION: u32 = 0x7AF3_2F20;
const FEATURES: usize = 768;
const HALF: usize = 512;
const BUCKETS: usize = 8;
const HIDDEN1: usize = 16;
const HIDDEN2: usize = 32;
/// The second dense layer's inputs, padded to 32 in the file.
const PADDED1: usize = 32;
const WEIGHT_SHIFT: u32 = 6;
const OUTPUT_SCALE: i32 = 16;
/// The hash of the first layer: HalfKAv2's with Fairy-Stockfish's variant
/// features, combined with its output size.
const TRANSFORMER_HASH: u32 = 0x5f23_4cb8 ^ (2 * HALF as u32);

/// One bucket's dense layers.
struct Stack {
    bias1: [i32; HIDDEN1],
    /// By output.
    weights1: Box<[[i8; 2 * HALF]; HIDDEN1]>,
    /// `weights1` by group of four inputs, then output: `[group][output * 4 + k]`.
    grouped1: Box<[[i8; 4 * HIDDEN1]; 2 * HALF / 4]>,
    bias2: [i32; HIDDEN2],
    weights2: [[i8; PADDED1]; HIDDEN2],
    /// `weights2` likewise.
    grouped2: [[i8; 4 * HIDDEN2]; HIDDEN1 / 4],
    bias3: i32,
    weights3: [i8; HIDDEN2],
}

/// For each byte, the positions of its set bits in order.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const SET_BITS: [[u16; 8]; 256] = {
    let mut table = [[0; 8]; 256];
    let mut byte = 0;
    while byte < 256 {
        let (mut bit, mut count) = (0, 0);
        while bit < 8 {
            if byte >> bit & 1 == 1 {
                table[byte][count] = bit as u16;
                count += 1;
            }
            bit += 1;
        }
        byte += 1;
    }
    table
};

pub struct Network {
    bias: Box<[i16; HALF]>,
    /// By feature: `weights[feature * HALF + output]`.
    weights: Vec<i16>,
    /// By feature: `material[feature * BUCKETS + bucket]`.
    material: Vec<i32>,
    stacks: Vec<Stack>,
}

/// The first layer of a position, for White's side and Black's.
#[derive(Clone)]
pub struct Accumulator {
    values: [[i16; HALF]; 2],
    material: [[i32; BUCKETS]; 2],
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let bytes = self
            .data
            .get(self.at..self.at + N)
            .ok_or("the network file ends early")?;
        self.at += N;
        Ok(bytes.try_into().expect("N bytes"))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.bytes()?))
    }

    fn i16(&mut self) -> Result<i16, String> {
        Ok(i16::from_le_bytes(self.bytes()?))
    }

    fn i8(&mut self) -> Result<i8, String> {
        Ok(i8::from_le_bytes(self.bytes()?))
    }
}

/// The hash of a dense layer of `outputs` after a layer hashed `previous`.
fn dense_hash(outputs: u32, previous: u32) -> u32 {
    let hash = 0xCC03_DAE4u32.wrapping_add(outputs);
    hash ^ (previous >> 1) ^ (previous << 31)
}

/// The hash of a layer stack, as each stack's header and the file's carry it.
fn stack_hash() -> u32 {
    let relu = |previous: u32| 0x538D_24C7u32.wrapping_add(previous);
    let input = 0xEC42_E90Du32 ^ (2 * HALF as u32);
    let first = relu(dense_hash(HIDDEN1 as u32, input));
    let second = relu(dense_hash(HIDDEN2 as u32, first));
    dense_hash(1, second)
}

impl Network {
    /// Reads a network file, refusing any other layout.
    pub fn load(path: &Path) -> Result<Network, String> {
        let data = std::fs::read(path).map_err(|err| format!("{}: {err}", path.display()))?;
        Network::parse(&data).map_err(|err| format!("{}: {err}", path.display()))
    }

    fn parse(data: &[u8]) -> Result<Network, String> {
        let mut reader = Reader { data, at: 0 };
        if reader.u32()? != VERSION {
            return Err("not a network of Stockfish 14's format".into());
        }
        if reader.u32()? != TRANSFORMER_HASH ^ stack_hash() {
            return Err("not Fairy-Stockfish's variant network layout".into());
        }
        let description = reader.u32()? as usize;
        reader.at += description;
        if reader.u32()? != TRANSFORMER_HASH {
            return Err("the first layer's hash differs".into());
        }
        let mut bias = Box::new([0; HALF]);
        for value in bias.iter_mut() {
            *value = reader.i16()?;
        }
        let weights = (0..FEATURES * HALF)
            .map(|_| reader.i16())
            .collect::<Result<_, _>>()?;
        let material = (0..FEATURES * BUCKETS)
            .map(|_| reader.i32())
            .collect::<Result<_, _>>()?;
        let mut stacks = Vec::with_capacity(BUCKETS);
        for _ in 0..BUCKETS {
            if reader.u32()? != stack_hash() {
                return Err("a layer stack's hash differs".into());
            }
            let mut stack = Stack {
                bias1: [0; HIDDEN1],
                weights1: Box::new([[0; 2 * HALF]; HIDDEN1]),
                grouped1: Box::new([[0; 4 * HIDDEN1]; 2 * HALF / 4]),
                bias2: [0; HIDDEN2],
                weights2: [[0; PADDED1]; HIDDEN2],
                grouped2: [[0; 4 * HIDDEN2]; HIDDEN1 / 4],
                bias3: 0,
                weights3: [0; HIDDEN2],
            };
            for value in &mut stack.bias1 {
                *value = reader.i32()?;
            }
            for row in stack.weights1.iter_mut() {
                for weight in row.iter_mut() {
                    *weight = reader.i8()?;
                }
            }
            for value in &mut stack.bias2 {
                *value = reader.i32()?;
            }
            for row in &mut stack.weights2 {
                for weight in row.iter_mut() {
                    *weight = reader.i8()?;
                }
            }
            stack.bias3 = reader.i32()?;
            for weight in &mut stack.weights3 {
                *weight = reader.i8()?;
            }
            for (output, row) in stack.weights1.iter().enumerate() {
                for (input, &weight) in row.iter().enumerate() {
                    stack.grouped1[input / 4][output * 4 + input % 4] = weight;
                }
            }
            for (output, row) in stack.weights2.iter().enumerate() {
                for (input, &weight) in row[..HIDDEN1].iter().enumerate() {
                    stack.grouped2[input / 4][output * 4 + input % 4] = weight;
                }
            }
            stacks.push(stack);
        }
        if reader.at != data.len() {
            return Err("the network file goes on past its layers".into());
        }
        Ok(Network {
            bias,
            weights,
            material,
            stacks,
        })
    }

    /// The feature of `piece` on `square` from `side`'s point of view, which
    /// sees the board from its own end.
    fn feature(side: Color, square: Square, piece: Piece) -> usize {
        let kind = match piece.role {
            Role::Pawn => 0,
            Role::Knight => 1,
            Role::Bishop => 2,
            Role::Rook => 3,
            Role::Queen => 4,
            Role::King => 5,
        };
        let theirs = usize::from(piece.color != side);
        let square = match side {
            Color::White => square as usize,
            Color::Black => square as usize ^ 56,
        };
        (2 * kind + theirs) * 64 + square
    }

    /// A position's first layer, computed afresh.
    pub fn refresh(&self, pos: &Antichess) -> Accumulator {
        self.refresh_board(pos.board())
    }

    /// A board's first layer, computed afresh.
    pub fn refresh_board(&self, board: &Board) -> Accumulator {
        let mut acc = Accumulator {
            values: [*self.bias; 2],
            material: [[0; BUCKETS]; 2],
        };
        for (square, piece) in board {
            self.change(&mut acc, square, piece, true);
        }
        acc
    }

    /// The first layer of `child`, a position after a move from `parent`,
    /// from `parent`'s: the squares whose pieces the move changed.
    pub fn update(&self, acc: &Accumulator, parent: &Antichess, child: &Antichess) -> Accumulator {
        self.update_board(acc, parent.board(), child.board())
    }

    /// `update` from the boards alone.
    pub fn update_board(&self, acc: &Accumulator, parent: &Board, child: &Board) -> Accumulator {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // Safety: the processor has AVX2.
            return unsafe { self.update_avx2(acc, parent, child) };
        }
        self.update_generic(acc, parent, child)
    }

    /// `update` compiled for AVX2, which adds sixteen values at a time.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn update_avx2(&self, acc: &Accumulator, parent: &Board, child: &Board) -> Accumulator {
        self.update_generic(acc, parent, child)
    }

    /// In one pass over each side's values: the parent's, plus the row of
    /// the piece the move put on a square, minus those of the pieces it
    /// took off.
    #[inline(always)]
    fn update_generic(&self, acc: &Accumulator, parent: &Board, child: &Board) -> Accumulator {
        let Some((on, off, also_off)) = moved(parent, child) else {
            return self.refresh_board(child);
        };
        let mut next = Accumulator {
            values: [[0; HALF]; 2],
            material: acc.material,
        };
        for (side, color) in Color::ALL.into_iter().enumerate() {
            let feature = |(square, piece)| Network::feature(color, square, piece);
            let (add, sub, also) = (feature(on), feature(off), also_off.map(feature));
            let row = |feature: usize| &self.weights[feature * HALF..(feature + 1) * HALF];
            let values = next.values[side].iter_mut().zip(&acc.values[side]);
            let terms = values.zip(row(add)).zip(row(sub));
            match also {
                None => terms.for_each(|(((value, &base), &add), &sub)| {
                    *value = base.wrapping_add(add).wrapping_sub(sub);
                }),
                Some(also) => {
                    terms
                        .zip(row(also))
                        .for_each(|((((value, &base), &add), &sub), &also)| {
                            *value = base.wrapping_add(add).wrapping_sub(sub).wrapping_sub(also);
                        })
                }
            }
            let material =
                |feature: usize| &self.material[feature * BUCKETS..(feature + 1) * BUCKETS];
            for (bucket, term) in next.material[side].iter_mut().enumerate() {
                *term += material(add)[bucket] - material(sub)[bucket];
                *term -= also.map_or(0, |also| material(also)[bucket]);
            }
        }
        next
    }

    /// Adds or removes a piece's rows on both sides.
    #[inline(always)]
    fn change(&self, acc: &mut Accumulator, square: Square, piece: Piece, add: bool) {
        for (side, color) in Color::ALL.into_iter().enumerate() {
            let feature = Network::feature(color, square, piece);
            let rows = &self.weights[feature * HALF..(feature + 1) * HALF];
            let material = &self.material[feature * BUCKETS..(feature + 1) * BUCKETS];
            let values = acc.values[side].iter_mut().zip(rows);
            let terms = acc.material[side].iter_mut().zip(material);
            if add {
                values.for_each(|(value, &weight)| *value = value.wrapping_add(weight));
                terms.for_each(|(term, &weight)| *term += weight);
            } else {
                values.for_each(|(value, &weight)| *value = value.wrapping_sub(weight));
                terms.for_each(|(term, &weight)| *term -= weight);
            }
        }
    }

    /// The evaluation for the side to move, from its first layer, in
    /// Fairy-Stockfish's units (an endgame pawn is about 200).
    pub fn evaluate(&self, acc: &Accumulator, pos: &Antichess) -> i32 {
        let pieces = pos.board().occupied().count();
        let bucket = ((pieces.max(1) - 1) / 4).min(BUCKETS - 1);
        let (us, them) = match pos.turn() {
            Color::White => (0, 1),
            Color::Black => (1, 0),
        };
        let material = (acc.material[us][bucket] - acc.material[them][bucket]) / 2;
        let halves = [&acc.values[us], &acc.values[them]];
        let stack = &self.stacks[bucket];
        let positional = stack
            .positional_vector(halves)
            .unwrap_or_else(|| stack.positional(halves));
        (material + positional) / OUTPUT_SCALE
    }
}

/// The piece a move put on a square and the one or two it took off: the
/// mover and any capture. `None` for any other change between positions.
fn moved(before: &Board, after: &Board) -> Option<Moved> {
    let (mut on, mut off, mut also_off) = (None, None, None);
    for color in Color::ALL {
        if before.by_color(color) == after.by_color(color) {
            continue;
        }
        for role in Role::ALL {
            let piece = Piece { color, role };
            let (was, is) = (before.by_piece(piece), after.by_piece(piece));
            for square in is & !was {
                if on.replace((square, piece)).is_some() {
                    return None;
                }
            }
            for square in was & !is {
                let slot = if off.is_none() {
                    &mut off
                } else {
                    &mut also_off
                };
                if slot.replace((square, piece)).is_some() {
                    return None;
                }
            }
        }
    }
    Some((on?, off?, also_off))
}

type Moved = ((Square, Piece), (Square, Piece), Option<(Square, Piece)>);

impl Stack {
    /// The dense layers' output from the first layer's halves, the side to
    /// move's first.
    fn positional(&self, halves: [&[i16; HALF]; 2]) -> i32 {
        let mut input = [0u8; 2 * HALF];
        for (clipped, values) in input.as_chunks_mut::<HALF>().0.iter_mut().zip(halves) {
            for (x, &value) in clipped.iter_mut().zip(values) {
                *x = value.clamp(0, 127) as u8;
            }
        }
        let first: [i32; HIDDEN1] = std::array::from_fn(|output| {
            let row = self.weights1[output].iter().zip(&input);
            let sum: i32 = row.map(|(&w, &x)| i32::from(w) * i32::from(x)).sum();
            ((self.bias1[output] + sum) >> WEIGHT_SHIFT).clamp(0, 127)
        });
        let second: [i32; HIDDEN2] = std::array::from_fn(|output| {
            let row = self.weights2[output].iter().zip(&first);
            let sum: i32 = row.map(|(&w, &x)| i32::from(w) * x).sum();
            ((self.bias2[output] + sum) >> WEIGHT_SHIFT).clamp(0, 127)
        });
        let row = self.weights3.iter().zip(&second);
        self.bias3 + row.map(|(&w, &x)| i32::from(w) * x).sum::<i32>()
    }

    /// `positional` with the processor's vector instructions, if it has them:
    /// AVX2 on x86-64, the dot-product instructions on ARM, as Apple's
    /// processors have.
    #[inline(always)]
    fn positional_vector(&self, halves: [&[i16; HALF]; 2]) -> Option<i32> {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // Safety: the processor has AVX2.
            return Some(unsafe { self.positional_avx2(halves) });
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // Safety: the processor has the dot-product instructions.
            return Some(unsafe { self.positional_dotprod(halves) });
        }
        let _ = halves;
        None
    }

    /// `positional` with AVX2. The first layer is clipped thirty-two values
    /// at a time, and only its groups of four that are not all zero, about
    /// two in five, are multiplied, each by its weights for all sixteen
    /// outputs at once; the second layer likewise. The inputs are at most
    /// 127, so a pair of products fits sixteen bits and every sum is exact.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn positional_avx2(&self, halves: [&[i16; HALF]; 2]) -> i32 {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero_si256();
        let ones = _mm256_set1_epi16(1);
        let mut input = [0u8; 2 * HALF];
        let mut groups = [0u16; 2 * HALF / 4];
        let mut count = 0;
        for (half, values) in halves.into_iter().enumerate() {
            for block in 0..HALF / 32 {
                let at = half * HALF + block * 32;
                // Safety: the block lies within the arrays, and `count` is at
                // most the number of groups before it, so its eight fit.
                unsafe {
                    let low = _mm256_loadu_si256(values.as_ptr().add(block * 32).cast());
                    let high = _mm256_loadu_si256(values.as_ptr().add(block * 32 + 16).cast());
                    // Saturated to bytes, clipped at zero, and put back in
                    // order across the lanes that packing interleaves.
                    let packed = _mm256_max_epi8(_mm256_packs_epi16(low, high), zero);
                    let clipped = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);
                    _mm256_storeu_si256(input.as_mut_ptr().add(at).cast(), clipped);
                    let zeros = _mm256_castsi256_ps(_mm256_cmpeq_epi32(clipped, zero));
                    let nonzero = !_mm256_movemask_ps(zeros) as u8;
                    let bits = _mm_loadu_si128(SET_BITS[nonzero as usize].as_ptr().cast());
                    let indices = _mm_add_epi16(bits, _mm_set1_epi16((at / 4) as i16));
                    _mm_storeu_si128(groups.as_mut_ptr().add(count).cast(), indices);
                    count += nonzero.count_ones() as usize;
                }
            }
        }
        let mut sums = [zero; HIDDEN1 / 8];
        for &group in &groups[..count] {
            let group = usize::from(group);
            // Safety: the group's inputs and weights lie within the arrays.
            unsafe {
                let four = input.as_ptr().add(group * 4).cast::<i32>().read_unaligned();
                let x = _mm256_set1_epi32(four);
                let weights = self.grouped1[group].as_ptr();
                for (eight, sum) in sums.iter_mut().enumerate() {
                    let w = _mm256_loadu_si256(weights.add(eight * 32).cast());
                    let products = _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), ones);
                    *sum = _mm256_add_epi32(*sum, products);
                }
            }
        }
        let top = _mm256_set1_epi32(127);
        let layer = |sums: &mut [__m256i], bias: &[i32]| {
            for (eight, sum) in sums.iter_mut().enumerate() {
                // Safety: the bias has eight values for each sum.
                let bias = unsafe { _mm256_loadu_si256(bias.as_ptr().add(eight * 8).cast()) };
                let shifted =
                    _mm256_srai_epi32::<{ WEIGHT_SHIFT as i32 }>(_mm256_add_epi32(*sum, bias));
                *sum = _mm256_min_epi32(_mm256_max_epi32(shifted, zero), top);
            }
        };
        layer(&mut sums, &self.bias1);
        // The sixteen outputs as bytes: groups of four at 32-bit elements 0,
        // 4, 1 and 5.
        let words = _mm256_packs_epi32(sums[0], sums[1]);
        let first = _mm256_packs_epi16(words, words);
        let mut second = [zero; HIDDEN2 / 8];
        for (group, element) in [0, 4, 1, 5].into_iter().enumerate() {
            let x = _mm256_permutevar8x32_epi32(first, _mm256_set1_epi32(element));
            let weights = self.grouped2[group].as_ptr();
            for (eight, sum) in second.iter_mut().enumerate() {
                // Safety: the group has eight outputs' weights for each sum.
                let w = unsafe { _mm256_loadu_si256(weights.add(eight * 32).cast()) };
                let products = _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), ones);
                *sum = _mm256_add_epi32(*sum, products);
            }
        }
        layer(&mut second, &self.bias2);
        let mut total = zero;
        for (eight, values) in second.into_iter().enumerate() {
            // Safety: there are eight weights for each eight values.
            let weights = unsafe { _mm_loadl_epi64(self.weights3.as_ptr().add(eight * 8).cast()) };
            let products = _mm256_mullo_epi32(values, _mm256_cvtepi8_epi32(weights));
            total = _mm256_add_epi32(total, products);
        }
        let quad = _mm_add_epi32(
            _mm256_castsi256_si128(total),
            _mm256_extracti128_si256::<1>(total),
        );
        let pair = _mm_add_epi32(quad, _mm_shuffle_epi32::<0b01_00_11_10>(quad));
        let one = _mm_add_epi32(pair, _mm_shuffle_epi32::<0b10_11_00_01>(pair));
        self.bias3 + _mm_cvtsi128_si32(one)
    }

    /// `positional` with NEON's dot product, as `positional_avx2`: the first
    /// layer is clipped sixteen values at a time, and only its groups of four
    /// that are not all zero are multiplied, each by its weights for all
    /// sixteen outputs, four outputs to an instruction; the second and third
    /// layers likewise. The inputs are at most 127, so they read as signed
    /// bytes, and every sum is exact.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "dotprod")]
    unsafe fn positional_dotprod(&self, halves: [&[i16; HALF]; 2]) -> i32 {
        use std::arch::aarch64::*;
        let mut input = [0u8; 2 * HALF];
        let mut groups = [0u16; 2 * HALF / 4];
        let mut count = 0;
        let ceiling = vdupq_n_u8(127);
        let (low_bits, high_bits) = (
            // Safety: each array holds four values.
            unsafe { vld1q_u32([1, 2, 4, 8].as_ptr()) },
            unsafe { vld1q_u32([16, 32, 64, 128].as_ptr()) },
        );
        for (half, values) in halves.into_iter().enumerate() {
            for block in 0..HALF / 32 {
                let at = half * HALF + block * 32;
                // Safety: the block lies within the arrays, and `count` is at
                // most the number of groups before it, so its eight fit.
                unsafe {
                    let clip = |from: usize| {
                        let low = vqmovun_s16(vld1q_s16(values.as_ptr().add(from)));
                        let high = vqmovun_s16(vld1q_s16(values.as_ptr().add(from + 8)));
                        vminq_u8(vcombine_u8(low, high), ceiling)
                    };
                    let (first, second) = (clip(block * 32), clip(block * 32 + 16));
                    vst1q_u8(input.as_mut_ptr().add(at), first);
                    vst1q_u8(input.as_mut_ptr().add(at + 16), second);
                    let (first, second) =
                        (vreinterpretq_u32_u8(first), vreinterpretq_u32_u8(second));
                    let nonzero = vaddvq_u32(vorrq_u32(
                        vandq_u32(vtstq_u32(first, first), low_bits),
                        vandq_u32(vtstq_u32(second, second), high_bits),
                    )) as usize;
                    let bits = vld1q_u16(SET_BITS[nonzero].as_ptr());
                    let indices = vaddq_u16(bits, vdupq_n_u16((at / 4) as u16));
                    vst1q_u16(groups.as_mut_ptr().add(count), indices);
                    count += nonzero.count_ones() as usize;
                }
            }
        }
        let mut sums = [vdupq_n_s32(0); HIDDEN1 / 4];
        for &group in &groups[..count] {
            let group = usize::from(group);
            // Safety: the group's inputs and weights lie within the arrays.
            unsafe {
                let four = input.as_ptr().add(group * 4).cast::<u32>().read_unaligned();
                let x = vreinterpretq_s8_u32(vdupq_n_u32(four));
                let weights = self.grouped1[group].as_ptr();
                for (quarter, sum) in sums.iter_mut().enumerate() {
                    *sum = vdotq_s32(*sum, vld1q_s8(weights.add(quarter * 16)), x);
                }
            }
        }
        let (zero, top) = (vdupq_n_s32(0), vdupq_n_s32(127));
        let layer = |sums: &mut [int32x4_t], bias: &[i32]| {
            for (quarter, sum) in sums.iter_mut().enumerate() {
                // Safety: the bias has four values for each sum.
                let bias = unsafe { vld1q_s32(bias.as_ptr().add(quarter * 4)) };
                let shifted = vshrq_n_s32::<{ WEIGHT_SHIFT as i32 }>(vaddq_s32(*sum, bias));
                *sum = vminq_s32(vmaxq_s32(shifted, zero), top);
            }
        };
        // Values from 0 to 127 narrow to bytes exactly.
        let bytes = |sums: &[int32x4_t]| {
            let low = vcombine_s16(vmovn_s32(sums[0]), vmovn_s32(sums[1]));
            let high = vcombine_s16(vmovn_s32(sums[2]), vmovn_s32(sums[3]));
            vcombine_s8(vmovn_s16(low), vmovn_s16(high))
        };
        layer(&mut sums, &self.bias1);
        let first = vreinterpretq_s32_s8(bytes(&sums));
        let mut second = [vdupq_n_s32(0); HIDDEN2 / 4];
        let fours = [
            vdupq_laneq_s32::<0>(first),
            vdupq_laneq_s32::<1>(first),
            vdupq_laneq_s32::<2>(first),
            vdupq_laneq_s32::<3>(first),
        ];
        for (group, four) in fours.into_iter().enumerate() {
            let x = vreinterpretq_s8_s32(four);
            let weights = self.grouped2[group].as_ptr();
            for (quarter, sum) in second.iter_mut().enumerate() {
                // Safety: the group has four outputs' weights for each sum.
                let w = unsafe { vld1q_s8(weights.add(quarter * 16)) };
                *sum = vdotq_s32(*sum, w, x);
            }
        }
        layer(&mut second, &self.bias2);
        // Safety: there are sixteen weights for each sixteen values.
        let total = unsafe {
            let low = vdotq_s32(zero, vld1q_s8(self.weights3.as_ptr()), bytes(&second[..4]));
            vdotq_s32(
                low,
                vld1q_s8(self.weights3.as_ptr().add(16)),
                bytes(&second[4..]),
            )
        };
        self.bias3 + vaddvq_s32(total)
    }
}

/// Networks for tests.
#[doc(hidden)]
pub mod testing {
    use super::*;

    /// A network of pseudo-random weights.
    pub fn random(seed: u64) -> Network {
        Network::parse(&random_network(seed)).unwrap()
    }

    /// A network file of the right layout with pseudo-random weights.
    pub fn random_network(seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut data = Vec::new();
        let description = b"random";
        data.extend(VERSION.to_le_bytes());
        data.extend((TRANSFORMER_HASH ^ stack_hash()).to_le_bytes());
        data.extend((description.len() as u32).to_le_bytes());
        data.extend(description);
        data.extend(TRANSFORMER_HASH.to_le_bytes());
        for _ in 0..HALF {
            data.extend((next() as i16 % 64).to_le_bytes());
        }
        for _ in 0..FEATURES * HALF {
            data.extend((next() as i16 % 48).to_le_bytes());
        }
        for _ in 0..FEATURES * BUCKETS {
            data.extend((next() as i32 % 4000).to_le_bytes());
        }
        for _ in 0..BUCKETS {
            data.extend(stack_hash().to_le_bytes());
            for _ in 0..HIDDEN1 {
                data.extend((next() as i32 % 3000).to_le_bytes());
            }
            for _ in 0..HIDDEN1 * 2 * HALF {
                data.push(next() as u8);
            }
            for _ in 0..HIDDEN2 {
                data.extend((next() as i32 % 3000).to_le_bytes());
            }
            for _ in 0..HIDDEN2 * PADDED1 {
                data.push(next() as u8);
            }
            data.extend((next() as i32 % 3000).to_le_bytes());
            for _ in 0..HIDDEN2 {
                data.push(next() as u8);
            }
        }
        data
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{random, random_network};
    use super::*;

    /// The vector layers, where the processor has them, are the plain ones,
    /// to the bit, on first layers that are mostly clipped, as the search's
    /// are, with the extremes.
    #[test]
    fn vector_layers_match() {
        let mut state = 9u64;
        for seed in 1..5 {
            let network = random(seed);
            for test in 0..200 {
                let mut values = [[0i16; HALF]; 2];
                // Of 32 values, how many are drawn from the whole range.
                let drawn = [0, 1, 4, 12, 32][test % 5];
                for value in values.iter_mut().flatten() {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let any = (state >> 16) as i16;
                    *value = match ((state >> 3) % 32 < drawn, (state >> 8) % 5) {
                        (false, _) => -(any & 0x7f),
                        (true, 0) => i16::MAX,
                        (true, 1) => i16::MIN,
                        (true, 2) => 127,
                        (true, 3) => any % 300,
                        (true, _) => any % 128,
                    };
                }
                for stack in &network.stacks {
                    let halves = [&values[0], &values[1]];
                    if let Some(vector) = stack.positional_vector(halves) {
                        assert_eq!(vector, stack.positional(halves));
                    }
                }
            }
        }
    }

    #[test]
    fn refuses_files_of_another_layout() {
        let data = random_network(1);
        assert!(Network::parse(&data).is_ok());
        assert!(Network::parse(&data[..data.len() - 1]).is_err());
        let mut longer = data.clone();
        longer.push(0);
        assert!(Network::parse(&longer).is_err());
        let mut other = data;
        other[4] ^= 1;
        assert!(Network::parse(&other).is_err());
    }

    /// Along random games the updated first layer is the one computed
    /// afresh, and a position seen from the other side, with the colours
    /// swapped and the board turned, evaluates the same.
    #[test]
    fn updates_match_refreshes_and_colours_are_symmetric() {
        let network = Network::parse(&random_network(7)).unwrap();
        let mut seed = 3u64;
        let mut random = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as usize % n
        };
        let mut evaluated = 0;
        for _ in 0..50 {
            let mut pos = Antichess::default();
            let mut acc = network.refresh(&pos);
            loop {
                let fresh = network.refresh(&pos);
                assert_eq!(fresh.values, acc.values);
                assert_eq!(fresh.material, acc.material);
                let mirrored = crate::antichess::rules::parse_fen(&mirror(
                    &crate::antichess::rules::fen(&pos),
                ))
                .unwrap();
                assert_eq!(
                    network.evaluate(&acc, &pos),
                    network.evaluate(&network.refresh(&mirrored), &mirrored)
                );
                evaluated += 1;
                let legal = pos.legal_moves();
                if legal.is_empty() || pos.us().is_empty() || evaluated % 97 == 0 {
                    break;
                }
                let mut child = pos.clone();
                child.play_unchecked(legal[random(legal.len())]);
                acc = network.update(&acc, &pos, &child);
                pos = child;
            }
        }
        assert!(evaluated > 1000, "{evaluated}");
    }

    /// The network prefers 1.e3, White's winning first move, to 1.d3, 1.d4
    /// and 1.e4, which lose: Black's evaluations after each. It reads
    /// Fairy-Stockfish's Antichess network, from `PROVARI_ANTICHESS_NETWORK`.
    #[test]
    #[ignore = "needs Fairy-Stockfish's network"]
    fn fairy_stockfish_network_prefers_e3() {
        let path = std::env::var_os("PROVARI_ANTICHESS_NETWORK")
            .map(std::path::PathBuf::from)
            .expect("PROVARI_ANTICHESS_NETWORK names Fairy-Stockfish's Antichess network");
        let network = Network::load(&path).unwrap();
        let evaluate = |fen: &str| {
            let pos = crate::antichess::rules::parse_fen(fen).unwrap();
            network.evaluate(&network.refresh(&pos), &pos)
        };
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1";
        assert_eq!(evaluate(start), 5);
        let after = |board: &str| evaluate(&format!("{board} b - - 0 1"));
        assert_eq!(after("rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR"), 45);
        assert_eq!(after("rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR"), 125);
        assert_eq!(after("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR"), 564);
        assert_eq!(after("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"), 593);
    }

    /// The FEN of a position with the colours swapped and the board turned.
    fn mirror(fen: &str) -> String {
        let fields: Vec<&str> = fen.split(' ').collect();
        let swap = |c: char| {
            if c.is_ascii_uppercase() {
                c.to_ascii_lowercase()
            } else {
                c.to_ascii_uppercase()
            }
        };
        let board: Vec<String> = fields[0]
            .split('/')
            .rev()
            .map(|rank| rank.chars().map(swap).collect())
            .collect();
        let turn = if fields[1] == "w" { "b" } else { "w" };
        let ep = match fields[3] {
            "-" => "-".to_string(),
            square => {
                let rank = if square.ends_with('3') { '6' } else { '3' };
                format!("{}{rank}", &square[..1])
            }
        };
        format!(
            "{} {turn} - {ep} {} {}",
            board.join("/"),
            fields[4],
            fields[5]
        )
    }
}
