//! Atomic-Stockfish's network, evaluated exactly as the engine does: the
//! legacy Atomic V1 format, with HalfKAv2Atomic inputs (each piece by its
//! square and the perspective's king square, kings sharing one plane), 512
//! values per side, then 1024 → 16 → 32 → 1 in eight layer stacks chosen by
//! the number of pieces. The network is a local input, as the tablebases are.
//!
//! A translation into Rust, modified for provari in 2026, of the evaluation
//! of Atomic-Stockfish (<https://github.com/Belzedar94/Atomic-Stockfish>),
//! which derives from Stockfish and Fairy-Stockfish: Stockfish, a UCI chess
//! playing engine derived from Glaurung 2.1, Copyright (C) 2004-2026 The
//! Stockfish developers (see Atomic-Stockfish's AUTHORS file), under the GNU
//! General Public License, version 3 or later.

use std::fs;
use std::path::Path;

use shakmaty::{Color, Role, Square};

use crate::board::Position;

const VERSION: u32 = 0x7AF3_2F20;
const HALF: usize = 512;
const INPUTS: usize = 64 * 11 * 64;
const BUCKETS: usize = 8;
/// The most input rows an update applies in one pass: a capture that blows
/// up every piece beside it changes ten squares.
const MANY: usize = 24;
const FC0: usize = 16;
const FC1: usize = 32;
/// The first layer in chunks of four outputs, the unit the dense layers skip
/// when it is all zero.
const CHUNKS: usize = 2 * HALF / 4;
/// The transformer's hash, and the dense layers'.
const TRANSFORMER_HASH: u32 = 0x5F23_48B8;
const OUTPUT_SCALE: i64 = 16;
const WEIGHT_SCALE_BITS: u32 = 6;

pub struct Network {
    biases: Box<[i16; HALF]>,
    weights: Vec<i16>,
    psqt: Vec<i32>,
    stacks: Vec<Stack>,
}

struct Stack {
    b0: [i32; FC0],
    /// The first dense layer's weights by chunk of the first layer: for each
    /// four inputs, the four weights of every output in turn.
    w0: Vec<i8>,
    b1: [i32; FC1],
    /// Rows padded to 32 inputs, of which the first 16 are used.
    w1: Vec<i8>,
    b2: i32,
    w2: [i8; FC1],
}

/// The hash the dense layers' architecture writes into the file.
const fn stack_hash() -> u32 {
    const fn affine(previous: u32, outputs: u32) -> u32 {
        (0xCC03_DAE4u32.wrapping_add(outputs) ^ (previous >> 1)) ^ (previous << 31)
    }
    const fn relu(previous: u32) -> u32 {
        0x538D_24C7u32.wrapping_add(previous)
    }
    let hash = 0xEC42_E90Du32 ^ (2 * HALF as u32);
    let hash = relu(affine(hash, FC0 as u32));
    let hash = relu(affine(hash, FC1 as u32));
    affine(hash, 1)
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let (head, rest) = self
            .bytes
            .split_first_chunk::<N>()
            .ok_or("the network file ends early")?;
        self.bytes = rest;
        Ok(*head)
    }

    fn u32(&mut self) -> Result<u32, String> {
        self.take().map(u32::from_le_bytes)
    }

    fn i32(&mut self) -> Result<i32, String> {
        self.take().map(i32::from_le_bytes)
    }

    fn i8(&mut self) -> Result<i8, String> {
        self.take().map(i8::from_le_bytes)
    }

    /// `count` values of `N` bytes each, read at once.
    fn many<const N: usize, T>(
        &mut self,
        count: usize,
        read: fn([u8; N]) -> T,
    ) -> Result<Vec<T>, String> {
        let (head, rest) = self
            .bytes
            .split_at_checked(count * N)
            .ok_or("the network file ends early")?;
        self.bytes = rest;
        let (values, _) = head.as_chunks::<N>();
        Ok(values.iter().map(|&bytes| read(bytes)).collect())
    }
}

impl Network {
    pub fn load(path: &Path) -> Result<Network, String> {
        let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
        Network::from_bytes(&bytes)
    }

    /// A network from its file's bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Network, String> {
        let mut file = Reader { bytes };
        let version = file.u32()?;
        let hash = file.u32()?;
        let description = file.u32()? as usize;
        if version != VERSION || hash != TRANSFORMER_HASH ^ stack_hash() {
            return Err("not a legacy Atomic V1 network".into());
        }
        file.bytes = file
            .bytes
            .get(description..)
            .ok_or("the network file ends early")?;
        if file.u32()? != TRANSFORMER_HASH {
            return Err("an unexpected feature transformer".into());
        }
        let biases = file
            .many(HALF, i16::from_le_bytes)?
            .into_boxed_slice()
            .try_into()
            .expect("512 biases");
        let weights = file.many(INPUTS * HALF, i16::from_le_bytes)?;
        let psqt = file.many(INPUTS * BUCKETS, i32::from_le_bytes)?;
        let mut stacks = Vec::with_capacity(BUCKETS);
        for _ in 0..BUCKETS {
            if file.u32()? != stack_hash() {
                return Err("unexpected dense layers".into());
            }
            let mut b0 = [0; FC0];
            for bias in &mut b0 {
                *bias = file.i32()?;
            }
            // The file has each output's 1024 weights in a row.
            let rows = file.many(FC0 * 2 * HALF, i8::from_le_bytes)?;
            let mut w0 = vec![0; FC0 * 2 * HALF];
            for (index, &weight) in rows.iter().enumerate() {
                let (output, input) = (index / (2 * HALF), index % (2 * HALF));
                w0[(input / 4 * FC0 + output) * 4 + input % 4] = weight;
            }
            let mut b1 = [0; FC1];
            for bias in &mut b1 {
                *bias = file.i32()?;
            }
            let w1 = file.many(FC1 * 32, i8::from_le_bytes)?;
            let b2 = file.i32()?;
            let mut w2 = [0; FC1];
            for weight in &mut w2 {
                *weight = file.i8()?;
            }
            stacks.push(Stack {
                b0,
                w0,
                b1,
                w1,
                b2,
                w2,
            });
        }
        if !file.bytes.is_empty() {
            return Err("the network file has trailing bytes".into());
        }
        Ok(Network {
            biases,
            weights,
            psqt,
            stacks,
        })
    }

    /// The evaluation for the side to move, in the engine's internal units
    /// (208 to a pawn), computed from scratch.
    pub fn evaluate(&self, pos: &Position) -> i32 {
        let mut accumulator = Accumulator::default();
        self.refresh(pos, &mut accumulator);
        self.output(&accumulator, pos)
    }

    /// The first layer for the side to move, computed from scratch.
    pub fn evaluate_layer(&self, pos: &Position) -> [u8; 2 * HALF] {
        let mut accumulator = Accumulator::default();
        self.refresh(pos, &mut accumulator);
        let mut transformed = Transformed::default();
        self.transform(&accumulator, pos, &mut transformed);
        transformed.values
    }

    /// Computes a position's first layer from scratch.
    pub fn refresh(&self, pos: &Position, accumulator: &mut Accumulator) {
        accumulator.squares = *pos.squares();
        for perspective in Color::ALL {
            accumulator.kings[perspective as usize] = pos.king(perspective);
            self.refresh_side(perspective, accumulator);
        }
    }

    /// Computes a position's first layer from its parent's: only the squares
    /// that changed, unless a side's own king moved, which starts that side
    /// from the cache's entry for the king's new square.
    pub fn update(
        &self,
        parent: &Accumulator,
        pos: &Position,
        accumulator: &mut Accumulator,
        cache: &mut RefreshCache,
    ) {
        let before = &parent.squares;
        let after = pos.squares();
        accumulator.squares = *after;
        // The squares that changed, eight at a time.
        let mut changed = [0usize; 64];
        let mut count = 0;
        let (before_chunks, _) = before.as_chunks::<8>();
        let (after_chunks, _) = after.as_chunks::<8>();
        for (chunk, (old, new)) in before_chunks.iter().zip(after_chunks).enumerate() {
            if old != new {
                for offset in 0..8 {
                    if old[offset] != new[offset] {
                        changed[count] = chunk * 8 + offset;
                        count += 1;
                    }
                }
            }
        }
        for perspective in Color::ALL {
            let side = perspective as usize;
            let king = pos.king(perspective);
            accumulator.kings[side] = king;
            if king.is_none() || king != parent.kings[side] {
                self.refresh_cached(perspective, accumulator, cache);
                continue;
            }
            let king = king.map_or(0, |king| orient(perspective, king));
            // Two rows, as a quiet move or a capture that blows up nothing
            // else changes, in one pass from the parent's values.
            let rows = changed[..count]
                .iter()
                .map(|&square| usize::from(before[square] != 0) + usize::from(after[square] != 0))
                .sum::<usize>();
            if (2..=MANY).contains(&rows) {
                let mut listed = [(0, false); MANY];
                let mut count_listed = 0;
                for &square in &changed[..count] {
                    let (old, new) = (before[square], after[square]);
                    let square = Square::new(square as u32);
                    for (code, add) in [(old, false), (new, true)] {
                        if code != 0 {
                            listed[count_listed] = (feature(perspective, square, code, king), add);
                            count_listed += 1;
                        }
                    }
                }
                if rows == 2 {
                    self.apply_two_from(parent, accumulator, side, [listed[0], listed[1]]);
                } else {
                    self.apply_rows_from(parent, accumulator, side, &listed[..rows]);
                }
                continue;
            }
            // The first change starts from the parent's values, which spares
            // copying them.
            let mut from = Some(parent);
            for &square in &changed[..count] {
                let (old, new) = (before[square], after[square]);
                let square = Square::new(square as u32);
                for (code, add) in [(old, false), (new, true)] {
                    if code != 0 {
                        let index = feature(perspective, square, code, king);
                        match from.take() {
                            Some(parent) => self.apply_from(parent, accumulator, side, index, add),
                            None => self.apply(accumulator, side, index, add),
                        }
                    }
                }
            }
            if from.is_some() {
                accumulator.values[side] = parent.values[side];
                accumulator.psqt[side] = parent.psqt[side];
            }
        }
    }

    /// A quiet move's first layer from its parent's, for a piece other than
    /// a king: for each side one input row taken away and one added, in one
    /// pass. The same values as `update`, without building the position.
    pub fn update_quiet(
        &self,
        parent: &Accumulator,
        from: Square,
        to: Square,
        accumulator: &mut Accumulator,
    ) {
        let code = parent.squares[from as usize];
        debug_assert!(code != 0 && usize::from(code & 7) != Role::King as usize);
        accumulator.squares = parent.squares;
        accumulator.squares[from as usize] = 0;
        accumulator.squares[to as usize] = code;
        accumulator.kings = parent.kings;
        for perspective in Color::ALL {
            let side = perspective as usize;
            let king = parent.kings[side].map_or(0, |king| orient(perspective, king));
            let old = feature(perspective, from, code, king);
            let new = feature(perspective, to, code, king);
            let taken = &self.weights[old * HALF..(old + 1) * HALF];
            let added = &self.weights[new * HALF..(new + 1) * HALF];
            for (((value, start), taken), added) in accumulator.values[side]
                .iter_mut()
                .zip(&parent.values[side])
                .zip(taken)
                .zip(added)
            {
                *value = start.wrapping_sub(*taken).wrapping_add(*added);
            }
            let taken = &self.psqt[old * BUCKETS..(old + 1) * BUCKETS];
            let added = &self.psqt[new * BUCKETS..(new + 1) * BUCKETS];
            for (((value, start), taken), added) in accumulator.psqt[side]
                .iter_mut()
                .zip(&parent.psqt[side])
                .zip(taken)
                .zip(added)
            {
                *value = start - taken + added;
            }
        }
    }

    /// A side's first layer after its own king moved: the cache's entry for
    /// the king's square, brought up to date by the squares that differ.
    fn refresh_cached(
        &self,
        perspective: Color,
        accumulator: &mut Accumulator,
        cache: &mut RefreshCache,
    ) {
        let side = perspective as usize;
        let Some(king) = accumulator.kings[side] else {
            self.refresh_side(perspective, accumulator);
            return;
        };
        let king = orient(perspective, king);
        let entry = &mut cache.entries[side * 64 + king];
        let (old_chunks, _) = entry.squares.as_chunks::<8>();
        let (new_chunks, _) = accumulator.squares.as_chunks::<8>();
        for (chunk, (old, new)) in old_chunks.iter().zip(new_chunks).enumerate() {
            if old == new {
                continue;
            }
            for offset in 0..8 {
                let (old, new) = (old[offset], new[offset]);
                if old == new {
                    continue;
                }
                let square = Square::new((chunk * 8 + offset) as u32);
                for (code, add) in [(old, false), (new, true)] {
                    if code != 0 {
                        let index = feature(perspective, square, code, king);
                        self.apply_to(&mut entry.values, &mut entry.psqt, index, add);
                    }
                }
            }
        }
        entry.squares = accumulator.squares;
        accumulator.values[side] = entry.values;
        accumulator.psqt[side] = entry.psqt;
    }

    fn refresh_side(&self, perspective: Color, accumulator: &mut Accumulator) {
        let side = perspective as usize;
        let squares = accumulator.squares;
        let king = accumulator.kings[side].map_or(0, |king| orient(perspective, king));
        accumulator.values[side] = *self.biases;
        accumulator.psqt[side] = [0; BUCKETS];
        for (square, &code) in squares.iter().enumerate() {
            if code != 0 {
                let index = feature(perspective, Square::new(square as u32), code, king);
                self.apply(accumulator, side, index, true);
            }
        }
    }

    /// The side's values and material: the parent's with an input's row added
    /// or taken away.
    fn apply_from(
        &self,
        parent: &Accumulator,
        accumulator: &mut Accumulator,
        side: usize,
        index: usize,
        add: bool,
    ) {
        let row = &self.weights[index * HALF..(index + 1) * HALF];
        let psqt = &self.psqt[index * BUCKETS..(index + 1) * BUCKETS];
        let values = accumulator.values[side]
            .iter_mut()
            .zip(&parent.values[side])
            .zip(row);
        let materials = accumulator.psqt[side]
            .iter_mut()
            .zip(&parent.psqt[side])
            .zip(psqt);
        if add {
            for ((value, start), weight) in values {
                *value = start.wrapping_add(*weight);
            }
            for ((value, start), weight) in materials {
                *value = start + weight;
            }
        } else {
            for ((value, start), weight) in values {
                *value = start.wrapping_sub(*weight);
            }
            for ((value, start), weight) in materials {
                *value = start - weight;
            }
        }
    }

    /// The side's values and material: the parent's with two inputs' rows
    /// added or taken away, in one pass. Sums wrap, so the values are those of
    /// applying the rows one after the other.
    fn apply_two_from(
        &self,
        parent: &Accumulator,
        accumulator: &mut Accumulator,
        side: usize,
        [(first, add_first), (second, add_second)]: [(usize, bool); 2],
    ) {
        let a = &self.weights[first * HALF..(first + 1) * HALF];
        let b = &self.weights[second * HALF..(second + 1) * HALF];
        let values = accumulator.values[side]
            .iter_mut()
            .zip(&parent.values[side])
            .zip(a)
            .zip(b);
        match (add_first, add_second) {
            (false, true) => {
                for (((value, start), a), b) in values {
                    *value = start.wrapping_sub(*a).wrapping_add(*b);
                }
            }
            (true, false) => {
                for (((value, start), a), b) in values {
                    *value = start.wrapping_add(*a).wrapping_sub(*b);
                }
            }
            (false, false) => {
                for (((value, start), a), b) in values {
                    *value = start.wrapping_sub(*a).wrapping_sub(*b);
                }
            }
            (true, true) => {
                for (((value, start), a), b) in values {
                    *value = start.wrapping_add(*a).wrapping_add(*b);
                }
            }
        }
        let a = &self.psqt[first * BUCKETS..(first + 1) * BUCKETS];
        let b = &self.psqt[second * BUCKETS..(second + 1) * BUCKETS];
        for (((value, start), a), b) in accumulator.psqt[side]
            .iter_mut()
            .zip(&parent.psqt[side])
            .zip(a)
            .zip(b)
        {
            let once = if add_first { start + a } else { start - a };
            *value = if add_second { once + b } else { once - b };
        }
    }

    /// The side's values and material: the parent's with several inputs' rows
    /// added or taken away (a capture that blows up pieces beside it, or
    /// castling), a tile of values at a time kept in registers. Sums wrap, so
    /// the values are those of applying the rows one after the other.
    fn apply_rows_from(
        &self,
        parent: &Accumulator,
        accumulator: &mut Accumulator,
        side: usize,
        rows: &[(usize, bool)],
    ) {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: every row's index is a feature's, below INPUTS, so each row
        // holds HALF values, as both sides' values do.
        unsafe {
            self.apply_rows_neon(&parent.values[side], &mut accumulator.values[side], rows);
        }
        #[cfg(target_feature = "simd128")]
        simd128::apply_rows(
            &self.weights,
            &parent.values[side],
            &mut accumulator.values[side],
            rows,
        );
        #[cfg(not(any(target_arch = "aarch64", target_feature = "simd128")))]
        apply_rows(
            &self.weights,
            &parent.values[side],
            &mut accumulator.values[side],
            rows,
        );
        let mut materials = parent.psqt[side];
        for &(index, add) in rows {
            let weights = &self.psqt[index * BUCKETS..(index + 1) * BUCKETS];
            for (value, weight) in materials.iter_mut().zip(weights) {
                if add {
                    *value += weight;
                } else {
                    *value -= weight;
                }
            }
        }
        accumulator.psqt[side] = materials;
    }

    /// `apply_rows_from`'s values with NEON: 64 at a time in eight registers,
    /// every row applied to them before they are stored.
    ///
    /// # Safety
    ///
    /// Every row's index must be below INPUTS.
    #[cfg(target_arch = "aarch64")]
    unsafe fn apply_rows_neon(
        &self,
        parent: &[i16; HALF],
        values: &mut [i16; HALF],
        rows: &[(usize, bool)],
    ) {
        use std::arch::aarch64::{
            int16x8_t, vaddq_s16, vdupq_n_s16, vld1q_s16, vst1q_s16, vsubq_s16,
        };
        const LANES: usize = 8;
        const REGISTERS: usize = 8;
        debug_assert!(rows.iter().all(|&(index, _)| index < INPUTS));
        for tile in (0..HALF).step_by(LANES * REGISTERS) {
            // SAFETY: the tile's 64 values lie inside `parent`, `values` and
            // each row, which holds HALF values from `index * HALF`.
            unsafe {
                let mut sums: [int16x8_t; REGISTERS] = [vdupq_n_s16(0); REGISTERS];
                for (register, sum) in sums.iter_mut().enumerate() {
                    *sum = vld1q_s16(parent.as_ptr().add(tile + register * LANES));
                }
                for &(index, add) in rows {
                    let row = self.weights.as_ptr().add(index * HALF + tile);
                    if add {
                        for (register, sum) in sums.iter_mut().enumerate() {
                            *sum = vaddq_s16(*sum, vld1q_s16(row.add(register * LANES)));
                        }
                    } else {
                        for (register, sum) in sums.iter_mut().enumerate() {
                            *sum = vsubq_s16(*sum, vld1q_s16(row.add(register * LANES)));
                        }
                    }
                }
                for (register, sum) in sums.iter().enumerate() {
                    vst1q_s16(values.as_mut_ptr().add(tile + register * LANES), *sum);
                }
            }
        }
    }

    /// The side's values and material with an input's row added or taken
    /// away.
    fn apply(&self, accumulator: &mut Accumulator, side: usize, index: usize, add: bool) {
        let (values, psqt) = (&mut accumulator.values[side], &mut accumulator.psqt[side]);
        self.apply_to(values, psqt, index, add);
    }

    fn apply_to(
        &self,
        values: &mut [i16; HALF],
        materials: &mut [i32; BUCKETS],
        index: usize,
        add: bool,
    ) {
        let row = &self.weights[index * HALF..(index + 1) * HALF];
        let psqt = &self.psqt[index * BUCKETS..(index + 1) * BUCKETS];
        if add {
            for (value, weight) in values.iter_mut().zip(row) {
                *value = value.wrapping_add(*weight);
            }
            for (value, weight) in materials.iter_mut().zip(psqt) {
                *value += weight;
            }
        } else {
            for (value, weight) in values.iter_mut().zip(row) {
                *value = value.wrapping_sub(*weight);
            }
            for (value, weight) in materials.iter_mut().zip(psqt) {
                *value -= weight;
            }
        }
    }

    /// The evaluation for the side to move from a position's first layer.
    pub fn output(&self, accumulator: &Accumulator, pos: &Position) -> i32 {
        let mut transformed = Transformed::default();
        self.transform(accumulator, pos, &mut transformed);
        self.output_from(&transformed, accumulator, pos)
    }

    /// The evaluation for the side to move from a position's first layer and
    /// its transformed form, `transform`'s.
    pub fn output_from(
        &self,
        transformed: &Transformed,
        accumulator: &Accumulator,
        pos: &Position,
    ) -> i32 {
        self.output_for(
            transformed,
            accumulator,
            pos.occupied().count() as u32,
            pos.turn(),
        )
    }

    /// `output_from` for a position of `pieces` pieces with `us` to move.
    pub fn output_for(
        &self,
        transformed: &Transformed,
        accumulator: &Accumulator,
        pieces: u32,
        us: Color,
    ) -> i32 {
        let bucket = ((i64::from(pieces) - 1) / 4).clamp(0, BUCKETS as i64 - 1) as usize;
        let stack = &self.stacks[bucket];
        let mut sums = stack.b0;
        first_layer(&stack.w0, transformed, &mut sums);
        let hidden0 = sums.map(relu);
        let positional = stack.b2 + later_layers(stack, &hidden0);
        let material =
            (accumulator.psqt[us as usize][bucket] - accumulator.psqt[!us as usize][bucket]) / 2;
        ((i64::from(material) + i64::from(positional)) / OUTPUT_SCALE) as i32
    }

    /// The first layer as the dense layers read it, into `out`: the side to
    /// move's half, then the other side's, each output clipped to 0-127; and
    /// which chunks of four outputs are not all zero.
    pub fn transform(&self, accumulator: &Accumulator, pos: &Position, out: &mut Transformed) {
        self.transform_for(accumulator, pos.turn(), out);
    }

    /// `transform` with `us` to move.
    pub fn transform_for(&self, accumulator: &Accumulator, us: Color, out: &mut Transformed) {
        out.fill([
            &accumulator.values[us as usize],
            &accumulator.values[!us as usize],
        ]);
    }
}

/// `Network::apply_rows_from`'s values without NEON or WebAssembly's SIMD:
/// with AVX2 where the processor has it.
#[cfg_attr(
    any(target_arch = "aarch64", target_feature = "simd128"),
    allow(dead_code)
)]
fn apply_rows(
    weights: &[i16],
    parent: &[i16; HALF],
    values: &mut [i16; HALF],
    rows: &[(usize, bool)],
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        assert!(
            rows.iter()
                .all(|&(index, _)| (index + 1) * HALF <= weights.len())
        );
        // SAFETY: the processor has AVX2, and every row lies inside `weights`.
        unsafe { avx2::apply_rows(weights, parent, values, rows) };
        return;
    }
    apply_rows_portable(weights, parent, values, rows);
}

/// `apply_rows` one row after the other.
#[cfg_attr(
    any(target_arch = "aarch64", target_feature = "simd128"),
    allow(dead_code)
)]
fn apply_rows_portable(
    weights: &[i16],
    parent: &[i16; HALF],
    values: &mut [i16; HALF],
    rows: &[(usize, bool)],
) {
    *values = *parent;
    for &(index, add) in rows {
        let row = &weights[index * HALF..(index + 1) * HALF];
        for (value, weight) in values.iter_mut().zip(row) {
            *value = if add {
                value.wrapping_add(*weight)
            } else {
                value.wrapping_sub(*weight)
            };
        }
    }
}

/// The first layer last computed for each king square of each side, with the
/// squares it was computed from, as Stockfish's refresh cache keeps them: a
/// side whose own king moved starts from its new square's entry and changes
/// only the squares that differ, instead of adding up every piece again. Its
/// sums wrap, so they come out the same in any order. One a search.
pub struct RefreshCache {
    entries: Vec<CacheEntry>,
}

#[derive(Clone)]
struct CacheEntry {
    values: [i16; HALF],
    psqt: [i32; BUCKETS],
    squares: [u8; 64],
}

impl RefreshCache {
    /// Every entry starts as the empty board's: the biases alone.
    pub fn new(network: &Network) -> RefreshCache {
        let empty = CacheEntry {
            values: *network.biases,
            psqt: [0; BUCKETS],
            squares: [0; 64],
        };
        RefreshCache {
            entries: vec![empty; 2 * 64],
        }
    }
}

/// A position's first layer, for each side's perspective, with the pieces it
/// was computed from.
#[derive(Clone)]
pub struct Accumulator {
    values: [[i16; HALF]; 2],
    psqt: [[i32; BUCKETS]; 2],
    squares: [u8; 64],
    kings: [Option<Square>; 2],
}

impl Default for Accumulator {
    fn default() -> Accumulator {
        Accumulator {
            values: [[0; HALF]; 2],
            psqt: [[0; BUCKETS]; 2],
            squares: [0; 64],
            kings: [None; 2],
        }
    }
}

/// The first layer as the dense layers read it, with the chunks of four of
/// its outputs that are not all zero: most are, and the first dense layer and
/// the provability model read only the others. The chunks are listed in
/// order, those at even indices apart from those at odd ones, as the model
/// sums them apart.
#[repr(C, align(64))]
pub struct Transformed {
    pub values: [u8; 2 * HALF],
    chunks: [[u16; CHUNKS / 2]; 2],
    counts: [usize; 2],
}

impl Default for Transformed {
    fn default() -> Transformed {
        Transformed {
            values: [0; 2 * HALF],
            chunks: [[0; CHUNKS / 2]; 2],
            counts: [0; 2],
        }
    }
}

impl Transformed {
    /// The nonzero chunks at even indices and at odd ones, each in order.
    pub fn nonzero(&self) -> [&[u16]; 2] {
        [
            &self.chunks[0][..self.counts[0]],
            &self.chunks[1][..self.counts[1]],
        ]
    }

    /// Clips the two halves of a first layer, the side to move's first, and
    /// lists the nonzero chunks.
    pub(crate) fn fill(&mut self, halves: [&[i16; HALF]; 2]) {
        #[cfg(target_arch = "aarch64")]
        self.fill_neon(halves);
        #[cfg(target_feature = "simd128")]
        simd128::fill(self, halves);
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the processor has AVX2.
            unsafe { avx2::fill(self, halves) };
            return;
        }
        #[cfg(not(any(target_arch = "aarch64", target_feature = "simd128")))]
        self.fill_portable(halves);
    }

    #[cfg_attr(
        any(target_arch = "aarch64", target_feature = "simd128"),
        allow(dead_code)
    )]
    fn fill_portable(&mut self, halves: [&[i16; HALF]; 2]) {
        for (half, values) in halves.into_iter().enumerate() {
            for (output, &value) in self.values[half * HALF..][..HALF].iter_mut().zip(values) {
                *output = value.clamp(0, 127) as u8;
            }
        }
        self.counts = [0; 2];
        for (index, chunk) in self.values.as_chunks::<4>().0.iter().enumerate() {
            let parity = index % 2;
            self.chunks[parity][self.counts[parity]] = index as u16;
            self.counts[parity] += usize::from(u32::from_ne_bytes(*chunk) != 0);
        }
    }

    /// `fill` with NEON, 32 outputs at a time: clipped by saturating to a
    /// byte and taking the minimum with 127, then their eight chunks'
    /// nonzero bits looked up as indices.
    #[cfg(target_arch = "aarch64")]
    fn fill_neon(&mut self, halves: [&[i16; HALF]; 2]) {
        use std::arch::aarch64::{
            vadd_u16, vaddvq_u32, vandq_u32, vcombine_u8, vdup_n_u16, vdupq_n_u8, vld1_u16,
            vld1q_s16, vld1q_u32, vminq_u8, vorrq_u32, vqmovun_s16, vreinterpretq_u32_u8, vst1_u16,
            vst1q_u8, vtstq_u32,
        };
        let mut counts = [0; 2];
        // SAFETY: every aarch64 processor has NEON. The loads read 32 values
        // inside a half, the stores write 32 bytes inside `values` and four
        // indices inside each list of `chunks`: before a group's indices are
        // written, at most four per earlier group are, so at most CHUNKS / 2
        // - 4.
        unsafe {
            let ceiling = vdupq_n_u8(127);
            let low = vld1q_u32([1, 2, 4, 8].as_ptr());
            let high = vld1q_u32([16, 32, 64, 128].as_ptr());
            for (half, values) in halves.into_iter().enumerate() {
                for group in 0..HALF / 32 {
                    let input = values.as_ptr().add(group * 32);
                    let clip = |at: usize| {
                        let low = vqmovun_s16(vld1q_s16(input.add(at)));
                        let high = vqmovun_s16(vld1q_s16(input.add(at + 8)));
                        vminq_u8(vcombine_u8(low, high), ceiling)
                    };
                    let (first, second) = (clip(0), clip(16));
                    let offset = half * HALF + group * 32;
                    vst1q_u8(self.values.as_mut_ptr().add(offset), first);
                    vst1q_u8(self.values.as_mut_ptr().add(offset + 16), second);
                    let (first, second) =
                        (vreinterpretq_u32_u8(first), vreinterpretq_u32_u8(second));
                    let bits = vorrq_u32(
                        vandq_u32(vtstq_u32(first, first), low),
                        vandq_u32(vtstq_u32(second, second), high),
                    );
                    let byte = vaddvq_u32(bits) as usize;
                    let base = vdup_n_u16((offset / 4) as u16);
                    for (parity, count) in counts.iter_mut().enumerate() {
                        let indices = vadd_u16(vld1_u16(SET_BITS[byte][parity].as_ptr()), base);
                        vst1_u16(self.chunks[parity].as_mut_ptr().add(*count), indices);
                        *count += usize::from(SET_COUNTS[byte][parity]);
                    }
                }
            }
        }
        self.counts = counts;
    }
}

/// For each byte, how many of its even bits and of its odd bits are set.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_feature = "simd128"
))]
static SET_COUNTS: [[u8; 2]; 256] = {
    let mut table = [[0; 2]; 256];
    let mut byte = 0;
    while byte < 256 {
        table[byte] = [
            (byte & 0x55).count_ones() as u8,
            (byte & 0xaa).count_ones() as u8,
        ];
        byte += 1;
    }
    table
};

/// For each byte, the positions of its set bits, the even ones then the odd
/// ones, in order.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_feature = "simd128"
))]
static SET_BITS: [[[u16; 4]; 2]; 256] = {
    let mut table = [[[0; 4]; 2]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut counts = [0; 2];
        let mut bit = 0;
        while bit < 8 {
            if byte & (1 << bit) != 0 {
                table[byte][bit % 2][counts[bit % 2]] = bit as u16;
                counts[bit % 2] += 1;
            }
            bit += 1;
        }
        byte += 1;
    }
    table
};

/// The second dense layer and the output's sum, before its bias.
fn later_layers(stack: &Stack, hidden0: &[u8; FC0]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        // SAFETY: the processor has the dot-product instructions.
        return unsafe { later_layers_dotprod(stack, hidden0) };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2.
        return unsafe { avx2::later_layers(stack, hidden0) };
    }
    #[cfg(target_feature = "simd128")]
    return simd128::later_layers(stack, hidden0);
    #[cfg(not(target_feature = "simd128"))]
    later_layers_portable(stack, hidden0)
}

#[cfg_attr(target_feature = "simd128", allow(dead_code))]
fn later_layers_portable(stack: &Stack, hidden0: &[u8; FC0]) -> i32 {
    let mut hidden1 = [0u8; FC1];
    for (index, output) in hidden1.iter_mut().enumerate() {
        let row = &stack.w1[index * 32..index * 32 + FC0];
        *output = relu(stack.b1[index] + dot(row, hidden0));
    }
    dot(&stack.w2, &hidden1)
}

/// `later_layers` with the dot-product instruction: each output's 16 inputs
/// in one, and the 32 hidden values into the output in two. Inputs are at
/// most 127, so they read as signed bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn later_layers_dotprod(stack: &Stack, hidden0: &[u8; FC0]) -> i32 {
    use std::arch::aarch64::{vaddvq_s32, vdotq_s32, vdupq_n_s32, vld1q_s8};
    assert!(stack.w1.len() >= FC1 * 32);
    let mut hidden1 = [0u8; FC1];
    // SAFETY: `hidden0` holds 16 bytes, each row of `w1` 32 (16 used), and
    // `w2` and `hidden1` 32.
    unsafe {
        let x = vld1q_s8(hidden0.as_ptr().cast());
        for (index, output) in hidden1.iter_mut().enumerate() {
            let row = vld1q_s8(stack.w1.as_ptr().add(index * 32));
            let sum = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), row, x));
            *output = relu(stack.b1[index] + sum);
        }
        let low = vdotq_s32(
            vdupq_n_s32(0),
            vld1q_s8(stack.w2.as_ptr()),
            vld1q_s8(hidden1.as_ptr().cast()),
        );
        let high = vdotq_s32(
            low,
            vld1q_s8(stack.w2.as_ptr().add(16)),
            vld1q_s8(hidden1.as_ptr().add(16).cast()),
        );
        vaddvq_s32(high)
    }
}

/// The first dense layer, over the first layer's nonzero chunks only: a chunk
/// of zeros adds nothing to any output.
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

#[cfg_attr(target_feature = "simd128", allow(dead_code))]
fn first_layer_portable(weights: &[i8], transformed: &Transformed, sums: &mut [i32; FC0]) {
    let [even, odd] = transformed.nonzero();
    for &chunk in even.iter().chain(odd) {
        let chunk = usize::from(chunk);
        let inputs = &transformed.values[chunk * 4..chunk * 4 + 4];
        let (rows, _) = weights[chunk * FC0 * 4..(chunk + 1) * FC0 * 4].as_chunks::<4>();
        for (sum, row) in sums.iter_mut().zip(rows) {
            *sum += dot(row, inputs);
        }
    }
}

/// The first dense layer with the dot-product instruction: each nonzero
/// chunk's four inputs against its 16 outputs' weights, the even chunks and
/// the odd ones into separate sums, so that they need not wait on each other.
/// Inputs are at most 127, so they read as signed bytes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn first_layer_dotprod(weights: &[i8], transformed: &Transformed, sums: &mut [i32; FC0]) {
    use std::arch::aarch64::{vaddq_s32, vdupq_n_s32, vld1q_s32, vst1q_s32};
    assert_eq!(weights.len(), CHUNKS * FC0 * 4);
    let [even, odd] = transformed.nonzero();
    let both = even.len().min(odd.len());
    let mut first = [vdupq_n_s32(0); 4];
    let mut second = [vdupq_n_s32(0); 4];
    // SAFETY: the chunks are below CHUNKS, as `chunk_products` needs, and
    // `sums` has 16 values, four per row.
    unsafe {
        for (&a, &b) in even[..both].iter().zip(&odd[..both]) {
            chunk_products(&mut first, weights, transformed, a);
            chunk_products(&mut second, weights, transformed, b);
        }
        for &chunk in even[both..].iter().chain(&odd[both..]) {
            chunk_products(&mut first, weights, transformed, chunk);
        }
        for (row, (first, second)) in first.into_iter().zip(second).enumerate() {
            let sum = sums.as_mut_ptr().add(row * 4);
            vst1q_s32(sum, vaddq_s32(vld1q_s32(sum), vaddq_s32(first, second)));
        }
    }
}

/// Adds a chunk's four inputs times its 16 outputs' weights to `totals`, four
/// outputs each.
///
/// # Safety
///
/// The chunk must be below CHUNKS, and `weights` hold CHUNKS * FC0 * 4 values.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn chunk_products(
    totals: &mut [std::arch::aarch64::int32x4_t; 4],
    weights: &[i8],
    transformed: &Transformed,
    chunk: u16,
) {
    use std::arch::aarch64::{vdotq_s32, vld1q_dup_u32, vld1q_s8, vreinterpretq_s8_u32};
    let chunk = usize::from(chunk);
    // SAFETY: the loads read 4 bytes inside `values`, aligned since `values`
    // is aligned to 64 bytes, and 64 inside `weights`.
    unsafe {
        let inputs = transformed.values.as_ptr().add(chunk * 4).cast();
        let x = vreinterpretq_s8_u32(vld1q_dup_u32(inputs));
        let rows = weights.as_ptr().add(chunk * FC0 * 4);
        for (row, total) in totals.iter_mut().enumerate() {
            *total = vdotq_s32(*total, vld1q_s8(rows.add(row * 16)), x);
        }
    }
}

fn dot(weights: &[i8], inputs: &[u8]) -> i32 {
    weights
        .iter()
        .zip(inputs)
        .map(|(&weight, &input)| i32::from(weight) * i32::from(input))
        .sum()
}

fn relu(sum: i32) -> u8 {
    (sum >> WEIGHT_SCALE_BITS).clamp(0, 127) as u8
}

/// A square as the perspective sees it: Black's board is flipped.
fn orient(perspective: Color, square: Square) -> usize {
    match perspective {
        Color::White => square as usize,
        Color::Black => square as usize ^ 56,
    }
}

/// A piece's input, by its square code, for a perspective whose king is on
/// `king`, oriented.
fn feature(perspective: Color, square: Square, code: u8, king: usize) -> usize {
    let role = usize::from(code & 7);
    let white = code & 8 != 0;
    let plane = if role == Role::King as usize {
        10
    } else {
        2 * (role - 1) + usize::from(white != perspective.is_white())
    };
    orient(perspective, square) + plane * 64 + king * 11 * 64
}

/// The network with AVX2, for x86-64 processors that have it, as most made
/// since 2013 do, chosen at run time: what NEON does on aarch64. Products of
/// bytes are summed in pairs in 16 bits, of first-layer outputs at most 127
/// and weights that are signed bytes, so no pair's sum saturates: every
/// product and sum is exact, and the results are the portable code's.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::{
        __m256i, _mm_add_epi16, _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_loadu_si128,
        _mm_set1_epi16, _mm_shuffle_epi32, _mm_storel_epi64, _mm256_add_epi16, _mm256_add_epi32,
        _mm256_broadcastsi128_si256, _mm256_castsi256_ps, _mm256_castsi256_si128,
        _mm256_cmpeq_epi32, _mm256_cvtepi8_epi32, _mm256_extracti128_si256, _mm256_hadd_epi32,
        _mm256_loadu_si256, _mm256_loadu2_m128i, _mm256_madd_epi16, _mm256_maddubs_epi16,
        _mm256_max_epi32, _mm256_min_epi32, _mm256_min_epu8, _mm256_movemask_ps,
        _mm256_mullo_epi32, _mm256_packus_epi16, _mm256_permute4x64_epi64,
        _mm256_permutevar8x32_epi32, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_set1_epi32,
        _mm256_setr_epi32, _mm256_setzero_si256, _mm256_srai_epi32, _mm256_storeu_si256,
        _mm256_sub_epi16,
    };

    use super::{
        CHUNKS, FC0, FC1, HALF, SET_BITS, SET_COUNTS, Stack, Transformed, WEIGHT_SCALE_BITS,
    };

    /// `apply_rows`: 128 values at a time in eight registers, every row
    /// applied to them before they are stored.
    ///
    /// # Safety
    ///
    /// The processor must have AVX2, and every row lie inside `weights`, as
    /// HALF values from `index * HALF`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn apply_rows(
        weights: &[i16],
        parent: &[i16; HALF],
        values: &mut [i16; HALF],
        rows: &[(usize, bool)],
    ) {
        const LANES: usize = 16;
        const REGISTERS: usize = 8;
        debug_assert!(
            rows.iter()
                .all(|&(index, _)| (index + 1) * HALF <= weights.len())
        );
        for tile in (0..HALF).step_by(LANES * REGISTERS) {
            // SAFETY: the tile's 128 values lie inside `parent`, `values` and
            // each row; the loads and stores need no alignment.
            unsafe {
                let mut sums = [_mm256_setzero_si256(); REGISTERS];
                for (register, sum) in sums.iter_mut().enumerate() {
                    *sum = _mm256_loadu_si256(parent.as_ptr().add(tile + register * LANES).cast());
                }
                for &(index, add) in rows {
                    let row = weights.as_ptr().add(index * HALF + tile);
                    if add {
                        for (register, sum) in sums.iter_mut().enumerate() {
                            let weight = _mm256_loadu_si256(row.add(register * LANES).cast());
                            *sum = _mm256_add_epi16(*sum, weight);
                        }
                    } else {
                        for (register, sum) in sums.iter_mut().enumerate() {
                            let weight = _mm256_loadu_si256(row.add(register * LANES).cast());
                            *sum = _mm256_sub_epi16(*sum, weight);
                        }
                    }
                }
                for (register, sum) in sums.iter().enumerate() {
                    let at = values.as_mut_ptr().add(tile + register * LANES);
                    _mm256_storeu_si256(at.cast(), *sum);
                }
            }
        }
    }

    /// `Transformed::fill`, 32 outputs at a time: clipped by packing to bytes
    /// with unsigned saturation and taking the minimum with 127, then their
    /// eight chunks' nonzero bits looked up as indices.
    ///
    /// # Safety
    ///
    /// The processor must have AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn fill(transformed: &mut Transformed, halves: [&[i16; HALF]; 2]) {
        let mut counts = [0; 2];
        let ceiling = _mm256_set1_epi8(127);
        let zero = _mm256_setzero_si256();
        for (half, values) in halves.into_iter().enumerate() {
            for group in 0..HALF / 32 {
                // SAFETY: the loads read 32 values inside a half, the store
                // writes 32 bytes inside `values` and four indices inside each
                // list of `chunks`: before a group's indices are written, at
                // most four per earlier group are, so at most CHUNKS / 2 - 4.
                unsafe {
                    let input = values.as_ptr().add(group * 32);
                    let low = _mm256_loadu_si256(input.cast());
                    let high = _mm256_loadu_si256(input.add(16).cast());
                    // Packing keeps to each 128-bit half, so the bytes come out
                    // as low's first eight, high's first eight, then the
                    // second eights: the middle two eights swap back.
                    let packed =
                        _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(low, high));
                    let clipped = _mm256_min_epu8(packed, ceiling);
                    let offset = half * HALF + group * 32;
                    let output = transformed.values.as_mut_ptr().add(offset);
                    _mm256_storeu_si256(output.cast(), clipped);
                    let zeros = _mm256_cmpeq_epi32(clipped, zero);
                    let byte = !_mm256_movemask_ps(_mm256_castsi256_ps(zeros)) as usize & 0xff;
                    let base = _mm_set1_epi16((offset / 4) as i16);
                    for (parity, count) in counts.iter_mut().enumerate() {
                        let bits = _mm_loadl_epi64(SET_BITS[byte][parity].as_ptr().cast());
                        let at = transformed.chunks[parity].as_mut_ptr().add(*count);
                        _mm_storel_epi64(at.cast(), _mm_add_epi16(bits, base));
                        *count += usize::from(SET_COUNTS[byte][parity]);
                    }
                }
            }
        }
        transformed.counts = counts;
    }

    /// `first_layer`: each nonzero chunk's four inputs, repeated, against its
    /// 16 outputs' weights, eight outputs a register, the even chunks and the
    /// odd ones into separate sums, as NEON does.
    ///
    /// # Safety
    ///
    /// The processor must have AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn first_layer(weights: &[i8], transformed: &Transformed, sums: &mut [i32; FC0]) {
        assert_eq!(weights.len(), CHUNKS * FC0 * 4);
        let [even, odd] = transformed.nonzero();
        let both = even.len().min(odd.len());
        let mut first = [_mm256_setzero_si256(); 2];
        let mut second = [_mm256_setzero_si256(); 2];
        // SAFETY: the chunks are below CHUNKS, as `chunk_products` needs, and
        // `sums` has 16 values, eight per register.
        unsafe {
            for (&a, &b) in even[..both].iter().zip(&odd[..both]) {
                chunk_products(&mut first, weights, transformed, a);
                chunk_products(&mut second, weights, transformed, b);
            }
            for &chunk in even[both..].iter().chain(&odd[both..]) {
                chunk_products(&mut first, weights, transformed, chunk);
            }
            for (half, (first, second)) in first.into_iter().zip(second).enumerate() {
                let at = sums.as_mut_ptr().add(half * 8).cast::<__m256i>();
                let total = _mm256_add_epi32(first, second);
                _mm256_storeu_si256(at, _mm256_add_epi32(_mm256_loadu_si256(at), total));
            }
        }
    }

    /// Adds a chunk's four inputs times its 16 outputs' weights to `totals`,
    /// eight outputs each: the products summed in pairs, then the pairs in
    /// fours.
    ///
    /// # Safety
    ///
    /// The processor must have AVX2, the chunk be below CHUNKS, and `weights`
    /// hold CHUNKS * FC0 * 4 values.
    #[target_feature(enable = "avx2")]
    unsafe fn chunk_products(
        totals: &mut [__m256i; 2],
        weights: &[i8],
        transformed: &Transformed,
        chunk: u16,
    ) {
        let chunk = usize::from(chunk);
        let ones = _mm256_set1_epi16(1);
        // SAFETY: the loads read 4 bytes inside `values` and 64 inside
        // `weights`.
        unsafe {
            let inputs = transformed.values.as_ptr().add(chunk * 4).cast::<i32>();
            let x = _mm256_set1_epi32(inputs.read_unaligned());
            let rows = weights.as_ptr().add(chunk * FC0 * 4);
            for (half, total) in totals.iter_mut().enumerate() {
                let pairs = _mm256_maddubs_epi16(x, _mm256_loadu_si256(rows.add(half * 32).cast()));
                *total = _mm256_add_epi32(*total, _mm256_madd_epi16(pairs, ones));
            }
        }
    }

    /// `later_layers`: two outputs' 16 inputs a register, in pairs then fours
    /// as `first_layer` takes them, each eight outputs' fours added up across
    /// registers, and the output's 32 products in 32 bits.
    ///
    /// # Safety
    ///
    /// The processor must have AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn later_layers(stack: &Stack, hidden0: &[u8; FC0]) -> i32 {
        assert!(stack.w1.len() >= FC1 * 32);
        // The additions across registers leave each eight outputs in the
        // order 0, 2, 4, 6, 1, 3, 5, 7, which this puts back.
        let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let (floor, ceiling) = (_mm256_setzero_si256(), _mm256_set1_epi32(127));
        let mut total = _mm256_setzero_si256();
        // SAFETY: `hidden0` holds 16 bytes, each row of `w1` 32 (16 used),
        // and `b1` and `w2` 32 values.
        unsafe {
            let x = _mm256_broadcastsi128_si256(_mm_loadu_si128(hidden0.as_ptr().cast()));
            for group in 0..FC1 / 8 {
                let rows = stack.w1.as_ptr().add(group * 8 * 32);
                let sums = _mm256_hadd_epi32(
                    _mm256_hadd_epi32(fours(x, rows), fours(x, rows.add(64))),
                    _mm256_hadd_epi32(fours(x, rows.add(128)), fours(x, rows.add(192))),
                );
                let sums = _mm256_add_epi32(
                    _mm256_permutevar8x32_epi32(sums, order),
                    _mm256_loadu_si256(stack.b1.as_ptr().add(group * 8).cast()),
                );
                let hidden = _mm256_min_epi32(
                    _mm256_max_epi32(
                        _mm256_srai_epi32::<{ WEIGHT_SCALE_BITS as i32 }>(sums),
                        floor,
                    ),
                    ceiling,
                );
                let weights = _mm_loadl_epi64(stack.w2.as_ptr().add(group * 8).cast());
                total = _mm256_add_epi32(
                    total,
                    _mm256_mullo_epi32(hidden, _mm256_cvtepi8_epi32(weights)),
                );
            }
        }
        let halves = _mm_add_epi32(
            _mm256_castsi256_si128(total),
            _mm256_extracti128_si256::<1>(total),
        );
        let pairs = _mm_add_epi32(halves, _mm_shuffle_epi32::<0b01_00_11_10>(halves));
        _mm_cvtsi128_si32(_mm_add_epi32(
            pairs,
            _mm_shuffle_epi32::<0b10_11_00_01>(pairs),
        ))
    }

    /// Two rows' 16 products with the inputs `x`, in fours: the row at `row`
    /// in the low half, the one 32 bytes on in the high half.
    ///
    /// # Safety
    ///
    /// The processor must have AVX2, and 16 bytes at `row` and at `row + 32`
    /// be readable.
    #[target_feature(enable = "avx2")]
    unsafe fn fours(x: __m256i, row: *const i8) -> __m256i {
        // SAFETY: the load reads 16 bytes at `row` and 16 at `row + 32`.
        let rows = unsafe { _mm256_loadu2_m128i(row.add(32).cast(), row.cast()) };
        _mm256_madd_epi16(_mm256_maddubs_epi16(x, rows), _mm256_set1_epi16(1))
    }
}

/// The network with WebAssembly's 128-bit SIMD, which browsers run on any
/// processor, for the browser build: what NEON does on aarch64. It has no
/// dot product of bytes, so products are taken in 16 bits, of first-layer
/// outputs at most 127 and weights that are signed bytes: every product and
/// sum is exact, and the results are the portable code's.
#[cfg(target_feature = "simd128")]
mod simd128 {
    use std::arch::wasm32::{
        i16x8_add, i16x8_sub, i32x4_add, i32x4_bitmask, i32x4_extract_lane, i32x4_ne,
        i32x4_shuffle, i32x4_splat, u8x16_min, u8x16_narrow_i16x8, u8x16_splat, u16x8_add,
        u16x8_splat, v128, v128_load, v128_load32_splat, v128_load64_zero, v128_store,
        v128_store64_lane,
    };

    use super::{CHUNKS, FC0, FC1, HALF, INPUTS, SET_BITS, SET_COUNTS, Stack, Transformed, relu};

    /// `apply_rows_from`'s values: 64 at a time in eight registers, every row
    /// applied to them before they are stored. Every row's index must be
    /// below INPUTS, and `weights` hold INPUTS rows of HALF values.
    pub fn apply_rows(
        weights: &[i16],
        parent: &[i16; HALF],
        values: &mut [i16; HALF],
        rows: &[(usize, bool)],
    ) {
        const LANES: usize = 8;
        const REGISTERS: usize = 8;
        assert!(rows.iter().all(|&(index, _)| index < INPUTS) && weights.len() == INPUTS * HALF);
        for tile in (0..HALF).step_by(LANES * REGISTERS) {
            // SAFETY: the tile's 64 values lie inside `parent`, `values` and
            // each row, which holds HALF values from `index * HALF`; the loads
            // and stores need no alignment.
            unsafe {
                let mut sums = [i32x4_splat(0); REGISTERS];
                for (register, sum) in sums.iter_mut().enumerate() {
                    *sum = v128_load(parent.as_ptr().add(tile + register * LANES).cast());
                }
                for &(index, add) in rows {
                    let row = weights.as_ptr().add(index * HALF + tile);
                    for (register, sum) in sums.iter_mut().enumerate() {
                        let weight = v128_load(row.add(register * LANES).cast());
                        *sum = if add {
                            i16x8_add(*sum, weight)
                        } else {
                            i16x8_sub(*sum, weight)
                        };
                    }
                }
                for (register, sum) in sums.iter().enumerate() {
                    v128_store(
                        values.as_mut_ptr().add(tile + register * LANES).cast(),
                        *sum,
                    );
                }
            }
        }
    }

    /// `Transformed::fill`, 32 outputs at a time: clipped by narrowing to
    /// bytes with unsigned saturation and taking the minimum with 127, then
    /// their eight chunks' nonzero bits looked up as indices.
    pub fn fill(transformed: &mut Transformed, halves: [&[i16; HALF]; 2]) {
        let mut counts = [0; 2];
        let ceiling = u8x16_splat(127);
        let zero = i32x4_splat(0);
        for (half, values) in halves.into_iter().enumerate() {
            for group in 0..HALF / 32 {
                // SAFETY: the loads read 32 values inside a half, the stores
                // write 32 bytes inside `values` and four indices inside each
                // list of `chunks`: before a group's indices are written, at
                // most four per earlier group are, so at most CHUNKS / 2 - 4.
                unsafe {
                    let input = values.as_ptr().add(group * 32);
                    let clip = |at: usize| {
                        let low = v128_load(input.add(at).cast());
                        let high = v128_load(input.add(at + 8).cast());
                        u8x16_min(u8x16_narrow_i16x8(low, high), ceiling)
                    };
                    let (first, second) = (clip(0), clip(16));
                    let offset = half * HALF + group * 32;
                    let output = transformed.values.as_mut_ptr().add(offset);
                    v128_store(output.cast(), first);
                    v128_store(output.add(16).cast(), second);
                    let byte = i32x4_bitmask(i32x4_ne(first, zero))
                        | i32x4_bitmask(i32x4_ne(second, zero)) << 4;
                    let byte = usize::from(byte);
                    let base = u16x8_splat((offset / 4) as u16);
                    for (parity, count) in counts.iter_mut().enumerate() {
                        let bits = v128_load64_zero(SET_BITS[byte][parity].as_ptr().cast());
                        let at = transformed.chunks[parity].as_mut_ptr().add(*count);
                        v128_store64_lane::<0>(u16x8_add(bits, base), at.cast());
                        *count += usize::from(SET_COUNTS[byte][parity]);
                    }
                }
            }
        }
        transformed.counts = counts;
    }

    /// `first_layer`: each nonzero chunk's four inputs, widened to 16 bits
    /// and repeated, against its 16 outputs' weights, four outputs in each of
    /// four registers' pairs; an output's two half sums are added at the end.
    #[cfg(not(target_feature = "relaxed-simd"))]
    pub fn first_layer(weights: &[i8], transformed: &Transformed, sums: &mut [i32; FC0]) {
        use std::arch::wasm32::{
            i16x8_extend_high_i8x16, i16x8_extend_low_i8x16, i32x4_dot_i16x8,
            u16x8_extend_low_u8x16,
        };
        assert_eq!(weights.len(), CHUNKS * FC0 * 4);
        let [even, odd] = transformed.nonzero();
        let mut totals = [[i32x4_splat(0); 2]; 4];
        for &chunk in even.iter().chain(odd) {
            let chunk = usize::from(chunk);
            // SAFETY: the chunk is below CHUNKS, so its four inputs lie inside
            // `values` and its 64 weights inside `weights`.
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
            // SAFETY: `sums` has 16 values, four per row.
            unsafe {
                let at = sums.as_mut_ptr().add(row * 4).cast::<v128>();
                v128_store(at, i32x4_add(v128_load(at), sum));
            }
        }
    }

    /// `later_layers`: each of the 32 outputs' 16 inputs in two 16-bit dot
    /// products, and the 32 hidden values into the output in four.
    #[cfg(not(target_feature = "relaxed-simd"))]
    pub fn later_layers(stack: &Stack, hidden0: &[u8; FC0]) -> i32 {
        use std::arch::wasm32::{u16x8_extend_high_u8x16, u16x8_extend_low_u8x16};
        assert!(stack.w1.len() >= FC1 * 32);
        let mut hidden1 = [0u8; FC1];
        // SAFETY: `hidden0` holds 16 bytes, each row of `w1` 32 (16 used), and
        // `w2` and `hidden1` 32.
        unsafe {
            let x = v128_load(hidden0.as_ptr().cast());
            let (low, high) = (u16x8_extend_low_u8x16(x), u16x8_extend_high_u8x16(x));
            for (index, output) in hidden1.iter_mut().enumerate() {
                let row = v128_load(stack.w1.as_ptr().add(index * 32).cast());
                *output = relu(stack.b1[index] + sum(products(row, low, high)));
            }
            let hidden = |at: usize| v128_load(hidden1.as_ptr().add(at).cast());
            let weights = |at: usize| v128_load(stack.w2.as_ptr().add(at).cast());
            let (first, second) = (hidden(0), hidden(16));
            sum(i32x4_add(
                products(
                    weights(0),
                    u16x8_extend_low_u8x16(first),
                    u16x8_extend_high_u8x16(first),
                ),
                products(
                    weights(16),
                    u16x8_extend_low_u8x16(second),
                    u16x8_extend_high_u8x16(second),
                ),
            ))
        }
    }

    /// Sixteen signed byte weights times sixteen inputs, as four sums.
    #[cfg(not(target_feature = "relaxed-simd"))]
    fn products(weights: v128, low: v128, high: v128) -> v128 {
        use std::arch::wasm32::{i16x8_extend_high_i8x16, i16x8_extend_low_i8x16, i32x4_dot_i16x8};
        i32x4_add(
            i32x4_dot_i16x8(i16x8_extend_low_i8x16(weights), low),
            i32x4_dot_i16x8(i16x8_extend_high_i8x16(weights), high),
        )
    }

    /// `first_layer` with relaxed SIMD's dot product of bytes: each nonzero
    /// chunk's four inputs, repeated, against its 16 outputs' weights, four
    /// outputs a register, the even chunks and the odd ones into separate
    /// sums, as NEON does. The inputs are at most 127, where the product is
    /// exact in every implementation.
    #[cfg(target_feature = "relaxed-simd")]
    pub fn first_layer(weights: &[i8], transformed: &Transformed, sums: &mut [i32; FC0]) {
        assert_eq!(weights.len(), CHUNKS * FC0 * 4);
        let [even, odd] = transformed.nonzero();
        let both = even.len().min(odd.len());
        let mut first = [i32x4_splat(0); 4];
        let mut second = [i32x4_splat(0); 4];
        for (&a, &b) in even[..both].iter().zip(&odd[..both]) {
            chunk_products(&mut first, weights, transformed, a);
            chunk_products(&mut second, weights, transformed, b);
        }
        for &chunk in even[both..].iter().chain(&odd[both..]) {
            chunk_products(&mut first, weights, transformed, chunk);
        }
        for (row, (first, second)) in first.into_iter().zip(second).enumerate() {
            // SAFETY: `sums` has 16 values, four per row.
            unsafe {
                let at = sums.as_mut_ptr().add(row * 4).cast::<v128>();
                v128_store(at, i32x4_add(v128_load(at), i32x4_add(first, second)));
            }
        }
    }

    /// Adds a chunk's four inputs times its 16 outputs' weights to `totals`,
    /// four outputs each. The chunk must be below CHUNKS.
    #[cfg(target_feature = "relaxed-simd")]
    fn chunk_products(
        totals: &mut [v128; 4],
        weights: &[i8],
        transformed: &Transformed,
        chunk: u16,
    ) {
        use std::arch::wasm32::i32x4_relaxed_dot_i8x16_i7x16_add;
        let chunk = usize::from(chunk);
        assert!(chunk < CHUNKS);
        // SAFETY: the chunk is below CHUNKS, so its four inputs lie inside
        // `values` and its 64 weights inside `weights`.
        unsafe {
            let x = v128_load32_splat(transformed.values.as_ptr().add(chunk * 4).cast());
            let rows = weights.as_ptr().add(chunk * FC0 * 4);
            for (row, total) in totals.iter_mut().enumerate() {
                let w = v128_load(rows.add(row * 16).cast());
                *total = i32x4_relaxed_dot_i8x16_i7x16_add(w, x, *total);
            }
        }
    }

    /// `later_layers` with relaxed SIMD's dot product of bytes: each output's
    /// 16 inputs in one, and the 32 hidden values into the output in two.
    #[cfg(target_feature = "relaxed-simd")]
    pub fn later_layers(stack: &Stack, hidden0: &[u8; FC0]) -> i32 {
        use std::arch::wasm32::i32x4_relaxed_dot_i8x16_i7x16_add;
        assert!(stack.w1.len() >= FC1 * 32);
        let zero = i32x4_splat(0);
        let mut hidden1 = [0u8; FC1];
        // SAFETY: `hidden0` holds 16 bytes, each row of `w1` 32 (16 used), and
        // `w2` and `hidden1` 32.
        unsafe {
            let x = v128_load(hidden0.as_ptr().cast());
            for (index, output) in hidden1.iter_mut().enumerate() {
                let row = v128_load(stack.w1.as_ptr().add(index * 32).cast());
                let products = i32x4_relaxed_dot_i8x16_i7x16_add(row, x, zero);
                *output = relu(stack.b1[index] + sum(products));
            }
            let low = i32x4_relaxed_dot_i8x16_i7x16_add(
                v128_load(stack.w2.as_ptr().cast()),
                v128_load(hidden1.as_ptr().cast()),
                zero,
            );
            sum(i32x4_relaxed_dot_i8x16_i7x16_add(
                v128_load(stack.w2.as_ptr().add(16).cast()),
                v128_load(hidden1.as_ptr().add(16).cast()),
                low,
            ))
        }
    }

    fn sum(values: v128) -> i32 {
        let pairs = i32x4_add(values, i32x4_shuffle::<2, 3, 0, 1>(values, values));
        i32x4_extract_lane::<0>(i32x4_add(pairs, i32x4_shuffle::<1, 0, 3, 2>(pairs, pairs)))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::Path;

    use super::{
        Accumulator, CHUNKS, FC0, FC1, HALF, MANY, Network, RefreshCache, Stack, Transformed,
        apply_rows, apply_rows_portable, first_layer, first_layer_portable, later_layers,
        later_layers_portable,
    };
    use crate::board::{Position, from_fen};

    /// A xorshift generator for test inputs.
    pub(crate) struct Random(pub(crate) u64);

    impl Random {
        pub(crate) fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// First-layer values like the network's, mostly at or below zero,
        /// with a share `zeros` of whole chunks of four at or below zero.
        pub(crate) fn halves(&mut self, zeros: u64) -> [[i16; HALF]; 2] {
            let mut halves = [[0; HALF]; 2];
            for chunk in halves
                .iter_mut()
                .flat_map(|half| half.as_chunks_mut::<4>().0)
            {
                let dead = self.next() % 100 < zeros;
                for value in chunk {
                    *value = match self.next() % 8 {
                        _ if dead => -((self.next() % 300) as i16),
                        0 | 1 => -((self.next() % 3000) as i16),
                        2 => 0,
                        3 => (self.next() % 128) as i16,
                        4 => 127 + (self.next() % 2000) as i16,
                        _ => (self.next() % 20) as i16 - 10,
                    };
                }
            }
            halves
        }
    }

    /// The clipped values and their nonzero chunks, from NEON and from the
    /// plain code, against the definition.
    #[test]
    fn transforms_as_defined() {
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        for trial in 0..2000 {
            let halves = random.halves(trial % 101);
            let mut transformed = Transformed::default();
            transformed.fill([&halves[0], &halves[1]]);
            let clipped: Vec<u8> = halves
                .iter()
                .flatten()
                .map(|&value| value.clamp(0, 127) as u8)
                .collect();
            assert_eq!(transformed.values.as_slice(), clipped.as_slice());
            for (parity, listed) in transformed.nonzero().into_iter().enumerate() {
                let nonzero: Vec<u16> = (0..CHUNKS as u16)
                    .filter(|&chunk| usize::from(chunk) % 2 == parity)
                    .filter(|&chunk| clipped[usize::from(chunk) * 4..][..4] != [0; 4])
                    .collect();
                assert_eq!(listed, nonzero.as_slice());
            }
            let mut portable = Transformed::default();
            portable.fill_portable([&halves[0], &halves[1]]);
            assert_eq!(portable.values, transformed.values);
            assert_eq!(portable.nonzero(), transformed.nonzero());
        }
    }

    /// The first dense layer over nonzero chunks, with the dot-product
    /// instruction and without, against every input's product.
    #[test]
    fn first_layer_skips_only_zeros() {
        let mut random = Random(0x2545_f491_4f6c_dd1d);
        let weights: Vec<i8> = (0..CHUNKS * FC0 * 4).map(|_| random.next() as i8).collect();
        for trial in 0..500 {
            let halves = random.halves(trial % 101);
            let mut transformed = Transformed::default();
            transformed.fill([&halves[0], &halves[1]]);
            let mut expected = [0i32; FC0];
            for (output, sum) in expected.iter_mut().enumerate() {
                for (input, &value) in transformed.values.iter().enumerate() {
                    let weight = weights[(input / 4 * FC0 + output) * 4 + input % 4];
                    *sum += i32::from(weight) * i32::from(value);
                }
            }
            let mut sums = [0; FC0];
            first_layer(&weights, &transformed, &mut sums);
            assert_eq!(sums, expected);
            let mut sums = [0; FC0];
            first_layer_portable(&weights, &transformed, &mut sums);
            assert_eq!(sums, expected);
        }
    }

    /// The dense layers after the first, with the processor's instructions
    /// and without, over random weights and inputs that reach both ends of
    /// the clipping.
    #[test]
    fn later_layers_as_portable() {
        let mut random = Random(0x853c_49e6_748f_ea9b);
        for _ in 0..2000 {
            let stack = Stack {
                b0: [0; FC0],
                w0: Vec::new(),
                b1: std::array::from_fn(|_| (random.next() % 40_001) as i32 - 20_000),
                w1: (0..FC1 * 32).map(|_| random.next() as i8).collect(),
                b2: 0,
                w2: std::array::from_fn(|_| random.next() as i8),
            };
            let hidden0 = std::array::from_fn(|_| match random.next() % 4 {
                0 => 0,
                1 => 127,
                _ => (random.next() % 128) as u8,
            });
            assert_eq!(
                later_layers(&stack, &hidden0),
                later_layers_portable(&stack, &hidden0)
            );
        }
    }

    /// Several rows added to and taken from a side's values in one pass, with
    /// the processor's instructions and without, over random weights whose
    /// sums wrap.
    #[test]
    fn applies_rows_as_portable() {
        let mut random = Random(0xda94_2042_e4dd_58b5);
        let weights: Vec<i16> = (0..64 * HALF).map(|_| random.next() as i16).collect();
        for trial in 0..500 {
            let parent = std::array::from_fn(|_| random.next() as i16);
            let rows: Vec<_> = (0..trial % (MANY + 1))
                .map(|_| ((random.next() % 64) as usize, random.next() & 1 == 0))
                .collect();
            let (mut values, mut expected) = ([0; HALF], [0; HALF]);
            apply_rows(&weights, &parent, &mut values, &rows);
            apply_rows_portable(&weights, &parent, &mut expected, &rows);
            assert_eq!(values, expected);
        }
    }

    /// The first layers a line updates match computing them from scratch,
    /// over random games, with the network bundled builds carry; the test
    /// passes without checking when `data/network.nnue` is absent.
    #[test]
    fn updates_match_refreshes() {
        let path = Path::new("data/network.nnue");
        if !path.exists() {
            return;
        }
        let network = Network::load(path).unwrap();
        let starts = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rn2k2r/pp3ppp/2p5/3pP3/8/2N5/PPP2PPP/R3K2R w KQkq d6 0 1",
            "r3k2r/1P4pp/8/2pP4/8/8/p5PP/R3K2R w KQkq c6 0 1",
            "r1bqBb2/pp1pk3/8/4P1pp/8/2P5/PP3P2/R1BK3R b - - 1 11",
        ];
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        let mut cache = RefreshCache::new(&network);
        let mut positions = 0;
        for start in starts {
            for _ in 0..40 {
                let mut pos: Position = from_fen(start).unwrap();
                let mut accumulator = Accumulator::default();
                network.refresh(&pos, &mut accumulator);
                for _ in 0..100 {
                    let moves: Vec<_> = pos
                        .pseudo_legal()
                        .as_slice()
                        .iter()
                        .copied()
                        .filter(|&m| pos.legal(m))
                        .collect();
                    if moves.is_empty() || pos.king(pos.turn()).is_none() {
                        break;
                    }
                    let next = pos.play(moves[(random.next() % moves.len() as u64) as usize]);
                    if next.king(!next.turn()).is_none() || next.king(next.turn()).is_none() {
                        break;
                    }
                    let mut updated = Accumulator::default();
                    network.update(&accumulator, &next, &mut updated, &mut cache);
                    let mut refreshed = Accumulator::default();
                    network.refresh(&next, &mut refreshed);
                    assert_eq!(updated.values, refreshed.values);
                    assert_eq!(updated.psqt, refreshed.psqt);
                    positions += 1;
                    pos = next;
                    accumulator = updated;
                }
            }
        }
        assert!(positions > 5000, "{positions}");
    }

    /// The engine's own evaluations of a few positions, in internal units,
    /// from Atomic-Stockfish's `eval` with the network bundled builds carry
    /// (`atomic_run3b_e202_l05.nnue`); the test passes without checking when
    /// `data/network.nnue` is absent.
    #[test]
    fn matches_the_engine() {
        let path = Path::new("data/network.nnue");
        if !path.exists() {
            return;
        }
        let network = Network::load(path).unwrap();
        let expected = [
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                204,
            ),
            (
                "rnbqkb1r/pppp1ppp/7n/4p1N1/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1",
                424,
            ),
            (
                "2bqkbnr/rp1p2p1/4pp2/7p/3PP3/3Q1N1P/PPP2PP1/R1B2RK1 w k - 0 1",
                552,
            ),
            (
                "rnbqkbnr/pp2p1pp/2pp4/5pN1/4P3/2N5/PPPP1PPP/R1BQKB1R w KQkq - 0 1",
                1341,
            ),
            (
                "rr1RBB2/ppk4p/4p3/5p1P/1b2P1P1/2N5/PPP2P2/4K2R w K - 11 13",
                4139,
            ),
            ("1k6/pp4p1/3p4/5Qbp/3P4/2P5/PP5P/R2K3R w - - 1 10", 3405),
        ];
        for (fen, value) in expected {
            assert_eq!(network.evaluate(&from_fen(fen).unwrap()), value, "{fen}");
        }
    }
}
