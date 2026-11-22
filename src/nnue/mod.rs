//! The networks that guide the search, in either of two formats, which a
//! file's first bytes tell apart: Atomic-Stockfish's legacy Atomic V1
//! (`legacy`), evaluated exactly as its engine does, and varifish's Atomic
//! networks in Stockfish's design, SFNNv20, pruned for provari (`sfnnv20`,
//! whose `prune` makes them from varifish's files). Both give the dense
//! layers, and the provability model, a first layer of 1024 values, the side
//! to move's half first. A network is a local input, as the tablebases are.

use std::fs;
use std::path::Path;

use shakmaty::{Color, Square};

use crate::board::Position;

mod legacy;
mod sfnnv20;

pub use legacy::Transformed;

/// The first layer's values, both sides': 1024 in either format.
const LAYER: usize = 1024;

pub enum Network {
    /// Legacy Atomic V1, as Atomic-Stockfish's networks.
    Legacy(legacy::Network),
    /// SFNNv20, varifish's Atomic networks, pruned.
    Sfnnv20(Box<sfnnv20::Network>),
}

/// A position's first layer, for each side's perspective, with the pieces it
/// was computed from: its network's. A search keeps one a level, all of one
/// network, so the smaller is not boxed.
#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Accumulator {
    Legacy(legacy::Accumulator),
    Sfnnv20(sfnnv20::Accumulator),
}

/// A search's cache of first layers by king square, for its network's
/// refreshes.
pub enum RefreshCache {
    Legacy(legacy::RefreshCache),
    Sfnnv20(sfnnv20::RefreshCache),
}

impl Default for Accumulator {
    /// A first layer of no position, which either network makes its own
    /// when it first writes to it.
    fn default() -> Accumulator {
        Accumulator::Legacy(legacy::Accumulator::default())
    }
}

impl Accumulator {
    /// The first layer to write as the legacy network's.
    fn legacy(&mut self) -> &mut legacy::Accumulator {
        if !matches!(self, Accumulator::Legacy(_)) {
            *self = Accumulator::Legacy(legacy::Accumulator::default());
        }
        match self {
            Accumulator::Legacy(accumulator) => accumulator,
            Accumulator::Sfnnv20(_) => unreachable!(),
        }
    }

    /// The first layer to write as an SFNNv20 network's.
    fn sfnnv20(&mut self) -> &mut sfnnv20::Accumulator {
        if !matches!(self, Accumulator::Sfnnv20(_)) {
            *self = Accumulator::Sfnnv20(sfnnv20::Accumulator::default());
        }
        match self {
            Accumulator::Sfnnv20(accumulator) => accumulator,
            Accumulator::Legacy(_) => unreachable!(),
        }
    }
}

impl RefreshCache {
    /// A network's cache: every entry starts as the empty board's.
    pub fn new(network: &Network) -> RefreshCache {
        match network {
            Network::Legacy(network) => RefreshCache::Legacy(legacy::RefreshCache::new(network)),
            Network::Sfnnv20(network) => RefreshCache::Sfnnv20(sfnnv20::RefreshCache::new(network)),
        }
    }
}

/// A first layer or a cache handed to the network it is not of: the search
/// keeps one network's.
const MIXED: &str = "a first layer of another network";

/// A pruned network for provari, the file's bytes, from one of varifish's
/// SFNNv20 files: the 256 outputs `keep` lists, in that order, and the
/// factor, a numerator and a denominator, that brings its values to run3b's
/// units (`sfnnv20::prune`).
pub fn prune(
    varifish: &[u8],
    keep: &[usize],
    scale: (i32, i32),
    description: &str,
) -> Result<Vec<u8>, String> {
    sfnnv20::prune(varifish, keep, scale, description)
}

impl Network {
    pub fn load(path: &Path) -> Result<Network, String> {
        let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
        Network::from_bytes(&bytes)
    }

    /// A network from its file's bytes, in the format its version says.
    pub fn from_bytes(bytes: &[u8]) -> Result<Network, String> {
        match bytes
            .first_chunk()
            .map(|&version| u32::from_le_bytes(version))
        {
            Some(legacy::VERSION) => legacy::Network::from_bytes(bytes).map(Network::Legacy),
            Some(sfnnv20::PRUNED | sfnnv20::VERSION) => sfnnv20::Network::from_bytes(bytes)
                .map(|network| Network::Sfnnv20(Box::new(network))),
            _ => Err("neither a legacy Atomic V1 network nor a pruned SFNNv20 Atomic one".into()),
        }
    }

    /// The evaluation for the side to move, in the engine's internal units
    /// (208 to a pawn in Atomic-Stockfish's networks), computed from scratch.
    pub fn evaluate(&self, pos: &Position) -> i32 {
        let mut accumulator = Accumulator::default();
        self.refresh(pos, &mut accumulator);
        self.output(&accumulator, pos)
    }

    /// The first layer for the side to move, computed from scratch.
    pub fn evaluate_layer(&self, pos: &Position) -> [u8; LAYER] {
        let mut accumulator = Accumulator::default();
        self.refresh(pos, &mut accumulator);
        let mut transformed = Transformed::default();
        self.transform(&accumulator, pos, &mut transformed);
        transformed.values
    }

    /// Computes a position's first layer from scratch.
    pub fn refresh(&self, pos: &Position, accumulator: &mut Accumulator) {
        match self {
            Network::Legacy(network) => network.refresh(pos, accumulator.legacy()),
            Network::Sfnnv20(network) => network.refresh(pos, accumulator.sfnnv20()),
        }
    }

    /// Computes a position's first layer from its parent's: only what the
    /// move changed, but for a side that starts over, as after a move of its
    /// own king, from the cache's entry for the king's new square.
    pub fn update(
        &self,
        parent: &Accumulator,
        pos: &Position,
        accumulator: &mut Accumulator,
        cache: &mut RefreshCache,
    ) {
        match (self, parent, cache) {
            (
                Network::Legacy(network),
                Accumulator::Legacy(parent),
                RefreshCache::Legacy(cache),
            ) => {
                network.update(parent, pos, accumulator.legacy(), cache);
            }
            (
                Network::Sfnnv20(network),
                Accumulator::Sfnnv20(parent),
                RefreshCache::Sfnnv20(cache),
            ) => network.update(parent, pos, accumulator.sfnnv20(), cache),
            _ => unreachable!("{MIXED}"),
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
        match (self, parent) {
            (Network::Legacy(network), Accumulator::Legacy(parent)) => {
                network.update_quiet(parent, from, to, accumulator.legacy());
            }
            (Network::Sfnnv20(network), Accumulator::Sfnnv20(parent)) => {
                network.update_quiet(parent, from, to, accumulator.sfnnv20());
            }
            _ => unreachable!("{MIXED}"),
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
        match (self, accumulator) {
            (Network::Legacy(network), Accumulator::Legacy(accumulator)) => {
                network.output_for(transformed, accumulator, pieces, us)
            }
            (Network::Sfnnv20(network), Accumulator::Sfnnv20(accumulator)) => {
                network.output_for(transformed, accumulator, us)
            }
            _ => unreachable!("{MIXED}"),
        }
    }

    /// The first layer as the dense layers read it, into `out`: the side to
    /// move's half, then the other side's, each output within 0-127; and
    /// which chunks of four outputs are not all zero.
    pub fn transform(&self, accumulator: &Accumulator, pos: &Position, out: &mut Transformed) {
        self.transform_for(accumulator, pos.turn(), out);
    }

    /// `transform` with `us` to move.
    pub fn transform_for(&self, accumulator: &Accumulator, us: Color, out: &mut Transformed) {
        match (self, accumulator) {
            (Network::Legacy(network), Accumulator::Legacy(accumulator)) => {
                network.transform_for(accumulator, us, out);
            }
            (Network::Sfnnv20(network), Accumulator::Sfnnv20(accumulator)) => {
                network.transform_for(accumulator, us, out);
            }
            _ => unreachable!("{MIXED}"),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) use super::legacy::tests::Random;

    /// A network in varifish's format, of generated weights.
    pub(crate) fn sfnnv20() -> super::Network {
        super::Network::from_bytes(&super::sfnnv20::tests::pruned()).unwrap()
    }
}
