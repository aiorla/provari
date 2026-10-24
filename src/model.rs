//! A provability model: the chance that the attacker's win from a position
//! gets proven, from features the search has when it expands the position.
//! Logistic, with one set of weights for the attacker's positions and one for
//! the defender's. It may also read the network's first layer, as the
//! network's dense layers read it.

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::nnue::Transformed;

/// The features, all from the attacker's side, roughly within -3..3.
pub const NAMES: [&str; FEATURES] = [
    "net",
    "net_ahead",
    "net_behind",
    "material",
    "pieces",
    "touching",
    "check",
    "moves",
    "forcing",
    "strong",
    "cheapest",
    "open",
    "good",
    "losing",
    "fifty",
];
pub const FEATURES: usize = 15;
/// The network's first layer: the side to move's half, then the other's.
pub const LAYER: usize = 1024;

pub struct Model {
    attacker: Weights,
    defender: Weights,
}

struct Weights {
    bias: f32,
    weights: [f32; FEATURES],
    /// Weights on the network's first layer, each output taken over 127, if
    /// the model reads it.
    layer: Option<Box<[f32; LAYER]>>,
    /// The share proven among the positions it was fitted to.
    prior: f32,
}

impl Model {
    /// Reads a model file: `{"features": [...], "attacker": {"bias": b,
    /// "weights": [...], "layer": [...]}, "defender": {...}}`, with the
    /// features named and ordered as `NAMES`; `layer`, the weights on the
    /// network's first layer, is optional.
    pub fn load(path: &Path) -> Result<Model, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Model::from_text(&text)
    }

    /// A model from its file's text.
    pub fn from_text(text: &str) -> Result<Model, String> {
        let model: Value = serde_json::from_str(text).map_err(|e| format!("model: {e}"))?;
        let names: Vec<&str> = model["features"]
            .as_array()
            .ok_or("model: no features")?
            .iter()
            .filter_map(Value::as_str)
            .collect();
        if names != NAMES {
            return Err("model: features differ from the prover's".into());
        }
        let weights = |side: &str| -> Result<Weights, String> {
            let side = &model[side];
            let values: Vec<f32> = side["weights"]
                .as_array()
                .ok_or("model: no weights")?
                .iter()
                .filter_map(Value::as_f64)
                .map(|value| value as f32)
                .collect();
            let layer = match side["layer"].as_array() {
                Some(layer) => {
                    let layer: Vec<f32> = layer
                        .iter()
                        .filter_map(Value::as_f64)
                        .map(|value| value as f32 / 127.0)
                        .collect();
                    let layer: Box<[f32; LAYER]> = layer
                        .into_boxed_slice()
                        .try_into()
                        .map_err(|_| "model: wrong number of layer weights")?;
                    Some(layer)
                }
                None => None,
            };
            Ok(Weights {
                layer,
                prior: side["prior"].as_f64().unwrap_or(0.5) as f32,
                bias: side["bias"].as_f64().ok_or("model: no bias")? as f32,
                weights: values
                    .try_into()
                    .map_err(|_| "model: wrong number of weights")?,
            })
        };
        Ok(Model {
            attacker: weights("attacker")?,
            defender: weights("defender")?,
        })
    }

    /// The share proven among the positions the model was fitted to.
    pub fn prior(&self, attacking: bool) -> f32 {
        if attacking {
            self.attacker.prior
        } else {
            self.defender.prior
        }
    }

    /// Whether the model reads the network's first layer.
    pub fn reads_layer(&self) -> bool {
        self.attacker.layer.is_some() || self.defender.layer.is_some()
    }

    /// The chance that the position is proven, given the network's first
    /// layer when the model reads it.
    pub fn predict(
        &self,
        attacking: bool,
        features: &[f32; FEATURES],
        layer: Option<&Transformed>,
    ) -> f32 {
        let weights = if attacking {
            &self.attacker
        } else {
            &self.defender
        };
        let mut sum = weights.bias
            + weights
                .weights
                .iter()
                .zip(features)
                .map(|(weight, feature)| weight * feature)
                .sum::<f32>();
        if let (Some(weights), Some(layer)) = (&weights.layer, layer) {
            sum += dot(weights, layer);
        }
        1.0 / (1.0 + (-sum).exp())
    }
}

/// A dot product in eight lanes, which the compiler turns into vector
/// instructions where a single running sum would stay serial: input i goes to
/// lane i % 8, each lane adds its terms in order, then the lanes are added in
/// order. Only the layer's nonzero chunks of four are read, the even ones
/// into lanes 0-3 and the odd ones into lanes 4-7. A skipped chunk's terms are
/// zeros, and adding a zero leaves a lane as it is (a lane is never -0, as it
/// starts at +0 and a sum of nonzero terms that cancel is +0), so the result
/// is the one all 1024 terms give, to the bit.
fn dot(weights: &[f32; LAYER], layer: &Transformed) -> f32 {
    let terms = |chunk: u16| {
        let at = usize::from(chunk) * 4;
        let weights: &[f32; 4] = weights[at..at + 4].try_into().expect("four weights");
        let values: &[u8; 4] = layer.values[at..at + 4].try_into().expect("four values");
        std::array::from_fn::<f32, 4, _>(|lane| weights[lane] * f32::from(values[lane]))
    };
    let add = |lanes: &mut [f32; 4], chunk: u16| {
        for (lane, term) in lanes.iter_mut().zip(terms(chunk)) {
            *lane += term;
        }
    };
    let [even, odd] = layer.nonzero();
    let both = even.len().min(odd.len());
    let (mut low, mut high) = ([0.0f32; 4], [0.0f32; 4]);
    // The two halves' sums advance side by side, each in its own order.
    for (&a, &b) in even[..both].iter().zip(&odd[..both]) {
        add(&mut low, a);
        add(&mut high, b);
    }
    for &chunk in &even[both..] {
        add(&mut low, chunk);
    }
    for &chunk in &odd[both..] {
        add(&mut high, chunk);
    }
    let lanes = [
        low[0], low[1], low[2], low[3], high[0], high[1], high[2], high[3],
    ];
    lanes.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::{LAYER, dot};
    use crate::nnue::Transformed;
    use crate::nnue::tests::Random;

    /// The dot product over all 1024 terms.
    fn dense(weights: &[f32; LAYER], layer: &[u8; LAYER]) -> f32 {
        let mut lanes = [0.0f32; 8];
        let (weights, _) = weights.as_chunks::<8>();
        let (layer, _) = layer.as_chunks::<8>();
        for (weights, layer) in weights.iter().zip(layer) {
            for ((lane, weight), &value) in lanes.iter_mut().zip(weights).zip(layer) {
                *lane += weight * f32::from(value);
            }
        }
        lanes.iter().sum()
    }

    /// Skipping the zero chunks gives the same sum to the bit, whatever the
    /// weights' signs and the layer's sparsity.
    #[test]
    fn sparse_dot_is_exact() {
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        for trial in 0..3000 {
            let scale = [1e-3, 1.0, 1e3][trial % 3];
            let weights: Box<[f32; LAYER]> = (0..LAYER)
                .map(|_| ((random.next() % 2001) as f32 - 1000.0) / 127.0 * scale)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let halves = random.halves(trial as u64 % 101);
            let mut layer = Transformed::default();
            layer.fill([&halves[0], &halves[1]]);
            let expected = dense(&weights, &layer.values);
            assert_eq!(
                dot(&weights, &layer).to_bits(),
                expected.to_bits(),
                "trial {trial}"
            );
        }
    }
}
