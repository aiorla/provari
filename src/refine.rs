//! `provari refine`: a proof made smaller by trying other moves. Where its
//! subproofs are largest first, each attacker position's other moves are
//! searched with a budget in proportion to its subproof, and a move whose
//! proof comes out smaller replaces it, if the whole proof then still holds
//! and shrinks. It tries harder rather than claiming the smallest proof.
//! Proofs are maps from attacker boards to their winning moves, measured by
//! the proof's walk (`walk`). Shorter wins are `--shorten`'s, which does
//! that better within the search.

use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use shakmaty::{Color, Move, Position as _, variant::Atomic};

use crate::dfpn::{Dfpn, Guide};
use crate::format;
use crate::search::{self, Status, hash, key, status};
use crate::tablebase::{Tablebase, Verdict};
use crate::walk::{Measure, Moves, Walk, measure, measure_after, winning_capture};

/// What a refinement makes smaller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    /// The proof's tree, its attacker positions counted along every line.
    Size,
    /// The distinct attacker positions it lists, which follow a search's
    /// work best.
    Positions,
}

/// A refinement's settings.
#[derive(Clone, Copy)]
pub struct Settings<'a> {
    pub objective: Objective,
    /// The budget of all its searches together, whichever comes first: one
    /// at least. Nodes alone repeat a run exactly.
    pub nodes: Option<u64>,
    pub seconds: Option<f64>,
    pub seed: u64,
    pub guide: Guide<'a>,
}

/// Passes of tries, each from the largest subproof down: the 16 largest
/// with four times their subtree's nodes, where a smaller proof saves most,
/// within half the budget; then every one with as many nodes as its subtree,
/// where cheap gains lie; then every one with four and sixteen times as
/// many. Each: the budget's factor, the most positions, and the share of the
/// budget spent by its end, in percent.
const PASSES: [(u64, usize, u64); 4] = [
    (4, 16, 50),
    (1, usize::MAX, 100),
    (4, usize::MAX, 100),
    (16, usize::MAX, 100),
];

/// Makes the proof of `attacker`'s win from `root`, either side to move
/// there, smaller within the budget: the refined proof's moves, and what
/// changed as JSON. The refined proof is for the checker like any other.
pub fn refine(
    root: &Atomic,
    attacker: Color,
    mut moves: Moves,
    tablebase: Option<&Tablebase>,
    settings: &Settings,
) -> Result<(Moves, Value), String> {
    if settings.nodes.is_none() && settings.seconds.is_none() {
        return Err("refine needs --nodes or --seconds, the budget of all its tries".into());
    }
    let started = Instant::now();
    // From a defender's root, as a sampled position's proof starts, the
    // measures of its replies' subproofs.
    let walk_from_root = |moves: &Moves| -> Result<(Walk, Measure), String> {
        let mut walk = Walk::default();
        let measure = if root.turn() == attacker {
            measure(root, attacker, moves, tablebase, &mut walk)?
        } else {
            measure_after(root, attacker, moves, tablebase, &mut walk)?
        };
        Ok((walk, measure))
    };
    let (mut walk, before) = walk_from_root(&moves)?;
    let mut now = before;
    let listed_before = size(root, attacker, &moves, tablebase);
    let mut listed_now = listed_before;
    let distinct = settings.objective == Objective::Positions;
    if let Some(seconds) = settings.seconds
        && !(0.0..=1e9).contains(&seconds)
    {
        return Err("bad seconds".into());
    }
    let deadline = started + Duration::from_secs_f64(settings.seconds.unwrap_or(3.2e7));
    let total = settings.nodes;
    // A pass's end: its share of each budget given, of the nodes and of the
    // time, and whether nodes are left at all.
    let pass_cap = |share: u64| total.map_or(u64::MAX, |total| total / 100 * share);
    let pass_end = |share: u64| match settings.seconds {
        Some(seconds) => {
            deadline.min(started + Duration::from_secs_f64(seconds * share as f64 / 100.0))
        }
        None => deadline,
    };
    let left = |nodes: u64| total.is_none_or(|total| nodes < total);
    let mut tried = HashSet::new();
    let mut cached: Option<(usize, Dfpn<'_>)> = None;
    let (mut pass, mut in_pass) = (0, 0);
    let (mut tries, mut replaced, mut nodes) = (0u64, 0u64, 0u64);
    'positions: while Instant::now() < deadline && left(nodes) {
        let (factor, most, share) = PASSES[pass];
        let (cap, end) = (pass_cap(share), pass_end(share));
        // The attacker position with the largest subproof now, untried with
        // this budget.
        let next_candidate = (in_pass < most && nodes < cap && Instant::now() < end)
            .then(|| {
                walk.measures
                    .iter()
                    .filter(|(key, measure)| measure.tree >= 2 && !tried.contains(&(**key, factor)))
                    .max_by_key(|(key, measure)| (measure.tree, **key))
                    .map(|(key, measure)| (*key, *measure))
            })
            .flatten();
        let Some((here, current)) = next_candidate else {
            if pass + 1 == PASSES.len() {
                break;
            }
            (pass, in_pass) = (pass + 1, 0);
            continue;
        };
        in_pass += 1;
        tried.insert((here, factor));
        let Some(pos) = walk.positions.get(&here).cloned() else {
            continue;
        };
        let Some(Some(chosen)) = moves.get(&here).copied() else {
            continue;
        };
        let mut best: Option<(Move, Moves, u64)> = None;
        let current_size = if distinct {
            size(&pos, attacker, &moves, tablebase) as u64
        } else {
            current.tree
        };
        let mut spent = false;
        for m in pos.legal_moves() {
            if m == chosen {
                continue;
            }
            let mut next = pos.clone();
            next.play_unchecked(m);
            if !matches!(status(&next, attacker), Status::Open) {
                continue;
            }
            if Instant::now() >= deadline || !left(nodes) {
                spent = true;
                break;
            }
            if nodes >= cap || Instant::now() >= end {
                break;
            }
            let budget = current
                .tree
                .saturating_mul(factor)
                .max(4096)
                .min(cap - nodes);
            let filled = budget.div_ceil(50_000) as usize;
            let table = filled.next_power_of_two().clamp(16, 4096);
            let search = match &mut cached {
                Some((size, search)) if *size == table => {
                    search.clear();
                    search.nodes = 0;
                    search.set_deadline(end);
                    search
                }
                _ => {
                    // Drop the old allocation before taking a different size.
                    cached = None;
                    let search =
                        Dfpn::new(&next, tablebase, settings.guide, end, (table, table / 2));
                    &mut cached.insert((table, search)).1
                }
            };
            search.set_limit(budget);
            search.set_attacker(attacker);
            search.set_seed(settings.seed);
            let found = search.solve(&next).is_some();
            tries += 1;
            nodes += search.nodes;
            if !found {
                continue;
            }
            let mut found_moves = Moves::new();
            collect(
                &next,
                attacker,
                tablebase,
                |p, k| search.winning_move(p, k, false),
                &mut found_moves,
            )?;
            let size = if distinct {
                let mut local = found_moves.clone();
                local.insert(here, Some(m));
                size(&pos, attacker, &local, tablebase) as u64
            } else {
                let mut alone = Walk::default();
                measure_after(&next, attacker, &found_moves, tablebase, &mut alone)?.tree + 1
            };
            if size < current_size && best.as_ref().is_none_or(|best| size < best.2) {
                best = Some((m, found_moves, size));
            }
        }
        if spent {
            break 'positions;
        }
        if let Some((m, found_moves, _)) = best {
            let mut trial = moves.clone();
            for (key, found) in found_moves {
                trial.entry(key).or_insert(found);
            }
            trial.insert(here, Some(m));
            // Kept only if the whole proof still holds and shrinks: positions
            // it shares keep their old moves, and a repetition voids it.
            let trial_listed = if distinct {
                size(root, attacker, &trial, tablebase)
            } else {
                0
            };
            if let Ok((trial_walk, measure)) = walk_from_root(&trial)
                && if distinct {
                    trial_listed < listed_now
                } else {
                    measure.tree < now.tree
                }
            {
                moves = trial;
                walk = trial_walk;
                now = measure;
                if distinct {
                    listed_now = trial_listed;
                }
                replaced += 1;
            }
        }
    }
    // The positions the proof lists, as JSON lines would: the same measure
    // as before refining, whatever the format written.
    let listed = size(root, attacker, &moves, tablebase);
    let mut result = json!({
        "objective": match settings.objective {
            Objective::Size => "size",
            Objective::Positions => "positions",
        },
        "positions_before": listed_before,
        "positions_after": listed,
        "tree_before": before.tree,
        "tree_after": now.tree,
        "length_before": before.length,
        "length_after": now.length,
        "tries": tries,
        "replaced": replaced,
        "nodes": nodes,
        "ms": started.elapsed().as_millis() as u64,
    });
    if let Some(total) = total {
        result["node_limit"] = json!(total);
    }
    if let Some(seconds) = settings.seconds {
        result["seconds"] = json!(seconds);
    }
    Ok((moves, result))
}

/// Writes a proof of `attacker`'s win from its moves in the compact format
/// (`format`).
pub fn write(
    root: &Atomic,
    attacker: Color,
    tablebase: Option<&Tablebase>,
    path: &Path,
    moves: &Moves,
) -> Result<(), String> {
    let (header, indices) = format::walk(root, attacker, tablebase, |pos| {
        moves
            .get(&pos.hash())
            .copied()
            .flatten()
            .and_then(|m| format::board_move(pos, &m))
    })?;
    format::save(path, &format::file(&header, &indices)?)
}

/// Adds to `moves` the winning moves of a proof found from `start`, an
/// attacker or a defender position, as `winning_move` gives them, for the
/// positions `moves` does not hold yet.
pub fn collect(
    start: &Atomic,
    attacker: Color,
    tablebase: Option<&Tablebase>,
    winning_move: impl Fn(&Atomic, &search::Key) -> Option<Move>,
    moves: &mut Moves,
) -> Result<(), String> {
    let probe = |pos: &Atomic| tablebase.and_then(|tables| tables.probe(pos));
    let mut seen = HashSet::new();
    let mut pending = Vec::new();
    if start.turn() == attacker {
        pending.push(start.clone());
    } else {
        replies(start, attacker, tablebase, &mut pending);
    }
    while let Some(pos) = pending.pop() {
        let here = hash(&pos);
        if !seen.insert(key(&pos)) {
            continue;
        }
        if probe(&pos) == Some(Verdict::Win) {
            moves.entry(here).or_insert(None);
            continue;
        }
        if winning_capture(&pos, attacker).is_some() {
            continue;
        }
        let m = match moves.get(&here) {
            Some(known) => *known,
            None => {
                let m = winning_move(&pos, &key(&pos)).ok_or("no winning move recorded")?;
                moves.insert(here, Some(m));
                Some(m)
            }
        };
        let Some(m) = m else { continue };
        let mut after = pos.clone();
        after.play_unchecked(m);
        replies(&after, attacker, tablebase, &mut pending);
    }
    Ok(())
}

/// The distinct attacker positions a proof lists from its root, as its file
/// would: tablebase wins included, wins by a capture at once left out. From
/// a defender's root, those its replies' subproofs list.
pub fn size(root: &Atomic, attacker: Color, moves: &Moves, tablebase: Option<&Tablebase>) -> usize {
    let probe = |pos: &Atomic| tablebase.and_then(|tables| tables.probe(pos));
    let mut seen = HashSet::new();
    let mut listed = 0;
    let mut pending = Vec::new();
    if root.turn() == attacker {
        pending.push(root.clone());
    } else {
        replies(root, attacker, tablebase, &mut pending);
    }
    while let Some(pos) = pending.pop() {
        if !seen.insert(key(&pos)) || winning_capture(&pos, attacker).is_some() {
            continue;
        }
        listed += 1;
        if probe(&pos) == Some(Verdict::Win) {
            continue;
        }
        let Some(Some(m)) = moves.get(&hash(&pos)) else {
            continue;
        };
        let mut after = pos.clone();
        after.play_unchecked(*m);
        replies(&after, attacker, tablebase, &mut pending);
    }
    listed
}

/// A defender position's replies, each an attacker position to prove: none
/// once the game is over or the tablebase has the defender lost.
fn replies(
    after: &Atomic,
    attacker: Color,
    tablebase: Option<&Tablebase>,
    pending: &mut Vec<Atomic>,
) {
    if !matches!(status(after, attacker), Status::Open)
        || tablebase.and_then(|tables| tables.probe(after)) == Some(Verdict::Loss)
    {
        return;
    }
    for reply in after.legal_moves() {
        let mut next = after.clone();
        next.play_unchecked(reply);
        if matches!(status(&next, attacker), Status::Open) {
            pending.push(next);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use serde_json::Value;
    use shakmaty::{Color, EnPassantMode, Position as _, fen::Fen};

    use super::{Objective, Settings, collect, refine, write};
    use crate::dfpn::{Dfpn, Guide};
    use crate::walk::Moves;

    /// A win in four that the search proves in some 70 nodes.
    const WIN_IN_FOUR: &str =
        "r1bqkbnr/1p1p2pp/p1n2p2/1B2p3/3PP3/2P2N2/PP3PPP/R1BQK2R w KQkq - 0 1";

    /// A small budget and the hand-crafted guide.
    fn settings(objective: Objective) -> Settings<'static> {
        Settings {
            objective,
            nodes: Some(20_000),
            seconds: None,
            seed: 0,
            guide: Guide::Hand,
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("provari-refine-{name}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The search's proof of `fen`'s win for the side to move, as moves.
    fn found(fen: &str) -> (shakmaty::variant::Atomic, Moves) {
        let root = crate::position(fen).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut search = Dfpn::new(&root, None, Guide::None, deadline, (16, 8));
        assert!(search.solve(&root).is_some());
        let mut moves = Moves::new();
        collect(
            &root,
            root.turn(),
            None,
            |pos, key| search.winning_move(pos, key, false),
            &mut moves,
        )
        .unwrap();
        (root, moves)
    }

    /// The win in four refined by each objective with a small budget into
    /// proofs that the checker accepts, no worse by the objective's own
    /// measure.
    #[test]
    fn refines_into_proofs_that_check() {
        let dir = scratch("objectives");
        let (root, moves) = found(WIN_IN_FOUR);
        for (objective, measure) in [
            (Objective::Size, "tree"),
            (Objective::Positions, "positions"),
        ] {
            let (refined, result) = refine(
                &root,
                root.turn(),
                moves.clone(),
                None,
                &settings(objective),
            )
            .unwrap();
            let number = |when: &str| result[format!("{measure}_{when}")].as_u64().unwrap();
            assert!(number("after") <= number("before"), "{result}");
            let out = dir.join(format!("{measure}.pvr"));
            write(&root, root.turn(), None, &out, &refined).unwrap();
            assert!(crate::check::run(&[out.display().to_string()]).unwrap());
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    /// A proof from the defender's move, as a sampled position's starts, is
    /// refined as any other into a proof the checker accepts, no larger by
    /// its objective, still rooted there.
    #[test]
    fn refines_from_a_defender_position() {
        let root = crate::position(WIN_IN_FOUR).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut search = Dfpn::new(&root, None, Guide::None, deadline, (16, 8));
        assert!(search.solve(&root).is_some());
        let m = search
            .winning_move(&root, &crate::search::key(&root), false)
            .unwrap();
        let mut after = root.clone();
        after.play_unchecked(m);
        let mut moves = Moves::new();
        collect(
            &after,
            Color::White,
            None,
            |p, k| search.winning_move(p, k, false),
            &mut moves,
        )
        .unwrap();
        let dir = scratch("defender");
        for (objective, measure) in [
            (Objective::Size, "tree"),
            (Objective::Positions, "positions"),
        ] {
            let (refined, result) = refine(
                &after,
                Color::White,
                moves.clone(),
                None,
                &settings(objective),
            )
            .unwrap();
            let (before, after_refining) = (
                result[format!("{measure}_before")].as_u64().unwrap(),
                result[format!("{measure}_after")].as_u64().unwrap(),
            );
            assert!(after_refining <= before, "{result}");
            assert!(result["tries"].as_u64().unwrap() > 0, "{result}");
            let out = dir.join(format!("{measure}.pvr"));
            write(&after, Color::White, None, &out, &refined).unwrap();
            let bytes = fs::read(&out).unwrap();
            let checked = crate::check::check_proof(measure, &bytes, None);
            assert_eq!(checked["valid"], true, "{checked}");
            let header = bytes.split(|&b| b == b'\n').next().unwrap();
            let header: Value = serde_json::from_slice(header).unwrap();
            let fen = Fen::from_position(&after, EnPassantMode::Legal).to_string();
            assert_eq!(header["root"], fen);
            assert_eq!(header["attacker"], "white");
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn needs_a_budget() {
        let (root, moves) = found(WIN_IN_FOUR);
        let mut settings = settings(Objective::Size);
        settings.nodes = None;
        assert!(refine(&root, root.turn(), moves, None, &settings).is_err());
    }

    #[test]
    fn refuses_invalid_seconds() {
        let (root, moves) = found(WIN_IN_FOUR);
        for bad in [-1.0, f64::NAN, f64::INFINITY, 2e9] {
            let mut settings = settings(Objective::Size);
            settings.seconds = Some(bad);
            assert!(
                refine(&root, root.turn(), moves.clone(), None, &settings).is_err(),
                "accepted {bad}"
            );
        }
    }
}
