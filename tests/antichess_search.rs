//! Exhaustive cross-checks of provari's Antichess search, proof writer,
//! rules, guidance, and (with `antichess-tablebases`) depth-to-win tables.

use std::collections::{HashMap, HashSet};

use shakmaty::{Color, Position, variant::Antichess};

use provari::antichess::exhaustive;
use provari::antichess::format::{self, Header, Step};
use provari::antichess::rules::{self, Status};
use provari::antichess::search::{Dfpn, Limits, Outcome, Stop};
#[cfg(feature = "antichess-tablebases")]
use provari::antichess::search::{Settled, settle};
#[cfg(feature = "antichess-tablebases")]
use provari::antichess::tablebases::{self, Entry, Tablebases, WHOLE_SET};

/// The tables the tests read, from `PROVARI_ANTICHESS_TABLEBASES`: tolius's
/// whole set, or the few that antichess-tb-sys's repository holds for its own
/// tests. Without it, the tests that need them pass without them.
#[cfg(feature = "antichess-tablebases")]
fn tables() -> Option<Tablebases> {
    let dir = std::path::PathBuf::from(std::env::var_os("PROVARI_ANTICHESS_TABLEBASES")?);
    Some(Tablebases::open(&dir).unwrap())
}

fn position(fen: &str) -> Antichess {
    rules::parse_fen(fen).unwrap()
}

/// The proof of `root` within `most` that the search left, written in the
/// compact format and read back: its header and the positions it proves.
fn proof(dfpn: &Dfpn, root: &Antichess, most: u32) -> (Header, Vec<Step>) {
    let written = format::from_search(dfpn, root, most).unwrap();
    let (header, indices) = format::decode(&written.bytes).unwrap();
    assert_eq!(header, written.header);
    let steps = format::steps(&header, &indices).unwrap();
    assert_eq!(steps.len() as u64, written.walked);
    (header, steps)
}

fn solve(dfpn: &mut Dfpn, pos: &Antichess, most: u32) -> Outcome {
    dfpn.solve(pos, most, Limits::default())
}

/// Walks a proof's steps from the root, independently of the search, and
/// returns the longest line in White moves. A White position met again at the
/// same counter with as many moves left ends its lines as before, so it is
/// walked once, and its lines checked for repeats then: a long win by the
/// tables has few positions but far too many lines to walk one by one.
fn verify(root: &Antichess, most: u32, steps: &[Step]) -> u32 {
    let mut by_position: HashMap<&str, Vec<&Step>> = HashMap::new();
    for step in steps {
        by_position.entry(step.fen.as_str()).or_default().push(step);
    }
    let mut line = HashSet::new();
    white(root, most, &by_position, &mut line, &mut HashMap::new())
}

fn white(
    pos: &Antichess,
    budget: u32,
    steps: &HashMap<&str, Vec<&Step>>,
    line: &mut HashSet<u64>,
    walked: &mut HashMap<(u64, u32, u32), u32>,
) -> u32 {
    let legal = pos.legal_moves();
    match rules::status(pos, &legal) {
        Status::WhiteWon => return 0,
        Status::Open => {}
        other => panic!("{other:?} in {}", rules::fen(pos)),
    }
    let counter = pos.halfmoves();
    let key = rules::key(pos);
    assert!(!line.contains(&key), "a line repeats {}", rules::fen(pos));
    if let Some(&longest) = walked.get(&(key, counter, budget)) {
        return longest;
    }
    let fen = rules::short_fen(pos);
    let step = steps
        .get(fen.as_str())
        .into_iter()
        .flatten()
        .filter(|step| step.counter >= counter && step.length <= budget)
        .min_by_key(|step| step.length)
        .unwrap_or_else(|| panic!("no step for {fen} within {budget}"));
    line.insert(key);
    let m = rules::parse_uci(pos, &step.m).unwrap();
    let mut after = pos.clone();
    after.play_unchecked(m);
    let replies = after.legal_moves();
    assert_eq!(rules::status(&after, &replies), Status::Open);
    let mut longest = 0;
    for &reply in &replies {
        let mut next = after.clone();
        next.play_unchecked(reply);
        longest = longest.max(1 + white(&next, step.length - 1, steps, line, walked));
    }
    line.remove(&key);
    assert!(longest <= step.length);
    walked.insert((key, counter, budget), longest);
    longest
}

/// Proof steps name positions as python-chess keys them.
#[test]
fn short_fens_match_python_chess_keys() {
    let start = position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let after = |moves: &[&str]| {
        let mut pos = start.clone();
        for text in moves {
            let m = rules::parse_uci(&pos, text).unwrap();
            pos.play_unchecked(m);
        }
        rules::short_fen(&pos)
    };
    let named = [
        (&[][..], "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - -"),
        (
            &["e2e4"],
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b - -",
        ),
        (
            &["e2e4", "d7d5"],
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w - -",
        ),
        (
            &["e2e4", "h7h6", "e4e5", "d7d5"],
            "rnbqkbnr/ppp1ppp1/7p/3pP3/8/8/PPPP1PPP/RNBQKBNR w - d6",
        ),
    ];
    for (moves, fen) in named {
        assert_eq!(after(moves), fen);
    }
}

#[test]
fn a_position_already_won_needs_no_move() {
    let mut dfpn = Dfpn::new(1, 1);
    let won = position("8/8/8/8/8/8/8/k7 w - - 0 1");
    assert_eq!(solve(&mut dfpn, &won, 3), Outcome::Won { need: 0 });
    let stalemated = position("8/8/8/8/8/p7/P7/8 w - - 0 1");
    assert_eq!(solve(&mut dfpn, &stalemated, 0), Outcome::Won { need: 0 });
}

#[test]
fn giving_away_the_last_piece_wins_in_one() {
    let mut dfpn = Dfpn::new(1, 1);
    let root = position("7r/8/8/8/8/8/8/R7 w - - 0 1");
    assert_eq!(solve(&mut dfpn, &root, 0), Outcome::NotWithin);
    assert_eq!(solve(&mut dfpn, &root, 3), Outcome::Won { need: 1 });
    let (header, steps) = proof(&dfpn, &root, 3);
    assert_eq!((header.moves, header.positions), (1, 1));
    assert_eq!(steps.len(), 1);
    assert!(["a1a8", "a1h1"].contains(&steps[0].m.as_str()));
    assert_eq!(verify(&root, 3, &steps), 1);
}

#[test]
fn the_fifty_move_rule_draws_at_a_hundred() {
    let mut dfpn = Dfpn::new(1, 1);
    // The proof found at a low counter holds up to 98, and no higher.
    let early = position("7r/8/8/8/8/8/8/R7 w - - 0 50");
    assert_eq!(solve(&mut dfpn, &early, 3), Outcome::Won { need: 1 });
    let (header, steps) = proof(&dfpn, &early, 3);
    assert_eq!(header.root, "7r/8/8/8/8/8/8/R7 w - - 98 50");
    assert_eq!(steps[0].counter, 98);
    let last_chance = position("7r/8/8/8/8/8/8/R7 w - - 98 50");
    assert_eq!(solve(&mut dfpn, &last_chance, 3), Outcome::Won { need: 1 });
    let too_late = position("7r/8/8/8/8/8/8/R7 w - - 99 50");
    assert_eq!(solve(&mut dfpn, &too_late, 3), Outcome::NotWithin);
    assert_eq!(exhaustive::fewest(&too_late, 3), None);
}

/// Lichess's insufficient material, which leaves the side to move unable
/// ever to capture.
#[test]
fn insufficient_material_is_lichess_rule() {
    let cases = [
        // Bishops alone, on opposite colors or not.
        ("8/8/8/8/8/8/b7/B7 w - - 0 1", true),
        ("8/8/8/8/8/8/1b6/B7 w - - 0 1", false),
        // Pawns stuck behind pawns on their own bishops' color.
        ("8/7b/8/5p2/5P2/1p6/1P6/B7 b - - 0 1", true),
        ("8/7b/8/5p2/8/1p3P2/1P6/B7 w - - 0 1", false),
        ("2b5/8/4p3/3pP3/3P4/8/8/2B5 w - - 0 1", true),
        // The same, but White can take en passant.
        ("2b5/8/4p3/3pP3/3P4/8/8/2B5 w - d6 0 1", false),
        // Pawns on the other side's bishops' color, bishops on one color, or
        // another piece.
        ("8/7b/8/2p5/2P5/8/8/B7 b - - 0 1", false),
        ("8/7b/8/5p2/5P2/1p6/1P6/1B6 b - - 0 1", false),
        ("8/7b/8/5p2/5P2/1p6/1P6/BN6 b - - 0 1", false),
    ];
    for (fen, drawn) in cases {
        assert_eq!(rules::insufficient_material(&position(fen)), drawn, "{fen}");
    }
}

/// White's only move, 1. f4, leaves Lichess's insufficient material, a draw
/// before White, left without a move, would win.
#[test]
fn stuck_pawns_and_opposite_bishops_draw() {
    let root = position("8/7b/8/5p2/8/1p3P2/1P6/B7 w - - 0 1");
    let mut after = root.clone();
    after.play_unchecked(rules::parse_uci(&root, "f3f4").unwrap());
    assert_eq!(
        rules::status(&after, &after.legal_moves()),
        Status::Insufficient
    );
    let mut dfpn = Dfpn::new(1, 1);
    assert_eq!(solve(&mut dfpn, &root, 3), Outcome::NotWithin);
    assert_eq!(exhaustive::fewest(&root, 3), None);
}

/// A small random number generator, so that the tests need no crate.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Random legal positions with White to move, up to `most_pieces` a side,
/// and 50-move counters from `counters`.
fn random_positions(
    seed: u64,
    count: usize,
    most_pieces: u64,
    counters: std::ops::RangeInclusive<u64>,
) -> Vec<Antichess> {
    let mut random = Random(seed);
    let mut positions = Vec::new();
    while positions.len() < count {
        let mut board = [None; 64];
        for color in [Color::White, Color::Black] {
            for _ in 0..1 + random.below(most_pieces) {
                let square = random.below(64) as usize;
                let rank = square / 8;
                let role = "pnbrqk".as_bytes()[random.below(6) as usize] as char;
                if role == 'p' && (rank == 0 || rank == 7) {
                    continue;
                }
                board[square] = Some(match color {
                    Color::White => role.to_ascii_uppercase(),
                    Color::Black => role,
                });
            }
        }
        let counter = counters.start() + random.below(counters.end() - counters.start() + 1);
        if let Ok(pos) = rules::parse_fen(&fen(&board, counter)) {
            positions.push(pos);
        }
    }
    positions
}

/// The FEN of a board of pieces by square, a1 first, with White to move at
/// 50-move counter `counter`.
fn fen(board: &[Option<char>; 64], counter: u64) -> String {
    let mut fen = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            match board[rank * 8 + file] {
                Some(piece) => {
                    if empty > 0 {
                        fen.push_str(&empty.to_string());
                        empty = 0;
                    }
                    fen.push(piece);
                }
                None => empty += 1,
            }
        }
        if empty > 0 {
            fen.push_str(&empty.to_string());
        }
        if rank > 0 {
            fen.push('/');
        }
    }
    fen.push_str(&format!(" w - - {counter} 60"));
    fen
}

/// Random positions with White to move of the pieces one of `sides` gives,
/// White's in capitals, each on its own square, pawns off the back ranks.
#[cfg(feature = "antichess-tablebases")]
fn random_material(
    seed: u64,
    count: usize,
    sides: &[&str],
    counters: std::ops::RangeInclusive<u64>,
) -> Vec<Antichess> {
    let mut random = Random(seed);
    let mut positions = Vec::new();
    while positions.len() < count {
        let mut board = [None; 64];
        for piece in sides[random.below(sides.len() as u64) as usize].chars() {
            loop {
                let square = random.below(64) as usize;
                let rank = square / 8;
                if board[square].is_none()
                    && !(piece.eq_ignore_ascii_case(&'p') && (rank == 0 || rank == 7))
                {
                    board[square] = Some(piece);
                    break;
                }
            }
        }
        let counter = counters.start() + random.below(counters.end() - counters.start() + 1);
        if let Ok(pos) = rules::parse_fen(&fen(&board, counter)) {
            positions.push(pos);
        }
    }
    positions
}

/// Checks the search against exhaustive search on random positions, and
/// every proof it finds; returns the wins found.
fn cross_check(positions: &[Antichess], most: u32) -> usize {
    cross_check_with(&mut Dfpn::new(4, 12), positions, most)
}

fn cross_check_with(dfpn: &mut Dfpn, positions: &[Antichess], most: u32) -> usize {
    let mut won = 0;
    for (index, root) in positions.iter().enumerate() {
        let fewest = exhaustive::fewest(root, most);
        for most in 0..=most {
            // Alternate between a fresh search and one that keeps what
            // earlier positions taught it.
            if index % 2 == 0 {
                dfpn.clear();
            }
            let outcome = solve(dfpn, root, most);
            match fewest {
                Some(fewest) if fewest <= most => {
                    let Outcome::Won { need } = outcome else {
                        panic!(
                            "{} wins in {fewest}, not found within {most}",
                            rules::fen(root)
                        );
                    };
                    assert!(fewest <= need && need <= most, "{}", rules::fen(root));
                    let (header, steps) = proof(dfpn, root, most);
                    let longest = verify(root, most, &steps);
                    assert!(fewest <= longest && longest <= header.moves);
                    assert!(header.moves <= need, "{}", rules::fen(root));
                    won += 1;
                }
                _ => assert_eq!(outcome, Outcome::NotWithin, "{}", rules::fen(root)),
            }
        }
    }
    won
}

#[test]
fn agrees_with_exhaustive_search_and_its_proofs_check() {
    let positions = random_positions(0x5eed_1234_abcd_0001, 300, 3, 0..=0);
    let won = cross_check(&positions, 4);
    assert!(won > 300, "only {won} wins: the positions test little");
}

#[test]
fn agrees_with_exhaustive_search_near_the_fifty_move_limit() {
    let positions = random_positions(0x0dd_ba11_c0ff_ee00, 300, 3, 94..=99);
    let won = cross_check(&positions, 4);
    assert!(won > 200, "only {won} wins: the positions test little");
}

#[test]
fn agrees_with_exhaustive_search_on_longer_wins() {
    let positions = random_positions(0x1234_5678_9abc_def1, 300, 2, 0..=99);
    let won = cross_check(&positions, 5);
    assert!(won > 300, "only {won} wins: the positions test little");
}

#[test]
fn agrees_with_exhaustive_search_with_more_pieces() {
    let positions = random_positions(0xfeed_f00d_dead_beef, 200, 4, 0..=0);
    let won = cross_check(&positions, 3);
    assert!(won > 100, "only {won} wins: the positions test little");
}

/// A network's guidance changes the search's order, never its answers: a
/// network of random weights steers it as far from the unguided order as any.
#[test]
fn agrees_with_exhaustive_search_when_guided() {
    let mut dfpn = Dfpn::new(4, 12);
    dfpn.set_network(Box::leak(Box::new(
        provari::antichess::nnue::testing::random(3),
    )));
    dfpn.set_guide_from(1);
    let positions = random_positions(0x9a1d_ed5e_a7c4_0002, 200, 4, 0..=99);
    let won = cross_check_with(&mut dfpn, &positions, 4);
    assert!(won > 200, "only {won} wins: the positions test little");
}

/// A depth to win settles a position by the White moves left and the 50-move
/// counter.
#[cfg(feature = "antichess-tablebases")]
#[test]
fn tablebase_entries_settle_by_moves_left_and_counter() {
    let proven = |need, limit| Some(Settled::Won { need, limit });
    let refuted = Some(Settled::Refuted);
    // White, to move, wins in ten plies, five White moves, and the game ends
    // by the 50-move rule's draw at counter 90 or below.
    assert_eq!(settle(Entry::Win(10), Color::White, 0, 5), proven(5, 90));
    assert_eq!(settle(Entry::Win(10), Color::White, 90, 9), proven(5, 90));
    assert_eq!(settle(Entry::Win(10), Color::White, 91, 9), None);
    assert_eq!(settle(Entry::Win(10), Color::White, 91, 4), refuted);
    // Black, to move, loses in nine: four White moves after its reply.
    assert_eq!(settle(Entry::Loss(9), Color::Black, 91, 4), proven(4, 91));
    assert_eq!(settle(Entry::Loss(9), Color::Black, 0, 3), refuted);
    assert_eq!(settle(Entry::Loss(1), Color::Black, 99, 0), proven(0, 99));
    // Draws, and wins for Black, are no wins for White at any counter.
    for (entry, turn) in [
        (Entry::Draw, Color::White),
        (Entry::Draw, Color::Black),
        (Entry::Loss(4), Color::White),
        (Entry::Win(3), Color::Black),
    ] {
        assert_eq!(settle(entry, turn, 0, 50), refuted);
    }
    // Beyond 100 plies, a depth shows only that every win takes more.
    assert_eq!(settle(Entry::Win(148), Color::White, 0, 50), refuted);
    assert_eq!(settle(Entry::Win(148), Color::White, 0, 51), None);
    assert_eq!(settle(Entry::Loss(161), Color::Black, 0, 49), refuted);
    assert_eq!(settle(Entry::Loss(161), Color::Black, 0, 50), None);
}

/// The positions to check the tables on: with tolius's whole set, random
/// ones of two to four pieces; with the few of antichess-tb-sys's repository,
/// a king against a rook, either side with either, which they hold with every
/// position their lines reach, as each capture ends the game.
#[cfg(feature = "antichess-tablebases")]
fn tabled_positions(
    tables: Tablebases,
    seed: u64,
    count: usize,
    counters: std::ops::RangeInclusive<u64>,
) -> Vec<Antichess> {
    if tables.count >= WHOLE_SET {
        random_positions(seed, count, 2, counters)
    } else {
        random_material(seed, count, &["Kr", "Rk"], counters)
    }
}

/// Settling positions from tolius's tables changes no answer, and every
/// proof found still checks, at every counter.
#[cfg(feature = "antichess-tablebases")]
#[test]
fn agrees_with_exhaustive_search_with_tablebases() {
    let Some(tables) = tables() else {
        return;
    };
    let mut dfpn = Dfpn::new(4, 12);
    dfpn.set_tablebases(tables);
    let positions = tabled_positions(tables, 0x7ab1_e5e7_0000_0001, 300, 0..=99);
    let won = cross_check_with(&mut dfpn, &positions, 4);
    assert!(won > 200, "only {won} wins: the positions test little");
    assert!(dfpn.tablebase_hits() > 1000, "{}", dfpn.tablebase_hits());
}

/// Bishops and pawns, where Lichess's insufficient material may draw what
/// the tables play on: settling from them still changes no answer.
#[cfg(feature = "antichess-tablebases")]
#[test]
fn agrees_with_exhaustive_search_on_bishops_and_pawns_with_tablebases() {
    let Some(tables) = tables() else {
        return;
    };
    if tables.count < WHOLE_SET {
        return;
    }
    let mut dfpn = Dfpn::new(4, 12);
    dfpn.set_tablebases(tables);
    let sides = ["BPbp", "BPpp", "PPbp", "PPpp", "BPb", "Bbp", "BBbp", "BPbb"];
    let positions = random_material(0xb15b_0b0b_0000_0003, 300, &sides, 0..=99);
    let won = cross_check_with(&mut dfpn, &positions, 4);
    assert!(won > 50, "only {won} wins: the positions test little");
    assert!(dfpn.tablebase_hits() > 1000, "{}", dfpn.tablebase_hits());
}

/// The whole set answers every position of two to four pieces whose game
/// goes on, but those that may get stuck, with a depth a game can reach:
/// misread, the code a 12-bit table gives a draw looks like a win in 2013
/// plies, and a table the reader cannot read is dropped for good.
#[cfg(feature = "antichess-tablebases")]
#[test]
fn tablebases_answer_every_position_with_a_depth() {
    let Some(tables) = tables() else {
        return;
    };
    if tables.count < WHOLE_SET {
        return;
    }
    let (mut draws, mut decided) = (0, 0);
    for pos in random_positions(0xd7a3_5e70_0000_0004, 20_000, 2, 0..=0) {
        // Black has won once White took its last piece, so no game comes to
        // White's move with Black bare, and no table holds a lone piece.
        if rules::status(&pos, &pos.legal_moves()) != Status::Open
            || pos.them().is_empty()
            || tablebases::may_get_stuck(pos.board())
        {
            continue;
        }
        match tables.probe(&pos) {
            // The longest wins of four pieces take a few hundred plies.
            Some(Entry::Win(plies) | Entry::Loss(plies)) => {
                assert!(plies <= 1000, "{plies} plies in {}", rules::fen(&pos));
                decided += 1;
            }
            Some(Entry::Draw) => draws += 1,
            None => panic!("no answer for {}", rules::fen(&pos)),
        }
    }
    assert!(
        draws > 1000 && decided > 1000,
        "{draws} draws, {decided} decided"
    );
}

/// The tables settle long wins at once, within as many White moves as their
/// depth takes and no fewer, and a proof plays them out to the end of every
/// line, where they hold before the 50-move rule could draw.
#[cfg(feature = "antichess-tablebases")]
#[test]
fn tablebases_prove_long_wins() {
    let Some(tables) = tables() else {
        return;
    };
    let mut dfpn = Dfpn::new(4, 12);
    dfpn.set_tablebases(tables);
    let mut proven = 0;
    for root in tabled_positions(tables, 0x10e6_3a1e_0000_0002, 3000, 0..=0) {
        let Some(Entry::Win(plies)) = tables.probe(&root) else {
            continue;
        };
        if !(20..=100).contains(&plies) {
            continue;
        }
        let need = plies / 2;
        // Highest counter at which every line ends before the 50-move draw.
        let last = with_counter(&root, 100 - plies);
        for (root, most) in [(&root, need + 3), (&last, need)] {
            dfpn.clear();
            let before = dfpn.nodes;
            assert_eq!(
                solve(&mut dfpn, root, most),
                Outcome::Won { need },
                "{}",
                rules::fen(root)
            );
            assert!(dfpn.nodes - before < 1000, "{} nodes", dfpn.nodes - before);
            let (header, steps) = proof(&dfpn, root, most);
            assert_eq!(header.moves, need);
            assert_eq!(verify(root, need, &steps), need, "{}", rules::fen(root));
            assert_eq!(solve(&mut dfpn, root, need - 1), Outcome::NotWithin);
        }
        proven += 1;
        if proven == 20 {
            break;
        }
    }
    assert!(proven >= 5, "only {proven} long wins");
}

#[test]
fn stops_at_its_node_budget() {
    let mut dfpn = Dfpn::new(1, 2);
    let start =
        rules::parse_fen("rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b - - 0 1").unwrap();
    let mut root = start.clone();
    root.play_unchecked(rules::parse_uci(&start, "b7b5").unwrap());
    let limits = Limits {
        nodes: Some(500),
        deadline: None,
    };
    assert_eq!(dfpn.solve(&root, 20, limits), Outcome::Unknown(Stop::Nodes));
    assert!(dfpn.nodes <= 510);
}

/// The same position with another 50-move counter.
fn with_counter(pos: &Antichess, counter: u32) -> Antichess {
    let fen = rules::fen(pos);
    let mut fields: Vec<&str> = fen.split(' ').collect();
    let counter = counter.to_string();
    fields[4] = &counter;
    position(&fields.join(" "))
}

/// Settles positions at several counters with one search, in both orders, so
/// that what it learns at one counter must hold, or be ignored, at another.
#[test]
fn what_one_counter_teaches_holds_at_others() {
    let counters = [99, 98, 97, 95, 93, 90, 0];
    let mut differing = 0;
    for (order, seed) in [(0, 0xabad_cafe_0000_0001u64), (1, 0x5ca1_ab1e_0000_0002)] {
        let mut dfpn = Dfpn::new(4, 12);
        for root in random_positions(seed, 200, 3, 0..=0) {
            let mut results = Vec::new();
            let mut ordered = counters.to_vec();
            if order == 1 {
                ordered.reverse();
            }
            for counter in ordered {
                let pos = with_counter(&root, counter);
                let fewest = exhaustive::fewest(&pos, 4);
                let outcome = solve(&mut dfpn, &pos, 4);
                match fewest {
                    Some(fewest) => {
                        let Outcome::Won { need } = outcome else {
                            panic!("{} wins in {fewest}, not found", rules::fen(&pos));
                        };
                        assert!(fewest <= need, "{}", rules::fen(&pos));
                        let (_, steps) = proof(&dfpn, &pos, 4);
                        verify(&pos, 4, &steps);
                    }
                    None => assert_eq!(outcome, Outcome::NotWithin, "{}", rules::fen(&pos)),
                }
                results.push(fewest);
            }
            if results.iter().any(|result| *result != results[0]) {
                differing += 1;
            }
        }
    }
    assert!(
        differing > 20,
        "only {differing} positions depend on the counter"
    );
}

/// A position a proof proves at a counter is proven at any lower one, with
/// the steps after it in the same proof, as the format's walk takes it when
/// it skips a position proven before. Settled first at high counters, the
/// positions keep proofs that hold at different counters, and those read
/// back at a low one still hold.
#[test]
fn every_step_holds_at_its_counter_and_below() {
    let mut dfpn = Dfpn::new(4, 12);
    let mut checked = 0;
    for root in random_positions(0x5afe_11a1_7000_0003, 300, 3, 0..=0) {
        for counter in [99, 97, 93] {
            solve(&mut dfpn, &with_counter(&root, counter), 4);
        }
        let Outcome::Won { .. } = solve(&mut dfpn, &root, 4) else {
            continue;
        };
        let (_, steps) = proof(&dfpn, &root, 4);
        for step in &steps {
            for counter in 0..=step.counter {
                let pos = position(&format!("{} {counter} 60", step.fen));
                verify(&pos, step.length, &steps);
            }
            checked += 1;
        }
    }
    assert!(checked > 300, "only {checked} steps checked");
}

/// Along random games, from the initial position and from random ones, the
/// counted moves are the listed ones, and every child's key, taken from its
/// parent's, is the one computed afresh.
#[test]
fn counted_moves_and_child_keys_match_shakmatys() {
    let mut random = Random(11);
    let mut starts = vec![Antichess::default(); 300];
    starts.extend(random_positions(12, 1000, 8, 0..=0));
    let (mut positions, mut en_passant, mut promoting) = (0, 0, 0);
    for start in starts {
        let mut pos = start;
        for _ in 0..300 {
            let legal = pos.legal_moves();
            assert_eq!(
                rules::count_moves(&pos),
                legal.len(),
                "{}",
                rules::fen(&pos)
            );
            positions += 1;
            en_passant += usize::from(legal.iter().any(|m| m.is_en_passant()));
            promoting += usize::from(legal.iter().any(|m| m.is_promotion()));
            if legal.is_empty() || pos.us().is_empty() {
                break;
            }
            let key = rules::key(&pos);
            for &m in &legal {
                let mut child = pos.clone();
                child.play_unchecked(m);
                assert_eq!(
                    rules::child_key(&pos, key, m),
                    rules::key(&child),
                    "{} {}",
                    rules::fen(&pos),
                    rules::uci(m)
                );
            }
            pos.play_unchecked(legal[random.below(legal.len() as u64) as usize]);
        }
    }
    assert!(positions > 20_000, "{positions}");
    assert!(
        en_passant > 10 && promoting > 100,
        "{en_passant} {promoting}"
    );
}
