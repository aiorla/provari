//! A proof's lines, as `provari line` and the browser page show them: the
//! proof's move and every defence with the attacker moves it lasts, a
//! longest line, and the proof's tree as PGN; and for the page, how many
//! attacker moves the proof takes from any position it covers.

use std::collections::HashMap;

use serde_json::{Value, json};
use shakmaty::san::{San, SanPlus};
use shakmaty::{
    Color, EnPassantMode, Move, Position as _, fen::Fen, uci::UciMove, variant::Atomic,
};

use crate::search;
use crate::tablebase::{Tablebase, Verdict};
use crate::walk;

/// A proof's lines from the position after `played` (SAN or UCI, move
/// numbers allowed), as JSON: against the proof's moves for `attacker`, the
/// most attacker moves the defender can last from there, the proof's move,
/// each defence with the attacker moves it lasts, longest first, and a
/// longest line. With `pgn`, the proof's tree from there too, as PGN, ending
/// its lines where the proof wins within `until` attacker moves.
pub fn show(
    root: Atomic,
    attacker: Color,
    moves: walk::Moves,
    tablebase: Option<&Tablebase>,
    played: &str,
    pgn: bool,
    until: Option<u32>,
) -> Result<(Value, Option<String>), String> {
    let mut proof = Proof::new(attacker, moves);
    let mut quiet = |_| {};
    let mut lines = Lines {
        proof: &mut proof,
        tablebase,
        until,
        report: &mut quiet,
        written: 0,
    };
    let mut pos = root;
    for token in played.split_whitespace() {
        // A move number, alone or before its move: "6.", "6...", "7.e4".
        let text = match token.find(|c: char| !c.is_ascii_digit()) {
            Some(i) if token[i..].starts_with('.') => token[i..].trim_start_matches('.'),
            _ => token,
        };
        if text.is_empty() {
            continue;
        }
        if !lines.open(&pos) {
            return Err("the game is over here".into());
        }
        let m = text
            .parse::<San>()
            .ok()
            .and_then(|san| san.to_move(&pos).ok())
            .or_else(|| text.parse::<UciMove>().ok()?.to_move(&pos).ok())
            .ok_or(format!("{token} is not a legal move here"))?;
        let ends = "the proof ends here, at a tablebase win";
        if pos.turn() == attacker {
            match lines.proof_move(&pos)? {
                None => return Err(ends.into()),
                Some(proved) if proved != m => {
                    return Err(format!(
                        "the proof plays {} here, not {token}",
                        SanPlus::from_move(pos.clone(), proved)
                    ));
                }
                Some(_) => {}
            }
        } else if lines.probe(&pos) == Some(Verdict::Loss) {
            return Err(ends.into());
        }
        pos.play_unchecked(m);
    }
    if !lines.open(&pos) {
        return Err("the game is over here".into());
    }
    let fen = Fen::from_position(&pos, EnPassantMode::Legal).to_string();
    let mut result = json!({ "fen": fen });
    let mut after = pos.clone();
    if pos.turn() == attacker {
        let Some(m) = lines.proof_move(&pos)? else {
            result["moves"] = json!(0);
            result["line"] = json!("(tablebase win)");
            return Ok((result, None));
        };
        result["move"] = json!(SanPlus::from_move(pos.clone(), m).to_string());
        after.play_unchecked(m);
    }
    let defences = lines.replies(&after)?;
    let longest = defences.first().map_or(0, |d| d.0);
    result["moves"] = json!(longest + u32::from(pos.turn() == attacker));
    result["defences"] = json!(
        defences
            .iter()
            .map(|&(length, reply)| json!([
                SanPlus::from_move(after.clone(), reply).to_string(),
                length
            ]))
            .collect::<Vec<_>>()
    );
    let (longest, at_table) = lines.proof.longest_line(&pos, tablebase)?;
    let mut line = Vec::new();
    let mut here = pos.clone();
    for (i, &m) in longest.iter().enumerate() {
        line.push(play(&mut here, m, i == 0));
    }
    if at_table {
        line.push("(tablebase win)".to_owned());
    }
    result["line"] = json!(line.join(" "));
    let mut text = None;
    if pgn {
        let (tree, moves) = lines.tree(pos, &fen)?;
        text = Some(tree);
        result["pgn_moves"] = json!(moves);
    }
    Ok((result, text))
}

/// A proof read for its lines: its attacker's moves, and the measures of
/// the positions walked so far.
pub struct Proof {
    attacker: Color,
    moves: walk::Moves,
    walk: walk::Walk,
}

impl Proof {
    pub fn new(attacker: Color, moves: walk::Moves) -> Proof {
        Proof {
            attacker,
            moves,
            walk: walk::Walk::measures_only(),
        }
    }

    pub fn attacker(&self) -> Color {
        self.attacker
    }

    pub fn moves(&self) -> &walk::Moves {
        &self.moves
    }

    fn open(&self, pos: &Atomic) -> bool {
        matches!(search::status(pos, self.attacker), search::Status::Open)
    }

    /// The attacker's move as the proof's length counts it: none at a
    /// tablebase win, a capture that wins at once where there is one, and
    /// the proof's move otherwise.
    pub fn proof_move(
        &self,
        pos: &Atomic,
        tablebase: Option<&Tablebase>,
    ) -> Result<Option<Move>, String> {
        if probe(tablebase, pos) == Some(Verdict::Win) {
            return Ok(None);
        }
        if let Some(m) = walk::winning_capture(pos, self.attacker) {
            return Ok(Some(m));
        }
        self.moves.get(&search::hash(pos)).copied().ok_or_else(|| {
            format!(
                "the proof has no move for {}",
                Fen::from_position(pos, EnPassantMode::Legal)
            )
        })
    }

    /// The defender's replies with the attacker moves each lasts, longest
    /// first and the larger subproof first among equals; none once the game
    /// or the tablebase has decided.
    fn replies(
        &mut self,
        after: &Atomic,
        tablebase: Option<&Tablebase>,
    ) -> Result<Vec<(u32, Move)>, String> {
        if !self.open(after) || probe(tablebase, after) == Some(Verdict::Loss) {
            return Ok(Vec::new());
        }
        let mut replies = Vec::new();
        for reply in after.legal_moves() {
            let mut next = after.clone();
            next.play_unchecked(reply);
            let lasts = if self.open(&next) {
                let measure =
                    walk::measure(&next, self.attacker, &self.moves, tablebase, &mut self.walk)?;
                (measure.length, measure.tree)
            } else {
                (0, 0)
            };
            replies.push((lasts, reply));
        }
        replies.sort_by_key(|&(lasts, _)| std::cmp::Reverse(lasts));
        Ok(replies
            .into_iter()
            .map(|((length, _), reply)| (length, reply))
            .collect())
    }

    /// A longest line from `pos`: the proof's moves against the replies that
    /// last longest, and whether it ends at a tablebase win.
    pub fn longest_line(
        &mut self,
        pos: &Atomic,
        tablebase: Option<&Tablebase>,
    ) -> Result<(Vec<Move>, bool), String> {
        let mut line = Vec::new();
        let mut here = pos.clone();
        while self.open(&here) {
            let m = if here.turn() == self.attacker {
                self.proof_move(&here, tablebase)?
            } else {
                self.replies(&here, tablebase)?
                    .first()
                    .map(|&(_, reply)| reply)
            };
            let Some(m) = m else {
                return Ok((line, true));
            };
            here.play_unchecked(m);
            line.push(m);
        }
        Ok((line, false))
    }

    /// Whether the attacker's move here ends the game.
    fn wins_at_once(&self, pos: &Atomic, tablebase: Option<&Tablebase>) -> Result<bool, String> {
        Ok(self.proof_move(pos, tablebase)?.is_some_and(|m| {
            let mut next = pos.clone();
            next.play_unchecked(m);
            matches!(search::status(&next, self.attacker), search::Status::Won)
        }))
    }

    /// How many moves and comments the PGN tree from `pos` writes, its lines
    /// ending where the proof wins within `until` attacker moves, counted on
    /// the proof's positions rather than written: the tree writes a position
    /// again on every line that reaches it, so it can be far larger than the
    /// proof.
    fn tree_size(
        &mut self,
        pos: &Atomic,
        tablebase: Option<&Tablebase>,
        until: Option<u32>,
        counted: &mut HashMap<u64, (u64, u64)>,
    ) -> Result<(u64, u64), String> {
        if !self.open(pos) {
            return Ok((0, 0));
        }
        let key = search::hash(pos);
        if let Some(&size) = counted.get(&key) {
            return Ok(size);
        }
        let play = |pos: &Atomic, m| {
            let mut next = pos.clone();
            next.play_unchecked(m);
            next
        };
        let short = |length: u32| until.is_some_and(|until| length <= until);
        let size = if pos.turn() == self.attacker {
            match self.proof_move(pos, tablebase)? {
                Some(m) => {
                    let (moves, comments) =
                        self.tree_size(&play(pos, m), tablebase, until, counted)?;
                    (moves + 1, comments)
                }
                // "{tablebase win}"
                None => (0, 1),
            }
        } else {
            let replies = self.replies(pos, tablebase)?;
            match replies.first() {
                Some(&(lasts, main)) => {
                    let (mut moves, mut comments) = (1, 0);
                    let mut others = 0;
                    for &(length, reply) in &replies[1..] {
                        let next = play(pos, reply);
                        if self.wins_at_once(&next, tablebase)? {
                            continue;
                        }
                        // The defence and its "{mate in N}", and its line.
                        others += 1;
                        moves += 1;
                        comments += 1;
                        if !short(length) {
                            let (m, c) = self.tree_size(&next, tablebase, until, counted)?;
                            moves += m;
                            comments += c;
                        }
                    }
                    // The main line's "{mate in N}".
                    if others > 0 || short(lasts) {
                        comments += 1;
                    }
                    if !short(lasts) {
                        let (m, c) = self.tree_size(&play(pos, main), tablebase, until, counted)?;
                        moves += m;
                        comments += c;
                    }
                    (moves, comments)
                }
                None => (0, 1),
            }
        };
        counted.insert(key, size);
        Ok(size)
    }

    /// How many moves and comments the whole PGN tree from `pos` writes.
    pub fn pgn_size(
        &mut self,
        pos: &Atomic,
        tablebase: Option<&Tablebase>,
    ) -> Result<(u64, u64), String> {
        self.tree_size(pos, tablebase, None, &mut HashMap::new())
    }

    /// The whole tree from `root` as a PGN game, as `provari line --pgn`
    /// writes it, handing `report` the moves and comments written so far
    /// every few thousand.
    pub fn pgn(
        &mut self,
        root: &Atomic,
        tablebase: Option<&Tablebase>,
        report: &mut dyn FnMut(u64),
    ) -> Result<String, String> {
        let fen = Fen::from_position(root, EnPassantMode::Legal).to_string();
        let mut lines = Lines {
            proof: self,
            tablebase,
            until: None,
            report,
            written: 0,
        };
        Ok(lines.tree(root.clone(), &fen)?.0)
    }

    /// How many attacker moves the proof takes to win from `pos`, whoever
    /// is to move, a tablebase win counted as reached: none where it does not
    /// cover `pos`, and 0 where the attacker has won.
    pub fn value(&mut self, pos: &Atomic, tablebase: Option<&Tablebase>) -> Option<u32> {
        match search::status(pos, self.attacker) {
            search::Status::Won => return Some(0),
            search::Status::Over => return None,
            search::Status::Open => {}
        }
        let (attacker, moves, walk) = (self.attacker, &self.moves, &mut self.walk);
        let measure = if pos.turn() == attacker {
            walk::measure(pos, attacker, moves, tablebase, walk)
        } else {
            walk::measure_after(pos, attacker, moves, tablebase, walk)
        };
        measure.ok().map(|measure| measure.length)
    }
}

fn probe(tablebase: Option<&Tablebase>, pos: &Atomic) -> Option<Verdict> {
    tablebase.and_then(|tables| tables.probe(pos))
}

/// A proof read for its lines from a position, with the tables, and where
/// its PGN tree stops.
struct Lines<'a> {
    proof: &'a mut Proof,
    tablebase: Option<&'a Tablebase>,
    /// Where the tree stops: at positions the proof wins within this many
    /// attacker moves.
    until: Option<u32>,
    /// Hears the moves and comments written so far, every few thousand.
    report: &'a mut dyn FnMut(u64),
    written: u64,
}

impl Lines<'_> {
    fn probe(&self, pos: &Atomic) -> Option<Verdict> {
        probe(self.tablebase, pos)
    }

    fn open(&self, pos: &Atomic) -> bool {
        self.proof.open(pos)
    }

    fn proof_move(&self, pos: &Atomic) -> Result<Option<Move>, String> {
        self.proof.proof_move(pos, self.tablebase)
    }

    fn replies(&mut self, after: &Atomic) -> Result<Vec<(u32, Move)>, String> {
        self.proof.replies(after, self.tablebase)
    }

    fn wins_at_once(&self, pos: &Atomic) -> Result<bool, String> {
        self.proof.wins_at_once(pos, self.tablebase)
    }

    /// The proof's tree from `pos`, whose FEN is `fen`, as a PGN game, and
    /// the moves it writes.
    fn tree(&mut self, pos: Atomic, fen: &str) -> Result<(String, usize), String> {
        let attacker = self.proof.attacker;
        let (us, them) = if attacker.is_white() {
            ("White", "Black")
        } else {
            ("Black", "White")
        };
        let cut = self.until.map_or(String::new(), |until| {
            format!(" Lines end where the proof wins within {until} moves.")
        });
        let mut tokens = vec![format!(
            "{{The proof's tree. The main line is {them}'s longest defence and the \
             variations are its other defences, except those after which {us} wins \
             at once. At each branch, mate in N counts {us}'s moves in the proof \
             from there, a tablebase win counted as reached.{cut}}}"
        )];
        self.pgn(pos, &mut tokens, true)?;
        let outcome = if attacker.is_white() { "1-0" } else { "0-1" };
        let moves = tokens
            .iter()
            .filter(|token| !token.starts_with(['{', '(', ')']))
            .count();
        tokens.push(outcome.to_owned());
        let text = format!(
            "[Event \"Proof\"]\n[Site \"?\"]\n[Date \"????.??.??\"]\n[Round \"?\"]\n\
             [White \"?\"]\n[Black \"?\"]\n[Result \"{outcome}\"]\n[Variant \"Atomic\"]\n\
             [SetUp \"1\"]\n[FEN \"{fen}\"]\n\n{}\n\n",
            movetext(&tokens)
        );
        Ok((text, moves))
    }

    /// Writes a move or a comment of the tree, and reports every few
    /// thousand.
    fn write(&mut self, tokens: &mut Vec<String>, token: String) {
        tokens.push(token);
        self.written += 1;
        if self.written.is_multiple_of(4096) {
            (self.report)(self.written);
        }
    }

    /// Writes the proof's tree from `pos` as PGN movetext: the attacker's
    /// moves, the defence that lasts longest as the main line and the others
    /// as variations, but those after which the attacker wins at once, and at
    /// each branch the attacker moves each defence lasts; with `until`, a
    /// defence that lasts no longer ends its line there.
    fn pgn(
        &mut self,
        mut pos: Atomic,
        tokens: &mut Vec<String>,
        mut numbered: bool,
    ) -> Result<(), String> {
        while self.open(&pos) {
            if pos.turn() == self.proof.attacker {
                let Some(m) = self.proof_move(&pos)? else {
                    self.write(tokens, "{tablebase win}".to_owned());
                    break;
                };
                self.write(tokens, play(&mut pos, m, numbered));
                numbered = false;
                continue;
            }
            let replies = self.replies(&pos)?;
            let Some(&(lasts, main)) = replies.first() else {
                self.write(tokens, "{tablebase win}".to_owned());
                break;
            };
            let mut others = Vec::new();
            for &(length, reply) in &replies[1..] {
                let mut next = pos.clone();
                next.play_unchecked(reply);
                if !self.wins_at_once(&next)? {
                    others.push((length, reply));
                }
            }
            let until = self.until;
            let short = |length: u32| until.is_some_and(|until| length <= until);
            let ends = short(lasts);
            let before = pos.clone();
            self.write(tokens, play(&mut pos, main, numbered));
            if !others.is_empty() || ends {
                self.write(tokens, format!("{{mate in {lasts}}}"));
            }
            for &(length, reply) in &others {
                let mut next = before.clone();
                tokens.push("(".to_owned());
                self.write(tokens, play(&mut next, reply, true));
                self.write(tokens, format!("{{mate in {length}}}"));
                if !short(length) {
                    self.pgn(next, tokens, false)?;
                }
                tokens.push(")".to_owned());
            }
            if ends {
                break;
            }
            numbered = !others.is_empty();
        }
        Ok(())
    }
}

/// Plays a move and writes it in SAN, numbered as PGN numbers it: always for
/// White, for Black where `numbered`.
fn play(pos: &mut Atomic, m: Move, numbered: bool) -> String {
    let number = pos.fullmoves().get();
    let white = pos.turn().is_white();
    let san = SanPlus::from_move_and_play_unchecked(pos, m);
    match (white, numbered) {
        (true, _) => format!("{number}. {san}"),
        (false, true) => format!("{number}... {san}"),
        (false, false) => san.to_string(),
    }
}

/// PGN movetext from its tokens, in lines of at most 79 characters where
/// tokens allow, with variations in parentheses that hug their moves.
fn movetext(tokens: &[String]) -> String {
    let mut text = String::new();
    let mut width = 0;
    let mut glued = true;
    for token in tokens {
        if !glued && token != ")" {
            if width + 1 + token.len() > 79 {
                text.push('\n');
                width = 0;
            } else {
                text.push(' ');
                width += 1;
            }
        }
        text.push_str(token);
        width += token.len();
        glued = token == "(";
    }
    text
}

#[cfg(test)]
mod tests {
    use shakmaty::{Position as _, uci::UciMove, variant::Atomic};

    use super::Proof;
    use crate::format::{SMALL, decode, encode_jsonl, small_defended, strategy};

    fn play(pos: &Atomic, uci: &str) -> Atomic {
        let m = uci.parse::<UciMove>().unwrap().to_move(pos).unwrap();
        let mut next = pos.clone();
        next.play_unchecked(m);
        next
    }

    /// The PGN tree's moves and comments counted on the proof's positions
    /// are those it writes, wherever its lines end.
    #[test]
    fn counts_the_moves_the_tree_writes() {
        let (_, bytes) = encode_jsonl(SMALL, None).unwrap();
        let (header, indices) = decode(&bytes).unwrap().unwrap();
        let (root, attacker, moves) = strategy(&header, &indices, None).unwrap();
        for until in [None, Some(0), Some(1), Some(2), Some(3)] {
            let (shown, pgn) =
                super::show(root.clone(), attacker, moves.clone(), None, "", true, until).unwrap();
            let mut proof = Proof::new(attacker, moves.clone());
            let (counted, comments) = proof
                .tree_size(&root, None, until, &mut std::collections::HashMap::new())
                .unwrap();
            assert_eq!(shown["pgn_moves"], counted, "until {until:?}");
            // The comments but the one that opens the tree.
            let written = pgn.unwrap().matches('{').count() as u64 - 1;
            assert_eq!(written, comments, "until {until:?}");
        }
        let mut proof = Proof::new(attacker, moves.clone());
        let (shown, pgn) =
            super::show(root.clone(), attacker, moves, None, "", true, None).unwrap();
        let mut reports = 0;
        assert_eq!(
            proof.pgn(&root, None, &mut |_| reports += 1).unwrap(),
            pgn.unwrap()
        );
        // Too few moves and comments to report any.
        assert_eq!(
            (reports, shown["pgn_moves"].as_u64().unwrap() < 4096),
            (0, true)
        );
    }

    /// The attacker moves the page shows: 4 at the root, one fewer after the
    /// proof's move, none off the proof, and 0 once its longest line wins.
    #[test]
    fn values_follow_the_proof() {
        let (_, bytes) = encode_jsonl(SMALL, None).unwrap();
        let (header, indices) = decode(&bytes).unwrap().unwrap();
        let (root, attacker, moves) = strategy(&header, &indices, None).unwrap();
        let mut proof = Proof::new(attacker, moves);
        assert_eq!(proof.value(&root, None), Some(4));
        assert_eq!(proof.value(&play(&root, "g5f7"), None), Some(3));
        assert_eq!(proof.value(&play(&root, "h4h5"), None), None);
        // A failed walk leaves the next one unharmed.
        assert_eq!(proof.value(&root, None), Some(4));
        let (line, at_table) = proof.longest_line(&root, None).unwrap();
        assert!(!at_table);
        assert_eq!(line.len(), 7);
        let end = line.iter().fold(root, |mut pos, &m| {
            pos.play_unchecked(m);
            pos
        });
        assert_eq!(proof.value(&end, None), Some(0));
    }

    /// A proof whose root has the defender to move shows its defences
    /// first, with no move of its own there, and its tree starts with
    /// them.
    #[test]
    fn shows_a_proof_from_the_defender_to_move() {
        let (_, bytes) = small_defended();
        let (header, indices) = decode(&bytes).unwrap().unwrap();
        let (root, attacker, moves) = strategy(&header, &indices, None).unwrap();
        let (shown, pgn) =
            super::show(root.clone(), attacker, moves.clone(), None, "", true, None).unwrap();
        assert_eq!(
            (shown["moves"].as_u64(), shown.get("move")),
            (Some(3), None)
        );
        assert_eq!(shown["defences"][0][1], 3);
        let pgn = pgn.unwrap();
        assert!(
            pgn.contains("[Result \"1-0\"]") && pgn.contains("}\n1... "),
            "{pgn}"
        );
        assert_eq!(Proof::new(attacker, moves).value(&root, None), Some(3));
    }
}
