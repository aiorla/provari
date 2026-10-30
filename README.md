# provari

provari finds forced wins in [Atomic chess](https://lichess.org/variant/atomic)
and proves them.

Give it a position and it searches for a win for the side to move. When it
finds one, it writes a proof: the winning side's move in every position the
win can reach, against every defence. `provari check` verifies proofs
independently of the search, so a win can be trusted without trusting the
search that found it.

- **Proof-number search**, guided by a model built on Atomic-Stockfish's
  neural network.
- **Threads**: a search can use several cores, its threads sharing its
  memory.
- **Small proof files**: the 13-move win of the quick start below takes 657
  bytes.
- **An independent checker**, which shares no code with the search.
- **Tablebases**: Lichess's Atomic Syzygy tables, up to 6 pieces.
- **A UCI engine** for chess GUIs, and a version that runs in the browser.

## Installation

Download the archive for your system from the
[releases](https://github.com/aiorla/provari/releases): Linux (x86-64),
macOS (Apple silicon) or Windows (x86-64). Each archive holds the program,
with its network and model built in, this README, the proof format
([FORMAT.md](FORMAT.md)) and the license.

On Linux, provari needs glibc 2.34 or later, as in Ubuntu 22.04 or Debian 12.

Tablebases are optional, but they help in endings: download Lichess's Atomic
tables of wins and losses, the `.atbw` files, from
[tablebase.lichess.ovh](https://tablebase.lichess.ovh/tables/atomic/): those
of up to 5 pieces from its `3-4-5` folder (some 0.4 GB), and the 6-piece ones
from `6-wdl` if you have room. Pass their folders with `--tablebases`,
separated by `:` (`;` on Windows): `--tablebases 3-4-5:6-wdl`.

provari also runs in the browser, with nothing to install, at
[solvedchess.com/provari](https://solvedchess.com/provari).

## Quick start

```sh
provari prove 'r1b1kbnr/ppp3pp/n4q2/3pppN1/1P1PP3/8/P1P2PPP/RNBQKB1R w KQkq - 0 1' --proofs win.pvr
```

searches the position for up to 10 seconds and prints:

```
White mates in 13, found in 240 ms and 169k nodes.
Longest line: 1. dxe5 Nf6 2. exd5 Bd6 3. Bb5+ Nxb4 4. Qe2+ Be5 5. f4 Ng4 6. fxe5+ Ne5 7. Qh5+ g6 8. Qh6 Nd3+ 9. cxd3 Kd8 10. Ne6+ Bxe6 11. Qg5+ Kd7 12. Qe7+ Kc6 13. Qxc7#
Proof written to win.pvr (936 moves).
```

White wins in 13 moves, and the proof is in `win.pvr`. Then:

```sh
provari check win.pvr
provari line win.pvr
```

verify the proof, printing `"valid":true`, and show every defence to
`1. dxe5` with the moves it lasts.

In the Windows Command Prompt, quote the FEN with double quotes.

## Usage

| Command | What it does |
|---|---|
| `provari prove FEN` | Searches one position for a win. |
| `provari bench POSITIONS.jsonl --out RESULTS.jsonl` | Searches each position of a file, several at a time. |
| `provari check PROOF...` | Verifies proofs. |
| `provari line PROOF [MOVES]` | Shows a proof's lines, or writes its tree as PGN. |
| `provari convert PROOF.jsonl... --out DIR` | Converts JSON-lines proofs to the compact format. |
| `provari eval NETWORK < FENS` | Prints a network's evaluations. |
| `provari uci` | Runs as a UCI engine. |
| `provari --version` | Prints the version. |
| `provari --help` | Lists the commands and the main options. |

### Proving a position

In a win, every line ends with the defender's king exploded or mated, or in a
position that the tablebases score as won. provari follows Lichess's Atomic
rules with the 50-move rule, and one rule of its own: no line of a proof may
repeat a position. Any win counts; `--shorten` looks for shorter ones.

`provari prove` says what it found: a win with its length and a longest line,
or that there is none, or, when its budget runs out first, the line the search
was on, the hardest it met. If the game is already over, it says how it ended.
If something goes wrong, such as a bad FEN or a proof it cannot write, it says
what on standard error and exits with 2.

While it searches on a terminal, a status line, rewritten every second, shows
how it is going:

```
  4 s   836k nodes   181k/s  pn 2.1k, closing in    1. d6 (3 left) Bg7 2. dxc7 (1 left) d5
```

the time, the nodes searched and their speed, the proof number (`pn`, how
much work the search reckons a proof still needs), and the line it is on, with
the defences not yet beaten after each attacker move. The proof number shows
its peak, `pn 7.1k (peak 7.4k)`, until it falls below half of it: then
`closing in`, a sign that a proof may be near, though no promise.

With `--json`, for programs, it prints a line of JSON instead: whether it
`found` a win, the win's length in moves (`mate`), the nodes searched, the
time taken, and the budgets it had; without a win, the root's proof and
disproof numbers (`root`). A game already over has `over` and `winner`, and
an error `error`. `--progress S` then prints a JSON line every S seconds on
standard error with the status line's numbers and more, and a search cut
short gives its line (`line`, in UCI notation) with the moves still open
where each was played (`open`). Keeping that line costs some time, so
without `--progress` a search keeps none.

| Option | Default | What it does |
|---|---|---|
| `--seconds S` | 10 | The time budget. |
| `--nodes N` | none | A node budget; with `--seconds`, whichever comes first. |
| `--memory MB` | from the budget | Each search's memory: 192 MB for 10 s, 1.5 GB for 60 s. |
| `--threads N` | 1 | Each search's threads, which share its memory; its default memory grows with them. |
| `--max-memory MB` | half the machine's memory | The most memory the searches may take together. |
| `--attacker white\|black` | the side to move | Whose win to prove. |
| `--most N` | none | Only wins in at most N moves. |
| `--shorten` | off | After a win, looks for shorter ones within the budget. |
| `--tablebases DIR` | none | The tablebases' folders, separated by `:` (`;` on Windows). |
| `--proofs PATH` | none | Where to write the proof (for `bench`, a folder). |
| `--out PATH` | | Where `bench` and `convert` write. |
| `--workers N` | 1 | Searches, checks or conversions at a time. |
| `--progress S` | 1 for the status line, else none | How often a search reports its progress, in seconds; 0 for never. |
| `--json` | off | Prints the result, and the progress, as JSON. |
| `--guide model\|network\|hand\|none` | the strongest available | What guides the search. |
| `--model FILE`, `--network FILE` | built in | Another model or network. |
| `--no-eval` | | The same as `--guide none`. |
| `--seed S` | 0 | Varies the search, deterministically. |
| `--search dfpn\|exhaustive` | `dfpn` | Proof-number search, or a plain win-in-N search as a baseline. |

### Searching many positions

`provari bench` reads a JSON-lines file, a position on each line:

```json
{"id": "q1", "fen": "r1b1kbnr/ppp3pp/n4q2/3pppN1/1P1PP3/8/P1P2PPP/RNBQKB1R w KQkq - 0 1"}
```

with an optional `attacker`, and appends a result for each position to
`--out`. Positions already there are skipped, so an interrupted run picks up
where it stopped.

### Checking proofs

`provari check` replays each proof, generating every defence itself, with
the rules of [shakmaty](https://github.com/niklasf/shakmaty) and none of the
search's code. It prints a JSON line for each proof with `valid`: true,
false, or null when it cannot check it (tablebases missing, or past its
limits, such as `--seconds S` per proof). It exits with 1 unless every proof
is valid. `--compact-only` refuses proofs in JSON lines.

### Exploring a proof

`provari line PROOF [MOVES]` shows, after the moves given (SAN or UCI), the
proof's move, every defence with the moves it lasts, and a longest line.
`--pgn FILE` writes the proof's tree for a PGN viewer, and `--until N` stops
its lines once the proof wins within N moves, which keeps the tree small.

### UCI

`provari uci` makes provari an Atomic engine for chess GUIs. Each `go`
searches for a win as `provari prove` does, limited by `mate N`, `nodes`,
`movetime` or the clock, or with `infinite` until `stop`. A win is reported
as a mate (`score mate N`), or as a tablebase win (`score cp`, 20000 less the
line's plies) when its lines may end in the tablebases. Without a win the
best move is `0000`, and an `info string` says why. While it searches, every
second (`--progress S`), it sends the time, the nodes, their speed, how full
its memory is, and the root move it has searched most (`currmove`); then the
proof number and the line as `provari prove` shows them, in an
`info string`; and with `UCI_ShowCurrLine`, the line itself (`currline`).

| Option | Default | What it does |
|---|---|---|
| `Hash` | 1024 | Each search's memory in MB. |
| `Threads` | 1 | Each search's threads, which share its memory. |
| `SyzygyPath` | empty | The tablebases' folders. |
| `Attacker` | `Mover` | Whose win to search for: `Mover`, `White` or `Black`. |
| `Shorten` | false | After a win, looks for shorter ones and reports each. |
| `UCI_ShowCurrLine` | false | Sends the line being searched (`currline`). |
| `UCI_Variant` | `atomic` | `atomic` only. |

## Proofs

Proofs are written in the compact format of [FORMAT.md](FORMAT.md): only the
attacker's moves, compressed, as the checker generates the defences itself.
A proof file named `*.jsonl` is written in JSON lines instead, readable but
some 600 times larger, which `provari convert` turns into the compact format.

## Benchmark

provari is compared with the strongest public Atomic engines and with
[varifish](https://github.com/aiorla/varifish), our engine for chess and its
variants, on the 200 positions of [bench/high200.jsonl](bench/high200.jsonl).
They are positions with White to move that Atomic-Stockfish rates as won (+15
or more after at least 100 million nodes) without finding a mate. Each
program searched each position once at each time limit, on one thread of a
Mac mini (M2) with 1 GB of memory and the tablebases up to 5 pieces. Wins
found, of 200:

| Program | 1 s | 10 s | 60 s |
|---|---|---|---|
| provari 0.2.0 | **11** | **25** | **39** |
| varifish 0.1.0 | 5 | 8 | 15 |
| Atomic-Stockfish (38255c52, 2026-10-22) | 4 | 8 | 14 |
| Multi-Variant Stockfish (3851b9df, 2019-11-06) | 4 | 9 | 16 |

A win counts for provari once `provari check` verifies its proof, and for an
engine when its last score is a mate or a tablebase win. provari's row comes
from:

```sh
for seconds in 1 10 60; do
    provari bench bench/high200.jsonl --seconds $seconds --memory 1024 \
        --shorten --workers 4 --tablebases DIR --proofs proofs-$seconds \
        --out results-$seconds.jsonl
    provari check proofs-$seconds/*.pvr --tablebases DIR
done
```

The engines ran with `go movetime`, `Threads` 1, `Hash` 1024 and the same
tablebases, and varifish and Atomic-Stockfish used provari's network.

## Building from source

With a recent stable Rust:

```sh
cargo build --release
```

builds `target/release/provari` without a network or model, which you can
give as files (`--network FILE`, `--model FILE`). To build them in, as the
releases do:

```sh
data/fetch.sh
cargo build --release --features bundled
```

`data/fetch.sh` downloads the network and the model from the repository's
`networks` release and checks their SHA-256 (on Windows, run it from Git
Bash).

For the browser, `web/build.sh [--tablebases DIR] [OUT]` builds the
WebAssembly version into `web/dist` (or OUT), with `web/worker.js`, which
runs it in a page's worker and a search's other threads in workers of their
own, and the data it loads. Its threads share memory, which browsers allow
only on a cross-origin isolated page (served with
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`). It needs the network and the
model in `data/`, clang for zstd's C code,
`rustup target add wasm32-wasip1-threads` and
`rustup component add llvm-tools`. `node web/compare.mjs` checks the browser
build against the native one, with the worker running in Node
(`web/node.mjs`).

## Acknowledgements

provari is built on [shakmaty](https://github.com/niklasf/shakmaty) and
[shakmaty-syzygy](https://github.com/niklasf/shakmaty-syzygy) by Niklas
Fiekas. Its network evaluation translates
[Atomic-Stockfish](https://github.com/Belzedar94/Atomic-Stockfish)'s, which
derives from [Stockfish](https://github.com/official-stockfish/Stockfish)
(Copyright (C) 2004-2026 The Stockfish developers) and
[Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish). The
network that the releases carry, `atomic_run3b_e202_l05`, was trained by
Belzedar for Atomic-Stockfish.

## License

provari is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version ([LICENSE](LICENSE)). It is distributed in the hope that it will be
useful, but without any warranty; without even the implied warranty of
merchantability or fitness for a particular purpose.
