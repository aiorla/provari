<div align="center">

<h1>provari</h1>

Finds forced wins in Atomic chess, and proves them.

[Download](https://github.com/aiorla/provari/releases/latest)
·
[Try it in your browser](https://solvedchess.com/provari)
·
[How to use it](USAGE.md)
·
[Report a bug](https://github.com/aiorla/provari/issues)

[![Release](https://img.shields.io/github/v/release/aiorla/provari?label=release)](https://github.com/aiorla/provari/releases/latest)
[![License](https://img.shields.io/github/license/aiorla/provari)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/aiorla/provari/test.yml?label=tests)](https://github.com/aiorla/provari/actions/workflows/test.yml)

</div>

## Overview

Give provari a position of [Atomic chess](https://lichess.org/variant/atomic)
and it searches for a forced win for the side to move. When it finds one, it
writes a proof: the winning move in every position the win can reach,
against every defence. `provari check` then verifies the proof on its own,
so a win can be trusted without trusting the search that found it.

- **Proof-number search**, guided by a neural network, that finds wins
  engines miss.
- **Small proof files**: the 15-move win below takes 701 bytes.
- **An independent checker**, which shares no code with the search.
- **Threads** and **tablebases**: several cores, and Lichess's Atomic tables
  up to 6 pieces.
- **A UCI engine** for chess GUIs, and a version that runs in the browser.

provari is developed for [solvedchess.com](https://solvedchess.com), which
works out who wins in chess variants, and where you can
[try it without installing anything](https://solvedchess.com/provari).

## Benchmark

Wins found and proven in 200 positions that engines rate as won but find no
mate in ([bench/high200.jsonl](bench/high200.jsonl)), on one thread of a Mac
mini (M2) with 1 GB of memory:

| Program | 1 s | 10 s | 60 s |
|---|---|---|---|
| provari 0.5.0 | **12** | **25** | **37** |
| [varifish](https://github.com/aiorla/varifish) 0.1.0 | 5 | 8 | 15 |
| Atomic-Stockfish (38255c52, 2026-10-22) | 4 | 8 | 14 |
| Multi-Variant Stockfish (3851b9df, 2019-11-06) | 4 | 9 | 16 |

[How the benchmark was run](USAGE.md#the-benchmark).

## Getting started

### Download

Download the archive for your computer from the
[latest release](https://github.com/aiorla/provari/releases/latest): Windows
(x86-64), Linux (x86-64) or macOS (Apple silicon). Each holds the program,
with its network built in, this README, the [guide](USAGE.md), the
[proof format](FORMAT.md) and the license. On Linux, provari needs Ubuntu
22.04, Debian 12 or newer (glibc 2.34).

### Tablebases (optional)

Tablebases help in endings. Download Lichess's Atomic tables of wins and
losses, the `.atbw` files, from
[tablebase.lichess.ovh](https://tablebase.lichess.ovh/tables/atomic/): the
`3-4-5` folder (some 0.4 GB), and `6-wdl` if you have room. Pass their
folders with `--tablebases 3-4-5:6-wdl` (`;` between them on Windows).

### Quick start

```sh
provari prove 'r1b1kbnr/ppp3pp/n4q2/3pppN1/1P1PP3/8/P1P2PPP/RNBQKB1R w KQkq - 0 1' --proofs win.pvr
```

searches the position for up to 10 seconds and prints:

```
White mates in 15, found in 554 ms and 326k nodes.
Longest line: 1. dxe5 Bd7 2. Bb5 Nxb4 3. exd5 Bd6 4. Ba3 Rd8 5. Qh5+ g6 6. Qe2+ Be5 7. Ne6 Ra8 8. Ng7+ Kd8 9. Ne6+ Ke8 10. Qd2 Bd6 11. Ng7+ Kf8 12. Qd5 Be6 13. Nxf5 Ke8 14. Qf7+ Kd8 15. Qxc7#
Proof written to win.pvr (1.1k moves).
```

Then `provari check win.pvr` verifies the proof, and `provari line win.pvr`
shows every defence to `1. dxe5` with the moves it lasts. In the Windows
Command Prompt, quote the FEN with double quotes.

## Usage

| Command | What it does |
|---|---|
| `provari prove FEN` | Searches one position for a win. |
| `provari bench POSITIONS.jsonl --out RESULTS.jsonl` | Searches each position of a file, several at a time. |
| `provari check PROOF...` | Verifies proofs. |
| `provari line PROOF [MOVES]` | Shows a proof's lines, or writes its tree as PGN. |
| `provari refine PROOF --out REFINED.pvr` | Makes a proof smaller. |
| `provari convert PROOF.jsonl... --out DIR` | Converts JSON-lines proofs to the compact format. |
| `provari eval NETWORK < FENS` | Prints a network's evaluations. |
| `provari uci` | Runs as a UCI engine. |
| `provari --help` | Lists the commands and the main options. |

The main options of `prove` and `bench`:

| Option | Default | What it does |
|---|---|---|
| `--seconds S` | 10 | The time to search. |
| `--threads N` | 1 | The cores to search with. |
| `--memory MB` | from the time | The memory for each search. |
| `--tablebases DIR` | none | The tablebases' folders. |
| `--proofs PATH` | none | Where to write the proof. |
| `--shorten` | off | After a win, looks for shorter ones. |
| `--json` | off | Prints the result as JSON, for programs. |

The [guide](USAGE.md) explains every command and option, what a search
prints while it runs, the UCI engine's options and the proof files.

## Building from source

With [Rust](https://rustup.rs):

```sh
data/fetch.sh
cargo build --release --features bundled
```

builds `target/release/provari` with its network built in, as the releases
are; `data/fetch.sh` downloads it and checks it (on Windows, run it from Git
Bash). Without `--features bundled`, give the network and the model as files
(`--network FILE`, `--model FILE`). `cargo test` runs the tests, and
[the guide](USAGE.md#the-browser-build) explains the browser build.

## Acknowledgements

- [shakmaty](https://github.com/niklasf/shakmaty) and
  [shakmaty-syzygy](https://github.com/niklasf/shakmaty-syzygy) by Niklas
  Fiekas, for the rules and the tablebases.
- [Atomic-Stockfish](https://github.com/Belzedar94/Atomic-Stockfish), which
  derives from [Stockfish](https://github.com/official-stockfish/Stockfish)
  (Copyright (C) 2004-2026 The Stockfish developers) and
  [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish), and
  [varifish](https://github.com/aiorla/varifish), which derives from
  Stockfish and Atomic-Stockfish: provari's network evaluation translates
  theirs.
- The network that the releases carry is varifish's `atomic-b0d279b9e021`,
  pruned for provari, trained on positions of varifish's games labelled by
  networks that derive from `atomic_run3b_e202_l05`, which Belzedar trained
  for Atomic-Stockfish and which provari carried before it.

## License

provari is free software under the
[GNU General Public License, version 3](LICENSE) or later. You may use it,
share it, sell it or build on it; whenever you distribute it, include the
license and the source code, or where to find it, and share your changes
under the same license. It comes without any warranty.
