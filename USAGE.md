# Using provari

This guide explains every command and option of provari. For what provari
is and a quick start, see the [README](README.md).

- [Proving a position](#proving-a-position)
- [Searching many positions](#searching-many-positions)
- [Checking proofs](#checking-proofs)
- [Exploring a proof](#exploring-a-proof)
- [Making a proof smaller](#making-a-proof-smaller)
- [UCI](#uci)
- [Proof files](#proof-files)
- [The browser build](#the-browser-build)
- [The benchmark](#the-benchmark)

## Proving a position

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
| `--most N` | none | Only wins in at most N moves, a tablebase win counting where it is reached. |
| `--shorten` | off | After a win, looks for shorter ones within the budget. |
| `--tablebases DIR` | none | The tablebases' folders, separated by `:` (`;` on Windows). |
| `--proofs PATH` | none | Where to write the proof (for `bench`, a folder). |
| `--out PATH` | | Where `bench` and `convert` write. |
| `--workers N` | 1 | Searches, checks or conversions at a time. |
| `--progress S` | 1 for the status line, else none | How often a search reports its progress, in seconds; 0 for never. |
| `--json` | off | Prints the result, and the progress, as JSON. |
| `--guide model\|network\|hand\|none` | the strongest available | What guides the search. |
| `--model FILE`, `--network FILE` | built in | Another model or network. A network is in Atomic-Stockfish's format (legacy Atomic V1) or one of varifish's for Atomic (SFNNv20) pruned for provari, as the releases carry; a model is fitted to one network. |
| `--no-eval` | | The same as `--guide none`. |
| `--seed S` | 0 | Varies the search, deterministically. |
| `--search dfpn\|exhaustive` | `dfpn` | Proof-number search, or a plain win-in-N search as a baseline. |

## Searching many positions

`provari bench` reads a JSON-lines file, a position on each line:

```json
{"id": "q1", "fen": "r1b1kbnr/ppp3pp/n4q2/3pppN1/1P1PP3/8/P1P2PPP/RNBQKB1R w KQkq - 0 1"}
```

with an optional `attacker`, and appends a result for each position to
`--out`. Positions already there are skipped, so an interrupted run picks up
where it stopped.

## Checking proofs

`provari check` replays each proof, generating every defence itself, with
the rules of [shakmaty](https://github.com/niklasf/shakmaty) and none of the
search's code. It prints a JSON line for each proof with `valid`: true,
false, or null when it cannot check it (tablebases missing, or past its
limits, such as `--seconds S` per proof). It exits with 1 unless every proof
is valid. `--compact-only` refuses proofs in JSON lines. A check walks at
most 20 million positions, some 10 GB; for larger proofs, `--memory MB` sets
the memory its walks may take, 2,000 positions a MB, shared among
`--workers`.

## Exploring a proof

`provari line PROOF [MOVES]` shows, after the moves given (SAN or UCI), the
proof's move, every defence with the moves it lasts, and a longest line.
`--pgn FILE` writes the proof's tree for a PGN viewer, and `--until N` stops
its lines once the proof wins within N moves, which keeps the tree small.

## Making a proof smaller

`provari refine PROOF --out REFINED.pvr --nodes N` makes a proof smaller:
where its subproofs are largest first, it searches the winning side's other
moves and keeps one whose proof comes out smaller, if the whole proof still
holds and shrinks. `--nodes N`, `--seconds S` or both budget all its
searches together; nodes alone repeat a run exactly. `--objective size`, the
default, makes the proof's tree smaller, its positions counted along every
line; `--objective positions` the distinct positions it lists. It reads
proofs in the compact format, from either side's move, or in JSON lines,
writes them in the compact format, and prints what changed as a JSON line.
The searches take `--tablebases`, `--network`, `--model`, `--guide` and
`--seed` as `prove` does. Check the refined proof like any other.

## UCI

`provari uci` makes provari an Atomic engine for chess GUIs. Each `go`
searches for a win as `provari prove` does, limited by `mate N`, `nodes`,
`movetime` or the clock, or with `infinite` until `stop`. A win is reported
as a mate (`score mate N`), or as a tablebase win (`score cp`, 20000 less the
line's plies) when its lines may end in the tablebases, which under `mate N`
they never do: that search plays tablebase wins out. Without a win the
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

## Proof files

Proofs are written in the compact format of [FORMAT.md](FORMAT.md): only the
attacker's moves, compressed, as the checker generates the defences itself.
A proof file named `*.jsonl` is written in JSON lines instead, readable but
some 600 times larger, which `provari convert` turns into the compact format.

## The browser build

For the browser, `web/build.sh [--tablebases DIR] [OUT]` builds the
WebAssembly version into `web/dist` (or OUT), with `web/worker.js`, which
runs it in a page's worker and a search's other threads in workers of their
own, and the data it loads, the tables in a folder named after their
contents, which the worker fetches all at once when a proof or a search
first needs them. Its threads share memory, which browsers allow
only on a cross-origin isolated page (served with
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`). It needs the network and the
model in `data/`, clang for zstd's C code,
`rustup target add wasm32-wasip1-threads` and
`rustup component add llvm-tools`. `node web/compare.mjs` checks the browser
build against the native one, with the worker running in Node
(`web/node.mjs`).

## The benchmark

The [README](README.md#benchmark)'s table compares provari with the
strongest public Atomic engines on the 200 positions of
[bench/high200.jsonl](bench/high200.jsonl): positions with White to move that
Atomic-Stockfish rates as won (+15 or more after at least 100 million nodes)
without finding a mate. Each program searched each position once at each time
limit, on one thread of a Mac mini (M2) with 1 GB of memory and the
tablebases up to 5 pieces. A win counts for provari once `provari check`
verifies its proof, and for an engine when its last score is a mate or a
tablebase win. provari's row comes from:

```sh
for seconds in 1 10 60; do
    provari bench bench/high200.jsonl --seconds $seconds --memory 1024 \
        --shorten --workers 4 --tablebases DIR --proofs proofs-$seconds \
        --out results-$seconds.jsonl
    provari check proofs-$seconds/*.pvr --tablebases DIR
done
```

The engines ran with `go movetime`, `Threads` 1, `Hash` 1024 and the same
tablebases, and varifish and Atomic-Stockfish used the network provari
carried then, Atomic-Stockfish's `atomic_run3b_e202_l05`.
