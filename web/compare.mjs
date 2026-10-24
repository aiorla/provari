// Checks the browser build against the native one: the same searches with
// node budgets, so that both repeat exactly, must find the same wins in the
// same nodes and write the same moves; each proof must pass the other
// build's checker; the page's view of a proof must agree with `provari
// line`, move by move along a longest line, and its PGN tree with
// `provari line --pgn`; searches limited to short wins must agree too, and
// so must shortening a win, each shorter win's proof handed over valid, and
// a win for the side not to move.
// It reports the speeds, and checks that the worker gives a browser without
// relaxed SIMD the plain build.
// Usage, after web/build.sh and a native release build:
//   node web/compare.mjs [NODES] [DIR] [BUILD]
// where DIR is web/build.sh's output, web/dist by default, and BUILD the
// manifest's build: wasm (the plain one, by default) or relaxed.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import vm from "node:vm";
import { zstdDecompressSync } from "node:zlib";

const nodes = Number(process.argv[2] ?? 2_000_000);
const dist = process.argv[3] ?? new URL("./dist/", import.meta.url).pathname;
const native = new URL("../target/release/provari", import.meta.url).pathname;
const manifest = JSON.parse(readFileSync(join(dist, "manifest.json"), "utf8"));
const build = process.argv[4] ?? "wasm";
const memory = 256;

// The worker's choice, with and without relaxed SIMD.
const worker = readFileSync(new URL("./worker.js", import.meta.url), "utf8");
const choice = (validates) => {
  const context = vm.createContext({ WebAssembly: { validate: () => validates }, TextDecoder, TextEncoder });
  vm.runInContext(worker, context);
  return context.chosen(manifest);
};
const choices = [choice(true), choice(false)];
const chooses = choices[0] === (manifest.relaxed ?? manifest.wasm) && choices[1] === manifest.wasm;
console.log(`the worker chooses ${choices.join(" with relaxed SIMD, ")} without: ${chooses ? "right" : "WRONG"}`);
const positions = [
  "rnbqkb1r/1pp1n1pp/p2p4/3P1pN1/4p2P/2N5/PPP1PPP1/R1BQKB1R w KQkq - 0 1",
  "r1b1kbnr/ppp3pp/n4q2/3pppN1/1P1PP3/8/P1P2PPP/RNBQKB1R w KQkq - 0 1",
  "rnbq1knr/p2pp2p/2p2ppb/1p2P1N1/3P4/2N5/PPP2PPP/R1BQKB1R w KQ - 0 1",
  "rnbqkbnr/pp1p4/2p2pp1/3Np2p/3PPB2/5N2/PPP2PPP/R2QKB1R w KQkq - 0 1",
  "rn1qkbnr/ppp3pp/3pb3/3P1pN1/4p3/2N5/PPP1PPPP/R1BQKB1R w KQkq - 0 1",
];

const decoder = new TextDecoder();
const encoder = new TextEncoder();
let wasm;
console.log(`checking ${manifest[build]}`);
// The proofs a shortening search hands over, one for each shorter win.
const handed = [];
const { instance } = await WebAssembly.instantiate(readFileSync(join(dist, manifest[build])), {
  provari: {
    now_ms: () => performance.now(),
    progress: () => {},
    found: (ptr, len) => handed.push(new Uint8Array(wasm.memory.buffer, ptr, len).slice()),
  },
});
wasm = instance.exports;
const put = (bytes) => {
  const ptr = wasm.alloc(bytes.length);
  new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
};
const take = (packed) => {
  const ptr = Number(packed & 0xffffffffn);
  const len = Number(packed >> 32n);
  const bytes = new Uint8Array(wasm.memory.buffer, ptr, len).slice();
  wasm.release(ptr, len);
  return bytes;
};
const json = (packed) => JSON.parse(decoder.decode(take(packed)));
const text = (value) => encoder.encode(JSON.stringify(value));
const must = (reply) => {
  if (reply.error) throw new Error(reply.error);
  return reply;
};

must(json(wasm.load_network(...put(readFileSync(join(dist, manifest.network))))));
must(json(wasm.load_model(...put(readFileSync(join(dist, manifest.model))))));
const tables = join(dist, "tables");
for (const name of manifest.tables) {
  must(json(wasm.add_table(...put(encoder.encode(name)), ...put(readFileSync(join(tables, name))))));
}
must(json(wasm.open_tables()));

const moves = (proof) => zstdDecompressSync(proof.subarray(proof.indexOf(10) + 1));
const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
const scratch = mkdtempSync(join(tmpdir(), "provari-compare-"));
const inputs = ["--tablebases", tables, "--network", join(dist, manifest.network), "--model", join(dist, manifest.model)];
const run = (...args) => JSON.parse(execFileSync(native, args));

/** Walks a longest line of the open proof, asking the page's view and `provari line` at each step. */
function lines(file, root, longest) {
  let fen = root;
  for (let ply = 0; ply <= longest.length; ply += 1) {
    const here = must(json(wasm.show_position(...put(text({ fen })))));
    if (here.outcome !== null) return here.value === 0;
    const theirs = run("line", file, longest.slice(0, ply).join(" "), "--tablebases", tables);
    if (theirs.error !== undefined || here.value !== theirs.moves) return false;
    if (here.proof_move !== undefined && here.proof_move !== longest[ply]) return false;
    if (here.tablebase) return ply === longest.length;
    const next = here.moves.find((move) => move.uci === longest[ply]);
    if (next === undefined) return false;
    fen = next.fen;
  }
  return false;
}

let ok = chooses;
const rates = [];
for (const [i, fen] of positions.entries()) {
  const ours = must(json(wasm.prove(...put(text({ fen, nodes, memory_mb: memory })))));
  const proof = ours.found ? take(wasm.take_proof()) : null;
  const nativeProof = join(scratch, `native-${i}.pvr`);
  const theirs = run("prove", fen, "--nodes", String(nodes), "--memory", String(memory), ...inputs, "--proofs", nativeProof);
  const agree = ours.found === theirs.found && ours.nodes === theirs.nodes && (ours.mate ?? null) === (theirs.mate ?? null);
  let checks = "";
  if (proof) {
    const webProof = join(scratch, `web-${i}.pvr`);
    writeFileSync(webProof, proof);
    const nativeBytes = readFileSync(nativeProof);
    const sameMoves = same(moves(proof), moves(nativeBytes));
    const byNative = run("check", webProof, "--tablebases", tables);
    const byWeb = json(wasm.check(...put(nativeBytes)));
    const opened = must(json(wasm.open_proof(...put(proof))));
    const view = opened.mate === ours.mate && lines(webProof, opened.root, opened.longest);
    must(json(wasm.proof_pgn()));
    const tree = decoder.decode(take(wasm.take_pgn()));
    const nativePgn = join(scratch, `native-${i}.pgn`);
    run("line", webProof, "--tablebases", tables, "--pgn", nativePgn);
    const samePgn = tree === readFileSync(nativePgn, "utf8");
    checks = `, same moves ${sameMoves}, checked ${byNative.valid}/${byWeb.valid}, ` +
      `page view ${view} along ${opened.longest.length} plies, same PGN ${samePgn}`;
    ok &&= sameMoves && byNative.valid && byWeb.valid && view && samePgn;
  }
  ok &&= agree;
  rates.push(ours.ms / theirs.ms);
  console.log(`${i}: found ${ours.found}/${theirs.found}, nodes ${ours.nodes}/${theirs.nodes}, ` +
    `mate ${ours.mate}/${theirs.mate}, ${ours.ms}/${theirs.ms} ms${checks}`);
}
// Searches limited to wins in at most so many attacker moves.
for (const [fen, most] of [[positions[1], 13], [positions[1], 9], [positions[2], 20]]) {
  const ours = must(json(wasm.prove(...put(text({ fen, nodes, memory_mb: memory, most })))));
  if (ours.found) take(wasm.take_proof());
  const theirs = run("prove", fen, "--nodes", String(nodes), "--memory", String(memory), "--most", String(most), ...inputs);
  const agree = ours.found === theirs.found && ours.nodes === theirs.nodes &&
    (ours.mate ?? null) === (theirs.mate ?? null) && (ours.refuted ?? false) === (theirs.refuted ?? false);
  ok &&= agree && (!ours.found || ours.mate <= most);
  console.log(`most ${most}: found ${ours.found}/${theirs.found}, refuted ${ours.refuted ?? false}/${theirs.refuted ?? false}, ` +
    `nodes ${ours.nodes}/${theirs.nodes}, mate ${ours.mate}/${theirs.mate}`);
}
// Shortening a win: the same lengths, passes and nodes as natively, every
// proof handed over valid and each shorter than the one before.
const header = (bytes) => JSON.parse(decoder.decode(bytes.subarray(0, bytes.indexOf(10))));
for (const fen of [positions[0], positions[1]]) {
  handed.length = 0;
  const ours = must(json(wasm.prove(...put(text({ fen, nodes, memory_mb: memory, shorten: true })))));
  if (ours.found) take(wasm.take_proof());
  const theirs = run("prove", fen, "--nodes", String(nodes), "--memory", String(memory), "--shorten", ...inputs);
  const lengths = handed.map((proof) => header(proof).moves);
  const valid = handed.every((proof) => json(wasm.check(...put(proof))).valid === true);
  const shrinking = lengths.every((length, i) => i === 0 || length < lengths[i - 1]);
  const agree = ours.found === theirs.found && ours.nodes === theirs.nodes && ours.mate === theirs.mate &&
    ours.first_mate === theirs.first_mate && ours.passes === theirs.passes && ours.shortest === theirs.shortest &&
    (!ours.found || (valid && shrinking && lengths[0] === ours.first_mate && lengths.at(-1) >= ours.mate));
  ok &&= agree;
  console.log(`shorten: found ${ours.found}/${theirs.found}, mate ${ours.first_mate}->${ours.mate}/` +
    `${theirs.first_mate}->${theirs.mate}, passes ${ours.passes}/${theirs.passes}, shortest ${ours.shortest}/` +
    `${theirs.shortest}, nodes ${ours.nodes}/${theirs.nodes}; handed over ${lengths.join(" ")}, valid ${valid}`);
}
// A win for the side not to move: White's, after its first move in the first
// position. The same search and moves as natively, each proof passing the
// other's checker, and the page's view naming White and following the proof.
{
  const fen = "rnbqkb1r/1pp1nNpp/p2p4/3P1p2/4p2P/2N5/PPP1PPP1/R1BQKB1R b KQkq - 1 1";
  const ours = must(json(wasm.prove(...put(text({ fen, attacker: "white", nodes, memory_mb: memory })))));
  const proof = ours.found ? take(wasm.take_proof()) : null;
  const nativeProof = join(scratch, "native-attacker.pvr");
  const theirs = run("prove", fen, "--attacker", "white", "--nodes", String(nodes), "--memory", String(memory),
    ...inputs, "--proofs", nativeProof);
  let agree = ours.found && theirs.found && ours.nodes === theirs.nodes && ours.mate === theirs.mate &&
    ours.attacker === "white" && theirs.attacker === "white";
  if (agree) {
    const webProof = join(scratch, "web-attacker.pvr");
    writeFileSync(webProof, proof);
    const nativeBytes = readFileSync(nativeProof);
    const opened = must(json(wasm.open_proof(...put(proof))));
    agree = same(moves(proof), moves(nativeBytes)) &&
      run("check", webProof, "--tablebases", tables).valid === true &&
      json(wasm.check(...put(nativeBytes))).valid === true &&
      opened.attacker === "white" && lines(webProof, opened.root, opened.longest);
  }
  ok &&= agree;
  console.log(`attacker: White to win with Black to move, found ${ours.found}/${theirs.found}, ` +
    `nodes ${ours.nodes}/${theirs.nodes}, mate ${ours.mate}/${theirs.mate}, ${agree ? "agree" : "DISAGREE"}`);
}
// Games already over, a king exploded and a stalemate: not searched, both
// say how each ended.
for (const fen of ["rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQ - 0 1", "k7/8/1Q6/8/8/8/8/6K1 b - - 0 1"]) {
  const ours = must(json(wasm.prove(...put(text({ fen, nodes, memory_mb: memory })))));
  const theirs = run("prove", fen, "--nodes", String(nodes), "--memory", String(memory), ...inputs);
  ok &&= !ours.found && ours.nodes === 0 && ours.over === theirs.over && ours.winner === theirs.winner;
  console.log(`over: ${ours.over}/${theirs.over}, winner ${ours.winner}/${theirs.winner}, nodes ${ours.nodes}`);
}
const mean = Math.exp(rates.reduce((sum, r) => sum + Math.log(r), 0) / rates.length);
console.log(`${ok ? "all agree" : "DISAGREEMENT"}; the browser build takes x${mean.toFixed(2)} the native time`);
process.exit(ok ? 0 : 1);
