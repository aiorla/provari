// Checks the browser build against the native one: the same searches with
// node budgets, so that both repeat exactly, must find the same wins in the
// same nodes and write the same moves; each proof must pass the other
// build's checker; the page's view of a proof must agree with `provari
// line`, move by move along a longest line, and its PGN tree with
// `provari line --pgn`; searches limited to short wins must agree too, and
// so must shortening a win, each shorter win's proof handed over valid, and
// a win for the side not to move. Threads must prove again what one thread
// proves, each proof passing both checkers, and a thread that traps must say
// so to the page. A search that the page stops must reply at once, its
// threads' workers closing, as each must once its search ends.
// It reports the speeds, and checks that the worker gives a browser without
// relaxed SIMD the plain build. The build runs in the worker, as a page runs
// it, under Node (node.mjs).
// Usage, after web/build.sh and a native release build:
//   node web/compare.mjs [NODES] [DIR] [BUILD]
// where DIR is web/build.sh's output, web/dist by default, and BUILD the
// manifest's build: wasm (the plain one, by default) or relaxed.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import vm from "node:vm";
import { zstdDecompressSync } from "node:zlib";
import { BrowserWorker } from "./node.mjs";

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
// The proofs a shortening search hands over, one for each shorter win.
const handed = [];
// The worker, as a page starts it, answering requests by id.
const page = new BrowserWorker(new URL("./worker.js", import.meta.url));
const pending = new Map();
let next = 0;
page.onmessage = ({ data }) => {
  if (data.found !== undefined) handed.push(data.found);
  if (data.id === undefined) return;
  const { resolve, reject } = pending.get(data.id);
  pending.delete(data.id);
  if (data.error === undefined) resolve(data.result);
  else reject(new Error(data.error));
};
const request = (type, payload = {}) =>
  new Promise((resolve, reject) => {
    const id = next++;
    pending.set(id, { resolve, reject });
    page.postMessage({ id, type, ...payload });
  });
const must = (reply) => {
  if (reply.error) throw new Error(reply.error);
  return reply;
};

// The build asked for: the plain one from a manifest without the relaxed one.
// The page's word that stops a search, 1 to stop it.
const stop = new Int32Array(new SharedArrayBuffer(4));
const base = pathToFileURL(join(dist, "/")).href;
const loaded = await request("load", {
  base,
  manifest: build === "relaxed" ? manifest : { ...manifest, relaxed: undefined },
  search: true,
  stop: stop.buffer,
});
console.log(`checking ${loaded.build}`);
if (loaded.build !== manifest[build]) throw new Error(`the worker loaded ${loaded.build}, not ${manifest[build]}`);
// A thread that fails leaves its search waiting.
const failures = new BroadcastChannel(loaded.channel);
failures.onmessage = ({ data }) => {
  console.log(`a thread failed: ${data}`);
  process.exit(1);
};
const tables = join(dist, "tables");

const moves = (proof) => zstdDecompressSync(proof.subarray(proof.indexOf(10) + 1));
const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
const scratch = mkdtempSync(join(tmpdir(), "provari-compare-"));
// A build without tables is checked against native searches without them.
const withTables = manifest.tables.length > 0 ? ["--tablebases", tables] : [];
const inputs = [...withTables, "--network", join(dist, manifest.network), "--model", join(dist, manifest.model)];
const run = (...args) => JSON.parse(execFileSync(native, args));

/** Walks a longest line of the open proof, asking the page's view and `provari line` at each step. */
async function lines(file, root, longest) {
  let fen = root;
  for (let ply = 0; ply <= longest.length; ply += 1) {
    const here = await request("position", { fen });
    if (here.outcome !== null) return here.value === 0;
    const theirs = run("line", file, longest.slice(0, ply).join(" "), ...withTables);
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
// The positions one thread proves, for the threads to prove again.
const proven = [];
for (const [i, fen] of positions.entries()) {
  const ours = must(await request("prove", { request: { fen, nodes, memory_mb: memory } }));
  const proof = ours.proof;
  if (ours.found) proven.push(fen);
  const nativeProof = join(scratch, `native-${i}.pvr`);
  const theirs = run("prove", fen, "--nodes", String(nodes), "--memory", String(memory), ...inputs, "--proofs", nativeProof);
  const agree = ours.found === theirs.found && ours.nodes === theirs.nodes && (ours.mate ?? null) === (theirs.mate ?? null);
  let checks = "";
  if (proof) {
    const webProof = join(scratch, `web-${i}.pvr`);
    writeFileSync(webProof, proof);
    const nativeBytes = readFileSync(nativeProof);
    const sameMoves = same(moves(proof), moves(nativeBytes));
    const byNative = run("check", webProof, ...withTables);
    const byWeb = await request("check", { proof: nativeBytes });
    const opened = await request("open", { proof });
    const view = opened.mate === ours.mate && (await lines(webProof, opened.root, opened.longest));
    const tree = decoder.decode((await request("pgn")).pgn);
    const nativePgn = join(scratch, `native-${i}.pgn`);
    run("line", webProof, ...withTables, "--pgn", nativePgn);
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
  const ours = must(await request("prove", { request: { fen, nodes, memory_mb: memory, most } }));
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
  const ours = must(await request("prove", { request: { fen, nodes, memory_mb: memory, shorten: true } }));
  const theirs = run("prove", fen, "--nodes", String(nodes), "--memory", String(memory), "--shorten", ...inputs);
  const lengths = handed.map((proof) => header(proof).moves);
  let valid = true;
  for (const proof of handed) valid &&= (await request("check", { proof })).valid === true;
  const shrinking = lengths.every((length, i) => i === 0 || length < lengths[i - 1]);
  const agree = ours.found === theirs.found && ours.nodes === theirs.nodes &&
    (ours.mate ?? null) === (theirs.mate ?? null) && ours.first_mate === theirs.first_mate &&
    ours.passes === theirs.passes && ours.shortest === theirs.shortest &&
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
  const ours = must(await request("prove", { request: { fen, attacker: "white", nodes, memory_mb: memory } }));
  const proof = ours.proof;
  const nativeProof = join(scratch, "native-attacker.pvr");
  const theirs = run("prove", fen, "--attacker", "white", "--nodes", String(nodes), "--memory", String(memory),
    ...inputs, "--proofs", nativeProof);
  let agree = ours.found && theirs.found && ours.nodes === theirs.nodes && ours.mate === theirs.mate &&
    ours.attacker === "white" && theirs.attacker === "white";
  if (agree) {
    const webProof = join(scratch, "web-attacker.pvr");
    writeFileSync(webProof, proof);
    const nativeBytes = readFileSync(nativeProof);
    const opened = await request("open", { proof });
    agree = same(moves(proof), moves(nativeBytes)) &&
      run("check", webProof, ...withTables).valid === true &&
      (await request("check", { proof: nativeBytes })).valid === true &&
      opened.attacker === "white" && (await lines(webProof, opened.root, opened.longest));
  }
  ok &&= agree;
  console.log(`attacker: White to win with Black to move, found ${ours.found}/${theirs.found}, ` +
    `nodes ${ours.nodes}/${theirs.nodes}, mate ${ours.mate}/${theirs.mate}, ${agree ? "agree" : "DISAGREE"}`);
}
// Games already over, a king exploded and a stalemate: not searched, both
// say how each ended.
for (const fen of ["rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQ - 0 1", "k7/8/1Q6/8/8/8/8/6K1 b - - 0 1"]) {
  const ours = must(await request("prove", { request: { fen, nodes, memory_mb: memory } }));
  const theirs = run("prove", fen, "--nodes", String(nodes), "--memory", String(memory), ...inputs);
  ok &&= !ours.found && ours.nodes === 0 && ours.over === theirs.over && ours.winner === theirs.winner;
  console.log(`over: ${ours.over}/${theirs.over}, winner ${ours.winner}/${theirs.winner}, nodes ${ours.nodes}`);
}
// Threads: what one thread proves, 2 and 4 threads prove again with twice
// its nodes, as their searches overlap, each proof passing both checkers.
for (const threads of [2, 4]) {
  for (const [i, fen] of proven.entries()) {
    const ours = must(await request("prove", { request: { fen, nodes: 2 * nodes, memory_mb: memory, threads } }));
    let valid = ours.found && ours.threads === threads;
    if (valid) {
      const webProof = join(scratch, `threads-${threads}-${i}.pvr`);
      writeFileSync(webProof, ours.proof);
      valid = run("check", webProof, ...withTables).valid === true &&
        (await request("check", { proof: ours.proof })).valid === true;
    }
    ok &&= valid;
    console.log(`${threads} threads, ${i}: found ${ours.found}, mate ${ours.mate}, nodes ${ours.nodes}, ` +
      `${ours.ms} ms, ${valid ? "valid" : "WRONG"}`);
  }
}
// Shortening with threads, spawned again for each pass, on the last win one
// thread proved: every proof handed over valid and each shorter than the one
// before.
if (proven.length > 0) {
  handed.length = 0;
  const ours = must(await request("prove", {
    request: { fen: proven.at(-1), nodes: 2 * nodes, memory_mb: memory, shorten: true, threads: 2 },
  }));
  const lengths = handed.map((proof) => header(proof).moves);
  let valid = ours.found && lengths[0] === ours.first_mate && lengths.at(-1) >= ours.mate &&
    lengths.every((length, i) => i === 0 || length < lengths[i - 1]);
  for (const proof of handed) valid &&= (await request("check", { proof })).valid === true;
  ok &&= valid;
  console.log(`shorten, 2 threads: mate ${ours.first_mate}->${ours.mate}, passes ${ours.passes}; ` +
    `handed over ${lengths.join(" ")}, ${valid ? "valid" : "WRONG"}`);
}
// Stops: a search stopped through the page's word replies at once, with one
// thread and with four, and the next, the word cleared, proves as before;
// one stopped as it starts spawns no threads. Each search's threads' workers
// close once it ends, where /proc lists this process's threads.
const tasks = () => {
  try {
    return readdirSync("/proc/self/task").length;
  } catch {
    return null;
  }
};
const settle = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
{
  const start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  const alone = tasks();
  const ended = () => settle(1500).then(() => alone === null || tasks() === alone);
  for (const threads of [1, 4]) {
    Atomics.store(stop, 0, 0);
    const begun = performance.now();
    const searching = request("prove", { request: { fen: start, seconds: 60, memory_mb: memory, threads } });
    setTimeout(() => Atomics.store(stop, 0, 1), 1000);
    const ours = must(await searching);
    const ms = Math.round(performance.now() - begun);
    const closed = await ended();
    ok &&= !ours.found && ms < 2000 && closed;
    console.log(`stopped after 1 s, ${threads} thread${threads > 1 ? "s" : ""}: replied after ${ms} ms, ` +
      `${ours.nodes} nodes, threads' workers closed ${closed}`);
  }
  Atomics.store(stop, 0, 0);
  const next = must(await request("prove", { request: { fen: positions[1], nodes: 2 * nodes, memory_mb: memory, threads: 4 } }));
  const closed = await ended();
  ok &&= next.found && closed;
  console.log(`the word cleared, 4 threads: found ${next.found}, mate ${next.mate}, nodes ${next.nodes}, ` +
    `threads' workers closed ${closed}`);
  Atomics.store(stop, 0, 1);
  const begun = performance.now();
  const early = must(await request("prove", { request: { fen: start, seconds: 60, memory_mb: memory, threads: 8 } }));
  const ms = Math.round(performance.now() - begun);
  Atomics.store(stop, 0, 0);
  const none = await ended();
  ok &&= !early.found && early.nodes <= 4096 && none;
  console.log(`stopped as it starts, 8 threads: replied after ${ms} ms, ${early.nodes} nodes, ` +
    `threads' workers closed ${none}`);
  if (alone === null) console.log("(no /proc: the threads' workers not counted)");
}
// A thread that traps says why on the channel that `load` names, as its
// search waits for it in vain: a thread's worker given a module whose
// thread does nothing else, and a thread as thread-spawn hands it over.
{
  const TRAPS = new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, // the header
    1, 6, 1, 96, 2, 127, 127, 0, // a type: two i32 parameters, no results
    2, 16, 1, 3, ...encoder.encode("env"), 6, ...encoder.encode("memory"), 2, 3, 1, 1, // a shared page
    3, 2, 1, 0, // a function of that type
    7, 21, 1, 17, ...encoder.encode("wasi_thread_start"), 0, 0, // exported
    10, 5, 1, 3, 0, 0, 11, // its code, without locals: unreachable
  ]);
  const slots = new Int32Array(new SharedArrayBuffer(4 * (4 * 256 + 2)));
  const channel = `provari-compare-${process.pid}`;
  const listening = new BroadcastChannel(channel);
  const failure = new Promise((resolve) => {
    listening.onmessage = ({ data }) => resolve(data);
    setTimeout(() => resolve(null), 10_000);
  });
  const thread = new BrowserWorker(new URL("./worker.js", import.meta.url), { name: "provari thread" });
  const ready = new Promise((resolve) => {
    thread.onmessage = ({ data }) => data.ready !== undefined && resolve();
  });
  const shared = new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true });
  const module = await WebAssembly.compile(TRAPS);
  thread.postMessage({ thread: { module, memory: shared, slots: slots.buffer, slot: 0, channel } });
  await ready;
  Atomics.store(slots, 1, 1);
  Atomics.store(slots, 0, 3);
  Atomics.notify(slots, 0);
  const said = await failure;
  listening.close();
  await thread.terminate();
  ok &&= typeof said === "string" && said.includes("unreachable");
  console.log(`a thread that traps: ${said === null ? "SAID NOTHING" : `said "${said}"`}`);
  // A thread's worker waiting in its slot closes itself when told to, or
  // once its search stops, and otherwise waits on: terminating one that has
  // closed finds nothing to end.
  const waiting = async (told) => {
    const slots = new Int32Array(new SharedArrayBuffer(4 * (4 * 256 + 2)));
    const word = new Int32Array(new SharedArrayBuffer(4));
    const worker = new BrowserWorker(new URL("./worker.js", import.meta.url), { name: "provari thread" });
    const ready = new Promise((resolve) => {
      worker.onmessage = ({ data }) => data.ready !== undefined && resolve();
    });
    worker.postMessage({ thread: { module, memory: shared, slots: slots.buffer, slot: 0, channel, stop: word.buffer, at: 0 } });
    await ready;
    told(slots, word);
    await settle(1500);
    return (await worker.terminate()) === undefined;
  };
  const closing = await waiting((slots) => {
    Atomics.store(slots, 0, 4);
    Atomics.notify(slots, 0);
  });
  const stopped = await waiting((_, word) => Atomics.store(word, 0, 1));
  const untold = await waiting(() => {});
  ok &&= closing && stopped && !untold;
  console.log(`a thread's worker waiting: closes when told ${closing}, when its search stops ${stopped}, ` +
    `otherwise ${untold ? "CLOSES" : "waits on"}`);
}
const mean = Math.exp(rates.reduce((sum, r) => sum + Math.log(r), 0) / rates.length);
console.log(`${ok ? "all agree" : "DISAGREEMENT"}; the browser build takes x${mean.toFixed(2)} the native time`);
process.exit(ok ? 0 : 1);
