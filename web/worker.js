// provari in a worker: loads the WebAssembly build and the data the page
// asks for once, then answers its requests one at a time. The page sends
// {id, type, ...} and gets back {id, result} or {id, error}, with `fatal`
// once the module can no longer be trusted; a search reports {progress}
// as it goes, and while it shortens a win, {found} with each shorter win's
// proof; loading reports {loading} steps.
//
// The module's memory is shared with a search's other threads, each in a
// worker that this one starts from this script before the search spawns it
// (a thread spawned cannot wait for a worker to load), for that search
// alone: a browser may never end a worker that waits for a thread (WebKit
// does not), which then keeps the memory, so each closes itself once its
// search ends. Browsers share memory only on a cross-origin isolated page. A
// thread that fails says why on the BroadcastChannel that `load` names, as
// this worker cannot answer while its search waits for that thread: the page
// then ends this worker, and its threads' with it.
//
// Nor can it answer while it searches, so the page stops a search through an
// Int32Array of one word that it hands `load` (`stop`): 1 to stop the search,
// 0 before the next. Each thread's look at the clock, as every 1,024 nodes,
// copies that word into the module's stop flag, which the search's threads
// read: the page never holds the module's memory, which its garbage would
// keep. A thread's worker whose search has stopped closes at once, as its
// page may have gone; `end`, the last request, answers once the search
// before it has ended.
//
// Bytes pass through the module's memory: `alloc` for what we write, replies
// packed as pointer (low 32 bits) and length (high 32 bits), freed with
// `release` once read.

const decoder = new TextDecoder();
const encoder = new TextEncoder();
let provari = null;
let memory = null;
// What the module wrote to standard error during the request, as a panic's
// message, for the error that follows.
let complaints = [];
// The page's word that stops a search, and the module's stop flag.
let asked = null;
let stopFlag = null;

// The running search's threads' workers, a slot each of [state, thread id,
// start argument, unused], then the last thread id given and a count of
// slots freed. A slot is FREE for a thread, CLAIMED while one is handed to
// it, RUNNING it, and CLOSING once its worker is to close itself.
const SLOTS = 256;
const FREE = 1;
const CLAIMED = 2;
const RUNNING = 3;
const CLOSING = 4;
const LAST_ID = 4 * SLOTS;
const FREED = 4 * SLOTS + 1;
let pool = null;

function put(bytes) {
  const ptr = provari.alloc(bytes.length);
  new Uint8Array(memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}

/** A reply's bytes, copied out of the shared memory, which decoders and transfers refuse. */
function take(packed) {
  const ptr = Number(packed & 0xffffffffn);
  const len = Number(packed >> 32n);
  const bytes = new Uint8Array(memory.buffer, ptr, len).slice();
  provari.release(ptr, len);
  return bytes;
}

const json = (packed) => JSON.parse(decoder.decode(take(packed)));

/**
 * A module of one function using relaxed SIMD's dot product of bytes
 * (i32x4.relaxed_dot_i8x16_i7x16_add_s), the instruction the faster build
 * adds: a browser that accepts it gets that build, and others the plain one.
 * Validating runs nothing.
 */
const RELAXED_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, // the header
  1, 5, 1, 96, 0, 1, 123, // a type: no parameters, one v128
  3, 2, 1, 0, // a function of that type
  10, 19, 1, 17, 0, // its code, without locals:
  65, 1, 253, 15, 65, 2, 253, 15, 65, 3, 253, 17, // three splats,
  253, 147, 2, // the dot product,
  11, // end
]);

/** The build this browser runs: with relaxed SIMD where it has it. */
function chosen(manifest) {
  return manifest.relaxed !== undefined && WebAssembly.validate(RELAXED_PROBE)
    ? manifest.relaxed
    : manifest.wasm;
}
const text = (value) => encoder.encode(JSON.stringify(value));

/** A reply that must not be an error. */
function ok(reply) {
  if (reply.error !== undefined) throw new Error(reply.error);
  return reply;
}

async function fetched(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

/**
 * Fetches a large file, reporting its share received as `{loading: what,
 * share, bytes}` every tenth of a second: `bytes` is what crosses the network
 * (compressed, if the server compresses it), and the share counts the file
 * as received, against its size `total`.
 */
async function fetchedWithProgress(url, what, total) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const bytes = Number(response.headers.get("Content-Length")) || total;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  let reported = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (performance.now() - reported > 100) {
      reported = performance.now();
      postMessage({ loading: what, share: total > 0 ? Math.min(1, received / total) : null, bytes });
    }
  }
  const whole = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    whole.set(chunk, offset);
    offset += chunk.length;
  }
  return whole;
}

/** The pages of memory the module takes at least, from its import of it. */
function leastPages(bytes) {
  let i = 8;
  const leb = () => {
    let result = 0;
    let shift = 0;
    let byte;
    do {
      byte = bytes[i++];
      result += (byte & 0x7f) * 2 ** shift;
      shift += 7;
    } while (byte & 0x80);
    return result;
  };
  while (i < bytes.length) {
    const section = bytes[i++];
    const size = leb();
    const end = i + size;
    if (section === 2) {
      for (let imports = leb(); imports > 0; imports--) {
        // The module's name, then the import's.
        for (let names = 0; names < 2; names++) {
          const len = leb();
          i += len;
        }
        const kind = bytes[i++];
        if (kind === 2) {
          i++; // the limits' flags
          return leb();
        }
        if (kind !== 0) throw new Error(`provari imports something of kind ${kind}`);
        leb(); // a function's type
      }
    }
    i = end;
  }
  throw new Error("provari imports no memory");
}

/**
 * The memory the module and its threads share, from the pages it takes up to
 * as much as the browser gives, of 4, 2 and 1 GB and 512 MB.
 */
function sharedMemory(initial) {
  if (!globalThis.crossOriginIsolated) {
    throw new Error("provari needs a cross-origin isolated page, for the memory its threads share");
  }
  let error;
  for (const mb of [4096, 2048, 1024, 512]) {
    try {
      return new WebAssembly.Memory({ initial, maximum: mb * 16, shared: true });
    } catch (e) {
      error = e;
    }
  }
  throw error;
}

/** Milliseconds on one clock for every worker: each one's performance.now() counts from its own start. */
const now = () => performance.timeOrigin + performance.now();

/** Whether the page has asked the search to stop, as it does when it goes. */
const stopping = () => asked !== null && Atomics.load(asked, 0) !== 0;

/** The search's clock, which first copies the page's word into the module's stop flag where they differ. */
function clock() {
  const word = stopping() ? 1 : 0;
  if (stopFlag !== null && stopFlag[0] !== word) Atomics.store(stopFlag, 0, word);
  return now();
}

// What the module writes, line by line, to the console; and to standard
// error, kept for the request's error.
const unfinished = { 1: "", 2: "" };
function write(fd, bytes) {
  const lines = (unfinished[fd] + decoder.decode(bytes)).split("\n");
  unfinished[fd] = lines.pop();
  for (const line of lines) {
    if (fd === 2) {
      complaints.push(line);
      console.error(line);
    } else {
      console.log(line);
    }
  }
}

/**
 * What the module imports in this worker: the shared memory, the clock, the
 * page's handlers for its progress and its shorter wins, the few WASI calls
 * that its C library and Rust's make, and WASI's thread-spawn.
 */
function imports(module, handlers) {
  const view = () => new DataView(memory.buffer);
  const wasi = {
    environ_sizes_get(count, size) {
      view().setUint32(count, 0, true);
      view().setUint32(size, 0, true);
      return 0;
    },
    environ_get: () => 0,
    random_get(ptr, len) {
      // getRandomValues takes no view of shared memory: a copy.
      const random = new Uint8Array(len);
      for (let i = 0; i < len; i += 65536) crypto.getRandomValues(random.subarray(i, i + 65536));
      new Uint8Array(memory.buffer, ptr, len).set(random);
      return 0;
    },
    clock_time_get(id, _precision, out) {
      const ms = id === 0 ? Date.now() : now();
      view().setBigUint64(out, BigInt(Math.round(ms * 1e6)), true);
      return 0;
    },
    fd_write(fd, iovs, count, written) {
      if (fd !== 1 && fd !== 2) return 8; // EBADF
      let total = 0;
      for (let i = 0; i < count; i++) {
        const ptr = view().getUint32(iovs + 8 * i, true);
        const len = view().getUint32(iovs + 8 * i + 4, true);
        write(fd, new Uint8Array(memory.buffer, ptr, len).slice());
        total += len;
      }
      view().setUint32(written, total, true);
      return 0;
    },
    proc_exit(code) {
      throw new Error(`provari exited with ${code}`);
    },
    sched_yield: () => 0,
  };
  const imported = {
    env: { memory },
    provari: { now_ms: clock, ...handlers },
    wasi: { "thread-spawn": spawn },
    wasi_snapshot_preview1: {},
  };
  // A call that the module starts making unknown to this file fails as
  // ENOSYS, rather than the module's instantiation.
  for (const { module: from, name, kind } of WebAssembly.Module.imports(module)) {
    if (from === "wasi_snapshot_preview1" && kind === "function") {
      imported[from][name] = wasi[name] ?? (() => 52);
    }
  }
  return imported;
}

/**
 * WASI's thread-spawn: hands the thread to a ready worker, or fails as
 * EAGAIN, at once for a search stopped, whose threads' workers close. A
 * worker frees its slot a moment after the search has joined its thread, so
 * a search that spawns again at once waits for that, a second at most.
 */
function spawn(arg) {
  if (stopping()) return -6;
  const slots = pool.slots;
  const until = performance.now() + 1000;
  for (;;) {
    const freed = Atomics.load(slots, FREED);
    for (let slot = 0; slot < SLOTS; slot++) {
      if (Atomics.compareExchange(slots, 4 * slot, FREE, CLAIMED) !== FREE) continue;
      const tid = Atomics.add(slots, LAST_ID, 1) + 1;
      Atomics.store(slots, 4 * slot + 1, tid);
      Atomics.store(slots, 4 * slot + 2, arg);
      Atomics.store(slots, 4 * slot, RUNNING);
      Atomics.notify(slots, 4 * slot);
      return tid;
    }
    const left = until - performance.now();
    if (left <= 0) return -6;
    Atomics.wait(slots, FREED, freed, left);
  }
}

const newSlots = () => new Int32Array(new SharedArrayBuffer(4 * (4 * SLOTS + 2)));

/** Starts threads' workers until `count` of them, at most SLOTS, wait ready in their slots. */
function grow(count) {
  const wanted = Math.min(count, SLOTS);
  while (pool.workers.length < wanted) {
    const slot = pool.workers.length;
    const { module, slots, channel } = pool;
    const worker = new Worker(self.location.href, { name: `provari thread ${slot + 1}` });
    // A worker of a search before says nothing to this one's.
    worker.onmessage = ({ data }) => {
      if (pool.slots !== slots) return;
      if (data.ready !== undefined) pool.ready += 1;
      else if (data.failed !== undefined) pool.failed = new Error(`a thread's worker failed: ${data.failed}`);
      pool.waiting = pool.waiting.filter((settled) => !settled());
    };
    worker.onerror = (event) => {
      event.preventDefault();
      if (pool.slots !== slots) return;
      pool.failed = new Error(`a thread's worker failed: ${event.message}`);
      pool.waiting = pool.waiting.filter((settled) => !settled());
    };
    const stop = asked?.buffer;
    const at = stopFlag.byteOffset;
    worker.postMessage({ thread: { module, memory, slots: slots.buffer, slot, channel, stop, at } });
    pool.workers.push(worker);
  }
  return new Promise((resolve, reject) => {
    const settled = () => {
      if (pool.ready >= wanted) resolve();
      else if (pool.failed !== null) reject(pool.failed);
      else return false;
      return true;
    };
    if (!settled()) pool.waiting.push(settled);
  });
}

/**
 * Has the search's threads' workers close themselves, and gives the next
 * search's slots of their own, which none of these reads, going on with the
 * thread ids.
 */
function closeThreads() {
  if (pool === null || pool.workers.length === 0) return;
  for (let slot = 0; slot < pool.workers.length; slot++) {
    Atomics.store(pool.slots, 4 * slot, CLOSING);
    Atomics.notify(pool.slots, 4 * slot);
  }
  const slots = newSlots();
  Atomics.store(slots, LAST_ID, Atomics.load(pool.slots, LAST_ID));
  Object.assign(pool, { slots, workers: [], ready: 0, waiting: [], failed: null });
}

/**
 * A thread's worker: an instance of the module on the shared memory, which
 * runs each thread of its search spawned into its slot, then waits for the
 * next, until told to close or the search stops.
 */
function thread({ module, memory: shared, slots, slot, channel, stop, at }) {
  memory = shared;
  pool = { slots: new Int32Array(slots) };
  if (stop !== undefined) asked = new Int32Array(stop);
  stopFlag = new Uint8Array(memory.buffer, at, 1);
  try {
    provari = new WebAssembly.Instance(module, imports(module, { progress() {}, found() {} })).exports;
  } catch (error) {
    postMessage({ failed: String(error?.message ?? error) });
    return;
  }
  const state = 4 * slot;
  // Free for threads, unless told to close while it loaded.
  if (Atomics.compareExchange(pool.slots, state, 0, FREE) !== 0) {
    close();
    return;
  }
  postMessage({ ready: slot });
  for (;;) {
    for (let seen; (seen = Atomics.load(pool.slots, state)) !== RUNNING; ) {
      // Told to close, or its search stopped while no thread is handed to it.
      const stopped = seen === FREE && stopping() && Atomics.compareExchange(pool.slots, state, FREE, CLOSING) === FREE;
      if (seen === CLOSING || stopped) {
        close();
        return;
      }
      // A second at most, to see a stop that no one tells it of.
      Atomics.wait(pool.slots, state, seen, 1000);
    }
    complaints = [];
    try {
      provari.wasi_thread_start(Atomics.load(pool.slots, state + 1), Atomics.load(pool.slots, state + 2));
    } catch (error) {
      // The search waits for this thread in vain, and its slot stays taken.
      new BroadcastChannel(channel).postMessage(said(error));
      return;
    }
    // Told to close while it ran a thread, or its search stopped, it closes now.
    if (stopping() || Atomics.compareExchange(pool.slots, state, RUNNING, FREE) !== RUNNING) {
      Atomics.store(pool.slots, state, CLOSING);
      close();
      return;
    }
    Atomics.add(pool.slots, FREED, 1);
    Atomics.notify(pool.slots, FREED);
  }
}

/** An error's message, with what the module said before it, as a panic's message. */
function said(error) {
  const message = String(error?.message ?? error);
  const why = complaints.filter((line) => line !== "" && !line.startsWith("note: ")).join(" ");
  return why === "" ? message : `${message}: ${why}`;
}

/**
 * Loads the module and, from the build's manifest, the tables, and for a
 * search the network and the model too. Files are named after their
 * contents, so that browsers never mix two builds.
 */
async function load({ base, manifest, search, stop }) {
  const started = performance.now();
  postMessage({ loading: "provari" });
  const build = chosen(manifest);
  const bytes = await fetched(base + build);
  const module = await WebAssembly.compile(bytes);
  memory = sharedMemory(leastPages(bytes));
  const instance = await WebAssembly.instantiate(
    module,
    imports(module, {
      progress: (ptr, len) => {
        const line = decoder.decode(new Uint8Array(memory.buffer, ptr, len).slice());
        postMessage({ progress: JSON.parse(line) });
      },
      found: (ptr, len) => {
        const proof = new Uint8Array(memory.buffer, ptr, len).slice();
        postMessage({ found: proof }, [proof.buffer]);
      },
    }),
  );
  provari = instance.exports;
  if (stop !== undefined) asked = new Int32Array(stop);
  stopFlag = new Uint8Array(memory.buffer, provari.stop_flag(), 1);
  // The C library's start, once, in the first thread: it knows its threads from it.
  provari._initialize();
  const channel = `provari-${crypto.randomUUID()}`;
  pool = { module, slots: newSlots(), channel, workers: [], ready: 0, waiting: [], failed: null };
  if (search) {
    postMessage({ loading: "network" });
    const network = await fetchedWithProgress(base + manifest.network, "network", manifest.network_bytes ?? 0);
    ok(json(provari.load_network(...put(network))));
    ok(json(provari.load_model(...put(await fetched(base + manifest.model)))));
  }
  postMessage({ loading: "tablebases" });
  for (const name of manifest.tables) {
    const bytes = await fetched(`${base}tables/${name}`);
    ok(json(provari.add_table(...put(encoder.encode(name)), ...put(bytes))));
  }
  const { pieces } = ok(json(provari.open_tables()));
  return { result: { pieces, build, channel, ms: Math.round(performance.now() - started) } };
}

const handlers = {
  load,
  async prove({ request }) {
    // Each of the search's other threads needs a worker, ready before the
    // search spawns it, and closed once it ends; provari refuses threads that
    // are no count.
    const threads = request.threads;
    try {
      await grow(Number.isInteger(threads) && threads > 1 ? threads - 1 : 0);
      const result = json(provari.prove(...put(text(request))));
      // A request refused before any search, as for more memory than the
      // browser gives, has no result.
      if (result.found === undefined) throw new Error(result.error);
      const proof = result.found ? take(provari.take_proof()) : null;
      return { result: { ...result, proof }, transfer: proof ? [proof.buffer] : [] };
    } finally {
      closeThreads();
    }
  },
  position: ({ fen }) => ({ result: ok(json(provari.show_position(...put(text({ fen }))))) }),
  open: ({ proof }) => ({ result: ok(json(provari.open_proof(...put(new Uint8Array(proof))))) }),
  close() {
    provari.close_proof();
    return { result: {} };
  },
  pgn() {
    ok(json(provari.proof_pgn()));
    const pgn = take(provari.take_pgn());
    return { result: { pgn }, transfer: [pgn.buffer] };
  },
  check: ({ proof }) => ({ result: json(provari.check(...put(new Uint8Array(proof)))) }),
  // The last request, answered once the requests before it are.
  end() {
    closeThreads();
    return { result: {} };
  },
};

async function answer(data) {
  const { id, type } = data;
  complaints = [];
  try {
    const handler = handlers[type];
    if (handler === undefined) throw new Error(`no request ${type}`);
    const { result, transfer = [] } = await handler(data);
    postMessage({ id, result }, transfer);
  } catch (error) {
    // A trap leaves the module's state unknown: the page starts another.
    const fatal = error instanceof WebAssembly.RuntimeError;
    postMessage({ id, error: fatal ? said(error) : String(error?.message ?? error), fatal });
  }
}

let queue = Promise.resolve();
onmessage = ({ data }) => {
  // A thread's worker is told so first, and runs threads from then on.
  if (data.thread !== undefined) {
    onmessage = null;
    thread(data.thread);
    return;
  }
  queue = queue.then(() => answer(data));
};
