// provari in a worker: loads the WebAssembly build and the data the page
// asks for once, then answers its requests one at a time. The page sends
// {id, type, ...} and gets back {id, result} or {id, error}, with `fatal`
// once the module can no longer be trusted; a search reports {progress}
// as it goes, and while it shortens a win, {found} with each shorter win's
// proof; loading reports {loading} steps.
//
// Bytes pass through the module's memory: `alloc` for what we write, replies
// packed as pointer (low 32 bits) and length (high 32 bits), freed with
// `release` once read.

const decoder = new TextDecoder();
const encoder = new TextEncoder();
let provari = null;

function put(bytes) {
  const ptr = provari.alloc(bytes.length);
  new Uint8Array(provari.memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}

function take(packed) {
  const ptr = Number(packed & 0xffffffffn);
  const len = Number(packed >> 32n);
  const bytes = new Uint8Array(provari.memory.buffer, ptr, len).slice();
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

/**
 * Loads the module and, from the build's manifest, the tables, and for a
 * search the network and the model too. Files are named after their
 * contents, so that browsers never mix two builds.
 */
async function load({ base, manifest, search }) {
  const started = performance.now();
  const imports = {
    provari: {
      now_ms: () => performance.now(),
      progress: (ptr, len) => {
        const line = decoder.decode(new Uint8Array(provari.memory.buffer, ptr, len));
        postMessage({ progress: JSON.parse(line) });
      },
      found: (ptr, len) => {
        const proof = new Uint8Array(provari.memory.buffer, ptr, len).slice();
        postMessage({ found: proof }, [proof.buffer]);
      },
    },
  };
  postMessage({ loading: "provari" });
  const build = chosen(manifest);
  const { instance } = await WebAssembly.instantiateStreaming(fetch(base + build), imports);
  provari = instance.exports;
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
  return { result: { pieces, build, ms: Math.round(performance.now() - started) } };
}

const handlers = {
  load,
  prove({ request }) {
    const result = json(provari.prove(...put(text(request))));
    // A request refused before any search, as for more memory than the
    // browser gives, has no result.
    if (result.found === undefined) throw new Error(result.error);
    const proof = result.found ? take(provari.take_proof()) : null;
    return { result: { ...result, proof }, transfer: proof ? [proof.buffer] : [] };
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
};

onmessage = async ({ data }) => {
  const { id, type } = data;
  try {
    const handler = handlers[type];
    if (handler === undefined) throw new Error(`no request ${type}`);
    const { result, transfer = [] } = await handler(data);
    postMessage({ id, result }, transfer);
  } catch (error) {
    // A trap leaves the module's state unknown: the page starts another.
    const fatal = error instanceof WebAssembly.RuntimeError;
    postMessage({ id, error: String(error?.message ?? error), fatal });
  }
};
