// Runs web/worker.js in Node as a cross-origin isolated page's browser runs
// it, for checks of the browser build without a browser (compare.mjs): in a
// worker thread with a worker's globals, the workers it starts in worker
// threads of their own, and its files fetched from disk by their file: URLs.
//
//     import { BrowserWorker } from "./node.mjs";
//     const worker = new BrowserWorker(new URL("./worker.js", import.meta.url));
//     worker.onmessage = ({ data }) => console.log(data);
//     worker.postMessage({ id: 0, type: "load", base, manifest, search: true });
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import { isMainThread, parentPort, Worker as Thread, workerData } from "node:worker_threads";

/** A browser's Worker, running the classic script at `url`. */
export class BrowserWorker {
  #thread;
  onmessage = null;
  onerror = null;

  constructor(url, { name } = {}) {
    this.#thread = new Thread(new URL(import.meta.url), { workerData: { script: String(url), name } });
    this.#thread.on("message", (data) => this.onmessage?.({ data }));
    this.#thread.on("error", (error) =>
      this.onerror?.({ message: String(error?.message ?? error), preventDefault() {} }),
    );
  }

  postMessage(message, transfer = []) {
    this.#thread.postMessage(message, transfer);
  }

  terminate() {
    return this.#thread.terminate();
  }
}

/** A file from disk as a server sends it, or HTTP 404 where none is. */
async function fetchFile(url) {
  let body;
  try {
    body = readFileSync(fileURLToPath(new URL(url)));
  } catch {
    return new Response(null, { status: 404 });
  }
  const type = String(url).endsWith(".wasm") ? "application/wasm" : "application/octet-stream";
  return new Response(body, { headers: { "Content-Length": String(body.length), "Content-Type": type } });
}

if (!isMainThread && workerData?.script !== undefined) {
  const { script, name } = workerData;
  let handler = null;
  Object.assign(globalThis, {
    self: globalThis,
    name,
    location: { href: script },
    crossOriginIsolated: true,
    Worker: BrowserWorker,
    fetch: fetchFile,
    postMessage: (message, transfer = []) => parentPort.postMessage(message, transfer),
    // A worker's own end: in a worker thread, process.exit ends the thread.
    close: () => process.exit(0),
  });
  Object.defineProperty(globalThis, "onmessage", {
    get: () => handler,
    set: (value) => {
      handler = value;
    },
  });
  parentPort.on("message", (data) => handler?.({ data }));
  vm.runInThisContext(readFileSync(fileURLToPath(new URL(script)), "utf8"), { filename: script });
}
