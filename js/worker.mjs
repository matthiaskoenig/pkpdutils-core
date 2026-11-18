/**
 * One instance of the module in one worker.
 *
 * The same file runs as a module worker in a browser and as a `worker_threads` worker in Node,
 * which it tells apart by whether `self` is defined; nothing else differs between the two. The
 * protocol is three messages in and two out:
 *
 * - `{kind: "init", module}`: compile and instantiate, answered with `{kind: "ready", about}`.
 * - `{kind: "analyze" | "summarize", id, request}`: answered with
 *   `{kind: "result", id, header, columns, values}`, the payload transferred, or with
 *   `{kind: "error", id, error: {kind, message}}`.
 *
 * The module is compiled once on the thread which owns the pool and posted as a
 * `WebAssembly.Module`, which both hosts carry through structured clone: a worker neither
 * fetches nor compiles the bytes again, so a pool of eight pays for one compilation.
 */

import { createRuntime } from "./loader.mjs";

/** The runtime of this worker, once `init` has arrived. */
let runtime = null;

/**
 * Answer one message.
 *
 * @param {object} message what the pool posted
 * @param {(message: object, transfer?: object[]) => void} post how to answer
 */
async function handle(message, post) {
  if (message.kind === "init") {
    runtime = await createRuntime({ module: message.module });
    post({ kind: "ready", about: runtime.about });
    return;
  }
  const { id, kind, request } = message;
  if (runtime === null) {
    post({ kind: "error", id, error: { kind: "internal", message: "this worker has no module" } });
    return;
  }
  try {
    const answer = kind === "summarize" ? runtime.summarize(request) : runtime.analyze(request);
    post(
      {
        kind: "result",
        id,
        header: answer.header,
        columns: answer.columns,
        values: answer.values,
      },
      [answer.values.buffer],
    );
  } catch (bad) {
    post({
      kind: "error",
      id,
      error: { kind: bad.kind ?? "internal", message: bad.message ?? String(bad) },
    });
  }
}

if (typeof self === "undefined") {
  const { parentPort } = await import("node:worker_threads");
  parentPort.on("message", (message) => {
    handle(message, (answer, transfer) => parentPort.postMessage(answer, transfer));
  });
} else {
  self.addEventListener("message", (event) => {
    handle(event.data, (answer, transfer) => self.postMessage(answer, transfer));
  });
}
