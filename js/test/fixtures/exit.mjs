// A script which never calls `close()`: it prints the row count of an answer and has to exit.
//
// `js/test/pool.test.mjs` runs it as a subprocess, which is the only way to see it: the workers
// of a node pool are unreferenced while it is idle, so a script which forgets `close()` still
// exits, and they are referenced again while a job is in flight, so an answer is never lost to
// an exit. The test runner of node keeps its own event loop alive and would hide both.

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createPool } from "../../pool.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const pool = await createPool({
  module: readFileSync(path.join(HERE, "..", "..", "runtime", "pkpdutils-core.wasm")),
  workers: 2,
});
const { cases } = JSON.parse(readFileSync(path.join(HERE, "cases.json"), "utf8"));
const answer = await pool.analyze(cases.find((one) => one.name === "two_subjects").request);
console.log(answer.header.n_rows);
