// A pool which is created and never used: the script has to exit without `close()`.
//
// `js/test/pool.test.mjs` runs it as a subprocess. A seat is unreferenced as soon as it is ready
// and not only once it has run something, so a page or a script which builds a pool and then
// decides it has nothing to analyse does not hang on it.

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createPool } from "../../pool.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const pool = await createPool({
  module: readFileSync(path.join(HERE, "..", "..", "runtime", "pkpdutils-core.wasm")),
  workers: 2,
});
console.log(pool.workers, pool.about.version);
