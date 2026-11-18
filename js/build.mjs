// Builds the WebAssembly module of the npm package:
//
//   node build.mjs             # cargo build + js/runtime/pkpdutils-core.wasm
//   node build.mjs --check     # fails when the module is missing or older than the crate
//
// The module is not committed: it is built from the crates of this repository, by `npm run build`
// before the tests of `js/test/`, by the `js` job of continuous integration and by `prepack`
// before a publish. `--check` is what every one of them runs first, so a stale module is an error
// and never a test which passes against yesterday's kernel.

import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";
import { readFileSync } from "node:fs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, "..");
const TARGET = path.join(
  process.env.CARGO_TARGET_DIR ?? path.join(ROOT, "target"),
  "wasm32-unknown-unknown/release/pkpdutils_wasm.wasm",
);
const RUNTIME = path.join(HERE, "runtime", "pkpdutils-core.wasm");

/** The newest modification time under a directory, in milliseconds. */
function newest(directory) {
  let latest = 0;
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name);
    latest = Math.max(latest, entry.isDirectory() ? newest(full) : statSync(full).mtimeMs);
  }
  return latest;
}

/** The newest source of the kernel and of this host: the crates and the lock file. */
function newestSource() {
  return Math.max(newest(path.join(ROOT, "crates")), statSync(path.join(ROOT, "Cargo.lock")).mtimeMs);
}

if (process.argv.includes("--check")) {
  let built;
  try {
    built = statSync(RUNTIME);
  } catch {
    console.error(`${RUNTIME} is not built, run\n  npm run build`);
    process.exit(1);
  }
  if (built.mtimeMs < newestSource()) {
    console.error(`${RUNTIME} is older than the crates it was built from, run\n  npm run build`);
    process.exit(1);
  }
  console.log(`${path.relative(ROOT, RUNTIME)} is up to date (${built.size} bytes)`);
  process.exit(0);
}

execFileSync(
  "cargo",
  [
    "build",
    "-p",
    "pkpdutils-wasm",
    "--release",
    "--no-default-features",
    "--target",
    "wasm32-unknown-unknown",
    "--locked",
  ],
  { cwd: ROOT, stdio: "inherit" },
);
mkdirSync(path.dirname(RUNTIME), { recursive: true });
copyFileSync(TARGET, RUNTIME);
const bytes = readFileSync(RUNTIME);
const { version } = JSON.parse(readFileSync(path.join(HERE, "package.json"), "utf8"));
console.log(
  `pkpdutils-core ${version}: ${path.relative(ROOT, RUNTIME)}, ${bytes.length} bytes, ` +
    `${gzipSync(bytes).length} gzipped`,
);
