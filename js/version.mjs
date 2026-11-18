// The version of the crates and the version of the package, which are one version.
//
// The module reports its own (`pkpd_about`), so the three can be compared: `js/build.mjs` prints
// it, `js/prepack.mjs` refuses to pack when the first two differ and `js/test/package.test.mjs`
// compares all three. A dev file, not part of the published package.

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** The version of `[workspace.package]` of the root manifest, which every crate takes. */
export function crateVersion() {
  const manifest = readFileSync(path.join(HERE, "..", "Cargo.toml"), "utf8");
  const found = /^\[workspace\.package\][^[]*?^version = "([^"]+)"/ms.exec(manifest);
  if (found === null) {
    throw new Error("Cargo.toml carries no [workspace.package] version");
  }
  return found[1];
}

/** The manifest of the npm package. */
export function manifest() {
  return JSON.parse(readFileSync(path.join(HERE, "package.json"), "utf8"));
}
