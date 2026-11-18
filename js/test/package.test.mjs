// The package: its version, its files and the module it carries.

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { createRuntime } from "../loader.mjs";
import { crateVersion, manifest as read } from "../version.mjs";
import { wasmBytes } from "./helpers.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PACKAGE = path.join(HERE, "..");
const manifest = read();

test("the package version is the version of the crates", () => {
  assert.equal(manifest.version, crateVersion());
});

test("every file of the package exists and every export resolves", () => {
  for (const file of manifest.files) {
    assert.ok(existsSync(path.join(PACKAGE, file)), file);
  }
  for (const target of Object.values(manifest.exports)) {
    assert.ok(existsSync(path.join(PACKAGE, target)), target);
  }
  // the loader, the pool and the worker are the three files a page imports, and each of them is
  // packed: an export which resolves in the checkout and not in the tarball is the one mistake
  // `files` makes
  for (const name of ["loader.mjs", "pool.mjs", "worker.mjs"]) {
    assert.ok(manifest.files.includes(name), name);
  }
  assert.equal(manifest.dependencies, undefined, "the package has no dependency");
  assert.equal(manifest.license, "MIT");
  assert.equal(manifest.type, "module");
});

test("the module reports the version of the package", async () => {
  const runtime = await createRuntime({ module: wasmBytes() });
  assert.equal(runtime.about.version, manifest.version);
});
