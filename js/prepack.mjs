// What `npm pack` and `npm publish` run first:
//
//   node prepack.mjs
//
// It builds the module, copies the license of the repository next to the package and refuses to
// pack when the version of `package.json` is not the version of the crates or when a file of
// `files` is missing. A published tarball cannot be taken back, and every one of those three is
// a mistake which would only show up in it.

import { copyFileSync, existsSync } from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { crateVersion, manifest as read } from "./version.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, "..");

const manifest = read();
const crate = crateVersion();
if (manifest.version !== crate) {
  console.error(
    `js/package.json is ${manifest.version} and the crates are ${crate}; ` +
      "the npm package and the kernel it carries are one version",
  );
  process.exit(1);
}

copyFileSync(path.join(ROOT, "LICENSE"), path.join(HERE, "LICENSE"));
execFileSync("node", [path.join(HERE, "build.mjs")], { cwd: ROOT, stdio: "inherit" });

const missing = manifest.files.filter((file) => !existsSync(path.join(HERE, file)));
if (missing.length > 0) {
  console.error(`the package names files it does not have: ${missing.join(", ")}`);
  process.exit(1);
}
console.log(`pkpdutils-core ${manifest.version} is ready to pack (${manifest.files.length} files)`);
