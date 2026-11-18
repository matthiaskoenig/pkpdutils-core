// The loader of the module: what it reads, what it throws and what it copies.

import assert from "node:assert/strict";
import { test } from "node:test";

import { createRuntime, KernelError, OUTPUT_FORMAT } from "../loader.mjs";
import { options, packageVersion, request, wasmBytes } from "./helpers.mjs";

/** One runtime for every test of this file, since a call of one does not outlive itself. */
const runtime = await createRuntime({ module: wasmBytes() });

test("an analysis of one curve comes back with its columns", () => {
  const answer = runtime.analyze(request());
  assert.equal(answer.header.n_rows, 1);
  assert.deepEqual(answer.header.sample_dims, ["sample"]);
  assert.ok(answer.column("cmax") instanceof Float64Array);
  assert.equal(answer.column("cmax")[0], 8);
  assert.equal(answer.column("tmax")[0], 0);
  assert.ok(Math.abs(answer.column("thalf")[0] - 1) < 1e-12);
  // the peak of this oral curve is its first point, which is `NcaFlag::NO_ABSORPTION`
  assert.equal(answer.column("flags")[0], 16);
  assert.equal(answer.column("not_a_parameter"), undefined);
});

test("every column of the header is in the payload, once", () => {
  const answer = runtime.analyze(request());
  const declared = answer.columns.reduce(
    (total, column) => total + (column.width ?? 1) * answer.header.n_rows,
    0,
  );
  assert.equal(answer.values.length, declared);
  const names = new Set(answer.columns.map((column) => column.name));
  assert.equal(names.size, answer.columns.length);
  const units = answer.columns.find((column) => column.name === "auc_last");
  assert.equal(units.unit, "hour * milligram / liter");
});

test("about names the version, the layout and the entry points", () => {
  assert.equal(runtime.about.version, packageVersion());
  assert.equal(runtime.about.output_format, OUTPUT_FORMAT);
  assert.equal(runtime.about.contracts.analysis, "analysis/1");
  assert.ok(runtime.about.entry_points.includes("analyze"));
  assert.ok(runtime.about.entry_points.includes("summarize"));
});

test("a refusal is a RefusedError and a malformed request a KernelError", () => {
  const effect = { ...request(), options: options({ kind: "effect" }) };
  assert.throws(
    () => runtime.analyze(effect),
    (bad) => bad.name === "RefusedError" && bad.kind === "refused" && /Effect/.test(bad.message),
  );
  assert.throws(
    () => runtime.analyze("{"),
    (bad) => bad instanceof KernelError && bad.kind === "malformed",
  );
  // a contract this module does not speak is malformed as well, and not a refusal
  assert.throws(
    () => runtime.analyze({ ...request(), contract: "analysis/2" }),
    (bad) => bad.kind === "malformed",
  );
});

test("a NaN of the payload arrives as a NaN", () => {
  const missing = request();
  missing.batch.values = ["NaN", "NaN", "NaN", "NaN"];
  const answer = runtime.analyze(missing);
  assert.ok(Number.isNaN(answer.column("cmax")[0]));
  // `NcaFlag::NO_DATA`
  assert.equal(answer.column("flags")[0], 64);
});

test("the statistics of a parameter table come back per statistic", () => {
  const answer = runtime.summarize({
    values: [1, 2, 3, 10, 20, 30],
    n_vars: 2,
    n_groups: 1,
    n_samples: 3,
    ci_level: 0.95,
    lognormal: [false, true],
    discrete: [false, false],
  });
  assert.equal(answer.header.width, 2);
  const at = answer.header.statistics.indexOf("mean");
  assert.deepEqual(Array.from(answer.values.subarray(at * 2, at * 2 + 2)), [2, 20]);
});

test("two calls do not see each other's output", () => {
  const first = runtime.analyze(request());
  const cmax = first.column("cmax");
  runtime.analyze(request());
  // the copy the loader took at the first call still holds, although the module has written over
  // its own output buffer since
  assert.equal(cmax[0], 8);
  assert.equal(first.column("cmax")[0], 8);
});
