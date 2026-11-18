// The pool: the rows of a batch over several instances, and the answer of the whole batch back.

import assert from "node:assert/strict";
import { test } from "node:test";

import { createRuntime } from "../loader.mjs";
import { createPool, mergeAnswers, shardRequest, shardable } from "../pool.mjs";
import { batchRequest, bitsOf, mixedRouteRequest, options, request, wasmBytes } from "./helpers.mjs";

const bytes = wasmBytes();
const compiled = await WebAssembly.compile(bytes);
const runtime = await createRuntime({ module: compiled });

/** Every column of an answer as the bits of its values, by name. */
function columnsOf(answer) {
  return Object.fromEntries(
    answer.columns.map((column) => [column.name, bitsOf(answer.column(column.name))]),
  );
}

test("a batch of eight rows through four workers answers what one instance answers", async () => {
  const pool = await createPool({ module: compiled, workers: 4 });
  try {
    assert.equal(pool.workers, 4);
    const whole = runtime.analyze(batchRequest(8));
    const sharded = await pool.analyze(batchRequest(8));
    assert.equal(sharded.header.n_rows, 8);
    assert.deepEqual(sharded.header.sample_dims, ["subject"]);
    assert.deepEqual(
      sharded.columns.map((column) => column.name),
      whole.columns.map((column) => column.name),
    );
    assert.deepEqual(columnsOf(sharded), columnsOf(whole));
    // the labels of the rows travelled with their shards and came back in order
    const subject = sharded.header.coords.find((coord) => coord.name === "subject");
    assert.deepEqual(subject.values.str, ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]);
  } finally {
    pool.close();
  }
});

test("shards which report different columns merge into the union", async () => {
  const pool = await createPool({ module: compiled, workers: 2 });
  try {
    const whole = runtime.analyze(mixedRouteRequest());
    const sharded = await pool.analyze(mixedRouteRequest());
    assert.equal(sharded.header.n_rows, 2);
    // one instance answers the union itself, which is the answer the merge has to reproduce
    assert.deepEqual(new Set(sharded.columns.map((c) => c.name)), new Set(whole.columns.map((c) => c.name)));
    for (const column of whole.columns) {
      assert.deepEqual(
        bitsOf(sharded.column(column.name)),
        bitsOf(whole.column(column.name)),
        column.name,
      );
    }
    // `cl` is the intravenous row's and `cl_f` the extravascular row's, each `NaN` for the other
    assert.ok(Number.isFinite(sharded.column("cl")[0]) && Number.isNaN(sharded.column("cl")[1]));
    assert.ok(Number.isNaN(sharded.column("cl_f")[0]) && Number.isFinite(sharded.column("cl_f")[1]));
    // `c0_method` is the column whose fill is a code and not a missing number: the
    // extravascular row does not report it and contributes the 0 of "no rule was applied",
    // which is what one instance answers for it as well
    assert.deepEqual(Array.from(sharded.column("c0_method")), [1, 0]);
    assert.equal(
      sharded.columns.find((column) => column.name === "c0_method").fill,
      0,
      "the module declares the fill of the column",
    );
  } finally {
    pool.close();
  }
});

test("the merge of two answers is the union of their columns, padded with NaN", () => {
  const answerWith = (names, rows, width = 1) => {
    const values = Float64Array.from(
      names.flatMap((name, at) => Array.from({ length: rows * width }, (_, row) => at * 10 + row)),
    );
    const columns = names.map((name) => ({ name, unit: "mg", kind: "parameter", width }));
    const at = new Map(names.map((name, index) => [name, index * rows * width]));
    return {
      header: { contract: "analysis/1", n_rows: rows, sample_dims: ["subject"], coords: [] },
      columns,
      values,
      column: (name) =>
        at.has(name) ? values.subarray(at.get(name), at.get(name) + rows * width) : undefined,
    };
  };
  const merged = mergeAnswers([answerWith(["cl", "vz"], 1), answerWith(["cl_f", "vz"], 2)]);
  assert.deepEqual(
    merged.columns.map((column) => column.name),
    ["cl", "vz", "cl_f"],
  );
  assert.equal(merged.header.n_rows, 3);
  assert.deepEqual(Array.from(merged.column("cl")), [0, Number.NaN, Number.NaN]);
  assert.deepEqual(Array.from(merged.column("cl_f")), [Number.NaN, 0, 1]);
  assert.deepEqual(Array.from(merged.column("vz")), [10, 10, 11]);
});

test("a point variable of two widths takes the wider one and the short rows are padded", () => {
  const answerWith = (width, rows, start) => {
    const values = Float64Array.from({ length: width * rows }, (_, at) => start + at);
    return {
      header: { contract: "analysis/1", n_rows: rows, sample_dims: ["subject"], coords: [] },
      columns: [{ name: "interval_auc", unit: "mg", kind: "point", dim: "interval", width }],
      values,
      column: () => values,
    };
  };
  const merged = mergeAnswers([answerWith(1, 1, 0), answerWith(3, 1, 10)]);
  assert.equal(merged.columns[0].width, 3);
  assert.deepEqual(Array.from(merged.column("interval_auc")), [0, Number.NaN, Number.NaN, 10, 11, 12]);
});

test("a request of a row range carries that range and nothing else", () => {
  const whole = batchRequest(4);
  const shard = shardRequest(whole, 1, 3);
  assert.deepEqual(shard.batch.sample_shape, [2]);
  assert.equal(shard.batch.times.length, 2 * whole.batch.n_time);
  assert.deepEqual(shard.batch.times.slice(0, whole.batch.n_time), whole.batch.times.slice(whole.batch.n_time, 2 * whole.batch.n_time));
  assert.deepEqual(shard.batch.coords[0].values.str, ["s2", "s3"]);
  assert.deepEqual(shard.batch.dose_amount, [100, 100]);
  // the request it was cut from is untouched
  assert.deepEqual(whole.batch.sample_shape, [4]);
  assert.equal(whole.batch.times.length, 4 * whole.batch.n_time);
});

test("a batch which cannot be cut is not cut", async () => {
  const twoDims = batchRequest(4);
  twoDims.batch.sample_dims = ["subject", "period"];
  twoDims.batch.sample_shape = [2, 2];
  twoDims.batch.coords = [{ name: "period", dims: ["period"], values: { int: [1, 2] } }];
  assert.match(shardable(twoDims.batch), /two sample dimensions|2 sample dimensions/);
  const pool = await createPool({ module: compiled, workers: 4 });
  try {
    const answer = await pool.analyze(twoDims);
    assert.deepEqual(answer.header.sample_shape, [2, 2]);
    assert.equal(answer.header.n_rows, 4);
  } finally {
    pool.close();
  }
});

test("a single row batch runs whole and answers what one instance answers", async () => {
  const pool = await createPool({ module: compiled, workers: 4 });
  try {
    const answer = await pool.analyze(request());
    assert.equal(answer.header.n_rows, 1);
    assert.deepEqual(columnsOf(answer), columnsOf(runtime.analyze(request())));
  } finally {
    pool.close();
  }
});

test("a refusal of one shard rejects the whole call", async () => {
  const pool = await createPool({ module: compiled, workers: 4 });
  try {
    const effect = { ...batchRequest(8), options: options({ kind: "effect" }) };
    await assert.rejects(pool.analyze(effect), (bad) => bad.name === "RefusedError");
    const broken = batchRequest(8);
    broken.batch.values = broken.batch.values.slice(1);
    await assert.rejects(pool.analyze(broken), (bad) => bad.kind === "malformed");
    // and the pool still answers afterwards
    const answer = await pool.analyze(batchRequest(8));
    assert.equal(answer.header.n_rows, 8);
  } finally {
    pool.close();
  }
});

test("the statistics of a parameter table run on one worker", async () => {
  const pool = await createPool({ module: compiled, workers: 2 });
  try {
    const answer = await pool.summarize({
      values: [1, 2, 3, 10, 20, 30],
      n_vars: 2,
      n_groups: 1,
      n_samples: 3,
      ci_level: 0.95,
      lognormal: [false, true],
      discrete: [false, false],
    });
    const at = answer.header.statistics.indexOf("mean");
    assert.deepEqual(Array.from(answer.values.subarray(at * 2, at * 2 + 2)), [2, 20]);
  } finally {
    pool.close();
  }
});

test("close ends every worker and a call after it throws", async () => {
  const pool = await createPool({ module: compiled, workers: 2 });
  const before = await pool.analyze(request());
  assert.equal(before.header.n_rows, 1);
  pool.close();
  assert.throws(() => pool.summarize({}), (bad) => /closed/.test(bad.message));
  await assert.rejects(
    (async () => pool.analyze(request()))(),
    (bad) => /closed/.test(bad.message),
  );
});

test("the pool reports what the module is", async () => {
  const pool = await createPool({ module: bytes, workers: 1 });
  try {
    assert.equal(pool.about.version, runtime.about.version);
    assert.equal(pool.about.output_format, runtime.about.output_format);
  } finally {
    pool.close();
  }
});
