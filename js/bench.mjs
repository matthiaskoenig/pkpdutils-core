// What a page pays for an analysis:
//
//   node bench.mjs                      # the three shapes, one instance and a pool
//   node bench.mjs --shapes small       # one of them
//   node bench.mjs --json               # the same numbers as JSON
//
// The shapes are three rows of `docs/benchmark.md` of `pkpdutils`, so the table of this script
// and the table of the native engines are read side by side. The protocol is the one of
// `js/bench.mjs` of `sbmlsim-core`: three warm up calls, then the median of twenty, three
// rounds, and the best round's median is reported.
//
// Four columns, because a page pays for more than the analysis: `encode` is `JSON.stringify` of
// the request, `call` is the analysis of a request which is already a string (so the JSON of the
// request is paid by the column before it and not twice), `whole` is the two together, which is
// what a page which hands over an object pays, and `pool` is the same object through one
// instance per worker, where the encoding happens in the workers and in parallel.

import { createRuntime } from "./loader.mjs";
import { createPool, defaultWorkers } from "./pool.mjs";
import { manifest } from "./version.mjs";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** The shapes, each a row count and a point count per curve. */
const SHAPES = {
  small: { rows: 1, points: 12 },
  large: { rows: 100_000, points: 12 },
  scan: { rows: 10_000, points: 200 },
};

/** How many calls one measurement is made of. */
const RUNS = 20;

/** How many calls are thrown away before a round is measured. */
const WARMUP = 3;

/** How many rounds are measured; the best round's median is the number reported. */
const ROUNDS = 3;

/**
 * The request of one shape: single dose intravenous bolus curves, the defaults of `NCAOptions`.
 *
 * @param {{rows: number, points: number}} shape how many curves of how many points
 * @returns {object} an `analysis/1` request
 */
function requestOf({ rows, points }) {
  const time = Array.from({ length: points }, (_, at) => (24 * (at + 1)) / points);
  const times = new Array(rows * points);
  const values = new Array(rows * points);
  for (let row = 0; row < rows; row += 1) {
    const clearance = 0.08 + 0.04 * (row % 7);
    for (let at = 0; at < points; at += 1) {
      times[row * points + at] = time[at];
      values[row * points + at] = 20 * Math.exp(-clearance * time[at]);
    }
  }
  return {
    contract: "analysis/1",
    batch: {
      contract: "batch/1",
      sample_dims: ["subject"],
      sample_shape: [rows],
      n_time: points,
      times,
      values,
      sd: null,
      se: null,
      n: null,
      nominal_time: null,
      lloq: null,
      max_doses: 1,
      dose_amount: new Array(rows).fill(100),
      dose_time: new Array(rows).fill(0),
      dose_duration: new Array(rows).fill("NaN"),
      time_unit: "hr",
      unit: "mg/l",
      dose_unit: "mg",
      substance: "substance",
      substances: null,
      route: "iv_bolus",
      routes: null,
      tissue: null,
      coords: [],
    },
    options: JSON.parse(
      readFileSync(path.join(HERE, "test", "fixtures", "cases.json"), "utf8"),
    ).cases[0].request.options,
  };
}

/** The median of the numbers, which is what a round reports. */
function median(numbers) {
  const sorted = [...numbers].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

/**
 * The median wall time of one call, in milliseconds.
 *
 * @param {() => unknown | Promise<unknown>} call what to time
 * @returns {Promise<number>}
 */
async function time(call) {
  const rounds = [];
  for (let round = 0; round < ROUNDS; round += 1) {
    for (let run = 0; run < WARMUP; run += 1) {
      await call();
    }
    const runs = [];
    for (let run = 0; run < RUNS; run += 1) {
      const started = performance.now();
      await call();
      runs.push(performance.now() - started);
    }
    rounds.push(median(runs));
  }
  return Math.min(...rounds);
}

/** A number with its thousands separated, as the table of the host writes them. */
function thousands(value) {
  return value.toLocaleString("en-US").replace(/,/g, " ");
}

const asked = process.argv.includes("--shapes")
  ? process.argv[process.argv.indexOf("--shapes") + 1].split(",")
  : Object.keys(SHAPES);
const bytes = readFileSync(path.join(HERE, "runtime", "pkpdutils-core.wasm"));
const compiled = await WebAssembly.compile(bytes);
const runtime = await createRuntime({ module: compiled });
const workers = defaultWorkers();
const pool = await createPool({ module: compiled, workers });

const rows = [];
for (const name of asked) {
  const shape = SHAPES[name];
  if (shape === undefined) {
    console.error(`${name} is no shape: ${Object.keys(SHAPES).join(", ")}`);
    process.exit(1);
  }
  const request = requestOf(shape);
  const text = JSON.stringify(request);
  const encode = await time(() => JSON.stringify(request));
  const call = await time(() => runtime.analyze(text));
  const whole = await time(() => runtime.analyze(request));
  const sharded = await time(() => pool.analyze(request));
  rows.push({
    name,
    rows: shape.rows,
    points: shape.rows * shape.points,
    bytes: text.length,
    encode,
    call,
    whole,
    pool: sharded,
    speedup: whole / sharded,
  });
}
pool.close();

if (process.argv.includes("--json")) {
  console.log(
    JSON.stringify(
      {
        version: manifest().version,
        node: process.version,
        workers,
        module_bytes: bytes.length,
        runs: RUNS,
        rounds: ROUNDS,
        rows,
      },
      null,
      2,
    ),
  );
} else {
  console.log(
    `pkpdutils-core ${manifest().version} in node ${process.version}, ` +
      `${workers} workers, the median of ${RUNS} calls of ${ROUNDS} rounds\n`,
  );
  console.log(`| shape | size | points | request | encode ms | call ms | whole ms | pool ms | pool speedup |`);
  console.log(`|---|---|---|---|---|---|---|---|---|`);
  for (const row of rows) {
    const size = `${thousands(row.rows)} curve${row.rows === 1 ? "" : "s"} x ${row.points / row.rows} points`;
    console.log(
      `| ${row.name} | ${size} | ${thousands(row.points)} | ` +
        `${(row.bytes / 1e6).toFixed(1)} MB | ${row.encode.toFixed(2)} | ${row.call.toFixed(2)} | ` +
        `${row.whole.toFixed(2)} | ${row.pool.toFixed(2)} | ${row.speedup.toFixed(2)}x |`,
    );
  }
}
