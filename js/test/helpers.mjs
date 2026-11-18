// What the tests of the loader share: the module, the requests of the fixtures and the bits of a
// column.

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** The module `npm run build` writes, as bytes. */
export function wasmBytes() {
  return readFileSync(path.join(HERE, "..", "runtime", "pkpdutils-core.wasm"));
}

/** The version of the npm package, which is the version of the kernel. */
export function packageVersion() {
  return JSON.parse(readFileSync(path.join(HERE, "..", "package.json"), "utf8")).version;
}

/** The cases of `scripts/wasm_fixtures.py`, with the bits the native build answers. */
export function fixtures() {
  return JSON.parse(readFileSync(path.join(HERE, "fixtures", "cases.json"), "utf8"));
}

/**
 * One oral curve of four points which halves every hour, the curve of the binding tests.
 *
 * @returns {object} an `analysis/1` request
 */
export function request() {
  return {
    contract: "analysis/1",
    batch: {
      contract: "batch/1",
      sample_dims: ["sample"],
      sample_shape: [1],
      n_time: 4,
      times: [0, 1, 2, 3],
      values: [8, 4, 2, 1],
      sd: null,
      se: null,
      n: null,
      nominal_time: null,
      lloq: null,
      max_doses: 1,
      dose_amount: [10],
      dose_time: [0],
      dose_duration: ["NaN"],
      time_unit: "hr",
      unit: "mg/l",
      dose_unit: "mg",
      substance: "caffeine",
      substances: null,
      route: "oral",
      routes: null,
      tissue: null,
      coords: [],
    },
    options: options(),
  };
}

/**
 * A batch of `rows` oral curves of nine points, each with its own clearance.
 *
 * @param {number} rows how many samples
 * @returns {object} an `analysis/1` request
 */
export function batchRequest(rows) {
  const time = [0.25, 0.5, 1, 2, 4, 6, 9, 12, 24];
  const times = [];
  const values = [];
  const labels = [];
  for (let row = 0; row < rows; row += 1) {
    const clearance = 2 + 0.4 * row;
    times.push(...time);
    values.push(
      ...time.map((t) => 25 * Math.exp((-clearance * t) / 12) * (1 - Math.exp(-3 * t))),
    );
    labels.push(`s${row + 1}`);
  }
  const one = request();
  return {
    ...one,
    batch: {
      ...one.batch,
      sample_dims: ["subject"],
      sample_shape: [rows],
      n_time: time.length,
      times,
      values,
      dose_amount: new Array(rows).fill(100),
      dose_time: new Array(rows).fill(0),
      dose_duration: new Array(rows).fill("NaN"),
      coords: [{ name: "subject", dims: ["subject"], values: { str: labels } }],
    },
  };
}

/**
 * A batch of one intravenous and one extravascular curve, the two routes in one request.
 *
 * The intravenous row reports `cl`, `vz` and `c0` and the extravascular one `cl_f` and `vz_f`, so
 * a shard of one row reports columns the other does not: this is the batch the merge of the pool
 * is about.
 *
 * @returns {object} an `analysis/1` request
 */
export function mixedRouteRequest() {
  const one = request();
  return {
    ...one,
    batch: {
      ...one.batch,
      sample_dims: ["subject"],
      sample_shape: [2],
      times: [0, 1, 2, 3, 0, 1, 2, 3],
      values: [8, 4, 2, 1, 6, 3, 1.5, 0.75],
      dose_amount: [10, 10],
      dose_time: [0, 0],
      dose_duration: ["NaN", "NaN"],
      route: null,
      routes: ["iv_bolus", "oral"],
      coords: [{ name: "subject", dims: ["subject"], values: { str: ["iv", "po"] } }],
    },
  };
}

/** The default options of the kernel, as the wire carries them. */
export function options(changed = {}) {
  return {
    kind: "concentration",
    auc_method: "linear_log",
    terminal: {
      method: "best_fit",
      min_points: 3,
      exclude_cmax: true,
      n_points: null,
      points: null,
      min_adj_r2: null,
      tie_tolerance: 1e-4,
      relative_warning: 1e-8,
      keep_candidates: false,
      windows: null,
    },
    lloq: null,
    lloq_relative: null,
    blq: "nan",
    c0_method: "log_back_extrapolation",
    extrapolation_warning: 0.2,
    acceptance: {
      r2_adj_min: null,
      extrapolation_max: null,
      span_min: null,
      n_points_min: null,
      exclude: false,
    },
    partial_aucs: {},
    tau: null,
    tau_tolerance: 0.1,
    intervals: true,
    units: {},
    effect_threshold: null,
    n_workers: null,
    chunk_rows: 5000,
    uncertainty: null,
    n_boot: 1000,
    seed: null,
    ci_level: 0.95,
    bootstrap_spread: "se",
    bootstrap_distribution: "normal",
    delta_step: 0.01,
    ...changed,
  };
}

/**
 * The bits of every value of a column, as the hex strings the fixtures carry.
 *
 * A comparison of the bits is what tells a `NaN` from another `NaN` and a `-0` from a `0`, which
 * is the point of the same-bits test: the two builds of one kernel answer the same numbers or
 * they do not.
 *
 * @param {Float64Array} values one column of an answer
 * @returns {string[]} one `0x` prefixed 16 digit string per value
 */
export function bitsOf(values) {
  const view = new DataView(new ArrayBuffer(8));
  return Array.from(values, (value) => {
    view.setFloat64(0, value, true);
    const low = view.getUint32(0, true);
    const high = view.getUint32(4, true);
    return `0x${high.toString(16).padStart(8, "0")}${low.toString(16).padStart(8, "0")}`;
  });
}
