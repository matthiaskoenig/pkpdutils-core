// The wasm module answers the numbers of the native build.
//
// The cases come from `scripts/wasm_fixtures.py`, which runs the python extension of
// `crates/python` on the same requests. The two builds run the same code of `crates/core` on two
// targets, so a difference is never a difference of the analysis: it is the libm of the target.
// `f64::exp` and `f64::ln` are the platform's library natively and the `libm` crate on
// `wasm32-unknown-unknown`, two implementations of one function which differ in the last bit.
//
// So a case whose parameters are sums, differences and quotients of the data has to agree to the
// bit, which the fixture claims with `exact`, and every other case has to agree to `TOLERANCE`.
// The worst disagreement measured when this test was written was 8.9e-10, in the standard error
// of the terminal slope of the multiple dose case: a cancellation over values which differ in
// their last bit. The second worst was 1.2e-12 in the delta method, whose difference quotient
// divides by a step of a hundredth. A value which is not finite is of the target no more than a
// sum is, so `NaN`, an infinity and the sign of a zero have to match whatever the case.

import assert from "node:assert/strict";
import { test } from "node:test";

import { createRuntime } from "../loader.mjs";
import { bitsOf, fixtures, wasmBytes } from "./helpers.mjs";

/** The relative agreement a case which goes through `exp` and `ln` has to reach. */
const TOLERANCE = 1e-8;

const runtime = await createRuntime({ module: wasmBytes() });
const { cases, pkpdutils_core: wheel } = fixtures();

/** The value of a bit pattern of the fixture. */
function valueOf(hex) {
  const view = new DataView(new ArrayBuffer(8));
  view.setBigUint64(0, BigInt(hex), true);
  return view.getFloat64(0, true);
}

/**
 * Why two values disagree, or `null` when they do not.
 *
 * @param {number} have what the module answered
 * @param {string} hex the bits of the native build
 * @param {boolean} exact whether the bits have to be the ones of the native build
 * @returns {string | null}
 */
function disagreement(have, hex, exact) {
  const [bits] = bitsOf(Float64Array.of(have));
  if (bits === hex) {
    return null;
  }
  const want = valueOf(hex);
  if (Number.isNaN(want)) {
    return Number.isNaN(have) ? null : `${have} for a NaN`;
  }
  if (exact) {
    return `${have} (${bits}), the native build answers ${want} (${hex})`;
  }
  if (!Number.isFinite(want) || !Number.isFinite(have) || want === 0 || have === 0) {
    // an infinity, a zero and the sign of either are of the analysis and not of the libm
    return `${have} (${bits}), the native build answers ${want} (${hex})`;
  }
  const relative = Math.abs((have - want) / want);
  return relative <= TOLERANCE
    ? null
    : `${have}, the native build answers ${want}, which is ${relative.toExponential(2)} apart`;
}

test("the fixture was written by the kernel this module is built from", () => {
  assert.equal(runtime.about.version, wheel);
});

for (const kase of cases) {
  const how = kase.exact ? "to the bit" : `to ${TOLERANCE}`;
  test(`${kase.name}: the module answers the native build ${how} (${kase.note})`, () => {
    const answer = runtime.analyze(kase.request);
    assert.equal(answer.header.n_rows, kase.expected.n_rows);
    assert.deepEqual(
      answer.columns.map((column) => column.name),
      kase.expected.columns.map((column) => column.name),
      "the columns of the module are not the ones of the native build, or not in its order",
    );
    for (const column of kase.expected.columns) {
      const values = answer.column(column.name);
      assert.ok(values, `the module reports no ${column.name}`);
      assert.equal(values.length, column.bits.length, `${column.name} has another width`);
      for (let at = 0; at < column.bits.length; at += 1) {
        const bad = disagreement(values[at], column.bits[at], kase.exact);
        assert.equal(bad, null, `${kase.name}: ${column.name}[${at}] is ${bad}`);
      }
    }
  });
}
