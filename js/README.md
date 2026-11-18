# pkpdutils-core

[![npm](https://img.shields.io/npm/v/pkpdutils-core.svg?label=npm)](https://www.npmjs.com/package/pkpdutils-core)
[![CI](https://github.com/matthiaskoenig/pkpdutils-core/actions/workflows/ci.yml/badge.svg)](https://github.com/matthiaskoenig/pkpdutils-core/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/npm/l/pkpdutils-core.svg)](https://opensource.org/licenses/MIT)

Non-compartmental analysis of pharmacokinetic timecourses in a browser and in Node: the JavaScript host of [`pkpdutils-core`](https://github.com/matthiaskoenig/pkpdutils-core), the rust kernel of [`pkpdutils`](https://github.com/matthiaskoenig/pkpdutils), compiled to WebAssembly.

The module is the same code the python package runs: one kernel, two builds. `js/test/same_bits.test.mjs` compares every column of eight analyses against the numbers of the native build, five of them bit for bit.

```bash
npm install pkpdutils-core
```

## One analysis

```javascript
import { createRuntime } from "pkpdutils-core";
import { readFile } from "node:fs/promises";          // in a browser: await (await fetch(url)).arrayBuffer()

const url = import.meta.resolve("pkpdutils-core/runtime/pkpdutils-core.wasm");
const runtime = await createRuntime({ module: await readFile(new URL(url)) });

const answer = runtime.analyze({
  contract: "analysis/1",
  batch: {
    contract: "batch/1",
    sample_dims: ["subject"],
    sample_shape: [1],
    n_time: 4,
    times: [0, 1, 2, 3],
    values: [8, 4, 2, 1],
    max_doses: 1,
    dose_amount: [10],
    dose_time: [0],
    dose_duration: ["NaN"],
    time_unit: "hr",
    unit: "mg/l",
    dose_unit: "mg",
    route: "oral",
    coords: [],
  },
  options: { kind: "concentration", auc_method: "linear_log" /* and the rest of the defaults */ },
});

answer.column("auc_last");        // Float64Array, one value per sample
answer.column("thalf")[0];       // 0.9999999999999999, an hour
answer.header.columns;            // every column with its unit, width and fill
```

A request is the `analysis/1` contract of `pkpdutils-abi`: the `batch` carries the times and the values of every sample flat and row major, the dosing protocol and the units, and the `options` are the fields of `NCAOptions` of `pkpdutils`. A number which is not finite travels as a string (`"NaN"`, `"inf"`, `"-inf"`), because JSON has no other spelling for it. `crates/abi/src/` is the reference, and `scripts/wasm_fixtures.py` writes complete requests for seven cases into `js/test/fixtures/cases.json` - the shortest way to see a real one of every shape.

The answer is the header and one payload: `column(name)` is a view of the payload, `header.columns` names every column with its unit and its width (a point variable such as `interval_auc` carries several values per sample), and `header.coords` carries the sample labels through. The payload of an answer is a copy, so an answer stays valid after the next call.

## A pool of workers

A wasm instance is single threaded. The pool compiles the module once, runs one instance per worker and cuts the rows of a batch into one shard per worker:

```javascript
import { createPool } from "pkpdutils-core/pool";

const pool = await createPool({ module: bytes });     // workers: the cores of the machine, at most 8
const answer = await pool.analyze(request);           // the answer of the whole batch
pool.close();
```

The shards are merged back into one answer: the union of the columns, and a row which does not report one of them filled with what the module says it contributes, so `answer.column(name)` holds what one instance would have answered, value for value. The columns of a batch of mixed routes come out in the order the shards first report them, which is not the order one instance uses, so read a column by name and not by its position. `summarize` is a reduction over the samples and runs whole on one worker. A page which bundles its own workers passes `createWorker`.

## Errors

Nothing traps. Every call throws a `KernelError` with a `kind`:

| `kind` | what it means |
| --- | --- |
| `refused` | an analysis this build does not carry (a `RefusedError`, the one kind a host handles rather than reports) |
| `malformed` | the request, the contract or the data is wrong, which is for the caller to fix |
| `internal` | a bug of this build |
| `panic` | the kernel panicked; the message is the one the panic hook left behind |

## What this package is not

The data model of `pkpdutils` - `Timecourses`, the units, the figures, the statistics on parameters - is python and stays there. This package is the analysis: a request in, the columns of a result out. A page builds its own requests, and [`sbmlsim-core`](https://www.npmjs.com/package/sbmlsim-core) is the simulator whose results they can come from.

MIT, see LICENSE. The entry points a build carries are in `about.entry_points`; `analyze`, `summarize` and `about` are the ones of this version.
