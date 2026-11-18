# What an analysis costs in a browser

The WebAssembly module of `pkpdutils-core` through the JavaScript host of `js/`, measured by `node js/bench.mjs` and, in a browser, by `js/bench.html`. The three shapes are rows of [`docs/benchmark.md`](https://github.com/matthiaskoenig/pkpdutils/blob/develop/docs/benchmark.md) of `pkpdutils`, so this table and the table of the native engines are read side by side: `small` is one curve of 12 points, `large` 100 000 curves of 12 points and `scan` 10 000 curves of 200 points.

Neither table is a gate. A regression is read off this file, as the numbers of the host are read off that page; nothing in continuous integration times anything, because a number a cold shared runner produces is a number nobody can read back.

The protocol is the one of `sbmlsim-core`: three warm up calls, then the median of twenty, three rounds, and the best round's median is reported.

Four timing columns, because a page pays for more than the analysis.

| column | what it times |
| --- | --- |
| `encode` | `JSON.stringify` of the request, which is what a page pays to produce the bytes |
| `call` | `analyze` of a request which is already a string: the copy into the module, the parse, the analysis and the read |
| `whole` | the two together, which is what a page which hands over an object pays |
| `pool` | the same object through `createPool`, one instance per worker, the rows cut into one shard each |

<!-- numbers:start -->
## Node

Measured on 2026-11-18 with `pkpdutils-core` 0.6.0, the development tree of this release (the module 896 KB, 256 KB gzipped), node v24.21.0 on Linux 7.0.0 x86_64, a 12th Gen Intel Core i7-12700 with 20 cores, 8 workers.

| shape | size | points | request | encode ms | call ms | whole ms | pool ms | pool speedup |
|---|---|---|---|---|---|---|---|---|
| small | 1 curve x 12 points | 12 | 0.0 MB | 0.00 | 0.21 | 0.19 | 0.28 | 0.67x |
| large | 100 000 curves x 12 points | 1 200 000 | 27.0 MB | 115.15 | 235.65 | 340.61 | 163.45 | 2.08x |
| scan | 10 000 curves x 200 points | 2 000 000 | 48.4 MB | 242.22 | 250.46 | 494.59 | 207.21 | 2.39x |

## Chrome

Measured on 2026-11-18 on the same machine and the same module with headless Chrome 157, through `js/bench.html` served over `http://127.0.0.1`, 8 workers of 20 cores. Two page loads: `small` and `scan` in one, `large` in another, because the three largest requests do not fit one renderer comfortably (`?shapes=` is what splits them).

| shape | size | points | request | encode ms | call ms | whole ms | pool ms | pool speedup |
|---|---|---|---|---|---|---|---|---|
| small | 1 curve x 12 points | 12 | 0.0 MB | 0.00 | 0.50 | 1.75 | 0.50 | 3.50x |
| large | 100 000 curves x 12 points | 1 200 000 | 27.0 MB | 53.15 | 688.95 | 755.20 | 244.15 | 3.09x |
| scan | 10 000 curves x 200 points | 2 000 000 | 48.4 MB | 124.25 | 838.55 | 978.15 | 328.30 | 2.98x |

The `small` row of the other load read 0.65 encode, 0.60 call, 0.50 pool: a sub-millisecond number of a browser carries about a millisecond of spread between page loads, whatever the protocol, so the first column to read in a browser is `call` of the large shapes and not the ratio of the small one.

<!-- numbers:end -->

## Node and Chrome are not the same number

The two tables disagree in both directions and neither disagreement is about the kernel. Chrome encodes the same request about twice as fast as this Node (53 ms against 115 ms for `large`), and it runs the same module about three times slower (689 ms against 236 ms): a headless renderer is not what a browser does to a visible tab, its wasm tiering and its memory are not Node's, and the two are different V8 builds. What the Chrome table is for is the order of magnitude a page pays and the shape of the ratios, which it agrees with Node on: a small analysis is well under a millisecond, a batch of a hundred thousand curves is most of a second, and a pool of eight workers is worth two to three times and not eight.

## What a page pays per call

`small` is the number to read first: **0.19 ms** in Node and about half a millisecond in Chrome for one curve of twelve points, end to end from the object to the columns of the result. The same curve through `nca()` of `pkpdutils` costs 3.16 ms on this machine with the native kernel, almost all of it the python front end - the data model, the units and the xarray result, none of which exists here. So an interactive page which analyses a curve per keystroke is not waiting for this module.

The pool costs more than it saves on one row (0.28 ms against 0.19 ms) and is not expected to do otherwise: a batch of one row is not cut, so what the pool adds is one message each way. A page which analyses single curves should hold a `createRuntime` and not a pool.

## Where the pool pays off, and why it is not eightfold

On the two large shapes the pool is worth about 2x in Node and about 3x in Chrome on eight workers, and the columns say where the other five went: `encode` is a third of `whole` for `large` and a half for `scan`, and the shards are cut from the same object on the main thread, which copies every sliced array once and hands each shard to a worker through structured clone. Both of those are proportional to the bytes and both are serial. The analysis itself is what scales, and it is not what the time is spent on at this size: for `scan` a request which is already a string costs 250 ms, of which the parse of 48 MB of JSON is most.

That is a finding about the request format and not about the kernel: a page which wants the pool to scale has to hand it numbers and not JSON. A binary request path - the times and the values as `Float64Array` written straight into the module's memory - is the next step for it, and it is deliberately not in this version: the JSON wire is the contract the python host and the kernel's own tests are written in, and it is what makes a request readable, fixable and comparable with the native build to the bit. The numbers above are what the readable wire costs.

## Reading it against the native table

The native numbers are measured through `nca()` of `pkpdutils`, which converts the batch, calls the kernel with the row shards of the `parallel` feature over as many threads as the machine has, and builds an xarray result. One wasm instance has one thread and no `parallel` feature, so the two are not the same measurement and the ratio of a large shape is mostly the thread count: 68 ms for `large` and 23 ms for `scan` natively against 236 ms and 250 ms here, of which a good part is the JSON. What the two tables share is the shapes, which is what makes the orders of magnitude comparable, and what they do not share is written in this paragraph rather than in a ratio column.
