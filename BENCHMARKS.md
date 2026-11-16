# Benchmarks

What the port bought, measured against the python package it is a port of. Every number here comes from a run on one machine on 2026-11-16 and is reproducible with the commands below; nothing is scaled, extrapolated or rounded in the flattering direction.

## What is inside each timing

The two sides do not do the same work, so both are measured and both are named.

| column | what the timed call does | what it does not do |
|---|---|---|
| rust, serial | `analyze_batch_in_shards(batch, options, 1)` of a build **without** the `parallel` feature: the contract check, `Batch::validate`, `NcaOptions::validate`, the refusal of an analysis this build does not carry, the terminal windows and the manual columns of the options, the row loop over every row, the unit of every parameter through `pkpdutils-units`, the assembly of the `AnalysisResult` with its columns, flags, coordinates and attributes, and `AnalysisResult::validate` | build the `Batch` (not timed on either side), serialize anything: a wasm host pays for the JSON of the batch and of the result on top of this |
| rust, sharded | the same call with the plan `automatic_shards` chooses, on a build **with** the `parallel` feature | the same |
| python, `nca` | `nca(batch, options=NCAOptions())`: the option resolution, the times, values, dose and route arrays read out of the `Timecourses` dataset, the vectorized numpy core in chunks of `chunk_rows` (in the shared thread pool above `NCA_WORKER_THRESHOLD` = 20 000 rows), the pint unit of every variable, the assembly of the `xarray.Dataset` with its coordinates and `attrs` | build the `Timecourses` (not timed) |
| python, `run_rows` | `run_rows(t, c, dose_amount=..., dose_time=..., dose_duration=..., route=..., options=NCAOptions(n_workers=1))`: the numpy core alone, serial, which gives back a dict of numpy arrays | the units, the xarray result, the option resolution, any validation |

So the `nca` columns are the closest thing to the kernel's call: both read a batch in and hand a finished, unit-carrying result back. The `run_rows` column is the numeric core of python alone and the kernel still carries its own assembly in that comparison, which is the one place where the ratio understates the kernel rather than the other way round.

Both sides analyse **the same numbers**. The batch generator of `crates/core/examples/kernel_timing.rs` is written a second time in python for this measurement, operation for operation on the same xorshift64 stream and with no transcendental function in it, and every run prints the sum of the finite `auc_last` over its rows. The checksums agreed to all twelve digits printed on every one of the four shapes, for instance `1.077047357262e6` for 100 000 x 12 from the kernel and `1.077047357262e+06` from `nca`.

## The machine

| | |
|---|---|
| CPU | 12th Gen Intel Core i7-12700: 8 performance cores with 2 threads each and 4 efficiency cores, 20 logical CPUs, 4.9 GHz maximum, governor `powersave` |
| OS | Linux 7.0.0-43-generic |
| rust | rustc 1.100.0, `--release` with the default profile of this workspace (`opt-level = 3`, no LTO, 16 codegen units), rayon 1.12.0 |
| python | CPython 3.14.10, numpy 2.5.4, `pkpdutils` 1.4.0 |
| load | `mpstat` read the machine 89 to 90 percent idle before and after the runs, which is one and a half of the twenty logical CPUs busy with an unrelated video call; the one minute load average read 2.4 to 4.8 during them and that figure includes the runs themselves. That is the quietest the machine was: a first pass taken at load 15 to 19 earlier the same day was discarded, as the spike's own first pass at load 57 was. |

The serial rust runs are pinned to one performance core with `taskset -c 2`, the sharded runs are not pinned and use the whole machine, which is the spike's protocol. The python runs are not pinned either, which is the friendlier setting for them.

## The protocol

Three warm up calls, then three rounds of twenty timed calls; the median of each round, and the median of the three round medians is the number of a process. Every shape and mode runs in a process of its own, so its peak resident set size (`VmHWM`) is its own, and the whole measurement is repeated three times: the tables report the median of the three process medians. The spread between the three processes of a cell was below one percent except on 100 000 x 12, where it was six percent serially; over the whole session, which took more runs of that shape, it ranged from 104 to 132 ms serially and from 50 to 61 ms sharded, so its ratio carries about a tenth of uncertainty either way.

The two builds write the same binary, so each is run before the other is built:

```bash
cargo build --release --example kernel_timing                      # the serial build
taskset -c 2 ./target/release/examples/kernel_timing 1000x200
cargo build --release --example kernel_timing --features parallel  # the sharded build
./target/release/examples/kernel_timing 1000x200
./target/release/examples/kernel_timing sweep                      # the crossover
```

The python column is `scripts/python_timing.py` of this repository, which builds the same batches and times the three variants above; it imports `pkpdutils`, so it runs in that checkout:

```bash
uv run --no-sync python path/to/pkpdutils-core/scripts/python_timing.py
uv run --no-sync python scripts/benchmark.py nca-large   # the benchmark of the package itself
```

## The measured table

Milliseconds per call, median of medians as above.

| rows x points | work | rust serial | rust sharded | shards | python `nca` | python `nca`, `n_workers=20` | python `run_rows`, serial |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 x 12 | 12 | 0.0479 | 0.0479 | 1 | 2.9447 | 2.9392 | 0.7961 |
| 1 000 x 200 | 200 000 | 7.9145 | 1.0740 | 80 | 55.7415 | 55.8386 | 53.5569 |
| 10 000 x 200 | 2 000 000 | 80.4993 | 11.0489 | 80 | 554.3993 | 294.9913 | 546.5306 |
| 100 000 x 12 | 1 200 000 | 107.7013 | 55.7648 | 80 | 139.9280 | 202.5560 | 292.1922 |

The ratios of the same numbers:

| rows x points | `nca` / rust serial | `nca` / rust sharded | `run_rows` / rust serial | rust serial / rust sharded |
|---|---:|---:|---:|---:|
| 1 x 12 | 61x | 61x | 17x | 1.0x (one shard, below the threshold) |
| 1 000 x 200 | 7.0x | 52x | 6.8x | 7.4x |
| 10 000 x 200 | 6.9x | 50x | 6.8x | 7.3x |
| 100 000 x 12 | 1.3x | 2.5x | 2.7x | 1.9x |

Peak resident set size of the process, one process per shape and mode:

| rows x points | rust serial | rust sharded | python |
|---|---:|---:|---:|
| 1 x 12 | 2.8 MB | 3.4 MB | |
| 1 000 x 200 | 6.4 MB | 6.8 MB | |
| 10 000 x 200 | 41.3 MB | 41.4 MB | |
| 100 000 x 12 | 101.8 MB | 101.4 MB | 425 MB |

The python figure is `scripts/benchmark.py nca-large` in a fresh interpreter, which is the one python process of this measurement that held a single shape; the four shape process of the table peaked at 791 MB.

That benchmark of the package itself is also the independent check on the harness here: `nca-large` is 100 000 curves of 12 points through `nca`, on its own data (a numpy generator with normal noise) and with its own three repeats, and it reported 143 ms where the matched run above reports 139.9 ms. Two percent apart on a shape where the process to process spread is six percent, so the python column is measuring what the package's own benchmark measures.

### What the table says

- On dense curves the kernel is **7x the python front end serially and about 50x with its shards**, and the shards are worth 7.4x of that on twenty logical CPUs.
- On 100 000 short curves it is **1.3x serially and 2.5x sharded**, which is the weakest result of the port and the honest one: python's own thread pool engages at that shape (`NCA_WORKER_THRESHOLD` is 20 000 rows) while the kernel stops being bound by its arithmetic. One row of parameters is about 420 bytes, so the row loop of this batch writes 42 MB and the columns of the result another 38 MB, next to the 19 MB of the batch itself: that is the 102 MB the process peaks at, and that traffic is what the shards cannot divide. The row loop needs 0.58 us for a row of twelve points at a thousand rows and 1.08 us at a hundred thousand, and the difference is the memory.
- A single curve costs the kernel 48 us against the front end's 2.9 ms, which is **61x**, but almost none of that 48 us is the curve: the fixed cost of a call is about 41 us and the curve itself about 0.6 us (see below). A host which analyses one curve at a time is paying for the assembly of the result, not for the analysis.
- `nca` with `n_workers=20` is no faster than the default at 1 000 x 200, because `split_rows` keeps a slice at `min_rows` = 1 000 rows or more and a thousand rows are therefore one slice and one worker. It is 1.9x faster at 10 000 x 200 (ten slices, 1.9x of them, since only the numpy calls release the GIL) and **slower** than the default at 100 000 x 12, where the automatic setting takes eight workers (`max_workers`) and twenty of them contend for the same interpreter. Python's parallelism is bounded by `min_rows`, by `chunk_rows` and by the GIL, not by the machine.

### The cost of a call, decomposed

The serial build, one performance core, from the shapes `1x12`, `1x200`, `10x12`, `100x12`, `1000x12` and `1000x24`:

| part | measured |
|---|---|
| fixed per call | about 41 us: the contract and option checks, the unit of each of the 48 parameter columns of an intravenous bolus through `pkpdutils-units`, the assembly and the validation of the result |
| per row | about 0.17 us |
| per observation | about 34 ns |

Two consequences. A batch is cheap per curve and a single call is not, so a host which has a thousand curves should send them as one batch. And the kernel's advantage over numpy is largest where the rows are long: numpy is vectorized along the rows, so a short curve is where it is already efficient. Per observation, `run_rows` needs 268 ns on 200 point curves against the kernel's 40, and 243 ns on twelve point curves against the kernel's 90.

## The threshold

`SHARD_WORK_THRESHOLD` is the work of a batch, its rows times its time points, below which `automatic_shards` plans a single shard. The figure is 4 096 and it comes from this sweep, `kernel_timing --features parallel -- sweep`, one process, milliseconds per call:

| rows x points | work | 1 shard | 2 shards | 20 shards | 80 shards | best of them / 1 shard |
|---|---:|---:|---:|---:|---:|---:|
| 10 x 12 | 120 | 0.1307 | 0.1661 | | | 0.79x |
| 20 x 12 | 240 | 0.0653 | 0.0746 | 0.1794 | | 0.88x |
| 50 x 12 | 600 | 0.1335 | 0.0730 | 0.1209 | | 1.83x |
| 100 x 12 | 1 200 | 0.1022 | 0.0920 | 0.1045 | 0.1371 | 1.11x |
| 200 x 12 | 2 400 | 0.1753 | 0.2264 | 0.1888 | 0.1483 | 1.18x |
| 500 x 12 | 6 000 | 0.4060 | 0.4594 | 0.2585 | 0.3099 | 1.57x |
| 1 000 x 12 | 12 000 | 0.6234 | 0.4578 | 0.2571 | 0.2553 | 2.44x |
| 2 000 x 12 | 24 000 | 1.2026 | 0.7515 | 0.4999 | 0.4304 | 2.79x |
| 5 000 x 12 | 60 000 | 3.0835 | 1.9408 | 1.1970 | 1.2006 | 2.58x |
| 10 000 x 12 | 120 000 | 7.0144 | 4.5739 | 2.7549 | 2.5913 | 2.71x |
| 100 000 x 12 | 1 200 000 | 131.6712 | 107.4442 | 87.4951 | 87.6228 | 1.51x |
| 5 x 200 | 1 000 | 0.0795 | 0.0791 | | | 1.01x |
| 10 x 200 | 2 000 | 0.1177 | 0.1033 | | | 1.14x |
| 20 x 200 | 4 000 | 0.1996 | 0.1396 | 0.1508 | | 1.43x |
| 50 x 200 | 10 000 | 0.4321 | 0.2861 | 0.2101 | | 2.06x |
| 100 x 200 | 20 000 | 0.8826 | 0.5330 | 0.3762 | 0.3006 | 2.94x |
| 1 000 x 200 | 200 000 | 8.2467 | 4.6374 | 1.7312 | 1.3435 | 6.14x |
| 10 000 x 200 | 2 000 000 | 82.6558 | 43.2692 | 14.0560 | 11.7355 | 7.04x |

A cell is empty where the number of rows caps the plan below that count. The plan the kernel itself uses is the whole pool, four shards per thread: the 80 shard column where the batch has at least eighty rows and one shard per row below that. It is not always the best column of its row, and where it is not the difference is at most a fifth (500 x 12: 0.3099 against 0.2585). The sweep ran every shape in one process, which is why its 100 000 x 12 row is 22 percent above the fresh process of the table: that batch allocates and frees 56 MB per call and the heap it lands in matters. Within a row the counts are comparable, which is all the sweep is read for.

Where the curve crosses one, with the plan the kernel really uses, which is the whole pool: at 1 200 observations the shards lose (0.1371 against 0.1022), at 2 400 they win by a fifth (0.1483 against 0.1753), at 4 000 by a third (0.1508 against 0.1996, twenty rows and so twenty shards) and from 10 000 on by more than two. **4 096 is above the crossing and not at it**: a batch which is only just worth sharding should not spend the whole pool on a tenth of a gain, and no shape of the sweep is made slower by a plan it did not ask for. Python's `NCA_WORKER_THRESHOLD` is 20 000 *rows* and counts no points at all, so a thousand curves of two hundred points - 200 000 observations, 56 ms of work in `nca`, 6x off what the shards do here - stays in a single thread there.

`SHARDS_PER_THREAD` is 4 and comes from the same sweep: four shards per thread beat one per thread at 100 x 200 (0.3006 against 0.3762), 1 000 x 200 (1.3435 against 1.7312), 2 000 x 12 (0.4304 against 0.4999), 10 000 x 12 and 10 000 x 200, drew at 5 000 x 12 and 100 000 x 12 and lost once, at 500 x 12 (0.3099 against 0.2585). The rows of a batch are not equally expensive - the terminal regression of a row evaluates every candidate window after its peak - so a shard per thread leaves the thread which drew the expensive ones finishing alone, and the pool steals the difference when there are more shards than threads. A plan which grows with the work instead, two shards at the threshold and twenty at ten times it, was measured against the full pool and was slower everywhere in between: 0.47 ms against 0.26 ms at 12 000 observations, 0.25 against 0.21 at 10 000.

## On wasm

The same example compiled for `wasm32-wasip1` and run under wasmtime 50.0.1, unpinned, same protocol, same checksums as the native build down to all twelve digits:

| rows x points | native serial | wasm32-wasip1 | wasm / native |
|---|---:|---:|---:|
| 1 x 12 | 0.0479 | 0.0602 | 1.26x |
| 1 000 x 200 | 7.9145 | 9.7630 | 1.23x |
| 100 000 x 12 | 107.7013 | 122.4929 | 1.14x |

So the kernel costs about a quarter more inside a wasm runtime than outside it, and the 100 000 x 12 shape, which is memory bound rather than compute bound, loses the least. The build was made with the `parallel` feature deliberately, to see what it does there: rayon reports one thread, `automatic_shards` therefore plans one shard, and an explicit plan of two shards runs them one after the other in the calling thread without trapping. The feature buys a wasm instance nothing, which is why the wasm build leaves it out.

## Against the spike

`spike/RESULTS.md` is the first measurement of this port: a kernel of eight outputs, no options, no flags, no BLQ or dose handling, no units and no result assembly, built with LTO and one codegen unit. The full kernel is slower than it and this is the number that matters, since this is what a host gets:

| rows x points | spike serial | kernel serial | spike rayon | kernel sharded |
|---|---:|---:|---:|---:|
| 1 x 12 | 0.0003 | 0.0479 (160x) | 0.0008 | 0.0479 (60x) |
| 1 000 x 200 | 4.12 | 7.9145 (1.9x) | 0.371 | 1.0740 (2.9x) |
| 10 000 x 200 | 41.19 | 80.4993 (2.0x) | 3.66 | 11.0489 (3.0x) |
| 100 000 x 12 | 19.75 | 107.7013 (5.5x) | 2.04 | 55.7648 (27x) |

The two dense shapes are explained by scope: the kernel computes 48 parameter columns, the count of subjects among them, where the spike computed eight numbers, so twice the time per observation on a dense curve is the price of the other forty parameters, the flags, the BLQ rules, the dose handling and the units. The 100 000 x 12 column is explained by memory: 420 bytes of parameters per row against the spike's 64, so 42 MB per call against 6.4 MB and 38 MB of result columns against 6.4 MB, which is why the shards there buy 1.9x and not the spike's 9.7x. The ratio of the port's own promise - the spike's "8 300x against python on one curve" - is likewise mostly the 41 us of assembly that the spike did not have; the honest single curve figure is 61x.

LTO and one codegen unit, which the spike used and this workspace does not configure, were measured on the kernel as well: `--config profile.release.lto=true --config profile.release.codegen-units=1` gave 0.0440 ms at 1 x 12, 8.05 at 1 000 x 200, 101.03 at 100 000 x 12 serially and 1.050 and 50.97 sharded, so between 9 percent faster and 2 percent slower than the default profile. The table is the default profile, which is what `cargo run --release` gives.

## Caveats

- Between one and two of the twenty logical CPUs were busy with an unrelated application throughout. That costs the sharded runs some of their speedup and the pinned serial runs next to nothing, so the sharded ratios here are if anything conservative.
- The data is synthetic: a geometric decline with five percent multiplicative noise, one intravenous bolus at time zero, no missing values, no values below a limit of quantification, one route for the whole batch. A batch with holes, with BLQ rules or with mixed routes takes other paths in both implementations, and the kernel splits nothing by route while python runs one group per route.
- The python side runs with the `pkpdutils` logger at `ERROR`, since the kernel has no logging at all and one `logger.warning` inside a timed call would be measuring stderr. The data was chosen so that no warning fires either way (the terminal values stay above `TerminalPhase.relative_warning`).
- Neither side includes serialization. A wasm host hands the batch over as JSON and reads the result back the same way today, which for 100 000 x 12 is megabytes of text that this table does not count. Measuring that belongs to the task which builds the host.
- `nca` carries a few things the kernel refuses outright and which therefore cost it nothing here: the uncertainty of group curves, the multiple dose intervals, the partial areas, the acceptance criteria and the unit conversion of a result. The kernel's `KernelError::Unsupported` names each of them.
- Absolute times are specific to this machine. The ratios between the columns of one row are the portable part, and even those shift with the number of cores: twenty logical CPUs is what the shard columns are divided over.
