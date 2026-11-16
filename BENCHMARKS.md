# Benchmarks

What the port bought, measured against the python package it is a port of. Every number here was measured on one machine on 2026-11-16 and is reproducible with the commands below; each table says which run it comes from, and nothing is scaled, extrapolated or rounded in the flattering direction.

## What is inside each timing

The two sides do not do the same work, so both are measured and both are named.

| column | what the timed call does | what it does not do |
|---|---|---|
| rust, serial | `analyze_batch_in_shards(batch, options, 1)` of a build **without** the `parallel` feature: the contract check, `Batch::validate`, `NcaOptions::validate`, the refusal of an analysis this build does not carry, the terminal windows and the manual columns of the options, the row loop over every row, the unit of every parameter through `pkpdutils-units`, the assembly of the `AnalysisResult` with its columns, flags, coordinates and attributes, and `AnalysisResult::validate` | build the `Batch` (not timed on either side), serialize anything: a wasm host pays for the JSON of the batch and of the result on top of this |
| rust, sharded | the same call with the plan `automatic_shards` chooses, on a build **with** the `parallel` feature | the same |
| python, `nca` | `nca(batch, options=NCAOptions())`: the option resolution, the times, values, dose and route arrays read out of the `Timecourses` dataset, the vectorized numpy core in chunks of `chunk_rows` (in the shared thread pool above `NCA_WORKER_THRESHOLD` = 20 000 rows), the pint unit of every variable, the assembly of the `xarray.Dataset` with its coordinates and `attrs` | build the `Timecourses` (not timed) |
| python, `run_rows` | `run_rows(t, c, dose_amount=..., dose_time=..., dose_duration=..., route=..., options=NCAOptions(n_workers=1))`: the numpy core alone, serial, which gives back a dict of numpy arrays | the units, the xarray result, the option resolution, any validation |

So the `nca` columns are the closest thing to the kernel's call: both read a batch in and hand a finished, unit-carrying result back. The `run_rows` column is the numeric core of python alone and the kernel still carries its own assembly in that comparison, which is the one place where the ratio understates the kernel rather than the other way round. Each python column is its own process per shape, like each rust column, so no cell of either side inherits the heap of the cell before it.

Both sides analyse **the same numbers**. The batch generator of `crates/core/examples/kernel_timing.rs` is written a second time in python for this measurement, operation for operation on the same xorshift64 stream and with no transcendental function in it, and every run prints the sum of the finite `auc_last` over its rows. The checksums agreed to all twelve digits printed on every one of the four shapes, for instance `1.077047357262e6` for 100 000 x 12 from the kernel and `1.077047357262e+06` from `nca`.

## The machine

| | |
|---|---|
| CPU | 12th Gen Intel Core i7-12700: 8 performance cores with 2 threads each and 4 efficiency cores, 20 logical CPUs, 4.9 GHz maximum, governor `powersave` |
| OS | Linux 7.0.0-43-generic |
| rust | rustc 1.100.0, `--release` with the default profile of this workspace (`opt-level = 3`, no LTO, 16 codegen units), rayon 1.12.0 |
| python | CPython 3.14.10, numpy 2.5.4, `pkpdutils` 1.4.0 |
| load | `mpstat` read the machine 92 to 94 percent idle before and after the runs of the table below, which is one to one and a half of the twenty logical CPUs busy with unrelated work; the one minute load average read 2.2 to 3.4 during them and that figure includes the runs themselves. The machine is shared and is not always that quiet, so every table here says which window it comes from: a pass taken at load 15 to 19 earlier the same day was discarded, a later window in which an unrelated job took the load to 15 and the same serial 1 000 x 200 call from 7.9 to 16 ms was discarded as well, and so was the spike's own first pass at load 57. The table below comes from two windows of this day, **A** from 14:58 to 15:11, whose two small shapes ran at load 3.4 to 4.4 between controls of 7.83 and 7.92 ms, and **C** from 15:28 to 15:43 at load 2.3 to 4.1 with a control under 8.1 ms before and after every single process. Each row says which one it is and no ratio divides across them. Window A's two large shape blocks are where the table has to say more: the 10 000 x 200 block carries no control of its own and the author's own build ran during it, so not one of its cells is published, while the 100 000 x 12 block was bracketed by controls which had drifted to 8.26 and 9.37 ms against the clean 7.90, and the two cells of it which window C could not re-take **are** published, marked **(a)** in the measured table, which names them and every ratio which divides them; two further cells of that block, the kernel's serial and python's default, are cited in the prose and named as window A's there. |

The serial rust runs are pinned to one performance core with `taskset -c 2`, the sharded runs are not pinned and use the whole machine, which is the spike's protocol. The python runs are not pinned either, which is the friendlier setting for them.

## The protocol

Three warm up calls, then three rounds of twenty timed calls; the median of each round, and the median of the three round medians is the number of a process. Every cell of every column runs in a process of its own - one process per shape and mode on the rust side, one process per shape and variant on the python side (`python_timing.py --shape 1000x200 --variant nca`) - so that no cell carries the allocator history of the cell before it, and its peak resident set size is its own. Every cell is three such processes and the tables report the median of their three medians.

That is a change from the first pass, where the python column ran all four shapes and all three variants in one process. It is the matched protocol, since every rust cell was always one process, and that is reason enough; what it is worth was not isolated here. At 10 000 x 200 the warm processes of the first pass read 554.40 ms for `nca` and 546.53 for `run_rows`, where the fresh processes of window C read 463.96 and 470.97, sixteen to twenty percent apart and in the same direction - but those are two different windows, and only the second has a control on both sides of every process of it. So the protocol is matched because item 3 of the review is right that it should be, not because a number was put on it. The kernel is far less sensitive to either: its serial cells reproduce to within one to two percent across every window and pass of the day at the dense shapes, 7.9452 against 7.9075 ms at 1 000 x 200 and 80.3194, 80.3335 against 79.2051 at 10 000 x 200, with the one exception named below, the 100 000 x 12 serial cell, which spans 95.8 to 107.7 ms across four windows.

**Both halves of the table come from one window.** The first two passes measured the rust columns in one window and reused the python column from an earlier one, and that mixing is not harmless: a python cell measured in another window moved by five percent on its own, which is the same size as the ratios that were read off it. The table below was taken in a single window, rust and python alternating shape by shape, with a rust control (the serial 1 000 x 200 call) run before and after every shape's python block so that the window itself is pinned to a number. Every control of the run is reported with the table.

One asymmetry of the protocol is named here because it favours python: `scripts/python_timing.py` calls `gc.collect()` before every timed call, outside the timer, so a collection the previous call made due does not land inside the next one. The kernel has no garbage collector and nothing on the rust side corresponds to it. A python caller pays that cost eventually; this measurement does not charge it to python.

The two builds write the same binary, so both are built and copied aside before the window opens and nothing is compiled inside it:

```bash
cargo build --release --example kernel_timing                      # the serial build
cp target/release/examples/kernel_timing /tmp/kt_serial
cargo build --release --example kernel_timing --features parallel  # the sharded build
cp target/release/examples/kernel_timing /tmp/kt_parallel
taskset -c 2 /tmp/kt_serial 1000x200     # a cell of the serial column, pinned
/tmp/kt_parallel 1000x200                # the same cell sharded, unpinned
/tmp/kt_parallel sweep                   # the crossover, its own run
```

The python column is `scripts/python_timing.py` of this repository, which builds the same batches and times the three variants above, one process per shape and variant; it imports `pkpdutils`, so it runs in that checkout:

```bash
uv run --no-sync python path/to/pkpdutils-core/scripts/python_timing.py --shape 1000x200 --variant nca
# and the other two columns of the same shape, each in its own process
uv run --no-sync python path/to/pkpdutils-core/scripts/python_timing.py --shape 1000x200 --variant nca-explicit
uv run --no-sync python path/to/pkpdutils-core/scripts/python_timing.py --shape 1000x200 --variant run-rows
uv run --no-sync python scripts/benchmark.py nca-large   # the benchmark of the package itself
```

## The measured table

Milliseconds per call, median of medians as above.

| rows x points | work | window | rust serial | rust sharded | shards | python `nca` | python `nca`, `n_workers=20` | python `run_rows`, serial |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 x 12 | 12 | A | 0.0482 | 0.0482 | 1 | 2.8874 | 2.9270 | 0.7946 |
| 1 000 x 200 | 200 000 | A | 7.9075 | 1.1126 | 80 | 54.5857 | 57.3242 | 54.0801 |
| 10 000 x 200 | 2 000 000 | C | 79.2051 | 10.6999 | 80 | 463.9564 | 218.9121 | 470.9740 |
| 100 000 x 12 | 1 200 000 | C | 95.8309 | 45.1613 | 80 | 116.7392 | 201.9856 (a) | 293.1906 (a) |

Every cell of a row was measured in the window the row names, so every ratio below divides two numbers from one window. In window A the controls ran before and after each shape's block: 7.8294 and 7.9167 ms around 1 x 12, 7.9005 and 7.9173 around 1 000 x 200. In window C every single process of every cell is bracketed by its own pair of controls, each under 8.1 ms against the clean 7.90, and a process whose closing control came in above that was discarded and run again: twenty six accepted processes are in the cells above, three per cell except the 10 000 x 200 `nca` cell which has five because it was run in two batches, one more belongs to a cell which is not published, and five were thrown away. **(a)** marks the two cells which could not be re-measured in window C before the machine filled up again and which are therefore window A's, as is every ratio which divides them: the `run_rows` / rust serial cell of that row, and the bolded 1.6x further down by which `n_workers=20` is slower than the default at that shape, both of which divide window A by window A. No ratio of the kernel against python uses one. The single `run_rows` process of that shape which window C did certify read 315.25 ms against window A's 293.19, so that cell is not obviously on either side of its value, which is why it is marked rather than corrected.

The two large shapes were measured in window A as well. Of its 100 000 x 12 block four cells are in this document: the two marked **(a)** in the table, and the kernel's serial 96.29 ms and python's default 125.67 ms, which the prose below cites by name as window A's wherever it uses them - the 3.0x of the ratio table divides by the first of those two and the bolded 1.6x by the second. Of its 10 000 x 200 block nothing is published. An earlier draft of this document read those discarded cells as a property of the window: python's dense cells moving by a fifth with the rest of the machine while the kernel's stayed put. The document's own `run_rows` cell refutes that reading. The front end's overhead over the bare numpy core is measurable in window A's controlled block, `nca` minus `run_rows` = 54.5857 - 54.0801 = +0.51 ms over a thousand rows, half a microsecond a row, so about 5 ms at ten thousand rows. That model predicts window C's `nca` cell from its own `run_rows` cell to 2.5 percent (470.97 + 5 against 463.96) and a third window measured independently by this task's reviewer to a twentieth of a percent (438.29 + 5 against 442.81), but window A's to **+27 percent** (462.27 + 5 against 592.66). The excess sits in window A's two `nca` variants, 592.66 against window C's 463.96 and 299.43 against 218.91, and nowhere else: `run_rows` walks the same 16 to 32 MB arrays and does not move with them. A property of the window would have moved all three cells. What was in that stretch instead was the author's own `cargo clippy` and `cargo test`, as the caveats below record, and that block carries no control of its own, so those cells are what a measurement discards and not what it reasons from. Window C is what this table publishes.

Two things about the 1 x 12 row. It is below the threshold, so `automatic_shards` plans one shard and the sharded column is the serial number of the same code path, not a second measurement of it.

And it is the one row where pinning decides the ratio, so both pairs were measured. Pinned to one performance core the kernel needs 0.0482 ms; unpinned, over six processes of the same one shard path, it reads 0.0497 to 0.1087 ms, a median of 0.0755, because 48 us is short enough for the core it lands on to be the whole story. Python does not care: 2.8874 ms unpinned against 2.8930 ms pinned, two parts in a thousand, since 2.9 ms of interpreter work is not core sensitive. **Pinned against pinned is 60x and unpinned against unpinned is 38x**, and the honest range of the unpinned pair is 26x to 59x. The table's 61.2x divided the pinned kernel cell by the unpinned python one, which is the flattering end of that: the number to quote at this shape is 60x, measured with both sides on the same core.

Every number above is a median of three processes. This is what those three spanned, as a percentage of the cell:

| rows x points | rust serial, pinned | rust sharded | python `nca` | python `nca`, `n_workers=20` | python `run_rows` |
|---|---:|---:|---:|---:|---:|
| 1 x 12 | 1.2 % | one shard | 2.6 % | 1.5 % | 1.7 % |
| 1 000 x 200 | 0.1 % | 7.6 % | 6.8 % | 4.9 % | 2.9 % |
| 10 000 x 200 | 0.3 % | 7.0 % | 9.1 % | 1.9 % | 9.3 % |
| 100 000 x 12 | 0.3 % | 3.7 % | 0.9 % | 5.8 % (a) | 2.7 % (a) |

The pinned serial column is the stable one, a tenth of a percent to a percent wherever the shape is large enough to measure. Everything which shares the machine is wider: the unpinned sharded cells span 3.7 to 7.6 percent and the python cells up to 9.3, and the widest processes are the ones which caught a neighbour - the 9.1 percent `nca` cell at 10 000 x 200 holds a process with one round of 1 004 ms among rounds of 454 and 505, which is what a shared machine looks like from inside a measurement. That is the figure to read a ratio against, and it is measured rather than assumed: a reader who wants three significant digits out of a python column does not have them.

The ratios of the same numbers. `nca` best is whichever of the two python front end columns is faster at that shape, so that the kernel is never divided by a python configuration python itself would not choose:

| rows x points | `nca` / rust serial | `nca` / rust sharded | `nca` best / rust sharded | `run_rows` / rust serial | rust serial / rust sharded | window |
|---|---:|---:|---:|---:|---:|---|
| 1 x 12 | 59.9x | 59.9x (one shard) | 59.9x (default) | 16.5x | 1.0x (one shard, below the threshold) | A |
| 1 000 x 200 | 6.9x | 49.1x | 49.1x (default) | 6.8x | 7.1x | A |
| 10 000 x 200 | 5.9x | 43.4x | 20.5x (`n_workers=20`) | 5.9x | 7.4x | C |
| 100 000 x 12 | 1.2x | 2.6x | 2.6x (default) | 3.0x (a, window A) | 2.1x | C |

Each of these divides two cells of one window, and each has one end rather than two, so nothing here is a range over windows. The one ratio which comes out of window A's large shape blocks is the 3.0x marked **(a, window A)**: the `run_rows` cell of 100 000 x 12, which the table publishes marked **(a)**, over the kernel's serial cell of that same block, 293.19 ms over 96.29 ms. Every cell of window A's 10 000 x 200 block is unpublished and no ratio here touches one.

The corroboration comes from outside instead. This task's reviewer re-measured the 10 000 x 200 shape in a third window and reports 5.65x serially and 41.7x sharded against the 5.9x and 43.4x above, so the figures published here are the generous end of that pair by about four percent, and the kernel's own serial cell at that shape reproduces within 1 to 2.5 percent of both of this document's windows. Within a window the three process pairings add their own range: 49.1x at 1 000 x 200 spans 46.7x to 53.8x and 43.4x at 10 000 x 200 spans 41.9x to 49.0x.

At 100 000 x 12 the same-window pairs which can be sourced are 2.50x, from the reviewer's first run of the shape, and 2.58x from window C, so **2.5x to 2.6x** is the figure. The 2.7x of the first pass divided a python cell of one pass by a rust cell of another and is retracted, and **nothing in this document is a cell of the kernel getting better**: every move in it is python's column being measured beside the rust column instead of in another window, or a window A number being dropped.

The one shape where the two sharded ratios differ is 10 000 x 200, where python's thread pool really does help: 43.4x is against the default `nca`, which leaves the pool out at that shape, and **20.5x is against python's own best**, which is `nca(n_workers=20)`. 20.5x is the number to quote for a host which would tune python, 43.4x the one for a host which would not.

Which configuration is "best" is itself inside the noise at two of the four shapes: at 1 x 12 the default leads by 1.4 percent and at 1 000 x 200 by 4.8 percent, both smaller than the spread of those cells, and in the first pass the order was the other way round at both. The label carries information only at 10 000 x 200, where `n_workers=20` is 2.12x the default, and at 100 000 x 12, where it is 1.6x slower than it, 201.99 ms against the 125.67 its own window read for the default: both cells of window A's last block, whose controls had drifted to 8.26 and 9.37 ms, so that ratio carries the direction and not its last digit.

Peak resident set size of the process, one process per shape and mode:

| rows x points | rust serial | rust sharded | python `nca` | python `nca`, `n_workers=20` | python `run_rows` |
|---|---:|---:|---:|---:|---:|
| 1 x 12 | 2.8 MB | 2.8 MB | 161.4 MB | 161.6 MB | 161.6 MB |
| 1 000 x 200 | 6.4 MB | 6.8 MB | 217.9 MB | 217.8 MB | 216.2 MB |
| 10 000 x 200 | 41.3 MB | 41.5 MB | 460.1 MB | 717.7 MB | 444.5 MB |
| 100 000 x 12 | 101.8 MB | 101.4 MB | 440.5 MB | 601.2 MB | 268.3 MB |

One process per shape and variant is what fills the python columns in, which the first pass could not: 161 MB is the floor of a process which has imported `pkpdutils` and numpy and analysed one curve, against 2.8 MB for the kernel on the same curve, and the thread pool is not free either, 718 MB against 460 MB at 10 000 x 200 and 601 MB against 440 MB at 100 000 x 12.

The sharded build costs the 1 x 12 shape nothing because it never starts the pool there: `automatic_shards` tests the work of the batch before it asks rayon how many threads it has, so a process which only ever sends small batches stays at one thread and 2.8 MB. The first pass of this table, taken before that order was fixed, read 3.4 MB on that shape, which was twenty thread stacks the kernel had no use for. Measured with `ps -o nlwp`: one thread over 300 batches of 300 x 12, twenty one over a batch of 1 000 x 200.

`scripts/benchmark.py nca-large` is the independent check on this harness: 100 000 curves of 12 points through `nca`, on its own data (a numpy generator with normal noise) and with its own three repeats. Run in window A it reported 134 ms over three processes of 154, 132 and 134 ms and a peak of 423 to 428 MB, where window C's matched cell is 116.7 ms and 440 MB. So it sits 15 percent above the cell this document publishes and 7 percent above window A's own 125.7 ms, which is the highest this document measures for that call: it is inside neither, and the two harnesses agree to within 15 percent and no more closely, which is what the cross-check establishes.

### What the table says

- On dense curves the kernel is **5.7x to 6.9x the python front end serially and 42x to 49x with its shards**, 49.1x at 1 000 x 200 and 43.4x at 10 000 x 200, and the shards are worth 7.1x to 7.4x of that on twenty logical CPUs. Against python's own best configuration it is 49.1x at 1 000 x 200, where python's pool cannot engage, and **20.5x at 10 000 x 200**, where it can. The reviewer's independent third window gives 41.7x and 5.65x for that shape, four percent under the cells above, and the lower end of each of the two ranges is that window rounded to the digits the range carries.
- On 100 000 short curves it is **1.2x serially and 2.6x sharded** (2.5x to 2.6x over the two same-window pairs which can be sourced), which is the weakest result of the port and the honest one: python's own thread pool engages at that shape (`NCA_WORKER_THRESHOLD` is 20 000 rows) while the kernel stops being bound by its arithmetic. One row of parameters is about 420 bytes, so the row loop of this batch writes 42 MB and the columns of the result another 38 MB, next to the 19 MB of the batch itself: that is the 102 MB the process peaks at, and that traffic is what the shards cannot divide. The row loop needs 0.57 us for a row of twelve points at a thousand rows and 0.96 us at a hundred thousand, and the difference is the memory. It is also the most window sensitive cell of the rust half of the document: the same pinned call read 95.8 ms in window C, 96.3 in window A and 105.4 to 107.7 ms in two earlier windows whose control was within a percent of all of them, a ten percent spread that no control of this document detects.
- A single curve costs the kernel 48 us against the front end's 2.9 ms, which is **60x with both sides pinned to one core** and 38x with neither of them pinned, but almost none of those 48 us is the curve: the fit below puts the fixed cost of a call at 48 us and the twelve observations of the curve at 0.6 us. A host which analyses one curve at a time is paying for the assembly of the result, not for the analysis.
- `nca` with `n_workers=20` is no faster than the default at 1 000 x 200, because `split_rows` keeps a slice at `min_rows` = 1 000 rows or more and a thousand rows are therefore one slice and one worker. It is 2.12x faster at 10 000 x 200 (ten slices, 2.12x of them, since only the numpy calls release the GIL), which is the configuration the `nca` best column divides by there, and **1.6x slower** than the default at 100 000 x 12, where the automatic setting takes eight workers (`max_workers`) and twenty of them contend for the same interpreter. Python's parallelism is bounded by `min_rows`, by `chunk_rows` and by the GIL, not by the machine.

### The cost of a call, decomposed

The serial build, one performance core, three processes per shape as above, from the six shapes `1x12`, `1x200`, `10x12`, `100x12`, `1000x12` and `1000x24`, which measured 0.0481, 0.0564, 0.0562, 0.1025, 0.6194 and 1.0411 ms. A least squares fit of `a + b * rows + c * rows * points` over those six numbers gives

| part | fitted |
|---|---|
| fixed per call | `a` = 48 us: the contract and option checks, the unit of each of the 48 parameter columns of an intravenous bolus through `pkpdutils-units`, the assembly and the validation of the result |
| per row | `b` = 0.15 us |
| per observation | `c` = 35 ns |

Those six shapes were measured in the window of the pass before this one, and the one of them which is also a cell of the table, 1 x 12, reads 0.0481 ms there against 0.0482 ms here, so the fit is anchored in the same place. The 48 us is a fitted intercept and not a measured quantity: nothing here times an empty call. What can be checked is whether the three parts add up to the six measurements, and they do, within 2.8 us everywhere and within four percent of each cell - +2.3 us at 10 x 12, -2.8 us at 100 x 12, -0.7 us at 1 x 12 and under 1.1 us on the other three. So the single curve of the table is the fixed cost of a call plus 0.6 us of curve, with the couple of percent the residuals carry.

Two consequences. A batch is cheap per curve and a single call is not, so a host which has a thousand curves should send them as one batch. And the kernel's advantage over numpy is largest where the rows are long: numpy is vectorized along the rows, so a short curve is where it is already efficient. Per observation, `run_rows` needs 270 ns on 200 point curves against the kernel's 40, and 244 ns on twelve point curves against the kernel's 80; that second pair is the `(a)` cell of the table divided by the kernel cell of the same window, 293.19 ms and 96.29 ms of window A.

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

The table compares 18 of the 20 shapes of `SWEEP`; the two it leaves out, 1 x 12 and 1 x 200, have a single row, so `shard_counts` caps them at one shard and there is nothing to compare. A cell is empty where the number of rows caps the plan below that count. The plan the kernel itself uses is the whole pool, four shards per thread: the 80 shard column where the batch has at least eighty rows and one shard per row below that. It is not always the best column of its row, and where it is not the difference is at most a fifth (500 x 12: 0.3099 against 0.2585). The sweep ran every shape in one process, which is why its 100 000 x 12 row is a fifth to a third above a fresh process of the same call: 131.6712 ms at one shard against 107.7013 in the window the sweep itself was taken in and 95.8309 in the quietest window of the day. That batch allocates and frees 56 MB per call, the heap it lands in matters, and so does the window. Within a row the counts are comparable, which is all the sweep is read for. It is the run the threshold was chosen from and it was taken in the first window, before the order of the tests inside `automatic_shards` was fixed; it passes every shard count explicitly, so that fix cannot have moved any of these numbers.

Where the curve crosses one, with the plan the kernel really uses, which is the whole pool: at 1 200 observations the shards lose (0.1371 against 0.1022), at 2 400 they win by a fifth (0.1483 against 0.1753), at 4 000 by a third (0.1508 against 0.1996, twenty rows and so twenty shards) and from 10 000 on by more than two (0.2101 against 0.4321 at 50 x 200, whose plan of 50 shards the sweep does not time, since `shard_counts` drops 80 as above the row count and keeps 20; 0.2553 against 0.6234 at 1 000 x 12 is the plan exactly). **4 096 is above that crossing and not at it**, and the reason is the size of what it gives up and not the ratio: between the crossing and the threshold the shards save 0.027 ms at 2 400 observations and 0.049 ms at 4 000, hundredths of a millisecond on calls which cost two tenths of one, and shapes that small move by about that much between runs - the sweep reads 10 x 12 at 0.1307 ms where a fresh pinned process of the same binary reads 0.0562. In exchange, a batch below the threshold never starts the pool at all, since a plan of one shard does not reach rayon, so a process which only sends small batches stays at one thread and pays for no thread stacks. A threshold at the crossing would buy those hundredths back and would rest on which of two neighbouring sweep cells one believes; 4 096 is the round number above both, and no shape of the sweep is made slower by a plan it did not ask for. Python's `NCA_WORKER_THRESHOLD` is 20 000 *rows* and counts no points at all, so a thousand curves of two hundred points - 200 000 observations, 56 ms of work in `nca`, 6x off what the shards do here - stays in a single thread there.

`SHARDS_PER_THREAD` is 4 and comes from the same sweep: four shards per thread beat one per thread at 100 x 200 (0.3006 against 0.3762), 1 000 x 200 (1.3435 against 1.7312), 2 000 x 12 (0.4304 against 0.4999), 10 000 x 12 and 10 000 x 200, drew at 5 000 x 12 and 100 000 x 12 and lost once, at 500 x 12 (0.3099 against 0.2585). The rows of a batch are not equally expensive - the terminal regression of a row evaluates every candidate window after its peak - so a shard per thread leaves the thread which drew the expensive ones finishing alone, and the pool steals the difference when there are more shards than threads. A plan which grows with the work instead, two shards at the threshold and twenty at ten times it, was measured against the full pool and was slower everywhere in between: 0.47 ms against 0.26 ms at 12 000 observations, 0.25 against 0.21 at 10 000.

## On wasm

The same example compiled for `wasm32-wasip1` and run under wasmtime 50.0.1, unpinned, three processes per shape as above, same checksums as the native build down to all twelve digits:

| rows x points | native serial | wasm32-wasip1 | wasm / native |
|---|---:|---:|---:|
| 1 x 12 | 0.0481 | 0.0603 | 1.25x |
| 1 000 x 200 | 7.9452 | 9.8411 | 1.24x |
| 100 000 x 12 | 105.4011 | 122.0083 | 1.16x |

The native column is the native run of the wasm window, not the table above: its 100 000 x 12 cell read 105.4011 ms there, where the quietest window of the day reads 95.8309 ms for the same pinned call. The pair measured together is what the ratio divides, which is why that cell is not replaced here. The first pass measured one process per shape and read 0.0602, 9.7630 and 122.4929, within a percent of these three. The window matters more than the process count: the same wasm binary read 13 to 26 ms at 1 000 x 200 while an unrelated job held the machine at load 15, where the native control read 11 to 16 ms against its 7.9, so that window was discarded and these runs were taken at load 3.0 to 3.6 with a native control of 7.9311 ms before and 7.9523 ms after them.

So the kernel costs about a quarter more inside a wasm runtime than outside it, and the 100 000 x 12 shape, which is memory bound rather than compute bound, loses the least. The build was made with the `parallel` feature deliberately, to see what it does there: rayon reports one thread, `automatic_shards` therefore plans one shard, and an explicit plan of two shards runs them one after the other in the calling thread without trapping. The feature buys a wasm instance nothing, which is why the wasm build leaves it out.

## Against the spike

`spike/RESULTS.md` is the first measurement of this port: a kernel of eight outputs, no options, no flags, no BLQ or dose handling, no units and no result assembly, built with LTO and one codegen unit. The full kernel is slower than it and this is the number that matters, since this is what a host gets:

| rows x points | spike serial | kernel serial | spike rayon | kernel sharded |
|---|---:|---:|---:|---:|
| 1 x 12 | 0.0003 | 0.0482 (161x) | 0.0008 | 0.0482 (60x) |
| 1 000 x 200 | 4.12 | 7.9075 (1.9x) | 0.371 | 1.1126 (3.0x) |
| 10 000 x 200 | 41.19 | 79.2051 (1.9x) | 3.66 | 10.6999 (2.9x) |
| 100 000 x 12 | 19.75 | 95.8309 (4.9x) | 2.04 | 45.1613 (22x) |

The spike column is its own run in its own window and is not re-measured here, so each of these ratios spans two windows and is read for its size and not its last digit.

The two dense shapes are explained by scope: the kernel computes 48 parameter columns, the count of subjects among them, where the spike computed eight numbers, so twice the time per observation on a dense curve is the price of the other forty parameters, the flags, the BLQ rules, the dose handling and the units. The 100 000 x 12 column is explained by memory: 420 bytes of parameters per row against the spike's 64, so 42 MB per call against 6.4 MB and 38 MB of result columns against 6.4 MB, which is why the shards there buy 2.1x and not the spike's 9.7x. The ratio of the port's own promise - the spike's "8 300x against python on one curve" - is likewise mostly the 48 us of fixed cost that the spike did not have; the honest single curve figure is 60x.

LTO and one codegen unit, which the spike used and this workspace does not configure, were measured on the kernel as well: `--config profile.release.lto=true --config profile.release.codegen-units=1` gave 0.0440 ms at 1 x 12, 8.05 at 1 000 x 200, 101.03 at 100 000 x 12 serially and 1.050 and 50.97 sharded. That was the first window, so it is read against that window's own numbers (0.0479, 7.9145, 107.7013, 1.0740 and 55.7648): between 9 percent faster and 2 percent slower than the default profile. The table is the default profile, which is what `cargo run --release` gives.

## Caveats

- Between one and two of the twenty logical CPUs were busy with an unrelated application throughout. That costs the sharded runs some of their speedup and the pinned serial runs next to nothing, so the sharded ratios here are if anything conservative.
- The data is synthetic: a geometric decline with five percent multiplicative noise, one intravenous bolus at time zero, no missing values, no values below a limit of quantification, one route for the whole batch. A batch with holes, with BLQ rules or with mixed routes takes other paths in both implementations, and the kernel splits nothing by route while python runs one group per route.
- The python side runs with the `pkpdutils` logger at `ERROR`, since the kernel has no logging at all and one `logger.warning` inside a timed call would be measuring stderr. The data was chosen so that no warning fires either way (the terminal values stay above `TerminalPhase.relative_warning`).
- Neither side includes serialization. A wasm host hands the batch over as JSON and reads the result back the same way today, which for 100 000 x 12 is megabytes of text that this table does not count. Measuring that belongs to the task which builds the host.
- `nca` carries a few things the kernel refuses outright and which therefore cost it nothing here: the uncertainty of group curves, the multiple dose intervals, the partial areas, the acceptance criteria and the unit conversion of a result. The kernel's `KernelError::Unsupported` names each of them.
- Absolute times are specific to this machine **and to the window they were taken in**. This is a shared machine and the same binary on the same shape read 7.9 ms in the window of the table and 11 to 16 ms in a window where an unrelated job held the load at 15; the wasm column went from 9.8 ms to 13 to 26 ms in that same window. Every table here therefore carries the controls of its run, and no two columns of one ratio come from different windows.
- The ratios between the columns of one row are the portable part, and even those shift with the number of cores: twenty logical CPUs is what the shard columns are divided over.
- A cell is a median of three processes and the spread of those three is published above, because on some cells it is as large as the differences a reader might want to read off them: the python 1 000 x 200 cell spans 6.8 percent, wider than the 4.8 percent between python's two configurations there, and the dense python cells of window C span 9 percent.
- Two python cells of window A, the `nca` variants at 10 000 x 200, read 28 and 37 percent above window C's. That block carries no control and the author's own build ran during it, and the `run_rows` cell of the same shape does not move with them, so those two cells are discarded rather than explained. A control which is pinned and compute bound certifies that a free core existed and nothing more; it read 7.9 ms around blocks of the pinned 100 000 x 12 call which themselves span 95.8 to 107.7 ms. What a measurement on this machine can carry is a cell with a control on both sides of every process of it, which is what window C is and what the table publishes.
- The author's own compiler and test runs polluted the first measurement of the last shape, and that is in the record because the controls caught it: the block was bracketed by 8.26 and 9.37 ms against the clean 7.90 and its python cells came out ten to twenty percent high. The shape was measured again in a quiet window; the two cells which could not be re-taken before the machine filled up again are marked (a) in the table. Nothing is gated on a number taken while something else of mine was running.
