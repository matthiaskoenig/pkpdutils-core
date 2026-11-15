# Spike: minimal Rust NCA kernel against the Python baseline

Machine: Intel Core i7-12700 (8 performance cores with 2 threads, 4 efficiency cores; 20 logical CPUs), Linux 7.0, governor `powersave`, rustc 1.100.0, `--release` (opt-level 3, lto, codegen-units 1), rayon 1.10. Serial runs are pinned with `taskset -c 2` (a performance core), rayon runs use all 20 logical CPUs and row chunks of `rows / (threads * 8)`. Protocol: median of 20 runs after 3 warm ups, 3 rounds, the median of the three round medians is reported. Peak RSS is `VmHWM` after the run, one process per shape and mode (includes the generated input).

Measure only on a quiet machine: a first pass taken at load average 57 (other jobs) was discarded; it showed rayon at 1.1x. This table was taken at load 4.4.

| rows x points | Python serial | Python threaded | Rust serial | Rust rayon | Python serial / Rust serial | Python threaded / Rust rayon | Rust serial / rayon | peak RSS (rayon) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 x 12 | 2.5 ms | 2.5 ms | 0.0003 ms | 0.0008 ms | 8300x | 3100x | 0.4x | 3.3 MB |
| 1 000 x 200 | 51.4 ms | 50.9 ms | 4.12 ms | 0.371 ms | 12.5x | 137x | 11.1x | 5.1 MB |
| 10 000 x 200 | 571.8 ms | 518.2 ms | 41.19 ms | 3.66 ms | 13.9x | 142x | 11.3x | 19.8 MB |
| 100 000 x 12 | 300.6 ms | 115.6 ms | 19.75 ms | 2.04 ms | 15.2x | 56.8x | 9.7x | 19.1 MB |

Extra, 100 000 x 12 rayon: 16 threads on cores 0-15 2.02 ms (9.8x), 8 threads on one thread of each performance core 2.56 ms (7.7x).

## Correctness (before timing)

`spike check`: noise-free `c = 10 exp(-0.2 t)`, 4801 points to t = 48: `lambda_z` 0.200000000, `thalf` 3.465735903 (ln2/k 3.465735903), `auc_inf_obs` 50.000000 (c0/k 50). Hand trapezoid (0,2),(1,4),(3,4): `auc_last` 11 (3 + 8), asserted equal. Log-down (0,4),(2,2),(4,2): 9.770780163555854 equal to `2/ln2*2 + 4`.

## Gate on 100 000 x 12

| criterion | threshold | measured | |
|---|---|---|---|
| Rust serial | at most 60 ms | 19.75 ms | pass |
| one 12-point curve | at most 5 us | 0.198 us | pass |
| rayon speed up on 20 logical CPUs | at least 8x | 9.7x | pass |

Caveats on the ratios (they overstate what the full port will reach):

- Output scope, the largest single reason. The kernel produces 8 outputs (`cmax`, `tmax`, `clast`, `tlast`, `auc_last`, `lambda_z`, `thalf`, `auc_inf_obs`) where Python's `nca()` produces dozens. Missing families: the moments and derived parameters (`aumc`, `mrt`, `cl`, `vz`, `vss`), the whole predicted set (`_pred` parameters and extrapolated fractions), the terminal window statistics (R2, span, candidates), flags, BLQ and dose handling, pint unit derivation and the xarray assembly of the result. The Python column pays for all of it and the Rust column for none of it.
- Terminal selection rule. `terminal.py` applies `tie_tolerance`, which prefers more points among windows whose adjusted R2 is near the best; the Rust kernel takes a strict greater-than on the adjusted R2, so on noisy data it can pick a different window. The effect on the timing is negligible, since the same windows are evaluated either way.
- No options, no error handling, windows ending at the last point only, as the brief says.

The 1 x 12 rayon time is pure pool dispatch (one chunk). Speedup beyond 8x comes from SMT and the efficiency cores. The spike is not a workspace member; Task 2's workspace manifest needs `exclude = ["spike"]`.
