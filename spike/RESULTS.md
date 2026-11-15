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

Notes: the kernel is not fully like-for-like with Python (no options, flags, BLQ, error handling, unit work; windows ending at the last point only, as the brief says), so the ratios are an upper bound of what the full port reaches. The 1 x 12 rayon time is pure pool dispatch (one chunk). Speedup beyond 8x comes from SMT and the efficiency cores. The spike is not a workspace member; Task 2's workspace manifest needs `exclude = ["spike"]`.
