"""Time `pkpdutils` on the batches of `crates/core/examples/kernel_timing.rs`.

This is the python column of `BENCHMARKS.md`. It builds the same batches as that
example, operation for operation on the same xorshift64 stream and with no
transcendental function anywhere, so both sides analyse the same numbers; every
row prints the sum of the finite `auc_last` over its rows, and those checksums
agree with the kernel's to all twelve digits printed. The protocol is the same
too: three warm up calls, three rounds of twenty timed calls, the median of each
round and the median of the three round medians.

Three variants per shape, since the two implementations do not do the same work
and `BENCHMARKS.md` names what is inside each timing:

* `nca` with the default options: the front end a user calls, which resolves the
  options, reads the times, values, dose and route arrays out of the
  `Timecourses` dataset, runs the vectorized numpy core in chunks (in the shared
  thread pool above `pkpdutils.parallel.NCA_WORKER_THRESHOLD`, 20 000 rows),
  derives the pint unit of every variable and assembles the `xarray` result.
* `nca` with `n_workers` set to the number of logical CPUs: the same, with the
  thread pool asked for explicitly, which is what a user has to do for a batch
  of fewer than 20 000 rows.
* `run_rows` with `n_workers=1`: the numpy core alone, serial, which gives back
  a dict of numpy arrays. No units, no xarray, no option resolution.

It imports `pkpdutils`, so it runs from a checkout or an install of it and not
from this repository's environment:

```bash
uv run --no-sync python path/to/pkpdutils-core/scripts/python_timing.py
uv run --no-sync python path/to/pkpdutils-core/scripts/python_timing.py --shape 1000x200
uv run --no-sync python path/to/pkpdutils-core/scripts/python_timing.py \
    --shape 1000x200 --variant nca
```

`--shape` and `--variant` are what the measurement of `BENCHMARKS.md` uses: one
process per shape and variant, as the rust side runs one process per shape and
mode, so that no cell carries the allocator history of the cells before it.

The `pkpdutils` logger is set to `ERROR` while the calls are timed: the kernel
has no logging at all, and one `logger.warning` inside a timed call would be
measuring stderr.
"""

import argparse
import gc
import logging
import os
import resource
import statistics
import time
from collections.abc import Callable

import numpy as np
import pkpdutils
from pkpdutils import Dose, NCAOptions, Route, Timecourses, nca
from pkpdutils.nca.nca import run_rows

#: 64 bits of ones, the word the xorshift stream wraps in
MASK = (1 << 64) - 1

#: the multiplier of the xorshift64 stream, which is also its seed
MULT = 0x2545F4914F6CDD1D

#: the mantissa a uniform value is divided by, as `f64` does it in rust
TWO_53 = 9007199254740992.0

#: timed calls of one round
RUNS = 20

#: rounds of a measurement
ROUNDS = 3

#: calls before the first round, whose times are thrown away
WARM_UPS = 3

#: the four shapes of `BENCHMARKS.md`, rows by time points
SHAPES = [(1, 12), (1000, 200), (10000, 200), (100000, 12)]


def uniforms(n: int) -> list[float]:
    """The first `n` values of the xorshift64 stream of the rust example.

    Args:
        n: how many values.

    Returns:
        The values, each in `[0, 1)`.
    """
    state = MULT
    out = []
    for _ in range(n):
        state ^= state >> 12
        state ^= (state << 25) & MASK
        state ^= state >> 27
        out.append((((state * MULT) & MASK) >> 11) / TWO_53)
    return out


def arrays(n_samples: int, n_time: int) -> tuple[np.ndarray, np.ndarray]:
    """The times and the values of one shape, as `batch_of` of the rust example.

    Args:
        n_samples: number of curves.
        n_time: number of time points of every curve.

    Returns:
        The times `(n_time,)` and the values `(n_samples, n_time)`.
    """
    stream = uniforms(n_samples * (n_time + 2))
    k = 0
    times = [
        (0.25 + 23.75 * j / (n_time - 1)) if n_time > 1 else 0.25
        for j in range(n_time)
    ]
    values = np.empty((n_samples, n_time), dtype=np.float64)
    for row in range(n_samples):
        c = 100.0 / (20.0 + 40.0 * stream[k])
        k += 1
        q = 1.0 - min((4.0 + 2.0 * stream[k]) / max(n_time - 1, 1), 0.5)
        k += 1
        for j in range(n_time):
            values[row, j] = c * (0.95 + 0.1 * stream[k])
            k += 1
            c *= q
    return np.asarray(times, dtype=np.float64), values


def batch_of(times: np.ndarray, values: np.ndarray) -> Timecourses:
    """The batch of one shape: an intravenous bolus of 100 mg at time zero.

    Args:
        times: the times of every curve.
        values: the values, one row per curve.

    Returns:
        The batch.
    """
    return Timecourses.from_arrays(
        times,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        dose=Dose(amount=100.0, unit="mg", route=Route.IV_BOLUS),
        substance="drug",
    )


def timing(call: Callable[[], float]) -> tuple[float, list[float], float]:
    """Time one call under the protocol of `BENCHMARKS.md`.

    Args:
        call: the call, which returns the checksum of its result.

    Returns:
        The median of the round medians in milliseconds, the round medians and
        the checksum of the last call.
    """
    checksum = 0.0
    for _ in range(WARM_UPS):
        checksum = call()
    medians = []
    for _ in range(ROUNDS):
        times = []
        for _ in range(RUNS):
            gc.collect()
            start = time.perf_counter()
            checksum = call()
            times.append((time.perf_counter() - start) * 1000.0)
        medians.append(statistics.median(times))
    return statistics.median(medians), medians, checksum


def report(label: str, rows: int, points: int, call: Callable[[], float]) -> None:
    """Time one variant of one shape and print the row of the table.

    Args:
        label: the variant.
        rows: number of curves.
        points: number of time points.
        call: the call to time.
    """
    median, medians, checksum = timing(call)
    rounds = " ".join(f"{t:.4f}" for t in medians)
    print(
        f"| {rows} x {points} | {rows * points} | {label} | {median:.4f} | "
        f"{rounds} | {median * 1000.0 / rows:.3f} | {checksum:.12e} |",
        flush=True,
    )


def main() -> int:
    """Time every variant of every shape.

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--shape",
        action="append",
        metavar="ROWSxPOINTS",
        help="one shape to time, repeatable; all four of BENCHMARKS.md by default",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=["nca", "nca-explicit", "run-rows"],
        help="one variant to time, repeatable; all three by default. One variant"
        " per process is how the rust side is measured, one process per shape and"
        " mode, so that no cell inherits the allocator history of another",
    )
    args = parser.parse_args()
    variants = set(args.variant or ["nca", "nca-explicit", "run-rows"])
    shapes = SHAPES
    if args.shape:
        shapes = [
            (int(text.split("x")[0]), int(text.split("x")[1])) for text in args.shape
        ]
    logging.getLogger("pkpdutils").setLevel(logging.ERROR)
    workers = os.cpu_count() or 1
    print(
        f"# pkpdutils timing\n\n"
        f"pkpdutils: {pkpdutils.__version__}\nnumpy: {np.__version__}\n"
        f"workers of the explicit variant: {workers}\n"
        f"load average: {os.getloadavg()[0]:.2f}\n"
    )
    print(
        "| rows x points | work | variant | median ms | rounds ms | us per row "
        "| checksum |"
    )
    print("|---|---:|---|---:|---|---:|---:|")
    for rows, points in shapes:
        times, values = arrays(rows, points)
        batch = batch_of(times, values)
        default = NCAOptions()
        explicit = NCAOptions(n_workers=workers)
        serial = NCAOptions(n_workers=1)
        t2d = np.ascontiguousarray(np.broadcast_to(times, (rows, points)))
        dose_amount = np.full((rows, 1), 100.0)
        dose_time = np.zeros((rows, 1))
        dose_duration = np.full((rows, 1), np.nan)

        def front_end(options: NCAOptions = default) -> float:
            result = nca(batch, options=options)
            return float(np.nansum(result.ds["auc_last"].values))

        def core(options: NCAOptions = serial) -> float:
            out = run_rows(
                t2d,
                values,
                dose_amount=dose_amount,
                dose_time=dose_time,
                dose_duration=dose_duration,
                route=Route.IV_BOLUS,
                options=options,
            )
            return float(np.nansum(out["auc_last"]))

        if "nca" in variants:
            report("nca, default", rows, points, front_end)
        if "nca-explicit" in variants:
            report(
                f"nca, n_workers={workers}",
                rows,
                points,
                lambda options=explicit: front_end(options),
            )
        if "run-rows" in variants:
            report("run_rows, n_workers=1", rows, points, core)
    print(
        f"\npeak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB\n"
        f"load average: {os.getloadavg()[0]:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
