"""Prints the reference tables of crates/core/src/percentile.rs from numpy.nanpercentile.

Usage: python scripts/reference_percentile.py {plain,hole,single,naive,grouping}

`single` are rows where `low + (high - low) * gamma` differs from numpy, `naive` rows where
`(1 - gamma) * low + gamma * high` does, both found by a seeded search. `grouping` are rows where
the virtual index `(k - 1) * q / 100` instead of numpy's `(k - 1) * (q / 100)` changes the output.
"""

import sys

import numpy as np


def lit(x: float) -> str:
    """The rust literal of a float, `f64::NAN` for a NaN."""
    if np.isnan(x):
        return "f64::NAN"
    return repr(float(x))


def row(v: np.ndarray, q: float) -> str:
    """One case of the table: the values, the percentile and what numpy answers."""
    cells = ", ".join(lit(x) for x in v)
    return f"    (&[{cells}], {float(q)!r}, {float(np.nanpercentile(v, q))!r}),"


def main(kind: str) -> None:
    """Write the cases of one kind of table to standard output."""
    rng = np.random.default_rng(7)
    if kind == "plain":
        for k in (5, 8):
            for q in (0, 10, 25, 50, 75, 90, 100):
                print(row(rng.lognormal(0, 1, k), q))
    elif kind == "hole":
        for _ in range(6):
            v = rng.lognormal(0, 1, 7)
            v[rng.choice(7, 3, replace=False)] = np.nan
            print(row(v, float(rng.choice([25, 50, 75]))))
    elif kind == "grouping":
        found = 0
        while found < 4:
            k = int(rng.integers(3, 9))
            v = np.sort(rng.lognormal(0, 1, k))
            q = float(rng.choice([10, 30, 70, 95, 99, 15, 35]))
            index = (k - 1) * q / 100
            b = int(np.floor(index))
            if b + 1 >= k:
                continue
            g = index - b
            lo, hi = v[b], v[b + 1]
            alt = hi - (hi - lo) * (1 - g) if g >= 0.5 else lo + (hi - lo) * g
            if alt != np.nanpercentile(v, q):
                print(row(v, q))
                found += 1
    else:
        found = 0
        while found < 4:
            k = int(rng.integers(3, 9))
            v = np.sort(rng.lognormal(0, 1, k))
            q = float(rng.choice([10, 25, 33.3, 50, 66.7, 75, 90, 95]))
            index = (k - 1) * (q / 100)
            b = int(np.floor(index))
            if b + 1 >= k:
                continue
            g = index - b
            lo, hi = v[b], v[b + 1]
            alt = lo + (hi - lo) * g if kind == "single" else (1 - g) * lo + g * hi
            if alt != np.nanpercentile(v, q) and (kind == "naive" or g >= 0.5):
                print(row(v, q))
                found += 1


main(sys.argv[1])
