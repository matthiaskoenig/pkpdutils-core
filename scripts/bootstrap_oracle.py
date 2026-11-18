"""Write the oracle of the bootstrap, python's whole answer for a batch of group curves.

`crates/core/src/uncertainty.rs` runs the parametric bootstrap inside
`analyze_batch`. This script analyses a handful of group batches with the
**python** engine of `pkpdutils` - `pkpdutils.nca.engine._force_python`, which
refuses every batch to the kernel, so nothing of the answer comes from the code
under test - and writes the batch, the options and every uncertainty variable to
`crates/core/tests/data/bootstrap_oracle.json`.

The batch and the options travel in the wire encoding of `pkpdutils-abi`
(`batch_dict`, `options_dict` of the host, numbers which are not finite as
`"NaN"`), so `crates/core/tests/bootstrap_oracle.rs` deserializes them straight
into `Batch` and `NcaOptions` and asks `analyze_batch` for the same answer.
Nothing of the comparison is transcribed here either: the two engines draw their
normals from the same generator, so they agree bit for bit and the tolerance of
the comparison is there for the transcendental functions alone.

Run from the repository root against the checkout the kernel mirrors:

    uv run --with-editable path/to/pkpdutils python scripts/bootstrap_oracle.py
    uv run --with-editable path/to/pkpdutils python scripts/bootstrap_oracle.py --check
"""

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pkpdutils
from pkpdutils import Dose, Dosing, Route, Timecourses
from pkpdutils.nca import NCAOptions, nca
from pkpdutils.nca.engine import batch_dict, options_dict
from pkpdutils.nca.options import BootstrapDistribution, BootstrapSpread
from pkpdutils.nca.uncertainty import DISCRETE_PARAMETERS
from pkpdutils.result import base_name

FIXTURE = (
    Path(__file__).resolve().parent.parent / "crates/core/tests/data/bootstrap_oracle.json"
)

T = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 24.0])


def encode(value: Any) -> Any:
    """One value of a batch or options dict in the encoding of the wire."""
    if isinstance(value, np.ndarray):
        # the arrays of the wire are flat and row major; the host hands the pyo3 path a
        # `(n_samples, n_time)` array and numpy flattens it on the way in, which the JSON
        # encoding has to do here instead
        return [encode(x) for x in value.ravel(order="C").tolist()]
    if isinstance(value, (list, tuple)):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        return {key: encode(x) for key, x in value.items()}
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) or isinstance(value, np.floating):
        number = float(value)
        if math.isfinite(number):
            return number
        if math.isnan(number):
            return "NaN"
        return "inf" if number > 0 else "-inf"
    raise TypeError(f"{type(value)} is no value of the wire")


def group_batch(n_rows: int = 4, cv: float = 0.18) -> Timecourses:
    """A batch of group curves with `se`, `sd` and the subject count of every point."""
    rng = np.random.default_rng(4711)
    rows, spread, counts = [], [], []
    for row in range(n_rows):
        clearance = 2.0 + 0.4 * row
        values = 25.0 * np.exp(-clearance * T / 12.0) * (1.0 - np.exp(-3.0 * T))
        rows.append(values)
        spread.append(cv * values * (1.0 + 0.2 * rng.standard_normal(T.size)))
        counts.append(np.full(T.size, 8.0 + row))
    values = np.vstack(rows)
    sd = np.abs(np.vstack(spread))
    n = np.vstack(counts)
    # a point without a spread and a point without a value: both are masked by the
    # resampling and both still consume their draw
    sd[1, 3] = np.nan
    values[2, -1] = np.nan
    return Timecourses.from_arrays(
        time=T,
        values=values,
        sd=sd,
        n=n,
        time_unit="hr",
        unit="mg/l",
        dose={"amount": 100.0, "unit": "mg"},
        route="oral",
        dims=("subject",),
    )


def steady_state_batch() -> Timecourses:
    """A multiple dose batch of group curves, so the steady state parameters travel too."""
    times = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    rows, spread = [], []
    for row in range(2):
        values = 8.0 + 6.0 * np.exp(-0.25 * (1.0 + 0.3 * row) * times)
        rows.append(values)
        spread.append(0.15 * values)
    return Timecourses.from_arrays(
        time=times,
        values=np.vstack(rows),
        se=np.vstack(spread),
        n=np.full(2, 10.0),
        time_unit="hr",
        unit="mg/l",
        dose=Dosing.from_doses(
            [
                Dose(amount=50.0, unit="mg", route=Route.ORAL, time=-12.0),
                Dose(amount=50.0, unit="mg", route=Route.ORAL, time=0.0),
            ]
        ),
        dims=("subject",),
    )


def uncertainty_of(result: Any) -> dict[str, list[Any]]:
    """Every uncertainty variable of a result, by name."""
    out = {}
    for name in sorted(result.ds.data_vars):
        base = base_name(name)
        if base is None or base in DISCRETE_PARAMETERS:
            continue
        values = np.asarray(result.ds[name].values, dtype=np.float64).ravel()
        out[name] = encode(values)
    return out


def case(name: str, timecourses: Timecourses, options: NCAOptions) -> dict[str, Any]:
    """One case: the batch and the options of the wire, and python's uncertainty for them."""
    from pkpdutils.nca import engine

    forced = engine._force_python
    engine._force_python = True
    try:
        result = nca(timecourses, options=options)
    finally:
        engine._force_python = forced
    # `contract` is no key of the host's dict: the bindings speak one contract and write it
    # themselves (`crates/python/src/convert.rs`), which this encoding has to do in their place.
    # A kernel which renamed it fails the comparison with the error it is there for.
    batch = {"contract": "batch/1", **batch_dict(timecourses)}
    kernel_options = options_dict(options)
    answer = uncertainty_of(result)
    if not answer:
        raise SystemExit(f"{name}: the python engine reported no uncertainty")
    return {
        "name": name,
        "batch": encode(batch),
        "options": encode(kernel_options),
        "expected": answer,
    }


def provenance() -> dict[str, str]:
    """Which `pkpdutils` answered, and from which tree."""
    source = Path(pkpdutils.__file__).resolve()
    commit = "unknown"
    try:
        head = subprocess.run(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(source.parent), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        commit = head + (" (dirty)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "pkpdutils": pkpdutils.__version__,
        "commit": commit,
        "numpy": np.__version__,
        "engine": "pkpdutils.nca.nca.nca with engine._force_python",
    }


def fixture() -> dict[str, Any]:
    """The whole fixture, one entry per case."""
    group = group_batch()
    base = {"n_boot": 200, "seed": 20261118, "ci_level": 0.90}
    return {
        "provenance": provenance(),
        "cases": [
            case("se_normal", group, NCAOptions(**base)),
            case(
                "sd_lognormal",
                group,
                NCAOptions(
                    bootstrap_spread=BootstrapSpread.SD,
                    bootstrap_distribution=BootstrapDistribution.LOGNORMAL,
                    **base,
                ),
            ),
            case(
                "se_lognormal_95",
                group,
                NCAOptions(
                    bootstrap_distribution=BootstrapDistribution.LOGNORMAL,
                    n_boot=64,
                    seed=5,
                    ci_level=0.95,
                ),
            ),
            case("steady_state", steady_state_batch(), NCAOptions(**base)),
        ],
    }


def main() -> int:
    """Write the fixture, or compare it with `--check`."""
    text = json.dumps(fixture(), indent=2) + "\n"
    if "--check" in sys.argv:
        if not FIXTURE.exists():
            print(f"{FIXTURE} is missing, run python scripts/bootstrap_oracle.py")
            return 1
        if FIXTURE.read_text() != text:
            print(f"{FIXTURE} is stale, run python scripts/bootstrap_oracle.py")
            return 1
        print(f"{FIXTURE} is what pkpdutils {pkpdutils.__version__} produces")
        return 0
    FIXTURE.write_text(text)
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
