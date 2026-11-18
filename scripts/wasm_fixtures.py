"""Write the cases the wasm module has to answer bit for bit like the native build.

The two builds of this kernel - the python extension of `crates/python` and the
WebAssembly module of `crates/wasm` - run the same code of `crates/core` on two
targets. They have to answer the same numbers, and the bits are what says so: a
tolerance here would hide exactly the class of bug this fixture exists for (a
different libm, a different optimization of an accumulation, a draw of the
bootstrap which ran in another order).

Every case is built with `pkpdutils`, dumped in the wire encoding of
`pkpdutils-abi` (`batch_dict`, `options_dict`, numbers which are not finite as
`"NaN"`, the arrays flat and row major) and analysed with
`pkpdutils_core.analyze_batch`, the **native** build. Every column of the answer
travels as the hex bits of its `f64` values, in the order the wasm header
declares them: the parameters, the flag word, the boolean variables and then the
point variables. `js/test/same_bits.test.mjs` reads the file, calls the module
and compares.

The wheel has to be the build of this checkout, as it has to be for
`scripts/uncertainty_oracle.py`: the fixture names the version it was written
with and the test of the loader pins the version of the module, so a fixture of
another kernel than the module is a failure and not a surprise.

Run from the repository root:

    uv run --with-editable path/to/pkpdutils python scripts/wasm_fixtures.py
    uv run --with-editable path/to/pkpdutils python scripts/wasm_fixtures.py --check
"""

import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pkpdutils
import pkpdutils_core
from pkpdutils import Dose, Dosing, Route, Timecourse, Timecourses
from pkpdutils.nca import NCAOptions, UncertaintyMethod
from pkpdutils.nca.engine import batch_dict, options_dict

FIXTURE = Path(__file__).resolve().parent.parent / "js/test/fixtures/cases.json"

ANALYSIS_CONTRACT = "analysis/1"
BATCH_CONTRACT = "batch/1"


def encode(value: Any) -> Any:
    """One value of a batch or options dict in the encoding of the wire."""
    if isinstance(value, np.ndarray):
        # the arrays of the wire are flat and row major; the pyo3 path takes the
        # `(n_samples, n_time)` arrays as they are and numpy flattens them on the way
        # in, which the JSON encoding has to do here instead
        return [encode(x) for x in value.ravel(order="C").tolist()]
    if isinstance(value, (list, tuple)):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        return {key: encode(x) for key, x in value.items()}
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isfinite(number):
            return number
        if math.isnan(number):
            return "NaN"
        return "inf" if number > 0 else "-inf"
    raise TypeError(f"{type(value)} is no value of the wire")


def bits(values: Any) -> list[str]:
    """The bits of every value of a column, as the hex strings the test compares."""
    return [
        f"0x{struct.unpack('<Q', struct.pack('<d', float(x)))[0]:016x}" for x in values
    ]


def columns_of(answer: dict[str, Any]) -> list[dict[str, Any]]:
    """Every numeric column of a native answer, in the order the wasm header declares.

    The flag word and the booleans travel as `f64` in the payload of the module, which
    is what `crates/wasm/src/analyze.rs` writes and what the python binding reports as
    an integer array and as booleans.
    """
    columns = [
        {"name": name, "width": 1, "bits": bits(np.atleast_1d(values))}
        for name, values in answer["parameters"].items()
    ]
    columns.append(
        {"name": "flags", "width": 1, "bits": bits(np.atleast_1d(answer["flags"]))}
    )
    columns += [
        {"name": name, "width": 1, "bits": bits(np.atleast_1d(values).astype(float))}
        for name, values in answer["booleans"].items()
    ]
    for name, variable in answer["point_variables"].items():
        # the binding reports a point variable as its width and its flat values, which is
        # what the module writes into the payload as well
        columns.append(
            {
                "name": name,
                "width": int(variable["size"]),
                "bits": bits(
                    np.asarray(variable["values"], dtype=float).ravel(order="C")
                ),
            }
        )
    return columns


def case(
    name: str, batch: Timecourses, options: NCAOptions, note: str, *, exact: bool
) -> dict[str, Any]:
    """One case: the request of the wire and the answer of the native build.

    `exact` is the claim that every column of this case agrees bit for bit. It holds for
    the cases whose parameters are sums, differences and quotients of the data, and it
    does not for the ones which go through `exp` and `ln` often enough for the last bits
    to matter: `f64::exp` is the libm of the platform natively and the `libm` crate on
    `wasm32-unknown-unknown`, which are two implementations of the same function and
    differ in the last bit. A standard error of the terminal slope and a numerical
    derivative of the delta method then amplify that bit, because a cancellation and a
    difference quotient are what they are. `js/test/same_bits.test.mjs` requires the bits
    where this says so and a relative agreement of 1e-8 otherwise.
    """
    wire_batch = encode(batch_dict(batch))
    wire_options = encode(options_dict(options))
    answer = pkpdutils_core.analyze_batch(batch_dict(batch), options_dict(options))
    return {
        "name": name,
        "note": note,
        "exact": exact,
        "request": {
            "contract": ANALYSIS_CONTRACT,
            "batch": {"contract": BATCH_CONTRACT, **wire_batch},
            "options": wire_options,
        },
        "expected": {
            "n_rows": int(np.prod(wire_batch["sample_shape"]))
            if wire_batch["sample_shape"]
            else 1,
            "columns": columns_of(answer),
        },
    }


T = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 24.0])


def single_dose() -> Timecourses:
    """One oral curve of four points which halves every hour."""
    curve = Timecourse(
        time=[0.0, 1.0, 2.0, 3.0],
        value=[8.0, 4.0, 2.0, 1.0],
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=10, unit="mg", route=Route.ORAL),
        substance="caffeine",
    )
    return curve.to_batch("sample", "s1")


def two_subjects() -> Timecourses:
    """Two individual curves with a lag, a peak and a terminal phase."""
    rows = []
    for row, clearance in enumerate((2.0, 3.2)):
        values = 25.0 * np.exp(-clearance * T / 12.0) * (1.0 - np.exp(-3.0 * T))
        rows.append(
            Timecourse(
                time=T,
                value=values,
                time_unit="hr",
                unit="mg/l",
                dose=Dose(amount=100, unit="mg", route=Route.ORAL),
                substance="caffeine",
                label=f"s{row + 1}",
            )
        )
    return Timecourses.from_timecourses(rows, dim="subject")


def multiple_dose() -> Timecourses:
    """One curve over three doses of twelve hours, the steady state path with intervals."""
    dosing = Dosing.regimen(
        dose=Dose(amount=50, unit="mg", route=Route.ORAL), interval=12.0, n_doses=3
    )
    time = np.concatenate(
        [
            step + np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 11.5])
            for step in (0.0, 12.0, 24.0)
        ]
    )
    value = np.zeros(time.size)
    for amount, start in zip(dosing.amounts, dosing.times, strict=True):
        after = np.clip(time - start, 0.0, None)
        value += (amount / 10.0) * (1.0 - np.exp(-1.5 * after)) * np.exp(-0.12 * after)
    curve = Timecourse(
        time=time,
        value=value,
        time_unit="hr",
        unit="mg/l",
        dosing=dosing,
        substance="caffeine",
    )
    return curve.to_batch("sample", "s1")


def group_batch() -> Timecourses:
    """Three group curves with the spread and the subject count of every point."""
    rng = np.random.default_rng(4711)
    values, spread, counts = [], [], []
    for row in range(3):
        row_values = (
            25.0 * np.exp(-(2.0 + 0.4 * row) * T / 12.0) * (1.0 - np.exp(-3.0 * T))
        )
        values.append(row_values)
        spread.append(
            np.abs(0.18 * row_values * (1.0 + 0.2 * rng.standard_normal(T.size)))
        )
        counts.append(np.full(T.size, 8.0 + row))
    return Timecourses.from_arrays(
        np.vstack([T] * 3),
        np.vstack(values),
        sd=np.vstack(spread),
        n=np.vstack(counts),
        time_unit="hr",
        unit="mg/l",
        dims=("group",),
        coords={"group": ["low", "mid", "high"]},
        dose={"amount": 100.0, "unit": "mg"},
        route=Route.ORAL,
        substance="caffeine",
    )


def infusion() -> Timecourses:
    """One intravenous infusion, the route whose clearance is `cl` and not `cl_f`."""
    time = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    value = (
        12.0
        * np.exp(-0.25 * np.clip(time - 0.5, 0.0, None))
        * np.minimum(time / 0.5, 1.0)
    )
    curve = Timecourse(
        time=time,
        value=value,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_INFUSION, duration=0.5),
        substance="caffeine",
    )
    return curve.to_batch("sample", "s1")


def zero_dose() -> Timecourses:
    """A curve whose dose is 0, which makes every dose normalized column infinite."""
    curve = Timecourse(
        time=[0.0, 1.0, 2.0, 3.0],
        value=[8.0, 4.0, 2.0, 1.0],
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=0, unit="mg", route=Route.IV_BOLUS),
        substance="caffeine",
    )
    return curve.to_batch("sample", "s1")


def fixture() -> dict[str, Any]:
    """The whole file: the provenance of the numbers and every case."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "source": (
            "scripts/wasm_fixtures.py, the native build of this kernel; every number is the "
            "answer of pkpdutils_core.analyze_batch as its bits"
        ),
        "pkpdutils": pkpdutils.__version__,
        "pkpdutils_core": pkpdutils_core.__version__,
        "commit": commit,
        "cases": [
            case(
                "single_dose",
                single_dose(),
                NCAOptions(),
                "one oral curve of four points",
                exact=True,
            ),
            case(
                "two_subjects",
                two_subjects(),
                NCAOptions(),
                "two individual curves, the batch path over a sample dimension",
                exact=True,
            ),
            case(
                "multiple_dose",
                multiple_dose(),
                NCAOptions(),
                "three doses of twelve hours, the steady state path with its intervals",
                exact=False,
            ),
            case(
                "bootstrap_group",
                group_batch(),
                NCAOptions(
                    uncertainty=UncertaintyMethod.BOOTSTRAP,
                    n_boot=200,
                    seed=20261118,
                    ci_level=0.90,
                ),
                "a seeded bootstrap, which pins the draws of the generator in wasm",
                exact=False,
            ),
            case(
                "delta_group",
                group_batch(),
                NCAOptions(uncertainty=UncertaintyMethod.DELTA, ci_level=0.90),
                "the delta method, which draws nothing and differentiates numerically",
                exact=False,
            ),
            case(
                "infusion",
                infusion(),
                NCAOptions(),
                "an intravenous infusion, the route of cl and vss rather than cl_f",
                exact=True,
            ),
            case(
                "zero_dose",
                zero_dose(),
                NCAOptions(),
                "a dose of 0, whose dose normalized columns are infinite",
                exact=True,
            ),
        ],
    }


def main() -> int:
    """Write the fixture, or compare it with `--check`."""
    text = json.dumps(fixture(), indent=2) + "\n"
    if "--check" in sys.argv:
        if not FIXTURE.exists():
            print(f"{FIXTURE} is missing, run python scripts/wasm_fixtures.py")
            return 1
        if FIXTURE.read_text() != text:
            print(f"{FIXTURE} is stale, run python scripts/wasm_fixtures.py")
            return 1
        print(f"{FIXTURE} is what pkpdutils-core {pkpdutils_core.__version__} produces")
        return 0
    FIXTURE.write_text(text)
    print(f"wrote {FIXTURE} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
