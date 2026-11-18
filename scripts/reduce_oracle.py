"""Write the oracle of the replicate reduction, python's own answer.

`crates/core/src/uncertainty.rs` is the port of
`pkpdutils.nca.uncertainty.reduce_replicates`. This script builds a handful of
replicate matrices which carry the cases the reduction has to get right - holes,
a row with a single finite replicate, non-positive replicates of a log-normal
parameter, a row without a usable spread, a missing subject count - calls the
python function on them and writes its answer to
`crates/core/tests/data/reduce_oracle.json`. Nothing of the answer is computed
here: the numbers come out of the imported `pkpdutils`, so the fixture is an
independent oracle and not a transcript of the port.

The fixture also carries `LOGNORMAL_PARAMETERS` and `DISCRETE_PARAMETERS` of the
same module. The kernel holds its own copy of that classification, because it
produces the parameters and a browser host has no python to ask, and the test
`the_classification_is_pythons` compares the two name by name inside
`cargo test`, so the copy cannot drift without a failure and continuous
integration needs no `pkpdutils` to notice.

The values are drawn from a fixed seed and are inputs, not a stream contract:
regenerating with another numpy writes other inputs, which is why the whole
fixture, inputs and answers together, is committed and compared as one file.

Run from the repository root against the checkout the kernel mirrors:

    uv run --with-editable path/to/pkpdutils python scripts/reduce_oracle.py
    uv run --with-editable path/to/pkpdutils python scripts/reduce_oracle.py --check
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pkpdutils
from pkpdutils.nca.options import BootstrapSpread
from pkpdutils.nca.uncertainty import (
    DISCRETE_PARAMETERS,
    LOGNORMAL_PARAMETERS,
    reduce_replicates,
)

FIXTURE = (
    Path(__file__).resolve().parent.parent / "crates/core/tests/data/reduce_oracle.json"
)

#: rows and replicates of every case, small enough to read in the fixture
N_ROWS, N_BOOT = 4, 25


def encode(values: np.ndarray) -> list[float | str]:
    """One array as the wire writes a number which may not be finite."""
    out: list[float | str] = []
    for x in np.asarray(values, dtype=np.float64).ravel(order="C"):
        value = float(x)
        if math.isfinite(value):
            out.append(value)
        elif math.isnan(value):
            out.append("NaN")
        else:
            out.append("inf" if value > 0 else "-inf")
    return out


def replicate_inputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The replicates and the estimates every case is reduced from.

    Five variables, one per path through the guards: a log-normal parameter, a
    plain continuous one, a discrete one, a name which is itself a derived
    variable and a name with no estimate. The log-normal rows carry the cases of
    the logarithms: holes, a single finite replicate, zeros and negatives.
    """
    rng = np.random.default_rng(20261118)
    auc = 20.0 * np.exp(0.25 * rng.standard_normal((N_ROWS, N_BOOT)))
    auc[0, [3, 7, 19]] = np.nan  # holes: the reductions see 22 of 25
    auc[1, 1:] = np.nan  # one finite replicate, so no spread and no interval
    auc[2, [0, 1]] = [0.0, -2.5]  # a log-normal parameter is not positive here
    fraction = 0.08 + 0.01 * rng.standard_normal((N_ROWS, N_BOOT))
    fraction[3, :] = np.nan  # no finite replicate at all
    replicates = {
        "auc_inf_obs": auc,
        "auc_extrap_fraction": fraction,
        "tmax": rng.uniform(0.5, 2.0, (N_ROWS, N_BOOT)),
        "auc_last_se": rng.uniform(0.5, 2.0, (N_ROWS, N_BOOT)),
        "ghost": rng.uniform(0.5, 2.0, (N_ROWS, N_BOOT)),
    }
    point = {
        "auc_inf_obs": np.array([20.1, 19.4, 21.0, np.nan]),
        "auc_extrap_fraction": np.array([0.081, 0.077, 0.090, 0.085]),
        "tmax": np.array([1.0, 1.0, 1.5, 1.0]),
        "auc_last_se": np.array([0.5, 0.6, 0.7, 0.8]),
    }
    return replicates, point


def case(
    name: str,
    spread_kind: BootstrapSpread,
    n_subjects: list[float] | None,
    usable_rows: list[bool],
    ci_level: float,
) -> dict[str, object]:
    """One case: its inputs and the answer `reduce_replicates` gives for them."""
    replicates, point = replicate_inputs()
    answer = reduce_replicates(
        replicates,
        point,
        spread_kind=spread_kind,
        n_subjects=None
        if n_subjects is None
        else np.array(n_subjects, dtype=np.float64),
        ci_level=ci_level,
        usable_rows=np.array(usable_rows, dtype=bool),
    )
    return {
        "name": name,
        "n_rows": N_ROWS,
        "n_boot": N_BOOT,
        "spread_kind": spread_kind.value,
        "ci_level": ci_level,
        "n_subjects": None if n_subjects is None else encode(np.array(n_subjects)),
        "usable_rows": usable_rows,
        "replicates": {key: encode(value) for key, value in replicates.items()},
        "point": {key: encode(value) for key, value in point.items()},
        "expected": {key: encode(value) for key, value in sorted(answer.items())},
    }


def provenance() -> dict[str, str]:
    """Which `pkpdutils` answered, and from which tree."""
    source = Path(pkpdutils.__file__).resolve()
    commit = "unknown"
    try:
        tree = subprocess.run(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(source.parent), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = tree.stdout.strip() + (" (dirty)" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "pkpdutils": pkpdutils.__version__,
        "commit": commit,
        "numpy": np.__version__,
        "function": "pkpdutils.nca.uncertainty.reduce_replicates",
    }


def fixture() -> dict[str, object]:
    """The whole fixture: the classification, then one entry per case."""
    return {
        "provenance": provenance(),
        "lognormal": sorted(LOGNORMAL_PARAMETERS),
        "discrete": sorted(DISCRETE_PARAMETERS),
        "cases": [
            case(
                "se_with_n",
                BootstrapSpread.SE,
                [12.0, 8.0, 5.0, 10.0],
                [True] * 4,
                0.90,
            ),
            case(
                "sd_with_n",
                BootstrapSpread.SD,
                [12.0, 8.0, 5.0, 10.0],
                [True] * 4,
                0.90,
            ),
            case("se_without_n", BootstrapSpread.SE, None, [True] * 4, 0.90),
            case("sd_without_n", BootstrapSpread.SD, None, [True] * 4, 0.95),
            # the third row carries no uncertainty however good its replicates are
            case(
                "row_not_usable",
                BootstrapSpread.SE,
                [12.0, 8.0, 5.0, 10.0],
                [True, True, False, True],
                0.90,
            ),
            # a subject count of its own per row, one of them missing
            case(
                "n_partly_missing",
                BootstrapSpread.SE,
                [12.0, float("nan"), 5.0, 10.0],
                [True] * 4,
                0.90,
            ),
        ],
    }


def main() -> int:
    """Write the fixture, or compare it with `--check`."""
    text = json.dumps(fixture(), indent=2) + "\n"
    if "--check" in sys.argv:
        if not FIXTURE.exists():
            print(f"{FIXTURE} is missing, run python scripts/reduce_oracle.py")
            return 1
        if FIXTURE.read_text() != text:
            print(f"{FIXTURE} is stale, run python scripts/reduce_oracle.py")
            return 1
        print(f"{FIXTURE} is what pkpdutils {pkpdutils.__version__} produces")
        return 0
    FIXTURE.write_text(text)
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
