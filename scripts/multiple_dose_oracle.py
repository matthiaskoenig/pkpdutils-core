"""Generate the multiple dose oracle of `crates/oracles/tests/multiple_dose.rs`.

Run from a `pkpdutils` checkout, so that the python core is the reference:

    uv run python /path/to/pkpdutils-core/scripts/multiple_dose_oracle.py
    uv run python /path/to/pkpdutils-core/scripts/multiple_dose_oracle.py --check

With `--check` nothing is written: the cases are recomputed, compared with the committed
file and the exit code is non-zero when it is stale, as `crosswalk_gate.py --check` does for
its fixture. The commit of the checkout is recorded and not compared, a file does not go
stale by moving repository; the version of `pkpdutils` and every case are compared.

Every case is built through `Timecourses.from_arrays`, the front end of the package, so a
protocol the front end rejects cannot be in the oracle. A batch of a multiple dose row is not
carried by the rust engine of `pkpdutils`, so `nca` runs the python core and the expected
numbers are the python kernel's; the script asserts that. The curves are deterministic
superpositions of one compartment models, rounded to four decimals and written as the inputs.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

import pkpdutils
from pkpdutils import NCAOptions, Route, Timecourses, nca
from pkpdutils.nca import engine


def oral(t, doses, ka=1.2, ke=0.17, v=1.0, f=0.9):
    """Concentration of an oral one compartment model after a list of `(time, amount)`."""
    out = np.zeros_like(t)
    for td, amount in doses:
        s = np.clip(t - td, 0.0, None)
        part = f * amount / v * ka / (ka - ke) * (np.exp(-ke * s) - np.exp(-ka * s))
        out += np.where(t >= td, part, 0.0)
    return np.round(out, 4)


def bolus(t, doses, ke=0.2, v=5.0):
    """Concentration of an intravenous bolus, a dose counts from its own time on."""
    out = np.zeros_like(t)
    for td, amount in doses:
        out += np.where(t >= td, amount / v * np.exp(-ke * (t - td)), 0.0)
    return np.round(out, 4)


def cell(x):
    """A number of the file: `null` is NaN, an infinity is the text `inf` or `-inf`."""
    if math.isnan(x):
        return None
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return float(x)


def pad(rows, n, fill=math.nan):
    return [list(r) + [fill] * (n - len(r)) for r in rows]


def case(name, why, times, values, doses, route, options):
    """One case: the arrays, the options and the python result of them."""
    n_dose = max(len(d) for d in doses)
    amount = np.array(pad([[a for _, a in d] for d in doses], n_dose))
    dtime = np.array(pad([[t for t, _ in d] for d in doses], n_dose))
    tc = Timecourses.from_arrays(
        np.array(times),
        np.array(values),
        time_unit="hour",
        unit="mg/l",
        dose={"amount": amount, "time": dtime, "unit": "mg"},
        route=route,
        substance="drug",
    )
    opts = NCAOptions(**options)
    assert engine.unsupported_reason(tc, opts) is not None, name
    res = nca(tc, options=opts).ds
    parameters, points = {}, {}
    for var in res.data_vars:
        v = res[var]
        unit = v.attrs["units"]
        flat = [cell(x) for x in np.ravel(v.values)]
        if v.dims == ("individual",):
            parameters[var] = {"unit": unit, "values": flat}
        elif v.dims[0] == "individual" and len(v.dims) == 2:
            points[var] = {"unit": unit, "dim": v.dims[1], "size": v.shape[1], "values": flat}
        else:
            raise SystemExit(f"{name}: unexpected dims {var} {v.dims}")
    return {
        "name": name,
        "why": why,
        "route": str(route),
        "times": [list(map(float, r)) for r in np.broadcast_to(times, np.shape(values))],
        "values": [list(map(float, r)) for r in values],
        "dose_amount": [[None if math.isnan(x) else float(x) for x in r] for r in amount],
        "dose_time": [[None if math.isnan(x) else float(x) for x in r] for r in dtime],
        "options": options,
        "parameters": parameters,
        "point_variables": points,
    }


#: the committed oracle, next to the gate which reads it
FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "crates/oracles/tests/data/multiple_dose.json"
)


def build_cases():
    """Every case of the oracle, computed by the python core."""
    cases = []
    # regular oral, four doses every 12 h, sampled over the last interval and the earlier ones
    t = np.array([0, 1, 2, 4, 8, 12, 13, 14, 16, 20, 24, 25, 26, 28, 32, 36, 37, 38, 40, 44, 48.0])
    d = [(0.0, 10.0), (12.0, 10.0), (24.0, 10.0), (36.0, 10.0)]
    cases.append(case("oral_regular", "regular oral protocol, last interval complete", t,
                      [oral(t, d)], [d], Route.ORAL, {}))
    cases.append(case("oral_regular_no_intervals", "same, without the interval variables", t,
                      [oral(t, d)], [d], Route.ORAL, {"intervals": False}))
    # regular bolus whose sample at each dose time carries the post-dose value
    tb = np.array([0, 0.5, 1, 2, 4, 8, 12, 12.5, 13, 14, 16, 20, 24, 24.5, 25, 26, 28, 32, 36.0])
    db = [(0.0, 100.0), (12.0, 100.0), (24.0, 100.0)]
    cases.append(case("bolus_regular_trough_regression",
                      "bolus, the sample at each dose time is the post-dose value", tb,
                      [bolus(tb, db)], [db], Route.IV_BOLUS, {}))
    # irregular protocol: the gaps are 8 and 16, the last differs from the first
    ti = np.array([0, 1, 2, 4, 8, 9, 10, 12, 16, 24, 25, 26, 28, 32, 40.0])
    di = [(0.0, 10.0), (8.0, 10.0), (24.0, 10.0)]
    cases.append(case("oral_irregular", "last gap 16 differs from the first gap 8", ti,
                      [oral(ti, di)], [di], Route.ORAL, {}))
    # last interval 0.8 h short of its end (0.067 of tau, within the 0.1 tolerance)
    tc_ = np.array([0, 1, 2, 4, 8, 12, 13, 14, 16, 20, 24, 25, 26, 28, 32, 35.2])
    dc = [(0.0, 10.0), (12.0, 10.0), (24.0, 10.0)]
    cases.append(case("oral_completed_last_interval",
                      "last sample 0.8 h before the end of the interval", tc_,
                      [oral(tc_, dc)], [dc], Route.ORAL, {}))
    # 3 h short, 0.25 of tau, beyond the tolerance, so the interval is flagged
    ts = np.array([0, 1, 2, 4, 8, 12, 13, 14, 16, 20, 24, 25, 26, 28, 32, 33.0])
    cases.append(case("oral_incomplete_last_interval",
                      "last sample 3 h before the end, beyond the tolerance", ts,
                      [oral(ts, dc)], [dc], Route.ORAL, {}))
    # mixed: single, multiple, single, multiple in one call
    tm = np.array([0, 1, 2, 4, 8, 12, 13, 14, 16, 20, 24, 25, 26, 28, 32, 36.0])
    s1, s2 = [(0.0, 10.0)], [(0.0, 20.0)]
    m1 = [(0.0, 10.0), (12.0, 20.0), (24.0, 30.0)]
    m2 = [(0.0, 5.0), (12.0, 5.0), (24.0, 5.0)]
    rows = [oral(tm, s1), oral(tm, m1), oral(tm, s2, ke=0.15), oral(tm, m2, ka=0.9)]
    cases.append(case("mixed_single_and_multiple", "single, multiple, single, multiple rows",
                      tm, rows, [s1, m1, s2, m2], Route.ORAL, {}))
    # a single dose row analysed at a steady state, with the dosing interval given
    tt = np.array([0, 1, 2, 4, 8, 12.0])
    st = [(0.0, 10.0)]
    cases.append(case("single_dose_with_tau", "one dose and an explicit tau", tt,
                      [oral(tt, st, ke=0.2)], [st], Route.ORAL, {"tau": 12.0}))
    return cases


def checkout_commit():
    """The commit of the checkout `pkpdutils` was imported from."""
    root = Path(pkpdutils.__file__).resolve().parents[2]
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def check(path, fresh):
    """Compare a fresh run with the committed file and return the exit code."""
    try:
        committed = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        print(f"{path} cannot be read ({error!r}), generate it without --check",
              file=sys.stderr)
        return 1
    differs = []
    if committed.get("pkpdutils_version") != fresh["pkpdutils_version"]:
        differs.append(f"pkpdutils_version: file {committed.get('pkpdutils_version')!r}, "
                       f"fresh {fresh['pkpdutils_version']!r}")
    old = {c["name"]: c for c in committed.get("cases", [])}
    new = {c["name"]: c for c in fresh["cases"]}
    differs += [f"{n}: not in the file" for n in new if n not in old]
    differs += [f"{n}: not computed any more" for n in old if n not in new]
    differs += [f"{n}: differs from a fresh run" for n in new if n in old and old[n] != new[n]]
    if differs:
        print(f"{path} is stale, regenerate it:", file=sys.stderr)
        print("\n".join(differs), file=sys.stderr)
        return 1
    print(f"{path} is up to date ({len(new)} cases, pkpdutils {fresh['pkpdutils_version']})")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", type=Path, default=FIXTURE_PATH,
                        help="file to write, or to compare with --check (default: the committed one)")
    parser.add_argument("--check", action="store_true",
                        help="compare the file with a fresh run instead of writing it")
    args = parser.parse_args()
    fresh = {
        "pkpdutils_version": pkpdutils.__version__,
        "pkpdutils_commit": checkout_commit(),
        "cases": json.loads(json.dumps(build_cases(), allow_nan=False)),
    }
    if args.check:
        return check(args.output, fresh)
    with args.output.open("w") as f:
        json.dump(fresh, f, indent=1, allow_nan=False)
        f.write("\n")
    print(f"wrote {len(fresh['cases'])} cases to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
