"""Generate the pint gate fixture of `pkpdutils-units`.

For every expression of `PARAMETER_UNITS` the script varies only the axes the
expression names (`{unit}`, `{time}`, `{dose}`, `{amount}`, `{volume}`), asks
pint through `pkpdutils` for the unit, the factor and the short form, and
writes the records as JSON next to a provenance header (the version and the
source, `checkout` or `release`, of the `pkpdutils` it imported). It exits
non-zero if any combination raises or yields a non-finite factor.

With `--check` it writes nothing: it recomputes the digest of `PARAMETER_UNITS`,
the parameter count, the record count and the records from the importable
`pkpdutils`, compares them with the committed fixture and exits non-zero when
the fixture is stale.
"""

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import pkpdutils
from pkpdutils.nca.nca import PARAMETER_UNITS
from pkpdutils.nca.result import parameter_unit
from pkpdutils.units import short_unit

REPO_DIR = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_DIR / "crates" / "units" / "tests" / "data" / "pint_gate.json"

AXES = {
    "unit": ["mg/l", "ng/ml", "ug/l", "mmol/l", "mg/L", "mmHg", "dimensionless"],
    "time": ["hr", "h", "hour", "min"],
    "dose": ["mg", "mmol", "IU", "mg/kg", "IU/kg"],
    "amount": ["mg", "mmol"],
    "volume": ["ml", "l"],
}


def check(path: Path, provenance: dict, records: list[dict]) -> int:
    """Compare a fresh run with the committed fixture and return the exit code."""
    try:
        committed = json.loads(path.read_text())
        header = committed["provenance"]
        committed_records = committed["records"]
        if not isinstance(header, dict):
            raise TypeError("provenance is not an object")
        if not isinstance(committed_records, list):
            raise TypeError("records is not a list")
    except FileNotFoundError:
        print(
            f"{path} does not exist, generate it with the same command without --check",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(
            f"{path} is not a pint gate fixture ({error!r}), regenerate it",
            file=sys.stderr,
        )
        return 1
    differs = [
        f"{key}: fixture {header.get(key)!r}, fresh {provenance[key]!r}"
        for key in (
            "pkpdutils_version",
            "parameter_units_digest",
            "n_parameters",
            "n_records",
        )
        if header.get(key) != provenance[key]
    ]
    if committed_records != records:
        differs.append("records: the fixture differs from a fresh run")
    if differs:
        print(f"{path} is stale, regenerate it:", file=sys.stderr)
        print("\n".join(differs), file=sys.stderr)
        return 1
    print(
        f"{path} is up to date ({len(records)} records, "
        f"digest {provenance['parameter_units_digest']})"
    )
    return 0


def main() -> int:
    """Write the fixture and return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=FIXTURE_PATH,
        help="fixture to write, or to compare with --check (default: the committed one)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the fixture with a fresh run instead of writing it",
    )
    args = parser.parse_args()

    records = []
    failures = []
    for parameter, expression in PARAMETER_UNITS.items():
        axes = [a for a in AXES if "{" + a + "}" in expression]
        for combo in itertools.product(*(AXES[a] for a in axes)):
            given = dict(zip(axes, combo))
            try:
                unit, factor = parameter_unit(
                    expression,
                    unit=given.get("unit", "mg/l"),
                    time_unit=given.get("time", "hr"),
                    dose_unit=given.get("dose"),
                    amount_unit=given.get("amount"),
                    volume_unit=given.get("volume"),
                )
                short = short_unit(unit)
            except Exception as error:  # noqa: BLE001
                failures.append(f"{parameter} {expression} {given}: {error}")
                continue
            if not math.isfinite(factor):
                failures.append(
                    f"{parameter} {expression} {given}: non-finite factor {factor!r}"
                )
                continue
            records.append(
                {
                    "parameter": parameter,
                    "expression": expression,
                    "unit": given.get("unit", "mg/l"),
                    "time": given.get("time", "hr"),
                    "dose": given.get("dose"),
                    "amount": given.get("amount"),
                    "volume": given.get("volume"),
                    "expected_unit": unit,
                    "expected_factor": factor,
                    "expected_short": short,
                }
            )
    module_dir = Path(pkpdutils.__file__).resolve().parent
    source = "release" if "site-packages" in module_dir.parts else "checkout"
    digest = hashlib.sha256(
        json.dumps(sorted(PARAMETER_UNITS.items())).encode()
    ).hexdigest()[:16]
    provenance = {
        "pkpdutils_version": pkpdutils.__version__,
        "pkpdutils_source": source,
        "parameter_units_digest": digest,
        "n_parameters": len(PARAMETER_UNITS),
        "n_records": len(records),
    }
    if failures:
        print(f"{len(failures)} combinations failed:", file=sys.stderr)
        print("\n".join(failures[:20]), file=sys.stderr)
        return 1
    if args.check:
        return check(args.output, provenance, records)
    with args.output.open("w") as f:
        json.dump({"provenance": provenance, "records": records}, f, indent=0)
        f.write("\n")
    print(f"{len(records)} records from {len(PARAMETER_UNITS)} parameters")
    print(
        f"generated with pkpdutils {provenance['pkpdutils_version']} "
        f"({provenance['pkpdutils_source']}, imported from {module_dir}), "
        f"PARAMETER_UNITS digest {digest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
