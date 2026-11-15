"""Generate the pint gate fixture of `pkpdutils-units`.

For every expression of `PARAMETER_UNITS` the script varies only the axes the
expression names (`{unit}`, `{time}`, `{dose}`, `{amount}`, `{volume}`), asks
pint through `pkpdutils` for the unit, the factor and the short form, and
writes the records as JSON. It exits non-zero if any combination raises.
"""

import argparse
import itertools
import json
import sys

from pkpdutils.nca.nca import PARAMETER_UNITS
from pkpdutils.nca.result import parameter_unit
from pkpdutils.units import short_unit

AXES = {
    "unit": ["mg/l", "ng/ml", "ug/l", "mmol/l", "mg/L", "mmHg", "dimensionless"],
    "time": ["hr", "h", "hour", "min"],
    "dose": ["mg", "mmol", "IU", "mg/kg", "IU/kg"],
    "amount": ["mg", "mmol"],
    "volume": ["ml", "l"],
}


def main() -> int:
    """Write the fixture and return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True)
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
    if failures:
        print(f"{len(failures)} combinations raised:", file=sys.stderr)
        print("\n".join(failures[:20]), file=sys.stderr)
        return 1
    with open(args.output, "w") as f:
        json.dump(records, f, indent=0)
        f.write("\n")
    print(f"{len(records)} records from {len(PARAMETER_UNITS)} parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
