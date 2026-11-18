"""Generate the pint gate fixture of `pkpdutils-units`.

The input spellings are not guessed. They are extracted from the `pkpdutils`
checkout the fixture is generated against: every string literal that `src/`,
`tests/`, `docs/` and `examples/` hand to a unit keyword (`unit=`, `time_unit=`,
`dose_unit=`, `amount_unit=`, `volume_unit=`, a `units=` mapping) or to one of
the unit functions of `pkpdutils.units`, read with `ast` from the python files
and from the `python` fences of the documentation. A hand-written list of
spellings is what let the gate grow to thousands of records while pinning one
axis only, so there is none here.

Three sections come out of that, each compared against pint through
`pkpdutils`:

* `spellings`: every extracted literal pint accepts, with pint's unit and short
  form. This is the gate on `parse_unit` and `short_unit` themselves.
* `rejected`: the few extracted literals pint accepts and the curated registry
  deliberately does not (see `REJECTED`), so the gap is a recorded and tested
  fact instead of a silent filter.
* `records`: for every distinct expression of `PARAMETER_UNITS`, every
  combination of the axes that expression names, with pint's unit, factor and
  short form. An expression is keyed once, not once per parameter: the unit
  derivation is a pure function of the expression and the input units, so the
  135 parameters sharing 16 expressions would be the same comparison repeated.
  `parameters` records the mapping, so the Rust side still proves that every
  parameter's expression is gated.

The axis of a spelling follows its dimension, which is what the expressions
branch on: a time for `{time}`, a volume for `{volume}`, a mass, an amount of
substance or an activity for `{amount}`, one of those or something which becomes
one per body weight for `{dose}` (pint reduces `[mass] / [mass]`, so the
dimensionless units `pkpdutils.check_dose_unit` accepts land here too), and
every spelling for `{unit}`.

It exits non-zero if any combination raises or yields a non-finite factor, if an
axis an expression names is empty, or if a `REJECTED` entry is no longer in the
source tree.

With `--check` it writes nothing: it recomputes everything from the importable
`pkpdutils` and the source tree, compares it with the committed fixture and
exits non-zero when the fixture is stale.
"""

import argparse
import ast
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pkpdutils
from pkpdutils.nca.nca import PARAMETER_UNITS
from pkpdutils.nca.result import parameter_unit
from pkpdutils.units import short_unit, ureg

REPO_DIR = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_DIR / "crates" / "units" / "tests" / "data" / "pint_gate.json"

#: directories of the source tree the spellings are extracted from
SOURCE_DIRS = ("src", "tests", "docs", "examples")

#: keyword arguments and mapping keys whose string value is a unit
UNIT_KEYWORDS = frozenset(
    {
        "unit",
        "value_unit",
        "concentration_unit",
        "time_unit",
        "dose_unit",
        "amount_unit",
        "volume_unit",
    }
)

#: keyword arguments whose value is a mapping of names to units
UNIT_MAPPINGS = frozenset({"units", "to_units"})

#: functions whose string arguments are units
UNIT_FUNCTIONS = frozenset(
    {
        "Q_",
        "Unit",
        "check_dose_unit",
        "is_per_bodyweight",
        "normalize_clearance",
        "normalize_volume",
        "parse_unit",
        "pkunit",
        "short_unit",
        "to",
        "to_units",
        "unit_str",
        "ureg",
    }
)

#: spellings pint accepts which the curated registry deliberately does not, with
#: the reason. `crates/units/tests/pint_gate.rs` asserts that `parse_unit`
#: rejects every one of them, so widening the registry fails the gate until the
#: entry is removed here.
REJECTED = {
    "degC": "the registry has no temperature, `tests/stats/` keeps pint",
    "grain": (
        "the registry has no apothecary mass, and `tests/nca/test_sparse.py` and "
        "`tests/nca/test_urine.py` of pkpdutils hand it to the kernel on purpose, as the "
        "unit whose refusal takes the python path"
    ),
    "kelvin": "the registry has no temperature, `tests/stats/` keeps pint",
}

FENCE = re.compile(r"^```python\b(.*?)^```", re.MULTILINE | re.DOTALL)

TIME = ureg.hour.dimensionality
VOLUME = ureg.liter.dimensionality
AMOUNTS = (ureg.gram.dimensionality, ureg.mole.dimensionality, ureg.IU.dimensionality)
BODY_WEIGHT = ureg.kilogram.dimensionality


def python_snippets(path: Path):
    """The python of a file: itself, or every `python` fence of a markdown page."""
    text = path.read_text()
    if path.suffix == ".py":
        yield text
    elif path.suffix == ".md":
        for fence in FENCE.finditer(text):
            yield fence.group(1)


def string_of(node: ast.AST) -> str | None:
    """The value of a non-empty string constant, `None` for anything else.

    The empty string is no spelling: `pkpdutils.units.parse_unit("")` raises and
    so does the Rust `parse_unit`, while `ureg.parse_units("")` is dimensionless,
    so gating it would compare the two helpers against pint instead of against
    each other.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value:
        return node.value
    return None


def called_name(node: ast.Call) -> str:
    """The name a call names, without its module."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def literals_of(tree: ast.AST) -> set[str]:
    """Every string literal of one snippet which is used as a unit."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword):
            if node.arg in UNIT_KEYWORDS and (value := string_of(node.value)):
                found.add(value)
            if node.arg in UNIT_MAPPINGS and isinstance(node.value, ast.Dict):
                found.update(v for item in node.value.values if (v := string_of(item)))
        if isinstance(node, ast.Dict):
            # a parsed dict has one key per value, `None` for a `**` entry, so
            # `strict` can only fire on a tree which `ast` did not build
            for key, value in zip(node.keys, node.values, strict=True):
                if string_of(key) in UNIT_KEYWORDS and (v := string_of(value)):
                    found.add(v)
        if isinstance(node, ast.Call) and called_name(node) in UNIT_FUNCTIONS:
            for argument in node.args:
                if value := string_of(argument):
                    found.add(value)
                if isinstance(argument, ast.Dict):
                    found.update(
                        v for item in argument.values if (v := string_of(item))
                    )
    return found


def source_literals(source: Path) -> tuple[set[str], int]:
    """Every unit literal of the source tree and the number of files read."""
    found: set[str] = set()
    files = 0
    for directory in SOURCE_DIRS:
        root = source / directory
        if not root.is_dir():
            sys.exit(f"refusing: {root} is not a directory of a pkpdutils checkout")
        for path in sorted(root.rglob("*")):
            if path.suffix not in (".py", ".md") or not path.is_file():
                continue
            files += 1
            for snippet in python_snippets(path):
                try:
                    found |= literals_of(ast.parse(snippet))
                except SyntaxError:
                    # a fragment fence of the documentation, `# not executed`
                    continue
    return found, files


def axes_of(literal: str) -> set[str]:
    """The placeholders a spelling can stand in for, by its dimension."""
    dimensionality = ureg.parse_units(literal).dimensionality
    axes = {"unit"}
    if dimensionality == TIME:
        axes.add("time")
    if dimensionality == VOLUME:
        axes.add("volume")
    if dimensionality in AMOUNTS:
        axes.add("amount")
        axes.add("dose")
    elif dimensionality * BODY_WEIGHT in AMOUNTS:
        # an amount per body weight, `mg/kg`; pint reduces `[mass] / [mass]`, so
        # the dimensionless units `check_dose_unit` accepts arrive here as well
        axes.add("dose")
    return axes


def collect(source: Path) -> tuple[dict[str, list[str]], list[str], int, int]:
    """The axes, the accepted spellings, and the file and literal counts."""
    literals, files = source_literals(source)
    unknown = sorted(set(REJECTED) - literals)
    if unknown:
        sys.exit(
            "refusing: REJECTED names spellings which are no longer in the "
            f"source tree, drop them: {', '.join(unknown)}"
        )
    names = ("unit", "time", "dose", "amount", "volume")
    axes: dict[str, set[str]] = {a: set() for a in names}
    spellings = []
    for literal in sorted(literals):
        try:
            ureg.parse_units(literal)
        except Exception:  # noqa: BLE001 - every way pint can refuse a string
            continue
        if literal in REJECTED:
            continue
        spellings.append(literal)
        for axis in axes_of(literal):
            axes[axis].add(literal)
    return {a: sorted(v) for a, v in axes.items()}, spellings, files, len(literals)


def records_of(axes: dict[str, list[str]]) -> tuple[list[dict], list[str]]:
    """Pint's answer for every expression and every combination of its axes."""
    records: list[dict] = []
    failures: list[str] = []
    for expression in sorted(set(PARAMETER_UNITS.values())):
        named = [a for a in axes if "{" + a + "}" in expression]
        empty = [a for a in named if not axes[a]]
        if empty:
            failures.append(
                f"{expression}: no spelling of the source tree fits the "
                f"{', '.join(empty)} axis, the gate would skip the expression"
            )
            continue
        for combination in itertools.product(*(axes[a] for a in named)):
            # the product runs over `named`, so a combination has one value per
            # axis and `strict` holds by construction
            given = dict(zip(named, combination, strict=True))
            unit = given.get("unit", "mg/l")
            time = given.get("time", "hr")
            try:
                derived, factor = parameter_unit(
                    expression,
                    unit=unit,
                    time_unit=time,
                    dose_unit=given.get("dose"),
                    amount_unit=given.get("amount"),
                    volume_unit=given.get("volume"),
                )
                short = short_unit(derived)
            except Exception as error:  # noqa: BLE001
                failures.append(f"{expression} {given}: {error}")
                continue
            if not math.isfinite(factor):
                failures.append(f"{expression} {given}: non-finite factor {factor!r}")
                continue
            records.append(
                {
                    "expression": expression,
                    "unit": unit,
                    "time": time,
                    "dose": given.get("dose"),
                    "amount": given.get("amount"),
                    "volume": given.get("volume"),
                    "expected_unit": derived,
                    "expected_factor": factor,
                    "expected_short": short,
                }
            )
    return records, failures


def spelling_records(spellings: list[str]) -> tuple[list[dict], list[str]]:
    """Pint's unit and short form for every accepted spelling."""
    records, failures = [], []
    for literal in spellings:
        unit = str(ureg.parse_units(literal))
        unit = unit or "dimensionless"
        try:
            short = short_unit(unit)
        except Exception as error:  # noqa: BLE001
            failures.append(f"short_unit({unit!r}): {error}")
            continue
        records.append({"literal": literal, "unit": unit, "short": short})
    return records, failures


def rejected_records() -> list[dict]:
    """The deliberate exclusions with pint's answer and the reason."""
    return [
        {
            "literal": literal,
            "unit": str(ureg.parse_units(literal)) or "dimensionless",
            "reason": reason,
        }
        for literal, reason in sorted(REJECTED.items())
    ]


def clean_commit(source: Path) -> str | None:
    """The commit of the source tree, `None` unless it is a clean checkout."""
    try:
        status = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        )
        if status.stdout.strip():
            return None
        head = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return head.stdout.strip()


def default_source() -> Path | None:
    """The checkout `pkpdutils` was imported from, `None` for a release."""
    module_dir = Path(pkpdutils.__file__).resolve().parent
    if "site-packages" in module_dir.parts:
        return None
    return module_dir.parent.parent


#: the keys of the provenance `--check` compares; the source and the commit are
#: recorded and not compared, a fixture does not go stale by moving repository
COMPARED = (
    "pkpdutils_version",
    "parameter_units_digest",
    "n_parameters",
    "n_expressions",
    "n_records",
    "n_spellings",
    "n_rejected",
    "axes",
)


def check(path: Path, fresh: dict) -> int:
    """Compare a fresh run with the committed fixture and return the exit code."""
    try:
        committed = json.loads(path.read_text())
        header = committed["provenance"]
        if not isinstance(header, dict):
            raise TypeError("provenance is not an object")
        for section in ("records", "spellings", "rejected"):
            if not isinstance(committed[section], list):
                raise TypeError(f"{section} is not a list")
        if not isinstance(committed["parameters"], dict):
            raise TypeError("parameters is not an object")
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
        f"{key}: fixture {header.get(key)!r}, fresh {fresh['provenance'][key]!r}"
        for key in COMPARED
        if header.get(key) != fresh["provenance"][key]
    ]
    differs += [
        f"{section}: the fixture differs from a fresh run"
        for section in ("parameters", "spellings", "rejected", "records")
        if committed[section] != fresh[section]
    ]
    if differs:
        print(f"{path} is stale, regenerate it:", file=sys.stderr)
        print("\n".join(differs), file=sys.stderr)
        return 1
    header = fresh["provenance"]
    print(
        f"{path} is up to date ({header['n_records']} records, "
        f"{header['n_spellings']} spellings, "
        f"digest {header['parameter_units_digest']})"
    )
    return 0


def main() -> int:
    """Write the fixture and return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=FIXTURE_PATH,
        help="fixture to write, or to compare with --check (default: the committed one)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="checkout of pkpdutils the spellings are extracted from "
        "(default: the checkout pkpdutils was imported from)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the fixture with a fresh run instead of writing it",
    )
    args = parser.parse_args()
    source = args.source or default_source()
    if source is None:
        parser.error(
            "pkpdutils was imported from a release, which carries no tests, "
            "documentation or examples: pass --source with a checkout"
        )
    source = source.resolve()

    axes, spellings, n_files, n_literals = collect(source)
    records, failures = records_of(axes)
    spellings_section, spelling_failures = spelling_records(spellings)
    failures += spelling_failures
    if failures:
        print(f"{len(failures)} combinations failed:", file=sys.stderr)
        print("\n".join(failures[:20]), file=sys.stderr)
        return 1

    module_dir = Path(pkpdutils.__file__).resolve().parent
    rejected = rejected_records()
    fixture = {
        "provenance": {
            "pkpdutils_version": pkpdutils.__version__,
            "pkpdutils_source": (
                "release" if "site-packages" in module_dir.parts else "checkout"
            ),
            "source_commit": clean_commit(source),
            "parameter_units_digest": hashlib.sha256(
                json.dumps(sorted(PARAMETER_UNITS.items())).encode()
            ).hexdigest()[:16],
            "n_parameters": len(PARAMETER_UNITS),
            "n_expressions": len(set(PARAMETER_UNITS.values())),
            "n_records": len(records),
            "n_spellings": len(spellings_section),
            "n_rejected": len(rejected),
            "axes": axes,
        },
        "parameters": dict(sorted(PARAMETER_UNITS.items())),
        "spellings": spellings_section,
        "rejected": rejected,
        "records": records,
    }
    if args.check:
        return check(args.output, fixture)
    with args.output.open("w") as f:
        json.dump(fixture, f, indent=0)
        f.write("\n")
    header = fixture["provenance"]
    print(
        f"{header['n_records']} records from {header['n_expressions']} expressions "
        f"of {header['n_parameters']} parameters, {header['n_spellings']} spellings "
        f"and {header['n_rejected']} deliberate exclusions, from {n_literals} unit "
        f"literals of {n_files} files of {source}"
    )
    print("axes: " + ", ".join(f"{a} {len(v)}" for a, v in axes.items()))
    print(
        f"generated with pkpdutils {header['pkpdutils_version']} "
        f"({header['pkpdutils_source']}, imported from {module_dir}, commit "
        f"{header['source_commit']}), PARAMETER_UNITS digest "
        f"{header['parameter_units_digest']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
