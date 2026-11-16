"""Generate the crosswalk fixture of the three-tool benchmark gate.

`crates/oracles/tests/benchmarks.rs` compares the kernel against the result
tables of Phoenix WinNonlin, PKNCA and NonCompart which
`testdata/oracles/docs/data/benchmarks/` carries. Every one of those tables is
written in the column names of its own tool (`AUCINF_obs`, `aucinf.obs`,
`AUCIFO` for the one parameter `auc_inf_obs`), and the map which translates
them lives in `src/pkpdutils/crosswalk.py`. That module is source and not an
oracle, so it was never vendored: without it nothing in this repository can read
the tables at all.

This script extracts the map. Nothing here is hand written: every name, set and
rule comes out of the imported `pkpdutils.crosswalk`, so a map which changes
upstream changes the fixture and a map which is renamed away fails the
extraction instead of silently keeping an old copy.

Three sections come out of it:

* `tools`: for each of the three tools the name map of a variable of
  `pkpdutils` to the column of the tool (`WINNONLIN_NAMES`, `PKNCA_NAMES`,
  `NONCOMPART_NAMES`), the map of the variables whose column depends on the
  route (`*_NAMES_BY_ROUTE`), the columns which carry a percentage where
  `pkpdutils` reports a fraction (`*_PERCENT`) and the columns the tool reports
  which have no variable at all (`*_UNMAPPED`, the correlation of the terminal
  regression and the intercept of NonCompart).
* `pknca_rules`: what reading the long PKNCA table needs beyond the names. The
  columns of that table (`PKNCA_COLUMNS`), the two sets which decide whether
  `cl.obs` is `cl` or `cl_f` (`_INTRAVENOUS_ONLY`, `_EXTRAVASCULAR_ONLY`) and
  the rules which recognize the route from the parameters PKNCA reports
  (`_pknca_route`).
* `provenance`: the `pkpdutils` version, whether it was imported from a
  checkout or from a release, the commit of a clean checkout, a digest over the
  two sections above and their sizes.

The route rules are read with `ast` from the source of `_pknca_route` and then
verified against the function itself: the extracted rules have to answer
exactly as it does for the empty set, for every single parameter of the PKNCA
maps and of the rules, for a parameter made from each prefix, and for every
ordered pair of those. An extraction which no longer matches the function it
came from exits non-zero rather than writing a fixture.

It also exits non-zero if a tool names two variables with one column, if the
columns a route resolves to are not one per variable, if a percentage column or
an unmapped column is no column of the tool, or if a map is empty.

With `--check` it writes nothing: it recomputes everything from the importable
`pkpdutils`, compares it with the committed fixture and exits non-zero when the
fixture is stale.
"""

import argparse
import ast
import hashlib
import inspect
import itertools
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pkpdutils
from pkpdutils import crosswalk
from pkpdutils.timecourse import Route

REPO_DIR = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_DIR / "crates" / "oracles" / "tests" / "data" / "crosswalk.json"

#: the three tools, each with the names of the module attributes it declares.
#: The attribute names are spelled out so that a map which is renamed upstream
#: fails here instead of being extracted as an empty one.
TOOLS: dict[str, dict[str, str | None]] = {
    "winnonlin": {
        "reader": "wide",
        "names": "WINNONLIN_NAMES",
        "names_by_route": None,
        "percent": "WINNONLIN_PERCENT",
        "unmapped": "WINNONLIN_UNMAPPED",
    },
    "pknca": {
        "reader": "long",
        "names": "PKNCA_NAMES",
        "names_by_route": "PKNCA_NAMES_BY_ROUTE",
        "percent": "PKNCA_PERCENT",
        "unmapped": None,
    },
    "noncompart": {
        "reader": "wide",
        "names": "NONCOMPART_NAMES",
        "names_by_route": "NONCOMPART_NAMES_BY_ROUTE",
        "percent": "NONCOMPART_PERCENT",
        "unmapped": "NONCOMPART_UNMAPPED",
    },
}


def attribute(name: str | None, default: object) -> object:
    """One attribute of `pkpdutils.crosswalk`, or `default` when it names none.

    A tool which declares no map of that kind passes `None`; a name which the
    module does not carry any more is an error, since the fixture would then be
    generated without the map instead of failing.
    """
    if name is None:
        return default
    try:
        return getattr(crosswalk, name)
    except AttributeError:
        sys.exit(
            f"refusing: pkpdutils.crosswalk has no {name} any more; the "
            "extraction has to follow the module, never a copy of it"
        )


def tool_section(tool: str, attributes: dict[str, str | None]) -> dict:
    """The name maps, the percentages and the unmapped columns of one tool."""
    names: dict[str, str] = dict(attribute(attributes["names"], {}))
    by_route_raw: dict = dict(attribute(attributes["names_by_route"], {}))
    percent = sorted(attribute(attributes["percent"], frozenset()))
    unmapped = sorted(attribute(attributes["unmapped"], frozenset()))
    if not names:
        sys.exit(f"refusing: the name map of {tool} is empty")
    by_route = {
        variable: {str(Route(route)): column for route, column in sorted(columns.items())}
        for variable, columns in sorted(by_route_raw.items())
    }
    columns = set(names.values())
    for variable, routes in by_route.items():
        if variable in names:
            sys.exit(
                f"refusing: {tool} names {variable} both route independently "
                f"({names[variable]}) and by route ({sorted(routes.values())}), "
                "which the readers would resolve in the order of a dict"
            )
        columns |= set(routes.values())
    stray = sorted(set(percent) - columns)
    if stray:
        sys.exit(f"refusing: {tool} marks {stray} as a percentage of no column of its own")
    # an unmapped column is by definition one the name map does not carry
    overlap = sorted(set(unmapped) & columns)
    if overlap:
        sys.exit(f"refusing: {tool} calls {overlap} unmapped and names them as well")
    return {
        # which of the two readers of `crosswalk.py` reads this tool's table:
        # `_read_wide` for the pivoted tables of WinNonlin and NonCompart,
        # `read_pknca_results` for the long table of PKNCA
        "reader": attributes["reader"],
        "names": dict(sorted(names.items())),
        "names_by_route": by_route,
        "percent": percent,
        "unmapped": unmapped,
    }


def check_injective(tool: str, section: dict) -> None:
    """A wide table is read by the inverse of its name map, so it has to be one.

    `_read_wide` of `crosswalk.py:330-366` builds `{column: variable}` from the
    name map and from every column of the by-route map. Two variables under one
    column would make that inverse depend on the order of a dict, so the readers
    of this gate reject it and the extraction refuses to write it.
    """
    reverse: dict[str, str] = {}
    for variable, column in section["names"].items():
        if column in reverse:
            sys.exit(
                f"refusing: {tool} names both {reverse[column]} and {variable} "
                f"{column}, so its table cannot be read back"
            )
        reverse[column] = variable
    for variable, routes in section["names_by_route"].items():
        for column in routes.values():
            if reverse.setdefault(column, variable) != variable:
                sys.exit(
                    f"refusing: {tool} names both {reverse[column]} and "
                    f"{variable} {column}, so its table cannot be read back"
                )


def check_pknca_routes(
    section: dict, intravenous: list[str], extravascular: list[str]
) -> None:
    """Every route resolves the PKNCA names to one variable per parameter.

    `read_pknca_results` (`crosswalk.py:649-657`) resolves the by-route names
    for the route it recognized and drops the variables of the other route, so
    that the one parameter `cl.obs` becomes `cl` after an intravenous dose and
    `cl_f` after an extravascular one. That has to leave one variable per
    parameter for every route, or the reader would again depend on the order of
    a dict.
    """
    for route in Route:
        resolved = dict(section["names"])
        for variable, routes in section["names_by_route"].items():
            if str(route) in routes:
                resolved[variable] = routes[str(route)]
        other = extravascular if route.is_iv else intravenous
        reverse: dict[str, str] = {}
        for variable, parameter in resolved.items():
            if variable in other:
                continue
            if reverse.setdefault(parameter, variable) != variable:
                sys.exit(
                    f"refusing: after an {route} dose PKNCA's {parameter} is "
                    f"both {reverse[parameter]} and {variable}"
                )


def rule_of(node: ast.If) -> dict:
    """One branch of `_pknca_route` as the codes and prefixes it tests for."""
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
        sys.exit("refusing: a branch of _pknca_route does not return straight away")
    returned = node.body[0].value
    if not (
        isinstance(returned, ast.Attribute)
        and isinstance(returned.value, ast.Name)
        and returned.value.id == "Route"
    ):
        sys.exit("refusing: a branch of _pknca_route does not return a Route")
    if node.orelse:
        sys.exit("refusing: a branch of _pknca_route has an else, which this reads as none")
    codes: set[str] = set()
    prefixes: set[str] = set()
    for inner in ast.walk(node.test):
        if isinstance(inner, ast.Set):
            codes.update(literal_strings(inner.elts))
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
            if inner.func.attr != "startswith":
                continue
            for argument in inner.args:
                if isinstance(argument, (ast.Tuple, ast.List)):
                    prefixes.update(literal_strings(argument.elts))
                else:
                    prefixes.update(literal_strings([argument]))
    if not codes and not prefixes:
        sys.exit("refusing: a branch of _pknca_route tests for no parameter at all")
    return {
        "route": str(Route(getattr(Route, returned.attr))),
        "codes": sorted(codes),
        "prefixes": sorted(prefixes),
    }


def literal_strings(nodes: list[ast.expr]) -> list[str]:
    """The string constants of a list of expressions, refusing anything else."""
    found = []
    for node in nodes:
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            sys.exit("refusing: _pknca_route tests against something which is no string")
        found.append(node.value)
    return found


def route_rules() -> list[dict]:
    """The rules `_pknca_route` recognizes the route by, in its own order.

    The branches are read from the source of the function and then verified
    against the function itself, so an extraction which has drifted away from
    what it came from fails here.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(crosswalk._pknca_route)))
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef):
        sys.exit("refusing: _pknca_route is no plain function any more")
    rules: list[dict] = []
    for node in function.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # the docstring
        if isinstance(node, ast.If):
            rules.append(rule_of(node))
            continue
        if isinstance(node, ast.Return):
            if not (isinstance(node.value, ast.Constant) and node.value.value is None):
                sys.exit("refusing: _pknca_route does not end in `return None`")
            continue
        sys.exit(f"refusing: _pknca_route carries a {type(node).__name__} this does not read")
    if not rules:
        sys.exit("refusing: no branch of _pknca_route was read")
    return rules


def apply_rules(rules: list[dict], codes: set[str]) -> str | None:
    """The route the extracted rules give a set of parameters, in their order."""
    for rule in rules:
        if codes & set(rule["codes"]):
            return rule["route"]
        prefixes = tuple(rule["prefixes"])
        if prefixes and any(code.startswith(prefixes) for code in codes):
            return rule["route"]
    return None


def verify_rules(rules: list[dict], section: dict) -> None:
    """The extracted rules answer exactly as `_pknca_route` does.

    The universe is every parameter the PKNCA maps name, every parameter the
    rules name and one parameter made from each prefix, which is what the rules
    can tell apart at all. Every single one of them and every ordered pair is
    compared, and so is the empty set.
    """
    universe: set[str] = set(section["names"].values())
    for routes in section["names_by_route"].values():
        universe |= set(routes.values())
    for rule in rules:
        universe |= set(rule["codes"])
        universe |= {prefix + "x" for prefix in rule["prefixes"]}
    cases: list[set[str]] = [set()]
    cases += [{code} for code in sorted(universe)]
    cases += [{a, b} for a, b in itertools.permutations(sorted(universe), 2)]
    for case in cases:
        theirs = crosswalk._pknca_route(set(case))
        expected = None if theirs is None else str(theirs)
        if apply_rules(rules, case) != expected:
            sys.exit(
                f"refusing: the extracted rules answer "
                f"{apply_rules(rules, case)!r} for {sorted(case)} where "
                f"_pknca_route answers {expected!r}"
            )


def canonical(body: dict) -> str:
    """The two sections of the fixture as one sorted line per leaf.

    The digest is checked by `crates/oracles/tests/benchmarks.rs` as well, so
    it is taken over a form both sides build the same way rather than over the
    JSON text, whose spacing and escaping are a property of the writer. Every
    leaf of the two sections is a string, so a line is the path of the leaf,
    a tab and the string.
    """
    lines: list[str] = []

    def walk(path: list[str], node: object) -> None:
        if isinstance(node, dict):
            for key in sorted(node):
                walk([*path, str(key)], node[key])
        elif isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                walk([*path, str(index)], item)
        elif isinstance(node, str):
            if "\t" in node or "\n" in node:
                sys.exit(f"refusing: {'/'.join(path)} carries a tab or a newline")
            lines.append("/".join(path) + "\t" + node)
        else:
            sys.exit(f"refusing: {'/'.join(path)} is a {type(node).__name__}, no string")

    walk([], body)
    return "\n".join(sorted(lines)) + "\n"


def digest(body: dict) -> str:
    """The digest of the two sections of the fixture, its fingerprint."""
    return hashlib.sha256(canonical(body).encode()).hexdigest()[:16]


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
    "crosswalk_digest",
    "n_tools",
    "n_names",
    "n_names_by_route",
    "n_percent",
    "n_unmapped",
    "n_columns",
    "n_route_rules",
)


def check(path: Path, fresh: dict) -> int:
    """Compare a fresh run with the committed fixture and return the exit code."""
    try:
        committed = json.loads(path.read_text())
        header = committed["provenance"]
        if not isinstance(header, dict):
            raise TypeError("provenance is not an object")
        for section in ("tools", "pknca_rules"):
            if not isinstance(committed[section], dict):
                raise TypeError(f"{section} is not an object")
    except FileNotFoundError:
        print(
            f"{path} does not exist, generate it with the same command without --check",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(
            f"{path} is not a crosswalk gate fixture ({error!r}), regenerate it",
            file=sys.stderr,
        )
        return 1
    differs: list[str] = []
    # the digest of the committed body against the digest the committed header
    # carries, which a hand edit of either one alone does not keep
    own = digest({section: committed[section] for section in ("tools", "pknca_rules")})
    if own != header.get("crosswalk_digest"):
        differs.append(
            f"crosswalk_digest: the header says {header.get('crosswalk_digest')!r} "
            f"and its own sections digest to {own!r}"
        )
    differs += [
        f"{key}: fixture {header.get(key)!r}, fresh {fresh['provenance'][key]!r}"
        for key in COMPARED
        if header.get(key) != fresh["provenance"][key]
    ]
    differs += [
        f"{section}: the fixture differs from a fresh run"
        for section in ("tools", "pknca_rules")
        if committed[section] != fresh[section]
    ]
    if differs:
        print(f"{path} is stale, regenerate it:", file=sys.stderr)
        print("\n".join(differs), file=sys.stderr)
        return 1
    header = fresh["provenance"]
    print(
        f"{path} is up to date ({header['n_columns']} columns of "
        f"{header['n_tools']} tools, digest {header['crosswalk_digest']})"
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
        help="checkout of pkpdutils whose commit is recorded "
        "(default: the checkout pkpdutils was imported from)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the fixture with a fresh run instead of writing it",
    )
    args = parser.parse_args()

    tools = {
        tool: tool_section(tool, attributes)
        for tool, attributes in sorted(TOOLS.items())
    }
    for tool, section in tools.items():
        if TOOLS[tool]["reader"] == "wide":
            check_injective(tool, section)
    intravenous = sorted(attribute("_INTRAVENOUS_ONLY", frozenset()))
    extravascular = sorted(attribute("_EXTRAVASCULAR_ONLY", frozenset()))
    check_pknca_routes(tools["pknca"], intravenous, extravascular)
    rules = route_rules()
    verify_rules(rules, tools["pknca"])
    pknca_rules = {
        "columns": list(attribute("PKNCA_COLUMNS", ())),
        "intravenous_only": intravenous,
        "extravascular_only": extravascular,
        "route_rules": rules,
    }

    columns: set[str] = set()
    for section in tools.values():
        columns |= set(section["names"].values())
        for routes in section["names_by_route"].values():
            columns |= set(routes.values())
    module_dir = Path(pkpdutils.__file__).resolve().parent
    source = (args.source or default_source() or REPO_DIR).resolve()
    body = {"tools": tools, "pknca_rules": pknca_rules}
    fixture = {
        "provenance": {
            "pkpdutils_version": pkpdutils.__version__,
            "pkpdutils_source": (
                "release" if "site-packages" in module_dir.parts else "checkout"
            ),
            "source_commit": clean_commit(source),
            "crosswalk_digest": digest(body),
            "n_tools": len(tools),
            "n_names": {tool: len(s["names"]) for tool, s in tools.items()},
            "n_names_by_route": {tool: len(s["names_by_route"]) for tool, s in tools.items()},
            "n_percent": {tool: len(s["percent"]) for tool, s in tools.items()},
            "n_unmapped": {tool: len(s["unmapped"]) for tool, s in tools.items()},
            "n_columns": len(columns),
            "n_route_rules": len(rules),
        },
        **body,
    }
    if args.check:
        return check(args.output, fixture)
    with args.output.open("w") as f:
        json.dump(fixture, f, indent=0)
        f.write("\n")
    header = fixture["provenance"]
    for tool, section in tools.items():
        print(
            f"{tool}: {len(section['names'])} names, "
            f"{len(section['names_by_route'])} by route, "
            f"{len(section['percent'])} percentages, "
            f"{len(section['unmapped'])} unmapped"
        )
    print(
        f"{header['n_columns']} distinct columns of {header['n_tools']} tools and "
        f"{header['n_route_rules']} PKNCA route rules"
    )
    print(
        f"generated with pkpdutils {header['pkpdutils_version']} "
        f"({header['pkpdutils_source']}, imported from {module_dir}, commit "
        f"{header['source_commit']}), crosswalk digest {header['crosswalk_digest']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
