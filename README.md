# pkpdutils-core

A Rust port of the non-compartmental analysis core of the Python library [pkpdutils](https://github.com/matthiaskoenig/pkpdutils), compiled natively and to WebAssembly.

Two contracts connect a host to the core: `batch/1`, the batch of timecourses, and `analysis/1`, the analysis request and its result. Their identifiers and the version of the serialized options are defined in `pkpdutils-abi` (`crates/abi`).

## Contracts

| Constant | Value |
| --- | --- |
| `BATCH_CONTRACT` | `batch/1` |
| `ANALYSIS_CONTRACT` | `analysis/1` |
| `OPTIONS_VERSION` | `1` |

`check_contract(found, expected)` accepts an exact match and otherwise returns a `ContractError` naming both sides.

`batch/1` carries the timecourses, the dosing protocol of every row, the units and the batch level descriptors `substance`, `route` and `tissue` (per-sample `substances` and `routes` when a batch is mixed; a mixed `tissue` does not exist in Python). `analysis/1` carries the request (a batch and its options) and the result: the parameters, the variables over an extra dimension, the boolean and the text variables (`accepted`, `excluded` and `excluded_reason`, which `NCAOptions.acceptance` drives), the flags, the coordinates (each with the sample dimensions it lives on, since a coordinate of `pkpdutils` lives on a subset of them) and the `attributes`, which are the `attrs` of `NCAResult.ds` - `substance`, `route`, `tissue` and the resolved options, the four a writer of the result reads back (`cdisc.to_pp` reads `tissue` for `PPSPEC`, `report.py` reads the options for its methods line).

The comparison is exact, so an additive field is a new contract as soon as a host speaks the old one. While nothing speaks them - no host, no release, no other repository - a field is added to `batch/1` and `analysis/1` in place and the contract strings stay; after a host ships, the same change is a `batch/2` with a coordinated bump across the repositories.

## The pint gate

`crates/units/tests/data/pint_gate.json` is pint's answer for the unit strings the package uses and produces, written by `scripts/pint_gate.py` from the `pkpdutils` checkout this crate mirrors. The tests in `crates/units/tests/pint_gate.rs` compare `parse_unit`, `unit_str`, `short_unit` and `parameter_unit` against it character for character, so the gate runs in `cargo test` without Python.

The input spellings are not hand written, they are extracted from the checkout: every string literal which `src/`, `tests/`, `docs/` and `examples/` hand to a unit keyword (`unit=`, `time_unit=`, `dose_unit=`, `amount_unit=`, `volume_unit=`, a `units=` mapping) or to one of the unit functions of `pkpdutils.units`, read with `ast` from the python files and from the `python` fences of the pages. The fixture has four sections:

| Section | What it pins |
| --- | --- |
| `spellings` | every extracted literal pint accepts, with pint's unit and short form: the gate on the input side, which a hand-written list of spellings cannot provide |
| `rejected` | the literals pint accepts and the curated registry deliberately does not (`REJECTED` in the script, with the reason), asserted to stay rejected |
| `records` | for every distinct expression of `PARAMETER_UNITS`, every combination of the axes it names, with pint's unit, factor and short form |
| `parameters` | every parameter with its expression, so the records are proven to cover all of them |

The axis a spelling may stand in for follows its dimension, which is what the expressions branch on: a time for `{time}`, a volume for `{volume}`, a mass, an amount of substance or an activity for `{amount}`, one of those or something which becomes one per body weight for `{dose}`, and every spelling for `{unit}`. The expressions are keyed once each and not once per parameter, since the derivation is a pure function of the expression and the input units.

The `provenance` header records the `pkpdutils` version, its source (`checkout` or `release`, by whether the module was imported from a `site-packages` directory), the commit of the source tree when it is a clean checkout, the digest of `PARAMETER_UNITS`, the section sizes and the axes. `crates/units/tests/pint_gate.rs` also carries those sizes as its own constants (`N_RECORDS`, `N_PARAMETERS`, `N_EXPRESSIONS`, `N_SPELLINGS`): a regenerated fixture regenerates its header with it, so the header alone cannot show that a fixture shrank, and continuous integration has no `pkpdutils` to compare against. Update the constants deliberately together with the fixture, never to make a failing gate pass.

Regenerate it against the checkout being mirrored (`path/to/pkpdutils` is that checkout) with

```bash
uv run --with-editable path/to/pkpdutils python scripts/pint_gate.py
```

The spellings come from the checkout `pkpdutils` was imported from, so a release install needs `--source path/to/pkpdutils` as well. The script writes the committed fixture by default, from any working directory; `-o PATH` writes another file instead. `--check` (same command with `--check` added) writes nothing, reads the committed fixture unless `-o PATH` names another one, and compares the `pkpdutils` version, the digest, the section sizes, the axes and all four sections with that fixture (the source and the commit are recorded, not compared); a non-zero exit means the fixture is stale and must be regenerated. The script prints the provenance, the section sizes and the axes, and exits non-zero if a combination raises in pint, if an axis an expression names has no spelling, or if a `REJECTED` entry is no longer in the source tree. Never edit the fixture by hand: if Rust disagrees, fix `pkpdutils-units`. Regeneration is needed when `PARAMETER_UNITS` changes, when the unit literals of `pkpdutils` change, and whenever the checkout moves ahead of the version recorded in the fixture.

## The vendored oracles

`testdata/oracles/` holds the files the Python tests pin their numbers to (the NCA and validation reference JSON, the validation README and `docs/data/benchmarks/` in full), copied with their relative paths, and `MANIFEST.json` with the sha256 of each of the 31 files and the commit of `pkpdutils` they were copied from. Vendor them from a clean checkout (`path/to/pkpdutils` is that checkout; the script refuses a tree with uncommitted changes) with

```bash
python3 scripts/vendor_oracles.py --source path/to/pkpdutils
```

The set of files is read from `git ls-files` of the source, so it is the set the recorded commit tracks. When a manifest exists and the set differs from it, the script prints the added and removed paths and refuses unless `--accept-changes` is passed. `python3 scripts/vendor_oracles.py --check` copies nothing and checks only the vendored files against the manifest, including a file under `testdata/oracles/` the manifest does not list. `python3 scripts/vendor_oracles.py --check --source path/to/pkpdutils` also compares the source files and its commit with the manifest; either form exits non-zero, naming the files, on any difference. `cargo test` (`crates/abi/tests/oracles_present.rs`) proves the vendored files match the manifest and that no reference case was dropped, but CI has no `pkpdutils` checkout and cannot compare against the source. Run `--check --source` by hand before a release and whenever the Python oracles change. Never edit the vendored files by hand.

## Toolchain

Stable Rust 1.100.0 with `rustfmt` and `clippy`, and the targets `wasm32-unknown-unknown` and `wasm32-wasip1`, pinned in `rust-toolchain.toml`.

## Checks

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## License

MIT, see `LICENSE`.
