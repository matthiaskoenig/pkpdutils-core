# pkpdutils-core

[![CI](https://github.com/matthiaskoenig/pkpdutils-core/actions/workflows/ci.yml/badge.svg)](https://github.com/matthiaskoenig/pkpdutils-core/actions/workflows/ci.yml)
[![Version](https://img.shields.io/pypi/v/pkpdutils-core.svg)](https://pypi.org/project/pkpdutils-core/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pkpdutils-core.svg)](https://pypi.org/project/pkpdutils-core/)
[![pkpdutils-abi](https://img.shields.io/crates/v/pkpdutils-abi.svg?label=pkpdutils-abi)](https://crates.io/crates/pkpdutils-abi)
[![MIT License](https://img.shields.io/pypi/l/pkpdutils-core.svg)](https://opensource.org/licenses/MIT)

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

## The kernel

`pkpdutils-core` (`crates/core`) is the port of `pkpdutils.nca`: the non-compartmental analysis of every row of a batch, single dose and multiple dose. It computes the areas under the curve and under the first moment curve with the three trapezoid rules, the peak, the trough and the last measurable point, the terminal regression under each of its four rules (the best adjusted R squared, the last n points, every point after the maximum, and the points a caller names) or in the time window a caller gives a sample, which replaces the rule, with the statistics of the chosen window and, on request, the table of every candidate, the extrapolations to infinity both observed and predicted, the moments, the clearance, the volumes and the mean residence time, the dose normalized parameters, `c0` and the back extrapolation of an intravenous bolus, `tlag`, the values below a limit of quantification under the BLQ rules, the flags of every row and the acceptance variables: 48 parameter columns for an intravenous bolus, 44 for an extravascular curve and 43 for an infusion, and 19 more for a batch with a multiple dose row (the steady state and accumulation columns: 67, 63 and 62), plus the 13 `interval_*` variables over the dimension `interval` while `NcaOptions.intervals` is on (`accepted`, `excluded` and the flags are not among these counts), each in the unit `pkpdutils-units` derives for it, assembled into an `AnalysisResult` which carries the coordinates and the attributes of the batch.

Three functions are its entry points. `analyze(request)` takes an `AnalysisRequest`, checks its contract and that of its batch, and analyses it; `analyze_batch(batch, options)` is the same analysis for a host which holds the two separately and plans the row shards itself; `analyze_batch_in_shards(batch, options, shards)` takes that plan from the caller, and the three give the same numbers down to the bit. `KernelError` is how a malformed batch, a contradictory set of options, an analysis this build does not carry and a foreign contract come back; it is `#[non_exhaustive]`, since every analysis this build refuses by name is a path a later plan implements with errors of its own. The wire types are re-exported as `pkpdutils_core::pkpdutils_abi`, so a host depends on this crate alone and cannot hold a version of them which disagrees with the build it calls. A panic would trap the wasm instance and take the host's session with it, so the kernel indexes nothing with a number from the input and unwraps nothing a host controls: a malformed input is a named error or a `NaN`.

Agreement with the python package is what the crate is measured by, down to the order of the operations inside a formula, and an upstream oddity is mirrored with a test pinning it rather than corrected. `docs/python-reference/` records what the python kernel computes, extracted from the commit `testdata/oracles/MANIFEST.json` names, and is the requirement the port is written against. Four oracle gates in `crates/oracles/tests/` hold it. `multiple_dose.rs` holds the multiple dose analysis against the python core of `pkpdutils` on nine cases written by `scripts/multiple_dose_oracle.py` (regular oral, bolus and infusion protocols, an irregular one, a completed and an incomplete last interval, a mixed batch and an explicit `tau`), the steady state and interval variables bit for bit. The other three hold the kernel against published numbers: `reference.rs` reproduces `pkdb_analysis` 0.3.1 on the nine curves of its own test data, `validation.rs` the published Phoenix WinNonlin and PKNCA values over the theophylline and indomethacin profiles, and `benchmarks.rs` every parameter of every subject of the eight benchmark scenarios against WinNonlin, PKNCA and NonCompart at a relative 1e-8.

What the kernel ports is `nca` and not `Timecourses`, so the validation of the data model belongs to the host: the kernel reads the batch as `nca` reads the arrays `Timecourses` hands it. Where that matters it is stated on the contract and checked by `Batch::validate` - the doses of a row sit at the front of its dose columns, because `reference_dose` picks positionally and a hole in the protocol would otherwise lose every dose-dependent parameter of the row without raising anything - and where it does not, the kernel simply does what the python kernel does: a non-finite sampling time, which `Timecourses` raises on at construction, is dropped from the row here exactly as `pack_valid` drops it there.

## Row shards and the `parallel` feature

The rows of a batch are independent, so the row loop is cut into contiguous shards and the shards run on a [rayon](https://docs.rs/rayon) pool. That is the `parallel` feature of `pkpdutils-core` and it is **off by default**:

```bash
cargo build --release -p pkpdutils-core --features parallel   # a native host
cargo build --release -p pkpdutils-core --no-default-features --target wasm32-unknown-unknown
```

A wasm build takes the kernel without the feature. A wasm instance has no threads to run a shard on, and parallelism on the web is one kernel instance per Web Worker, which is the deployment this port was planned for: the worker holds the instance, the host spreads the batches over the workers and no instance ever spawns anything. Rayon 1.12 does compile for both wasm targets, so nothing stops a build from asking for the feature there, and what it then gets is nothing: `wasm32-wasip1` under wasmtime reports one thread and runs every shard in the calling thread, measured in `BENCHMARKS.md`, while `wasm32-unknown-unknown` carries no thread support in `std` at all. The feature on a wasm target is a dependency the instance cannot use, so do not enable it there.

`analyze_batch` plans the shards itself with `automatic_shards`, from the work of the batch (its rows times its time points) and the size of the pool: one shard below `SHARD_WORK_THRESHOLD` of work, which is above where the measurement of `BENCHMARKS.md` found the crossover, and above it the whole pool, four shards per thread of it and never more shards than there are rows. A host which has its own pool, or its own worker per batch, calls `analyze_batch_in_shards(batch, options, 1)` instead and parallelizes outside; a count there is a request, the smallest stretch of rows the pool may cut, so the run can end up with fewer stretches than the count and never with more. Whatever the plan, the numbers are identical down to the bit - a shard writes its own rows in the row order of the batch and shares nothing - and `the_parallel_and_the_serial_path_produce_the_same_bits` compares them that way, bit for bit over six shard counts and five option sets (the best fit regression, manual point indices, the last n points with the linear up log down rule and no peak excluded, a window per sample, and a limit of quantification with the values below it zeroed before the peak), on a batch whose rows carry holes, a missing dose, an empty row, a single point row and three routes.

`BENCHMARKS.md` is the measured table against the python package. `crates/core/examples/kernel_timing.rs` is the rust half of what wrote it and `scripts/python_timing.py`, which builds the same batches in python and is run from a `pkpdutils` checkout, is the other:

```bash
cargo run --release --example kernel_timing                              # the serial build
cargo run --release --example kernel_timing --features parallel          # the sharded build
cargo run --release --example kernel_timing --features parallel -- sweep # the crossover
cargo run --release --example kernel_timing -- 1000x200                  # one shape of it
```

Both sides take one shape at a time, so that a measured cell never carries the heap of the cell before it: `kernel_timing <shape>` on this side, `python_timing.py --shape 1000x200 --variant nca` on the other.

## The python binding and the `pkpdutils-core` distribution

`pkpdutils-python` (`crates/python`) is the pyo3 binding, and the wheel built from it is the `pkpdutils-core` distribution on PyPI, which `pkpdutils` will depend on and route an eligible batch to. It is the only crate of the workspace which knows that python exists: it takes the numpy buffers of a batch and the fields of the options as plain dicts, copies them into the wire shapes of `pkpdutils-abi` and hands them to the kernel with the interpreter released. `analyze_batch` and `is_supported` are the two functions it exports, `UnsupportedError` is the refusal a host routes its own path on, and `crates/python/README.md`, which is also the long description of the distribution, is what a user of the wheel reads.

The crate is a workspace member and **not a default member**: pyo3 refuses to build against an interpreter older than its `abi3-py313` floor, so a crate in the default set would make a python 3.13 toolchain the price of `cargo test` on the kernel. `cargo test`, `cargo clippy` and `cargo doc` without a `-p` therefore build none of it, `--workspace` does reach it, which is why the wasm checks exclude it by name, and the `python` job of continuous integration is its gate.

The wheel is built by [maturin](https://www.maturin.rs) from `crates/python/pyproject.toml`, where two settings are not conveniences:

- `module-name = "pkpdutils_core"`. The `#[pymodule]` is named after the distribution and the lib target after the cargo package, so without it the extension ships under the wrong name and the import fails on a missing `PyInit_` symbol, on an installed wheel and nowhere earlier.
- `features = ["extension-module", "parallel"]`. `extension-module` leaves libpython to the interpreter which loads the wheel, which an extension module must do and `cargo test` must not; `parallel` is the row shards above, without which `n_workers` would be inert and the measured speedup over the python kernel most of itself smaller. The hazard it carries is a rayon pool across `fork`, which `crates/python/README.md` warns about: `pkpdutils` starts its own process pools with `forkserver` or `spawn` and never with `fork`, for that same reason.

`crates/python/tests/test_bindings.py` runs against the installed wheel and never against the source tree, because the module name, the `PyInit_` symbol, the `abi3` tag, the version the metadata claims and the presence of the row shards are all decided by the build and by nothing a rust test can see.

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

The set of files is read from `git ls-files` of the source, so it is the set the recorded commit tracks. When a manifest exists and the set differs from it, the script prints the added and removed paths and refuses unless `--accept-changes` is passed. `python3 scripts/vendor_oracles.py --check` copies nothing and checks only the vendored files against the manifest, including a file under `testdata/oracles/` the manifest does not list. `python3 scripts/vendor_oracles.py --check --source path/to/pkpdutils` also compares the source files and its commit with the manifest; either form exits non-zero, naming the files, on any difference. `cargo test` (`crates/oracles/tests/oracles_present.rs`, a crate which is never published, since the oracles live at the root of the workspace and cargo packages only what is inside a crate's own directory) proves the vendored files match the manifest and that no reference case was dropped, but CI has no `pkpdutils` checkout and cannot compare against the source. Run `--check --source` by hand before a release and whenever the Python oracles change. Never edit the vendored files by hand.

## The crosswalk

`crates/oracles/tests/data/crosswalk.json` is the map from the variables of `pkpdutils` to the column names of Phoenix WinNonlin, PKNCA and NonCompart, written by `scripts/crosswalk_gate.py` from the same checkout. `crates/oracles/tests/benchmarks.rs` compares the eight scenarios of `testdata/oracles/docs/data/benchmarks/scenarios.csv` against the result table of each of the three tools, and every one of those 24 tables is written in the names of its own tool (`AUCINF_obs`, `aucinf.obs` and `AUCIFO` are the one parameter `auc_inf_obs`). The map which translates them is `src/pkpdutils/crosswalk.py`, which is source and no oracle, so it was never vendored: without this fixture nothing here can read the tables at all.

Every name, set and rule in it is extracted and none is transcribed. The script imports `pkpdutils.crosswalk` and writes two sections: `tools`, for each tool the name map, the map of the names which depend on the route, the columns which carry a percentage where `pkpdutils` reports a fraction and the columns no variable corresponds to (`Corr_XY`, `CORRXY`, `b0`); and `pknca_rules`, what reading the long PKNCA table needs beyond the names, which is its columns, the two sets which decide whether `cl.obs` is `cl` or `cl_f`, and the rules which recognize the route from the parameters PKNCA reports. Those rules are read from the source of `_pknca_route` and then verified against the function itself for every parameter of the maps and every pair of them, so an extraction which has drifted fails instead of being written.

What the script does write by hand is only where to look: `TOOLS` names the module attribute of every map and which of the two readers reads each tool's table. That is deliberate rather than a shortcut, since a map renamed upstream then fails the extraction instead of being written as an empty one, and the script holds that table to the module as well: a map `crosswalk.py` declares which `TOOLS` does not name, a tool it gains, and a name the module no longer carries are each a non-zero exit.

The `provenance` header records the `pkpdutils` version, its source (`checkout` or `release`), the commit of a clean checkout, a digest of the two sections and their sizes. The gate carries the version, the digest and the column count as its own constants and recomputes the digest from the sections themselves, so a fixture edited by hand fails there even when the numbers happen to agree. Update the constants deliberately together with the fixture, never to make a failing gate pass.

Regenerate it against the checkout being mirrored with

```bash
uv run --with-editable path/to/pkpdutils python scripts/crosswalk_gate.py
```

The script writes the committed fixture by default, from any working directory; `-o PATH` writes another file instead. `--check` (the same command with `--check` added) writes nothing, compares the version, the digest, the sizes and both sections with the committed fixture, checks the committed header against the committed sections as well, and exits non-zero when anything differs. It also exits non-zero if a tool names two variables with one column, if a route leaves the PKNCA names ambiguous, or if the route rules no longer answer as `_pknca_route` does. Never edit the fixture by hand: if Rust disagrees, fix the gate. Regeneration is needed whenever `crosswalk.py` changes and whenever the checkout moves ahead of the version recorded in the fixture.

## Toolchain

Stable Rust 1.100.0 with `rustfmt` and `clippy`, and the targets `wasm32-unknown-unknown` and `wasm32-wasip1`, pinned in `rust-toolchain.toml`.

## Running the tests

`cargo test -p pkpdutils-core` runs the kernel alone and `cargo test` the default members (the kernel, the wire types, the units and the oracle gates), none of which needs python. The binding is not a default member: pyo3 refuses to build against an interpreter older than its `abi3-py313` floor, and `cargo test --workspace` on a machine whose default `python3` is older fails with an opaque `failed to run custom build command for pyo3-ffi`. Name a python 3.13 or newer for it, and for the tests of the binding activate a virtual environment of that interpreter with `numpy` installed (`pip install numpy`): the tests build real numpy arrays through the C API, and an interpreter without numpy, or one which is embedded without its environment, builds fine and then panics in 26 of them with `Failed to access NumPy array API capsule`.

```bash
cargo test -p pkpdutils-core
python3.13 -m venv .venv && . .venv/bin/activate && pip install numpy
PYO3_PYTHON=python3.13 cargo test --workspace
```

Continuous integration sets up python 3.13 for the same reason (`ci.yml` and `release.yml`).

`PYO3_PYTHON` points the build at an interpreter; it does not tell the embedded one where to import from. Against a virtual environment outside this repository, the `pkpdutils` checkout's for instance, the test binary computes `sys.path` from its own location and finds the system numpy, which fails every test that touches an array with `Failed to access NumPy array API capsule`. Pass the environment's `site-packages` as well, and do not set `PYTHONHOME`, which breaks the interpreter's own start-up:

```bash
PYO3_PYTHON=../pkpdutils/.venv/bin/python \
  PYTHONPATH=../pkpdutils/.venv/lib/python3.14/site-packages \
  cargo test -p pkpdutils-python
```

## Checks

What continuous integration runs, so `--locked` here as well: the committed `Cargo.lock` is what is checked and a resolver change is a failure rather than a silent difference. `missing_docs` is a workspace lint (`[workspace.lints.rust]`, inherited by every crate through `[lints] workspace = true`), so the convention that every public item carries a doc comment is a gate: `clippy -D warnings` fails on an undocumented one. Never silence it for a crate, write the doc comment.

```bash
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo clippy --locked --all-targets --features parallel -- -D warnings
cargo test --locked --features parallel
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps
cargo check --locked --workspace --exclude pkpdutils-python --target wasm32-unknown-unknown
cargo check --locked --workspace --exclude pkpdutils-python --target wasm32-wasip1
cargo check --locked --workspace --exclude pkpdutils-python --no-default-features --target wasm32-unknown-unknown
cargo check --locked --workspace --exclude pkpdutils-python --no-default-features --target wasm32-wasip1
```

The `parallel` feature is checked and tested as well as left out: a feature which compiles in one configuration only is a break waiting for the first host which asks for it. The wasm targets are checked both ways for the same reason: `--no-default-features` is how a wasm host builds the kernel, and the plain check is the one which fails the day a default feature stops compiling for wasm. `--workspace` reaches the python binding, which pyo3 cannot build for wasm at all, so it is excluded there by name; nothing is lost, since the binding carries no numerics of its own and the kernel it binds is what those four lines check.

The binding has a gate of its own, the `python` job, because none of the commands above builds the artifact a user installs. It needs a python 3.13 or newer on `PATH` with a real `libpython` (a standalone interpreter is not enough: numpy fails on `_contextvars` when it is embedded), and the lines after the cargo ones run in a virtual environment with `numpy`, `maturin` and `pytest`:

```bash
cargo clippy -p pkpdutils-python --all-targets --locked -- -D warnings
cargo clippy -p pkpdutils-python --all-targets --locked --features parallel -- -D warnings
cargo test -p pkpdutils-python --locked
RUSTDOCFLAGS="-D warnings" cargo doc -p pkpdutils-python --locked --no-deps
cmp LICENSE crates/python/LICENSE
maturin build --release --locked -m crates/python/Cargo.toml
pip install --no-index --find-links target/wheels pkpdutils-core
pytest crates/python/tests -q
maturin build --release --sdist --out target/sdist -m crates/python/Cargo.toml
```

The second clippy line is the configuration the wheel ships, which would otherwise be the one build users install and no lint has ever seen. `cmp` holds the two license files together: `license-files` of `pyproject.toml` may not reach out of the crate directory, so `crates/python/LICENSE` is a copy of the one at the root and the wheel ships the copy.

The last line is the source distribution of the release, and maturin proves it by building a wheel out of it: a file the build needs and the sdist does not carry fails there instead of failing every source install after a release. It is a command of its own and without `--locked`, because the sdist carries a workspace maturin assembled out of the binding and the path dependencies it needs, which is not the workspace the committed `Cargo.lock` was resolved for, so cargo has to rewrite that lock. The wheel which is installed and tested is the `--locked` one, built from the checkout.

Four fixtures are generated from a `pkpdutils` checkout instead of being written by hand, and each of their scripts compares the committed file with a fresh run and writes nothing. They need that checkout and so are not part of continuous integration, where the Rust gates carry the sizes and digests as their own constants instead; run both by hand whenever `pkpdutils` changes and before a release, from any working directory (`path/to/pkpdutils` is the checkout being mirrored). A non-zero exit means the fixture is stale and has to be regenerated with the same command without `--check`, never edited.

```bash
uv run --with-editable path/to/pkpdutils python scripts/pint_gate.py --check
uv run --with-editable path/to/pkpdutils python scripts/crosswalk_gate.py --check
uv run --with-editable path/to/pkpdutils python scripts/reduce_oracle.py --check
uv run --with-editable path/to/pkpdutils python scripts/bootstrap_oracle.py --check
```

`scripts/pint_gate.py` writes `crates/units/tests/data/pint_gate.json`, pint's answer for every unit spelling of the package and every derived parameter unit, and `scripts/crosswalk_gate.py` writes `crates/oracles/tests/data/crosswalk.json`, the map from the variables of `pkpdutils` to the column names of the three NCA tools; the sections above describe both. `scripts/reduce_oracle.py` writes `crates/core/tests/data/reduce_oracle.json`, which is `pkpdutils.nca.uncertainty.reduce_replicates` answering for the replicate reduction of `crates/core/src/uncertainty.rs`: six cases of replicates with their inputs and python's uncertainty variables for them, compared to within `1e-12` relative, together with `LOGNORMAL_PARAMETERS` and `DISCRETE_PARAMETERS`, which the kernel holds its own copy of and the test `the_classification_is_pythons` compares name by name, so a classification which changes upstream fails `cargo test` here. `scripts/bootstrap_oracle.py` writes `crates/core/tests/data/bootstrap_oracle.json`, the whole bootstrap of four batches of group curves: the batch and the options in the encoding of the wire and every uncertainty variable the **python** engine reports for them (`engine._force_python`, so no part of the answer comes from the kernel), which `crates/core/tests/bootstrap_oracle.rs` compares against `analyze_batch` to within `1e-12` relative. The two engines draw their normals from the same generator - python asks `pkpdutils_core.bootstrap_normals` - so what separates them is the last bit of a transcendental function and nothing else. `scripts/vendor_oracles.py --check` is the fifth, and it needs no checkout unless `--source` is given.

Three generators are committed for the numbers a test pins rather than computes, and none of them needs a `pkpdutils` checkout. `scripts/philox_oracle.py` writes `crates/core/tests/data/philox_oracle.json`, which is `numpy.random.Philox` answering for the integer stream of `pkpdutils_core::Draws`: numpy's Philox is the same Random123 philox4x64-10 the kernel draws with, so the fixture is an independent implementation of the algorithm rather than a second copy of ours, and `crates/core/tests/philox_oracle.rs` compares the first words of four seeds and a digest of a hundred thousand against it. It takes `--check` like the two above and needs numpy alone. The remaining two print rather than write. `scripts/reference_percentile.py` prints the rows of `crates/core/src/percentile.rs` from `numpy.nanpercentile` (`plain`, `hole`, `single`, `naive` and `grouping`: the last three are the rows a seeded search found where one of the plausible interpolations, or the plausible grouping of the virtual index, differs from numpy in the last bit, which is what makes the bit-exactness of `nan_percentile_rows` a measurement rather than a claim). The Student t references of `crates/core/src/stats.rs` come from the twin generator in the host repository, `scripts/reference_t_isf.py` of `pkpdutils`, which inverts the survival function with 60-digit mpmath and so reaches `p = 1e-300`, where scipy is no longer an oracle.

```bash
python3 scripts/reference_percentile.py single
```

## Release

A tag `vx.y.z` publishes the distribution to PyPI and creates the release of this repository from `release-notes/x.y.z.md`. [docs/release.md](docs/release.md) has the steps.

## License

MIT, see `LICENSE`.
