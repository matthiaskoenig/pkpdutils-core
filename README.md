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

## The pint gate

`crates/units/tests/data/pint_gate.json` is pint's answer for every unit string the package can produce: for every entry of `PARAMETER_UNITS` it varies only the input units the expression names (`{unit}`, `{time}`, `{dose}`, `{amount}`, `{volume}`) and records the unit, the factor and the short form from `pkpdutils`. The tests in `crates/units/tests/pint_gate.rs` compare `parameter_unit` and `short_unit` against it character for character, so the gate runs in `cargo test` without Python.

The fixture has a `provenance` header with the `pkpdutils` version and its source (`checkout` or `release`, by whether the module was imported from a `site-packages` directory), next to the `records`. It must be regenerated from the same `pkpdutils` the Rust side is meant to agree with, which is the checkout being mirrored, and the recorded version and source tell which one produced the file. Regenerate it against the checkout being mirrored (`path/to/pkpdutils` is that checkout) with

```bash
uv run --with-editable path/to/pkpdutils python scripts/pint_gate.py
```

Without a checkout, the release is the fallback, `uv run --with pkpdutils python scripts/pint_gate.py`. The script writes the committed fixture by default, from any working directory; `-o PATH` writes another file instead. `--check` (same command with `--check` added) writes nothing, reads the committed fixture unless `-o PATH` names another one, and compares the `pkpdutils` version, the digest of `PARAMETER_UNITS`, the parameter and record counts and the records with that fixture (the source is recorded, not compared); a non-zero exit means the fixture is stale and must be regenerated. The script prints the provenance and the record count and exits non-zero if any combination raises in pint. Never edit the fixture by hand: if Rust disagrees, fix `pkpdutils-units`. Regeneration is needed when `PARAMETER_UNITS` gains an entry or the accepted input units widen, and whenever the checkout moves ahead of the version recorded in the fixture.

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
