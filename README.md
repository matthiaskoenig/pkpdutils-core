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
