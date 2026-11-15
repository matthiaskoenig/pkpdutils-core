//! The gate over the vendored conformance oracles.
//!
//! The oracles live in `testdata/oracles/` at the root of the workspace, since
//! every crate of the core compares against them, and cargo packages only what
//! is inside a crate's own directory. A test over them inside a published crate
//! would therefore ship without its data and fail for whoever ran
//! `cargo test` on the package, so the test lives here, in a crate which is
//! never published, together with the `sha2` it needs for the checksums.
//!
//! The crate has no code of its own: `tests/oracles_present.rs` is all of it.
