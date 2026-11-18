//! The WebAssembly host of `pkpdutils-core`.
//!
//! One `cdylib` for `wasm32-unknown-unknown` with the raw C ABI of the engine of
//! `sbmlsim-core`: a page allocates bytes with `pkpd_alloc`, writes a JSON request into them,
//! calls an entry point, reads the answer at `pkpd_output_ptr` for `pkpd_output_len` bytes and
//! frees its request. There is no wasm-bindgen, no import of any kind
//! (`crates/wasm/tests/imports.rs` asserts it) and no thread: a page which wants several cores
//! runs one instance per Web Worker, which is what the loader of the npm package does.
//!
//! Every entry point returns 0 for an answer and 1 for an error, writes its output as
//! [`output`] describes and is a thin wrapper around a pure function of the same name without
//! the prefix, which is what the tests of this crate call. Nothing of the analysis itself lives
//! here: this crate is the layout, the error classes and the panic hook, and the kernel is
//! `pkpdutils-core`.
//!
//! ```bash
//! cargo build -p pkpdutils-wasm --release --no-default-features --target wasm32-unknown-unknown
//! ```

mod analyze;
mod output;

pub use analyze::pkpd_analyze;
pub use output::{pkpd_alloc, pkpd_free, pkpd_output_len, pkpd_output_ptr};
