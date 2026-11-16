//! What every test binary of this crate needs to reach a vendored oracle.
//!
//! `tests/common/mod.rs` is a module of each test binary which declares it and is not a test
//! binary of its own, so the two helpers below are compiled into every gate that says
//! `mod common;` instead of being copied into it.

use std::fs;
use std::path::PathBuf;

/// The vendored oracles, `testdata/oracles` at the root of the workspace.
///
/// `CARGO_MANIFEST_DIR` is a compile-time constant naming the directory of this package, not
/// the working directory the test binary happens to run in, so the path holds wherever cargo
/// is invoked from.
pub fn oracle_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/oracles")
}

/// Read a vendored file as text, after checking it is a regular file and no
/// symlink: opening a FIFO would block forever instead of failing.
pub fn read_text(rel: &str) -> String {
    let path = oracle_dir().join(rel);
    let kind = fs::symlink_metadata(&path)
        .unwrap_or_else(|e| panic!("missing: {rel}: {e}"))
        .file_type();
    assert!(kind.is_file(), "not a regular file: {rel}");
    fs::read_to_string(path).unwrap()
}
