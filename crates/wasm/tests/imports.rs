//! The module a page loads imports nothing at all.
//!
//! The one reason this host targets `wasm32-unknown-unknown` with a raw C ABI rather than
//! `wasm32-wasip1`, as the engine of `sbmlsim-core` has to, is that there is no C to link here
//! and therefore nothing to import: a page instantiates the module with an empty import object
//! and needs neither a WASI shim nor a memory of its own. An import which appears is that
//! property lost, and a loader which passes `{}` would fail with a `LinkError` instead.
//!
//! The model of this test is `wasm_has_no_imports` of `crates/mathlib/tests/values.rs` of
//! `sbmlsim-core`.

use std::path::{Path, PathBuf};

/// Where `cargo build --target wasm32-unknown-unknown --release` puts the module.
fn module_path() -> PathBuf {
    let target = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target")
            .to_string_lossy()
            .into_owned()
    });
    Path::new(&target).join("wasm32-unknown-unknown/release/pkpdutils_wasm.wasm")
}

/// The section ids of a wasm module, in the order a module carries them.
const IMPORT_SECTION: u8 = 2;

/// Read one LEB128 unsigned integer, and how many bytes it took.
fn leb128(bytes: &[u8]) -> Option<(usize, usize)> {
    let mut value = 0usize;
    let mut shift = 0;
    for (at, byte) in bytes.iter().enumerate() {
        value |= usize::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some((value, at + 1));
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
    None
}

/// The length of the import section of a module, `None` when it carries none.
fn import_section(module: &[u8]) -> Option<usize> {
    assert_eq!(&module[..4], b"\0asm", "no wasm magic");
    let mut at = 8;
    while at < module.len() {
        let id = module[at];
        let (len, read) = leb128(&module[at + 1..]).expect("a section length");
        if id == IMPORT_SECTION {
            return Some(len);
        }
        at += 1 + read + len;
    }
    None
}

#[test]
fn the_module_imports_nothing() {
    let path = module_path();
    if !path.exists() {
        println!(
            "skipped: {} is not built, run\n  cargo build -p pkpdutils-wasm --release \
             --no-default-features --target wasm32-unknown-unknown",
            path.display()
        );
        return;
    }
    let module = std::fs::read(&path).expect("the module");
    match import_section(&module) {
        None => (),
        // an empty section is a section of one zero byte, the count of its entries
        Some(1) => (),
        Some(len) => panic!(
            "{} carries an import section of {len} bytes; a page instantiates this module with \
             an empty import object",
            path.display()
        ),
    }
}
