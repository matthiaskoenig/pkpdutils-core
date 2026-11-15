//! The vendored conformance oracles are present, intact and complete.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use sha2::{Digest, Sha256};

const MANIFEST_ENTRIES: usize = 31;

fn oracle_dir() -> PathBuf {
    // cargo runs a test binary from the package root, so go up to the workspace
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/oracles")
}

/// Read a vendored file as text, after checking it is a regular file and no
/// symlink: opening a FIFO would block forever instead of failing.
fn read_text(rel: &str) -> String {
    let path = oracle_dir().join(rel);
    let kind = fs::symlink_metadata(&path)
        .unwrap_or_else(|e| panic!("missing: {rel}: {e}"))
        .file_type();
    assert!(kind.is_file(), "not a regular file: {rel}");
    fs::read_to_string(path).unwrap()
}

fn json(rel: &str) -> Value {
    let text = read_text(rel);
    serde_json::from_str(&text).unwrap()
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::new(), |mut out, b| {
            write!(out, "{b:02x}").unwrap();
            out
        })
}

fn walk(dir: &std::path::Path, out: &mut BTreeSet<String>) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        // the entry's own file type does not follow a symlink, which is then a file
        if entry.file_type().unwrap().is_dir() {
            walk(&path, out);
        } else {
            let rel = path.strip_prefix(oracle_dir()).unwrap();
            out.insert(rel.to_str().unwrap().replace('\\', "/"));
        }
    }
}

fn file_count(rel: &str) -> usize {
    let dir = oracle_dir().join(rel);
    let kind = fs::symlink_metadata(&dir).unwrap().file_type();
    assert!(kind.is_dir(), "not a real directory: {rel}");
    fs::read_dir(dir)
        .unwrap()
        .filter(|e| e.as_ref().unwrap().file_type().unwrap().is_file())
        .count()
}

#[test]
fn every_vendored_oracle_matches_its_recorded_checksum() {
    let manifest = json("MANIFEST.json");
    let files = manifest["files"].as_object().unwrap();
    // an empty manifest would pass the loop below, so pin its size
    assert_eq!(files.len(), MANIFEST_ENTRIES);
    assert_eq!(manifest["source_commit"].as_str().unwrap().len(), 40);
    let mut on_disk = BTreeSet::new();
    walk(&oracle_dir(), &mut on_disk);
    let mut expected: BTreeSet<String> = files.keys().cloned().collect();
    expected.insert("MANIFEST.json".to_string());
    let unexpected: Vec<_> = on_disk.difference(&expected).collect();
    let absent: Vec<_> = expected.difference(&on_disk).collect();
    assert!(
        unexpected.is_empty() && absent.is_empty(),
        "unrecorded files {unexpected:?}, missing files {absent:?}"
    );
    let bad: Vec<String> = files
        .iter()
        .filter_map(|(rel, digest)| {
            let path = oracle_dir().join(rel);
            // the link itself, never what it points at
            match fs::symlink_metadata(&path) {
                Err(_) => return Some(format!("missing: {rel}")),
                Ok(m) if m.file_type().is_symlink() => return Some(format!("symlink: {rel}")),
                Ok(m) if !m.file_type().is_file() => {
                    return Some(format!("not a regular file: {rel}"));
                }
                Ok(_) => {}
            }
            match fs::read(path) {
                Err(_) => Some(format!("missing: {rel}")),
                Ok(bytes) if hex_sha256(&bytes) != digest.as_str().unwrap() => {
                    Some(format!("differs: {rel}"))
                }
                Ok(_) => None,
            }
        })
        .collect();
    assert!(bad.is_empty(), "vendored oracles damaged: {bad:?}");
}

#[test]
fn the_reference_cases_are_all_there() {
    let nca = json("tests/data/reference/nca_reference.json");
    let nca = nca.as_array().unwrap();
    assert_eq!(nca.len(), 9);
    assert!(nca.iter().all(|c| !c.as_object().unwrap().is_empty()));

    let validation = json("tests/data/validation/reference.json");
    let cases = validation["cases"].as_object().unwrap();
    assert_eq!(cases.len(), 6);
    let references: usize = cases
        .values()
        .flat_map(|case| case["references"].as_object().unwrap().values())
        .map(|subject| subject.as_object().unwrap().len())
        .sum();
    assert_eq!(references, 1043);
    assert_eq!(validation["partial_cases"].as_object().unwrap().len(), 1);
    assert_eq!(validation["summary_cases"].as_object().unwrap().len(), 1);

    let scenarios = read_text("docs/data/benchmarks/scenarios.csv");
    let mut lines = scenarios.lines().filter(|l| !l.is_empty());
    assert!(lines.next().unwrap().starts_with("scenario,dataset"));
    let rows = lines.count();
    assert_eq!(rows, 8);
    for dir in ["winnonlin", "pknca", "noncompart"] {
        assert_eq!(
            file_count(&format!("docs/data/benchmarks/{dir}")),
            8,
            "{dir}"
        );
    }
}
