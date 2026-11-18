//! The bootstrap of `analyze_batch` against python's own answer.
//!
//! `crates/core/tests/data/bootstrap_oracle.json` is written by `scripts/bootstrap_oracle.py`
//! from the `pkpdutils` checkout this crate mirrors: four batches of group curves, the batch and
//! the options in the encoding of the wire, and every uncertainty variable the **python** engine
//! reports for them (`engine._force_python`, so no part of the answer comes from the kernel).
//!
//! The two engines draw their normals from the same generator, which is the whole point of the
//! design: python asks `pkpdutils_core.bootstrap_normals` for its draws, so the replicates are
//! the same curves on both paths and the comparison is of the analysis and the reduction alone.
//! What separates them is the last bit of a transcendental function - `exp`, `ln`, `sqrt` of the
//! shaping and of the geometric statistics, where numpy has its own implementations - so the
//! tolerance is relative `1e-12` and not bit equality.

use std::collections::BTreeMap;

use pkpdutils_abi::{Batch, NcaOptions};
use pkpdutils_core::analyze_batch;
use serde::Deserialize;

/// The relative tolerance of the comparison, the transcendental functions of the two engines.
const TOLERANCE: f64 = 1e-12;

/// One array of the fixture, in the encoding of the wire: a number, or `"NaN"`.
#[derive(Deserialize)]
struct Values(#[serde(with = "pkpdutils_abi::nonfinite::vec")] Vec<f64>);

/// One case: a batch, its options and python's uncertainty variables for them.
#[derive(Deserialize)]
struct Case {
    name: String,
    batch: Batch,
    options: NcaOptions,
    expected: BTreeMap<String, Values>,
}

/// The whole fixture.
#[derive(Deserialize)]
struct Oracle {
    cases: Vec<Case>,
}

fn oracle() -> Oracle {
    serde_json::from_str(include_str!("data/bootstrap_oracle.json"))
        .expect("the bootstrap oracle is readable")
}

#[test]
fn the_bootstrap_matches_the_python_oracle() {
    let oracle = oracle();
    assert_eq!(oracle.cases.len(), 4, "the fixture lost a case");
    let mut compared = 0usize;
    for case in &oracle.cases {
        let result = analyze_batch(&case.batch, &case.options)
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        let columns: BTreeMap<&str, &[f64]> = result
            .parameters
            .iter()
            .map(|column| (column.name.as_str(), column.values.as_slice()))
            .collect();
        assert!(
            !case.expected.is_empty(),
            "{}: the fixture carries no variable",
            case.name
        );
        for (name, expected) in &case.expected {
            let values = columns
                .get(name.as_str())
                .unwrap_or_else(|| panic!("{}: the kernel reports no {name}", case.name));
            assert_eq!(values.len(), expected.0.len(), "{}: {name}", case.name);
            for (row, (&value, &want)) in values.iter().zip(&expected.0).enumerate() {
                let at = format!("{} {name} row {row}", case.name);
                if want.is_nan() {
                    assert!(value.is_nan(), "{at}: {value} is no NaN");
                    continue;
                }
                assert!(
                    (value - want).abs() <= TOLERANCE * want.abs(),
                    "{at}: {value} against {want}"
                );
                compared += 1;
            }
        }
        // python reports nothing the kernel does not: a variable of the one and not of the other
        // is a difference of the two results and not a tolerance
        let theirs: Vec<&str> = case.expected.keys().map(String::as_str).collect();
        for name in columns.keys() {
            let derived = [
                "_sd", "_se", "_ci_low", "_ci_high", "_pi_low", "_pi_high", "_geomean", "_geocv",
            ]
            .iter()
            .any(|suffix| name.ends_with(suffix));
            assert!(
                !derived || theirs.contains(name),
                "{}: the kernel reports {name} and python does not",
                case.name
            );
        }
    }
    // the fixture is not a handful of NaN: most of what it carries is a number
    assert!(compared > 2000, "only {compared} finite values compared");
}
