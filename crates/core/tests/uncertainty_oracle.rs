//! The uncertainty of `analyze_batch` against python's own answer.
//!
//! `crates/core/tests/data/uncertainty_oracle.json` is written by `scripts/uncertainty_oracle.py`
//! from the `pkpdutils` checkout this crate mirrors: eight batches of group curves, the batch and
//! the options in the encoding of the wire, and every uncertainty variable and flag word the
//! **python** engine reports for them (`engine._force_python`, so no part of the answer comes
//! from the kernel). Four cases are the bootstrap and four the delta method, among them one
//! whose perturbations move the terminal window.
//!
//! The two engines draw their normals from the same generator, which is the whole point of the
//! design: python asks `pkpdutils_core.bootstrap_normals` for its draws, so the replicates are
//! the same curves on both paths and the comparison is of the analysis and the reduction alone.
//! What separates them is the last bit of a transcendental function - `exp`, `ln`, `sqrt` of the
//! shaping and of the geometric statistics, where numpy has its own implementations - so the
//! tolerance is relative `1e-12` and not bit equality. The delta method draws nothing at all,
//! so for its cases the tolerance covers those functions and nothing else.

use std::collections::BTreeMap;

use pkpdutils_abi::{Batch, NcaOptions, UncertaintyMethod};
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
    flags: Values,
    order: Vec<String>,
    expected: BTreeMap<String, Values>,
}

/// The whole fixture.
#[derive(Deserialize)]
struct Oracle {
    cases: Vec<Case>,
}

fn oracle() -> Oracle {
    serde_json::from_str(include_str!("data/uncertainty_oracle.json"))
        .expect("the uncertainty oracle is readable")
}

#[test]
fn the_uncertainty_matches_the_python_oracle() {
    let oracle = oracle();
    assert_eq!(oracle.cases.len(), 8, "the fixture lost a case");
    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut where_worst = String::new();
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
        // The bootstrap compares two runs of the same analysis of the same replicate curves, so
        // its tolerance is the agreement of the two analyses themselves. The delta method divides
        // by a step of `delta_step` times the standard error of the point, and a difference
        // quotient of two nearly equal numbers amplifies whatever separates the engines by one
        // over that step: the tolerance of a delta case is therefore `TOLERANCE / delta_step`,
        // which is derived and not fitted. The worst difference the fixture actually shows is
        // printed at the end of the test and is two orders below the loosest of these bounds.
        let tolerance = if case.options.uncertainty == Some(UncertaintyMethod::Delta) {
            TOLERANCE / case.options.delta_step
        } else {
            TOLERANCE
        };
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
                let relative = (value - want).abs() / want.abs().max(f64::MIN_POSITIVE);
                if relative > worst {
                    worst = relative;
                    where_worst = at.clone();
                }
                assert!(
                    (value - want).abs() <= tolerance * want.abs(),
                    "{at}: {value} against {want} (tolerance {tolerance:e})"
                );
                compared += 1;
            }
        }
        // The order of the columns is python's as well and not only their names: it is the
        // column order of `NCAResult.to_dataframe` and of every table rendered from a result,
        // and the host adopts the kernel's answer as it comes.
        let written: Vec<&str> = result
            .parameters
            .iter()
            .map(|column| column.name.as_str())
            .filter(|name| case.order.iter().any(|wanted| wanted == name))
            .collect();
        let wanted: Vec<&str> = case.order.iter().map(String::as_str).collect();
        assert_eq!(written, wanted, "{}: the column order differs", case.name);

        // the flags of the row are python's, which for the delta method includes
        // `DELTA_WINDOW_CHANGE` where a perturbation moved the terminal window
        let flags: Vec<f64> = result.flags.iter().map(|bits| f64::from(*bits)).collect();
        assert_eq!(flags, case.flags.0, "{}: the flags differ", case.name);

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
    println!("worst relative difference {worst:e} at {where_worst}");
    // and the window change of the delta method is really in it
    let flagged = oracle
        .cases
        .iter()
        .filter(|case| case.flags.0.iter().any(|bits| (*bits as u32) & 128 != 0))
        .count();
    assert_eq!(flagged, 1, "one case moves the terminal window");
}
