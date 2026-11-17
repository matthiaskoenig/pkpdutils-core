//! The multiple dose analysis of the kernel reproduces the python core of `pkpdutils`.
//!
//! The oracle is `tests/data/multiple_dose.json`, written by `scripts/multiple_dose_oracle.py`
//! from a `pkpdutils` checkout. Every case is built through `Timecourses.from_arrays`, so the
//! protocol of a case is one the python front end accepts, and analysed by `nca`, which runs the
//! python core for a multiple dose batch. No expected number here is a number of this kernel.
//!
//! Two tolerances, deliberately different:
//!
//! * the steady state variables (`STEADY_STATE` and `n_doses`) and every `interval_*` variable
//!   are compared **bit for bit**. Tasks 5 to 8 of the port verified them to be identical, and
//!   a tolerance there would hide the drift this gate exists to catch, such as a changed
//!   association of a sum.
//! * every other column is compared at a relative `1e-12`. Those come from the single dose
//!   core (`compute_row`), which predates the multiple dose port, and a handful of them
//!   (`aumc_inf`, `mrt`, `auc_inf_obs`, `cl_f`, `cl_f_pred`, `auc_inf_dn`, and `auc_last` or
//!   `aumc_last` depending on the data) differ from numpy by one unit in the last place, a
//!   difference which the shipped 0.2.2 kernel already shows against the python core. The
//!   project's engine gate is `rel=1e-6, abs=1e-12`, ten orders of magnitude looser.
//!
//! The sets of names must agree exactly, so a column the kernel stops writing or starts writing
//! is a failure, and the flags are compared as integers.

use std::collections::BTreeSet;

use pkpdutils_abi::{AnalysisResult, BATCH_CONTRACT, Batch, NcaOptions, Route};
use pkpdutils_core::analyze_batch;
use serde::Deserialize;

/// The number of cases of the oracle, an independent literal so that a truncated file fails
/// instead of passing vacuously.
const CASES: usize = 8;

/// The relative tolerance of the columns of the single dose core, see the module comment.
const REL: f64 = 1e-12;

/// The steady state variables of `STEADY_STATE_VARIABLES` (`cdisc.py`) and the protocol count:
/// compared exactly. `interval_*` is exact as well and is recognized by its prefix.
const EXACT: &[&str] = &[
    "auc_tau",
    "auc_tau_dn",
    "auc_tau_extrap_fraction",
    "cmax_ss",
    "cmax_ss_dn",
    "cmin_ss",
    "ctrough",
    "cavg",
    "cavg_dn",
    "fluctuation",
    "fluctuation_tau",
    "swing",
    "swing_tau",
    "ptr",
    "tau",
    "cl_ss",
    "cl_ss_f",
    "accumulation_ratio",
    "accumulation_ratio_obs",
    "accumulation_ratio_cmax_obs",
    "accumulation_ratio_cmin_obs",
    "accumulation_ratio_ctrough_obs",
    "stationarity_ratio",
    "n_doses",
];

/// The cases the oracle must carry, in order.
const NAMES: [&str; CASES] = [
    "oral_regular",
    "oral_regular_no_intervals",
    "bolus_regular_trough_regression",
    "oral_irregular",
    "oral_completed_last_interval",
    "oral_incomplete_last_interval",
    "mixed_single_and_multiple",
    "single_dose_with_tau",
];

/// The oracle file.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Oracle {
    /// The commit of `pkpdutils` the numbers came from.
    pkpdutils_commit: String,
    /// The cases.
    cases: Vec<Case>,
}

/// The options of a case, the ones python's `NCAOptions` was given.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields, default)]
struct CaseOptions {
    /// `NCAOptions.tau`.
    tau: Option<f64>,
    /// `NCAOptions.intervals`.
    intervals: Option<bool>,
}

/// A number of the file: `null` is NaN, the text `inf` and `-inf` the infinities, which json
/// has no spelling for.
#[derive(Deserialize)]
#[serde(untagged)]
enum Cell {
    /// A finite number.
    Number(f64),
    /// `null`, NaN.
    Missing(()),
    /// `"inf"` or `"-inf"`.
    Infinite(String),
}

impl Cell {
    /// The value.
    fn value(&self) -> f64 {
        match self {
            Self::Number(x) => *x,
            Self::Missing(()) => f64::NAN,
            Self::Infinite(s) if s == "inf" => f64::INFINITY,
            Self::Infinite(s) if s == "-inf" => f64::NEG_INFINITY,
            Self::Infinite(s) => panic!("not a number of the oracle: {s}"),
        }
    }
}

/// One expected column.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Column {
    /// The unit python wrote.
    unit: String,
    /// The values, `null` for NaN.
    values: Vec<Cell>,
}

/// One expected variable over the dimension `interval`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PointColumn {
    /// The unit python wrote.
    unit: String,
    /// The extra dimension.
    dim: String,
    /// Its size.
    size: usize,
    /// The values, `null` for NaN.
    values: Vec<Cell>,
}

/// One case of the oracle.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    /// The name.
    name: String,
    /// What the case is there to catch.
    why: String,
    /// The route.
    route: Route,
    /// The times of every row.
    times: Vec<Vec<f64>>,
    /// The values of every row.
    values: Vec<Vec<f64>>,
    /// The dose amounts of every row, `null` padded.
    dose_amount: Vec<Vec<Option<f64>>>,
    /// The dose times of every row, `null` padded.
    dose_time: Vec<Vec<Option<f64>>>,
    /// The options.
    options: CaseOptions,
    /// The expected columns by name.
    parameters: std::collections::BTreeMap<String, Column>,
    /// The expected interval variables by name.
    point_variables: std::collections::BTreeMap<String, PointColumn>,
}

/// The oracle, read from the file.
fn oracle() -> Oracle {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/multiple_dose.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// `null` is NaN.
fn nan(x: &Option<f64>) -> f64 {
    x.unwrap_or(f64::NAN)
}

/// The batch of a case, as `Timecourses.from_arrays` laid it out.
fn batch_of(case: &Case) -> Batch {
    let n_rows = case.times.len();
    let n_time = case.times[0].len();
    let max_doses = case.dose_amount[0].len();
    let flat = |rows: &[Vec<f64>]| rows.iter().flatten().copied().collect::<Vec<f64>>();
    let doses = |rows: &[Vec<Option<f64>>]| rows.iter().flatten().map(nan).collect::<Vec<f64>>();
    Batch {
        contract: BATCH_CONTRACT.to_string(),
        sample_dims: vec!["individual".to_string()],
        sample_shape: vec![n_rows],
        n_time,
        times: flat(&case.times),
        values: flat(&case.values),
        sd: None,
        se: None,
        n: None,
        nominal_time: None,
        lloq: None,
        max_doses,
        dose_amount: doses(&case.dose_amount),
        dose_time: doses(&case.dose_time),
        dose_duration: vec![f64::NAN; n_rows * max_doses],
        time_unit: "hour".to_string(),
        unit: "mg/l".to_string(),
        dose_unit: Some("mg".to_string()),
        substance: Some("drug".to_string()),
        substances: None,
        route: Some(case.route),
        routes: None,
        tissue: None,
        coords: Vec::new(),
    }
}

/// The kernel's analysis of a case.
fn analysis_of(case: &Case) -> AnalysisResult {
    let options = NcaOptions {
        tau: case.options.tau,
        intervals: case.options.intervals.unwrap_or(true),
        ..NcaOptions::default()
    };
    analyze_batch(&batch_of(case), &options)
        .unwrap_or_else(|e| panic!("{}: the kernel refused the batch: {e}", case.name))
}

/// Whether a variable is compared bit for bit.
fn is_exact(name: &str) -> bool {
    name.starts_with("interval_") || EXACT.contains(&name)
}

/// Compare one value, exactly or within `REL`, NaN against NaN.
fn compare(what: &str, actual: f64, expected: f64, exact: bool) {
    if expected.is_nan() || actual.is_nan() {
        assert!(
            expected.is_nan() && actual.is_nan(),
            "{what}: {actual} against {expected}"
        );
    } else if exact {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{what}: {actual:e} is not bit for bit {expected:e}"
        );
    } else {
        assert!(
            actual == expected || (actual - expected).abs() <= REL * expected.abs(),
            "{what}: {actual:e} against {expected:e}, relative {:e}",
            ((actual - expected) / expected).abs()
        );
    }
}

#[test]
fn the_oracle_carries_its_cases_and_names_its_source() {
    let oracle = oracle();
    assert_eq!(oracle.pkpdutils_commit.len(), 40);
    let names: Vec<&str> = oracle.cases.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, NAMES);
    for case in &oracle.cases {
        assert!(!case.why.is_empty(), "{}", case.name);
    }
}

#[test]
fn every_multiple_dose_case_reproduces_the_python_core() {
    for case in &oracle().cases {
        let result = analysis_of(case);
        let n_rows = case.times.len();
        let name = &case.name;

        // the names: the kernel writes what python writes, neither more nor fewer
        let skip = ["accepted", "excluded", "flags"];
        let kernel: BTreeSet<&str> = result.parameters.iter().map(|p| p.name.as_str()).collect();
        let python: BTreeSet<&str> = case
            .parameters
            .keys()
            .map(String::as_str)
            .filter(|n| !skip.contains(n))
            .collect();
        assert_eq!(kernel, python, "{name}: the parameter names differ");
        let kernel: BTreeSet<&str> = result
            .point_variables
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        let python: BTreeSet<&str> = case.point_variables.keys().map(String::as_str).collect();
        assert_eq!(kernel, python, "{name}: the interval variable names differ");

        for column in &result.parameters {
            let want = &case.parameters[&column.name];
            assert_eq!(column.unit, want.unit, "{name}/{}: unit", column.name);
            assert_eq!(column.values.len(), n_rows);
            for (row, (a, e)) in column.values.iter().zip(&want.values).enumerate() {
                compare(
                    &format!("{name}/{}[{row}]", column.name),
                    *a,
                    e.value(),
                    is_exact(&column.name),
                );
            }
        }
        for variable in &result.point_variables {
            let want = &case.point_variables[&variable.name];
            let what = format!("{name}/{}", variable.name);
            assert_eq!(variable.unit, want.unit, "{what}: unit");
            assert_eq!(variable.dim, want.dim, "{what}: dim");
            assert_eq!(variable.size, want.size, "{what}: size");
            assert_eq!(variable.values.len(), want.values.len(), "{what}: length");
            for (i, (a, e)) in variable.values.iter().zip(&want.values).enumerate() {
                compare(&format!("{what}[{i}]"), *a, e.value(), true);
            }
        }

        // the flags are integers, and `accepted` and `excluded` are booleans
        let flags: Vec<f64> = result.flags.iter().map(|&f| f64::from(f)).collect();
        let want: Vec<f64> = case.parameters["flags"]
            .values
            .iter()
            .map(Cell::value)
            .collect();
        assert_eq!(flags, want, "{name}: flags");
        for column in &result.boolean_variables {
            let want: Vec<f64> = case.parameters[&column.name]
                .values
                .iter()
                .map(Cell::value)
                .collect();
            let got: Vec<f64> = column
                .values
                .iter()
                .map(|&b| f64::from(u8::from(b)))
                .collect();
            assert_eq!(got, want, "{name}/{}", column.name);
        }
    }
}
