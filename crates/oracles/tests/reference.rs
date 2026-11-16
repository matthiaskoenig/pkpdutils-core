//! The kernel reproduces the results of `pkdb_analysis` 0.3.1 on its own test data.
//!
//! The oracle is `testdata/oracles/tests/data/reference/nca_reference.json`, vendored from the
//! python package, and this is the port of its gate `tests/nca/test_reference.py`: nine single
//! dose oral curves with their data inline, the linear trapezoid rule and a terminal regression
//! over every point after the maximum, compared at a relative 1e-6 and an absolute 1e-12.
//!
//! Nothing here is skipped. A parameter the kernel does not produce, a case the file does not
//! carry and a number which differs are each a failure which names what was missing or by how
//! much it differed.

mod common;
// the unit helpers of the numeric gates, which `tests/oracles_present.rs` does not declare
#[path = "common/units.rs"]
mod units;

use std::collections::{BTreeMap, BTreeSet};

use pkpdutils_abi::{
    AnalysisResult, AucMethod, BATCH_CONTRACT, Batch, NcaOptions, Route, TerminalMethod,
    TerminalPhase,
};
use pkpdutils_core::analyze_batch;
use serde_json::{Map, Value};

use common::read_text;
use units::{in_unit, unit_of};

/// The number of cases of the oracle, an independent literal so that a truncated or mis-parsed
/// file fails here instead of passing vacuously.
const CASES: usize = 9;

/// The relative tolerance of `pytest.approx(expected, rel=1e-6, abs=1e-12)`.
const REL: f64 = 1e-6;

/// The absolute tolerance of the same, which is also the default `pytest.approx` applies when
/// only `rel` is given, as `test_reference.py:122-127` does for the regression.
const ABS: f64 = 1e-12;

/// The old name of a parameter of the oracle and the variable of the kernel it became,
/// `MAPPING` of `tests/nca/test_reference.py:23-32`.
const MAPPING: &[(&str, &str)] = &[
    ("auc", "auc_last"),
    ("aucinf", "auc_inf_obs"),
    ("tmax", "tmax"),
    ("cmax", "cmax"),
    ("kel", "lambda_z"),
    ("thalf", "thalf"),
    ("vd", "vz_f"),
    ("cl", "cl_f"),
];

/// The parameters the oracle carries which `MAPPING` does not name, and which the python gate
/// therefore does not compare either: `dose` is an input and not a result, and `tmaxhalf`,
/// `cmaxhalf` and `vdss` are definitions of `pkdb_analysis` which `pkpdutils` did not keep.
/// They are named here so that a parameter which appears in the oracle without being accounted
/// for fails the name contract.
const NOT_COMPARED: &[&str] = &["cmaxhalf", "dose", "tmaxhalf", "vdss"];

/// The keys of a case of the oracle, so that a file of another shape fails here.
const CASE_KEYS: &[&str] = &[
    "concentration",
    "dose",
    "dose_unit",
    "file",
    "name",
    "parameters",
    "regression",
    "substance",
    "time",
    "time_unit",
    "unit",
];

/// The keys of the `regression` of a case. The gate asserts the slope and the intercept, as
/// `test_reference.py:114-129` does; `r_value`, `p_value`, `std_err` and `max_idx` are read by
/// no gate and are listed so that a key which appears or disappears is noticed.
const REGRESSION_KEYS: &[&str] = &[
    "intercept",
    "intercept_unit",
    "max_idx",
    "p_value",
    "r_value",
    "slope",
    "slope_unit",
    "std_err",
];

/// The keys of one expected parameter.
const PARAMETER_KEYS: &[&str] = &["magnitude", "unit"];

/// The name of every case of the oracle.
///
/// The counts alone would not notice a file of nine cases in which one was replaced by a copy
/// of another with the same shape, so the gate pins which curves it ran on and does not lean on
/// the checksum of `oracles_present.rs` for that.
const NAMES: [&str; 9] = [
    "Benowitz1995_caffeine_2mgkg",
    "Benowitz1995_caffeine_4mgkg",
    "Benowitz1995_paraxanthine_2mgkg",
    "Benowitz1995_paraxanthine_4mgkg",
    "Divoll1982_Fig1",
    "Kim2011_Fig2",
    "midazolam",
    "nan_values",
    "synthetic_monoexp",
];

/// The two cases whose first sample comes after the dose, which the triangle correction of
/// [`expected_parameters`] applies to.
const CORRECTED: [&str; 2] = ["Divoll1982_Fig1", "nan_values"];

/// One expected parameter of a case: its magnitude in its own unit, `None` for a `null`.
struct Expected {
    /// The expected value, `None` where the oracle has `null` and the kernel must give `NaN`.
    magnitude: Option<f64>,
    /// The unit the magnitude is written in, which is not the canonical unit of the kernel.
    unit: String,
}

/// The terminal regression of a case, of which the gate asserts the slope and the intercept.
struct Regression {
    /// The slope of the regression of the logarithm, negative, `None` where there is no fit.
    slope: Option<f64>,
    /// The unit of the slope, the reciprocal of the time unit.
    slope_unit: String,
    /// The intercept on the natural logarithm scale, compared without a unit.
    intercept: Option<f64>,
}

/// One case of the oracle: the curve, its dose and the expected results.
struct Case {
    /// The name of the case.
    name: String,
    /// The unit of the times.
    time_unit: String,
    /// The unit of the concentrations.
    unit: String,
    /// The dose amount.
    dose: f64,
    /// The unit of the dose.
    dose_unit: String,
    /// The substance.
    substance: String,
    /// The observation times.
    time: Vec<f64>,
    /// The concentrations, `None` for a `null`, which is a `NaN`.
    concentration: Vec<Option<f64>>,
    /// The terminal regression.
    regression: Regression,
    /// The expected parameters by their old name.
    parameters: BTreeMap<String, Expected>,
}

/// The object a value is, naming what was read instead.
#[track_caller]
fn object<'a>(value: &'a Value, what: &str) -> &'a Map<String, Value> {
    value
        .as_object()
        .unwrap_or_else(|| panic!("{what} is no object but {value}"))
}

/// Assert that an object carries exactly the keys expected of it.
#[track_caller]
fn keys_are(map: &Map<String, Value>, expected: &[&str], what: &str) {
    let found: BTreeSet<&str> = map.keys().map(String::as_str).collect();
    let expected: BTreeSet<&str> = expected.iter().copied().collect();
    assert_eq!(found, expected, "the keys of {what}");
}

/// The value of a key, which the key check before it has shown to be there.
#[track_caller]
fn entry<'a>(map: &'a Map<String, Value>, key: &str, what: &str) -> &'a Value {
    map.get(key)
        .unwrap_or_else(|| panic!("{what} has no {key}"))
}

/// A string entry.
#[track_caller]
fn text(map: &Map<String, Value>, key: &str, what: &str) -> String {
    let value = entry(map, key, what);
    value
        .as_str()
        .unwrap_or_else(|| panic!("{what}: {key} is no string but {value}"))
        .to_string()
}

/// A number entry which may be `null`.
#[track_caller]
fn maybe_number(map: &Map<String, Value>, key: &str, what: &str) -> Option<f64> {
    match entry(map, key, what) {
        Value::Null => None,
        value => Some(
            value
                .as_f64()
                .unwrap_or_else(|| panic!("{what}: {key} is no number but {value}")),
        ),
    }
}

/// A number entry which must be a number.
#[track_caller]
fn number(map: &Map<String, Value>, key: &str, what: &str) -> f64 {
    maybe_number(map, key, what).unwrap_or_else(|| panic!("{what}: {key} is null"))
}

/// An array entry of numbers, none of which may be `null`.
#[track_caller]
fn numbers(map: &Map<String, Value>, key: &str, what: &str) -> Vec<f64> {
    maybe_numbers(map, key, what)
        .into_iter()
        .enumerate()
        .map(|(i, value)| value.unwrap_or_else(|| panic!("{what}: {key}[{i}] is null")))
        .collect()
}

/// An array entry of numbers, a `null` of which is a missing value.
#[track_caller]
fn maybe_numbers(map: &Map<String, Value>, key: &str, what: &str) -> Vec<Option<f64>> {
    let value = entry(map, key, what);
    let array = value
        .as_array()
        .unwrap_or_else(|| panic!("{what}: {key} is no array but {value}"));
    array
        .iter()
        .enumerate()
        .map(|(i, item)| match item {
            Value::Null => None,
            item => Some(
                item.as_f64()
                    .unwrap_or_else(|| panic!("{what}: {key}[{i}] is no number but {item}")),
            ),
        })
        .collect()
}

/// The nine cases of the oracle, with every key of the file accounted for.
fn cases() -> Vec<Case> {
    let text_of_file = read_text("tests/data/reference/nca_reference.json");
    let document: Value = serde_json::from_str(&text_of_file).expect("the oracle is JSON");
    let array = document.as_array().expect("the oracle is a list of cases");
    let cases: Vec<Case> = array.iter().map(case_of).collect();
    assert_eq!(NAMES.len(), CASES, "the two literals of the gate disagree");
    let found: BTreeSet<&str> = cases.iter().map(|case| case.name.as_str()).collect();
    assert_eq!(
        found,
        NAMES.into_iter().collect::<BTreeSet<&str>>(),
        "the oracle does not carry the curves this gate was written against"
    );
    cases
}

/// One case of the oracle.
#[track_caller]
fn case_of(value: &Value) -> Case {
    let map = object(value, "a case");
    let name = text(map, "name", "a case");
    keys_are(map, CASE_KEYS, &name);
    let time = numbers(map, "time", &name);
    let concentration = maybe_numbers(map, "concentration", &name);
    assert_eq!(
        time.len(),
        concentration.len(),
        "{name}: the times and the concentrations are of different length"
    );
    Case {
        time_unit: text(map, "time_unit", &name),
        unit: text(map, "unit", &name),
        dose: number(map, "dose", &name),
        dose_unit: text(map, "dose_unit", &name),
        substance: text(map, "substance", &name),
        time,
        concentration,
        regression: regression_of(entry(map, "regression", &name), &name),
        parameters: parameters_of(entry(map, "parameters", &name), &name),
        name,
    }
}

/// The terminal regression of a case.
#[track_caller]
fn regression_of(value: &Value, name: &str) -> Regression {
    let what = format!("{name}: the regression");
    let map = object(value, &what);
    keys_are(map, REGRESSION_KEYS, &what);
    Regression {
        slope: maybe_number(map, "slope", &what),
        slope_unit: text(map, "slope_unit", &what),
        intercept: maybe_number(map, "intercept", &what),
    }
}

/// The expected parameters of a case, by their old name.
#[track_caller]
fn parameters_of(value: &Value, name: &str) -> BTreeMap<String, Expected> {
    let map = object(value, &format!("{name}: the parameters"));
    map.iter()
        .map(|(parameter, value)| {
            let what = format!("{name}: {parameter}");
            let entry = object(value, &what);
            keys_are(entry, PARAMETER_KEYS, &what);
            (
                parameter.clone(),
                Expected {
                    magnitude: maybe_number(entry, "magnitude", &what),
                    unit: text(entry, "unit", &what),
                },
            )
        })
        .collect()
}

/// The options of the python gate: the linear trapezoid rule and a terminal regression over
/// every point after the maximum (`test_reference.py:34-37`), everything else as it comes.
fn options() -> NcaOptions {
    NcaOptions {
        auc_method: AucMethod::Linear,
        terminal: TerminalPhase {
            method: TerminalMethod::AllAfterTmax,
            ..TerminalPhase::default()
        },
        ..NcaOptions::default()
    }
}

/// The curve of a case as a batch of one sample.
///
/// `to_timecourse` of `test_reference.py:40-49` builds one `Timecourse` with a single oral
/// `Dose`, whose time is 0 and which has no duration, and `nca_single` wraps it in a batch of
/// one sample along the dimension `_single` (`nca.py:1946-1962`).
fn batch_of(case: &Case) -> Batch {
    let values = case
        .concentration
        .iter()
        .map(|value| value.unwrap_or(f64::NAN))
        .collect();
    Batch {
        contract: BATCH_CONTRACT.to_string(),
        sample_dims: vec!["_single".to_string()],
        sample_shape: vec![1],
        n_time: case.time.len(),
        times: case.time.clone(),
        values,
        sd: None,
        se: None,
        n: None,
        nominal_time: None,
        lloq: None,
        max_doses: 1,
        dose_amount: vec![case.dose],
        dose_time: vec![0.0],
        dose_duration: vec![f64::NAN],
        time_unit: case.time_unit.clone(),
        unit: case.unit.clone(),
        dose_unit: Some(case.dose_unit.clone()),
        substance: Some(case.substance.clone()),
        substances: None,
        route: Some(Route::Oral),
        routes: None,
        tissue: None,
        coords: Vec::new(),
    }
}

/// The analysis of one case with the options of the python gate.
fn analysis_of(case: &Case) -> AnalysisResult {
    analyze_batch(&batch_of(case), &options())
        .unwrap_or_else(|e| panic!("{}: the kernel refused the batch: {e}", case.name))
}

/// The expected parameters of a case, with the area before its first sample.
///
/// The port of `expected_parameters` of `tests/nca/test_reference.py:52-81`, with one
/// deliberate difference: the `auc` entry of the oracle is looked up before the `t0 <= 0`
/// return, where python's `or` short-circuits past it, so an oracle without an `auc` key fails
/// here for a case python would have returned early from. Stricter, never looser.
/// `pkdb_analysis` 0.3.1 started the area at the first sample, where `pkpdutils` inserts the
/// value at the dose when the first sample comes later - 0 for an extravascular dose, as
/// Phoenix WinNonlin does. The oracle predates that rule, so the expected area gains the
/// triangle `0.5 * t0 * c0` from the dose to the first sample it has, and the clearance and the
/// volume, which divide by the area to infinity, shrink by `aucinf / (aucinf + triangle)`.
///
/// It touches four of the eight compared parameters, `auc`, `aucinf`, `cl` and `vd`, and only
/// the two cases of [`CORRECTED`]: `nan_values`, whose first sample is at 1 hour, and
/// `Divoll1982_Fig1`, whose sample at 0.083 hours is null so that its first value is the one at
/// 0.167 hours. Every other case starts at the dose and is compared as the oracle wrote it.
///
/// The second return value says whether the correction fired.
fn expected_parameters(case: &Case) -> (BTreeMap<String, Expected>, bool) {
    let mut parameters: BTreeMap<String, Expected> = case
        .parameters
        .iter()
        .map(|(name, value)| {
            (
                name.clone(),
                Expected {
                    magnitude: value.magnitude,
                    unit: value.unit.clone(),
                },
            )
        })
        .collect();
    let (t0, c0) = case
        .time
        .iter()
        .zip(&case.concentration)
        .find_map(|(time, value)| value.map(|value| (*time, value)))
        .unwrap_or_else(|| panic!("{}: every sample of the curve is null", case.name));
    let auc = expectation(&parameters, "auc", &case.name);
    let auc_unit = auc.unit.clone();
    let auc_magnitude = auc.magnitude;
    // `if t0 <= 0 or parameters["auc"]["magnitude"] is None: return parameters`
    if t0 <= 0.0 {
        return (parameters, false);
    }
    let Some(auc_magnitude) = auc_magnitude else {
        return (parameters, false);
    };
    let what = format!("{}: the triangle under the first sample", case.name);
    let product = unit_of(&case.time_unit, &what)
        .times(&unit_of(&case.unit, &what))
        .unwrap_or_else(|| panic!("{what}: the units do not multiply"));
    let target = unit_of(&auc_unit, &what);
    assert_eq!(
        product.dimension(),
        target.dimension(),
        "{what}: a time times a concentration is not of the dimension of {auc_unit}"
    );
    let triangle = 0.5 * t0 * c0 * product.factor() / target.factor();
    set_magnitude(&mut parameters, "auc", auc_magnitude + triangle, &case.name);
    let aucinf = expectation(&parameters, "aucinf", &case.name);
    assert_eq!(
        aucinf.unit, auc_unit,
        "{}: the triangle is converted into the unit of auc and added to aucinf as well, which \
         holds only while the two share a unit",
        case.name
    );
    let Some(aucinf) = aucinf.magnitude else {
        return (parameters, true);
    };
    set_magnitude(&mut parameters, "aucinf", aucinf + triangle, &case.name);
    let factor = aucinf / (aucinf + triangle);
    for name in ["cl", "vd"] {
        let magnitude = expectation(&parameters, name, &case.name)
            .magnitude
            .unwrap_or_else(|| {
                panic!("{}: {name} has no value although aucinf has one", case.name)
            });
        set_magnitude(&mut parameters, name, magnitude * factor, &case.name);
    }
    (parameters, true)
}

/// One expected parameter, naming the case and the parameter when the oracle has none.
#[track_caller]
fn expectation<'a>(
    parameters: &'a BTreeMap<String, Expected>,
    name: &str,
    case: &str,
) -> &'a Expected {
    parameters
        .get(name)
        .unwrap_or_else(|| panic!("{case}: the oracle has no parameter {name}"))
}

/// Overwrite the magnitude of an expected parameter, keeping its unit.
#[track_caller]
fn set_magnitude(
    parameters: &mut BTreeMap<String, Expected>,
    name: &str,
    magnitude: f64,
    case: &str,
) {
    parameters
        .get_mut(name)
        .unwrap_or_else(|| panic!("{case}: the oracle has no parameter {name}"))
        .magnitude = Some(magnitude);
}

/// One parameter of the single row of a result, converted into `unit`.
///
/// A parameter the kernel does not produce is a failure which names it and lists what the
/// kernel did produce, never a skip.
#[track_caller]
fn converted(result: &AnalysisResult, name: &str, case: &str, unit: &str) -> f64 {
    let column = result
        .parameters
        .iter()
        .find(|column| column.name == name)
        .unwrap_or_else(|| {
            let produced: Vec<&str> = result
                .parameters
                .iter()
                .map(|column| column.name.as_str())
                .collect();
            panic!("{case}: the kernel produces no parameter {name}; it produces {produced:?}")
        });
    assert_eq!(
        column.values.len(),
        1,
        "{case}: {name} is not one value of one sample"
    );
    in_unit(
        column.values[0],
        &column.unit,
        unit,
        &format!("{case}: {name}"),
    )
}

/// Assert that two numbers agree to [`REL`] and [`ABS`], naming both and their difference.
///
/// This is `pytest.approx(expected, rel=REL, abs=ABS)`: the difference must not exceed the
/// larger of the two tolerances. A `NaN` on either side makes the difference a `NaN`, which no
/// comparison accepts, so a kernel which answers a number of the reference with a `NaN` fails
/// here rather than passing.
#[track_caller]
fn expect_close(actual: f64, expected: f64, what: &str) {
    let tolerance = (REL * expected.abs()).max(ABS);
    let difference = (actual - expected).abs();
    assert!(
        difference <= tolerance,
        "{what}: the kernel gives {actual:?} where the reference has {expected:?}, a difference \
         of {difference:?} (relative {:?}) above the tolerance {tolerance:?}",
        difference / expected.abs()
    );
}

#[test]
fn every_reference_case_reproduces_the_pkdb_analysis_results() {
    let cases = cases();
    assert_eq!(cases.len(), CASES, "the oracle must carry {CASES} cases");
    let mut compared = 0usize;
    let mut required_nan = 0usize;
    let mut corrected: BTreeSet<String> = BTreeSet::new();
    for case in &cases {
        let result = analysis_of(case);
        let (expectations, triangle) = expected_parameters(case);
        if triangle {
            corrected.insert(case.name.clone());
        }
        for (old, new) in MAPPING {
            let expected = expectation(&expectations, old, &case.name);
            let what = format!("{}: {new} (the reference {old})", case.name);
            let actual = converted(&result, new, &case.name, &expected.unit);
            match expected.magnitude {
                None => {
                    assert!(
                        actual.is_nan(),
                        "{what}: the reference has no value, so the kernel must give NaN, and it \
                         gives {actual:?}"
                    );
                    required_nan += 1;
                }
                Some(magnitude) => {
                    expect_close(actual, magnitude, &what);
                    compared += 1;
                }
            }
        }
    }
    // eight parameters of nine cases, of which the two paraxanthine curves have no terminal
    // phase and so no aucinf, kel, thalf, vd and cl
    assert_eq!(
        compared + required_nan,
        CASES * MAPPING.len(),
        "every case must contribute one comparison per mapped parameter"
    );
    assert_eq!(
        required_nan, 10,
        "the aucinf, kel, thalf, vd and cl of the two paraxanthine curves are the null \
         magnitudes of the oracle, and nothing else is"
    );
    assert_eq!(
        compared, 62,
        "the magnitudes of the oracle which are numbers and were compared as numbers"
    );
    assert_eq!(
        corrected,
        CORRECTED
            .map(str::to_string)
            .into_iter()
            .collect::<BTreeSet<String>>()
    );
}

#[test]
fn the_reference_case_parameter_names_are_the_columns_this_kernel_produces() {
    let cases = cases();
    assert_eq!(cases.len(), CASES, "the oracle must carry {CASES} cases");
    let accounted: BTreeSet<&str> = MAPPING
        .iter()
        .map(|(old, _)| *old)
        .chain(NOT_COMPARED.iter().copied())
        .collect();
    assert_eq!(
        accounted.len(),
        MAPPING.len() + NOT_COMPARED.len(),
        "a name is both mapped and declared as not compared"
    );
    for case in &cases {
        let present: BTreeSet<&str> = case.parameters.keys().map(String::as_str).collect();
        assert_eq!(
            present, accounted,
            "{}: the parameters of the oracle are not the ones this gate accounts for",
            case.name
        );
        let produced: Vec<String> = analysis_of(case)
            .parameters
            .iter()
            .map(|column| column.name.clone())
            .collect();
        for (old, new) in MAPPING {
            assert!(
                produced.iter().any(|name| name == new),
                "{}: the kernel produces no parameter {new}, the reference {old}; it produces \
                 {produced:?}",
                case.name
            );
        }
    }
}

#[test]
fn the_terminal_regression_matches_the_reference_slope_and_intercept() {
    let cases = cases();
    assert_eq!(cases.len(), CASES, "the oracle must carry {CASES} cases");
    let mut slopes = 0usize;
    let mut intercepts = 0usize;
    let mut without_a_fit = 0usize;
    for case in &cases {
        let result = analysis_of(case);
        let lambda_z = converted(&result, "lambda_z", &case.name, &case.regression.slope_unit);
        let Some(slope) = case.regression.slope else {
            assert!(
                lambda_z.is_nan(),
                "{}: the reference has no regression, so lambda_z must be NaN, and it is \
                 {lambda_z:?}",
                case.name
            );
            without_a_fit += 1;
            continue;
        };
        expect_close(-lambda_z, slope, &format!("{}: the slope", case.name));
        slopes += 1;
        let intercept = case
            .regression
            .intercept
            .unwrap_or_else(|| panic!("{}: a regression with a slope and no intercept", case.name));
        // the intercept is on the natural logarithm scale and the kernel reports it as
        // dimensionless, so it is compared as it stands, as `test_reference.py:125-127` does
        let actual = converted(&result, "lambda_z_intercept", &case.name, "dimensionless");
        expect_close(actual, intercept, &format!("{}: the intercept", case.name));
        intercepts += 1;
    }
    assert_eq!(
        slopes, 7,
        "seven of the nine cases have a terminal regression with a slope"
    );
    assert_eq!(
        intercepts, 7,
        "and an intercept, compared on the natural logarithm scale"
    );
    assert_eq!(
        without_a_fit, 2,
        "the two paraxanthine curves have no terminal regression at all"
    );
}
