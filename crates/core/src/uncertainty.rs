//! The uncertainty of the parameters, the reduction of the bootstrap replicates.
//!
//! The port of `pkpdutils.nca.uncertainty.reduce_replicates`. The replicates of a parameter are
//! the parameter of every resampled curve of a row; this module turns them into the variables a
//! result carries beside the parameter itself: its spread, its interval and, for a parameter
//! whose distribution is log-normal, its geometric statistics.
//!
//! The two scales are the whole of the design and are easy to confuse. A point of a group curve
//! can be resampled with its standard error, in which case the replicates are replicates of the
//! **mean** curve and their spread is the standard error of the parameter, or with its standard
//! deviation, in which case the replicates are individual curves and their spread is the
//! deviation over subjects. Each is converted into the other by `se = sd / sqrt(n)`, which needs
//! the subject count: without it the other scale is unknown and is `NaN`, never a silent
//! identity. `x_ci_low`/`x_ci_high` are an interval of the estimate on both paths, which is why
//! they are the percentiles of the replicates under `se` draws and the normal approximation
//! around the estimate under `sd` draws, where the percentiles of the replicates bound the
//! individuals instead and are reported as `x_pi_low`/`x_pi_high`.
//!
//! The classification of the parameters lives here rather than on the wire. The kernel produces
//! the parameters, so it is the one place which can say what their distribution is, and a host in
//! the browser has no python to ask. The copy is held to python's own sets by
//! `the_classification_is_pythons`, which reads them out of the committed oracle
//! (`scripts/reduce_oracle.py`), so a set which changes upstream fails `cargo test` here without
//! a `pkpdutils` anywhere near it.

// the bootstrap of `analyze_batch` is the only caller of this module and arrives with the next
// commit of this branch; until it does, every item here is reached by the tests alone
#![allow(dead_code)]

use pkpdutils_abi::{BootstrapSpread, ParameterColumn};

use crate::percentile::nan_percentile_rows;
use crate::stats::normal_isf;
use crate::summarize::{mean_of, variance_of};

/// The replicates of one parameter, `(n_rows, n_boot)` row major.
///
/// The row order is the row order of the batch and the replicate order inside a row is the order
/// the draws were generated in ([`crate::random`]), which is what makes a reduction independent
/// of how the rows were cut into blocks.
#[derive(Debug, Clone)]
pub(crate) struct Replicates {
    /// The name of the parameter.
    pub name: String,
    /// `n_rows * n_boot` values, row major; a column of another length is reduced to `NaN`.
    pub values: Vec<f64>,
}

pub(crate) const LOGNORMAL_PARAMETERS: &[&str] = &[
    "accumulation_ratio_cmax_obs",
    "accumulation_ratio_cmin_obs",
    "accumulation_ratio_ctrough_obs",
    "accumulation_ratio_obs",
    "auc_all",
    "auc_all_dn",
    "auc_inf_dn",
    "auc_inf_obs",
    "auc_inf_pred",
    "auc_inf_pred_dn",
    "auc_last",
    "auc_last_dn",
    "auc_tau",
    "auc_tau_dn",
    "auec_tau",
    "aumc_all",
    "aumc_inf",
    "aumc_inf_pred",
    "aumc_last",
    "c0",
    "c0_dn",
    "cavg",
    "cavg_dn",
    "cl",
    "cl_f",
    "cl_f_pred",
    "cl_pred",
    "cl_ss",
    "cl_ss_f",
    "clast",
    "clast_pred",
    "cmax",
    "cmax_dn",
    "cmax_ss",
    "cmax_ss_dn",
    "cmin",
    "cmin_ss",
    "ctrough",
    "eavg",
    "lambda_z",
    "mrt",
    "mrt_last",
    "mrt_pred",
    "ptr",
    "thalf",
    "thalf_eff",
    "vss",
    "vss_pred",
    "vz",
    "vz_f",
    "vz_f_pred",
    "vz_pred",
];

pub(crate) const DISCRETE_PARAMETERS: &[&str] = &[
    "auc_last_df",
    "c0_method",
    "flags",
    "interval_dose",
    "interval_end",
    "interval_n_points",
    "interval_start",
    "interval_temax",
    "interval_tmax",
    "lambda_z_intercept",
    "lambda_z_n_points",
    "lambda_z_r2",
    "lambda_z_r2_adj",
    "lambda_z_span",
    "lambda_z_stderr",
    "lambda_z_t_first",
    "lambda_z_t_last",
    "n",
    "n_doses",
    "tau",
    "temax",
    "tlag",
    "tlast",
    "tmax",
    "tmax_half",
    "tmin",
];

/// The suffixes which make a name a variable derived from a parameter rather than a parameter.
///
/// `pkpdutils.result.base_name` reads them, and this is its `is not None`: a name which ends in
/// one of them belongs to a parameter and has no uncertainty of its own. The list is python's
/// `UNCERTAINTY_SUFFIXES` and `SUMMARY_SUFFIXES` together, and the order inside it does not
/// matter here, since only the answer and not the parameter is wanted.
const DERIVED_SUFFIXES: &[&str] = &[
    "_sd", "_se", "_ci_low", "_ci_high", "_pi_low", "_pi_high", "_geomean", "_geocv", "_cv",
    "_median", "_q25", "_q75", "_min", "_max", "_n",
];

/// The unit of a coefficient of variation, which is a ratio.
const DIMENSIONLESS: &str = "dimensionless";

/// Whether `name` is in the sorted `list`.
fn listed(list: &[&str], name: &str) -> bool {
    list.binary_search(&name).is_ok()
}

/// Whether the replicates of `name` are log-normal, so that it carries geometric statistics.
pub(crate) fn is_lognormal(name: &str) -> bool {
    listed(LOGNORMAL_PARAMETERS, name)
}

/// Whether `name` is discrete, so that a spread of it is not a quantity.
pub(crate) fn is_discrete(name: &str) -> bool {
    listed(DISCRETE_PARAMETERS, name)
}

/// Whether `name` is itself derived from a parameter, python's `base_name(name) is not None`.
fn is_derived(name: &str) -> bool {
    DERIVED_SUFFIXES
        .iter()
        .any(|suffix| name.len() > suffix.len() && name.ends_with(suffix))
}

/// The statistics of one row of replicates, before the validity mask.
struct Row {
    /// The finite replicates; an infinity is a hole, as `numpy.isfinite` has it.
    count: usize,
    /// The standard deviation of the finite replicates, `ddof = 1`.
    std: f64,
    /// The mean of the logarithms of the positive replicates.
    log_mean: f64,
    /// The variance of those logarithms, `ddof = 1`.
    log_var: f64,
    /// How many replicates are positive, the count the log statistics are guarded by.
    n_positive: usize,
}

/// The statistics of one row, both scales at once, since a second pass costs another sum.
fn reduce_row(row: &[f64], lognormal: bool) -> Row {
    let mut zeroed = Vec::with_capacity(row.len());
    let mut count = 0usize;
    for &value in row {
        if value.is_finite() {
            zeroed.push(value);
            count += 1;
        } else {
            zeroed.push(0.0);
        }
    }
    let std = variance_of(&zeroed, |i| row[i].is_finite(), count).sqrt();
    if !lognormal {
        return Row {
            count,
            std,
            log_mean: f64::NAN,
            log_var: f64::NAN,
            n_positive: 0,
        };
    }
    let mut logs = Vec::with_capacity(row.len());
    let mut n_positive = 0usize;
    for &value in row {
        if value.is_finite() && value > 0.0 {
            logs.push(value.ln());
            n_positive += 1;
        } else {
            logs.push(0.0);
        }
    }
    let positive = |i: usize| row[i].is_finite() && row[i] > 0.0;
    Row {
        count,
        std,
        log_mean: mean_of(&logs, n_positive),
        log_var: variance_of(&logs, positive, n_positive),
        n_positive,
    }
}

/// One column of the answer: `name` with the unit of its parameter, `NaN` where it is not defined.
fn column(name: &str, suffix: &str, unit: &str, values: Vec<f64>) -> ParameterColumn {
    ParameterColumn {
        name: format!("{name}{suffix}"),
        unit: unit.to_owned(),
        values,
    }
}

/// Reduce the replicates of every continuous parameter to its uncertainty variables.
///
/// The variables are `x_sd`, `x_se`, `x_ci_low` and `x_ci_high` for every parameter which is
/// neither discrete nor derived and has an estimate, `x_pi_low` and `x_pi_high` under `sd` draws
/// and `x_geomean` and `x_geocv` for a log-normal parameter. A parameter which is skipped carries
/// no variable at all, and a row which carries no uncertainty - no usable spread, fewer than two
/// finite replicates, or no finite estimate - is `NaN` in every variable of that parameter and is
/// never an error.
///
/// `x_geocv` is the geometric coefficient of variation over subjects on both paths,
/// `sqrt(exp(var(ln x)) - 1)` [^efron]: under `sd` draws the log variance of the replicates is
/// that variance already, under `se` draws it is the log variance of the mean and is scaled by
/// the subject count first, so it is `NaN` without one.
///
/// # Arguments
///
/// * `replicates` - the replicates of every parameter, `(n_rows, n_boot)` row major
/// * `point` - the parameters of the original curves, the estimates the intervals are built on
/// * `spread_kind` - whether the points were resampled with `se` or with `sd`
/// * `n_subjects` - the subjects of every row, `None` when the batch does not carry them and
///   `NaN` for a row without one
/// * `ci_level` - the level of the interval, strictly between 0 and 1
/// * `usable_rows` - the rows which carried a usable spread, one flag per row; its length is the
///   row count of the reduction
///
/// [^efron]: Efron & Tibshirani 1993, ch. 13.
pub(crate) fn reduce_replicates(
    replicates: &[Replicates],
    point: &[ParameterColumn],
    spread_kind: BootstrapSpread,
    n_subjects: Option<&[f64]>,
    ci_level: f64,
    usable_rows: &[bool],
) -> Vec<ParameterColumn> {
    let n_rows = usable_rows.len();
    let alpha = 1.0 - ci_level;
    let z = normal_isf(alpha / 2.0);
    let (q_low, q_high) = (100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0));
    let mut out = Vec::new();
    if n_rows == 0 {
        return out;
    }
    for reps in replicates {
        let name = reps.name.as_str();
        if is_discrete(name) || is_derived(name) {
            continue;
        }
        let Some(estimates) = point.iter().find(|parameter| parameter.name == name) else {
            continue;
        };
        let lognormal = is_lognormal(name);
        let unit = estimates.unit.as_str();
        // a column of another length is a bug of the caller and not of the request: it is
        // reduced to `NaN` rather than cutting the batch short or panicking
        let n_boot = if reps.values.len() % n_rows == 0 {
            reps.values.len() / n_rows
        } else {
            debug_assert!(
                false,
                "{name}: {} values for {n_rows} rows",
                reps.values.len()
            );
            0
        };
        let filled: Vec<f64> = reps
            .values
            .iter()
            .map(|&x| if x.is_finite() { x } else { f64::NAN })
            .collect();
        let (mut low, mut high) = if n_boot == 0 {
            (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows])
        } else {
            (
                nan_percentile_rows(&filled, n_boot, q_low),
                nan_percentile_rows(&filled, n_boot, q_high),
            )
        };
        let mut sd = vec![f64::NAN; n_rows];
        let mut se = vec![f64::NAN; n_rows];
        let mut geomean = vec![f64::NAN; n_rows];
        let mut geocv = vec![f64::NAN; n_rows];
        let mut percentile_low = vec![f64::NAN; n_rows];
        let mut percentile_high = vec![f64::NAN; n_rows];
        for r in 0..n_rows {
            let row = if n_boot == 0 {
                &[][..]
            } else {
                &reps.values[r * n_boot..(r + 1) * n_boot]
            };
            let stats = reduce_row(row, lognormal);
            let n = n_subjects.map(|values| values[r]);
            let sqrt_n = n.map(f64::sqrt);
            let (row_sd, row_se) = match spread_kind {
                BootstrapSpread::Se => {
                    (sqrt_n.map_or(f64::NAN, |root| stats.std * root), stats.std)
                }
                BootstrapSpread::Sd => {
                    (stats.std, sqrt_n.map_or(f64::NAN, |root| stats.std / root))
                }
            };
            let estimate = estimates.values[r];
            let valid = estimate.is_finite() && stats.count > 1 && usable_rows[r];
            if spread_kind == BootstrapSpread::Sd {
                // the replicates are individual curves: their percentiles bound the individuals
                // and the interval of the estimate is the normal approximation around it, on the
                // log scale for a log-normal parameter
                percentile_low[r] = if valid { low[r] } else { f64::NAN };
                percentile_high[r] = if valid { high[r] } else { f64::NAN };
                if lognormal {
                    let relative = row_se / estimate;
                    low[r] = estimate * (-z * relative).exp();
                    high[r] = estimate * (z * relative).exp();
                } else {
                    low[r] = estimate - z * row_se;
                    high[r] = estimate + z * row_se;
                }
            }
            if !valid {
                low[r] = f64::NAN;
                high[r] = f64::NAN;
                continue;
            }
            sd[r] = row_sd;
            se[r] = row_se;
            if lognormal {
                let log_var = match spread_kind {
                    // the log variance of the mean curve, scaled to the between subject variance
                    BootstrapSpread::Se => n.map_or(f64::NAN, |count| stats.log_var * count),
                    BootstrapSpread::Sd => stats.log_var,
                };
                if stats.n_positive > 1 {
                    geomean[r] = stats.log_mean.exp();
                    geocv[r] = log_var.exp_m1().sqrt();
                }
            }
        }
        if spread_kind == BootstrapSpread::Sd {
            out.push(column(name, "_pi_low", unit, percentile_low));
            out.push(column(name, "_pi_high", unit, percentile_high));
        }
        out.push(column(name, "_sd", unit, sd));
        out.push(column(name, "_se", unit, se));
        out.push(column(name, "_ci_low", unit, low));
        out.push(column(name, "_ci_high", unit, high));
        if lognormal {
            out.push(column(name, "_geomean", unit, geomean));
            out.push(column(name, "_geocv", DIMENSIONLESS, geocv));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde::Deserialize;

    use super::*;
    use crate::parameters::PARAMETERS;

    /// One array of the fixture, in the encoding of the wire: a number, or `"NaN"`.
    #[derive(Deserialize)]
    struct Values(#[serde(with = "pkpdutils_abi::nonfinite::vec")] Vec<f64>);

    /// One case of the fixture: its inputs and python's answer for them.
    #[derive(Deserialize)]
    struct Case {
        name: String,
        n_rows: usize,
        n_boot: usize,
        spread_kind: BootstrapSpread,
        ci_level: f64,
        n_subjects: Option<Values>,
        usable_rows: Vec<bool>,
        replicates: BTreeMap<String, Values>,
        point: BTreeMap<String, Values>,
        expected: BTreeMap<String, Values>,
    }

    /// The whole fixture, `crates/core/tests/data/reduce_oracle.json`.
    #[derive(Deserialize)]
    struct Oracle {
        lognormal: Vec<String>,
        discrete: Vec<String>,
        cases: Vec<Case>,
    }

    fn oracle() -> Oracle {
        serde_json::from_str(include_str!("../tests/data/reduce_oracle.json"))
            .expect("the oracle fixture is readable")
    }

    /// The unit of the fixture's parameters; the reduction copies it and never reads it.
    const UNIT: &str = "hour * milligram / liter";

    fn inputs(case: &Case) -> (Vec<Replicates>, Vec<ParameterColumn>) {
        let replicates = case
            .replicates
            .iter()
            .map(|(name, values)| Replicates {
                name: name.clone(),
                values: values.0.clone(),
            })
            .collect();
        let point = case
            .point
            .iter()
            .map(|(name, values)| ParameterColumn {
                name: name.clone(),
                unit: UNIT.to_owned(),
                values: values.0.clone(),
            })
            .collect();
        (replicates, point)
    }

    #[test]
    fn the_reduction_matches_the_python_oracle() {
        let oracle = oracle();
        assert_eq!(oracle.cases.len(), 6, "the fixture lost a case");
        for case in &oracle.cases {
            let (replicates, point) = inputs(case);
            let answer = reduce_replicates(
                &replicates,
                &point,
                case.spread_kind,
                case.n_subjects.as_ref().map(|values| values.0.as_slice()),
                case.ci_level,
                &case.usable_rows,
            );
            let got: BTreeMap<&str, &[f64]> = answer
                .iter()
                .map(|column| (column.name.as_str(), column.values.as_slice()))
                .collect();
            assert_eq!(answer.len(), got.len(), "{}: a variable twice", case.name);
            let names: Vec<&str> = got.keys().copied().collect();
            let wanted: Vec<&str> = case.expected.keys().map(String::as_str).collect();
            assert_eq!(names, wanted, "{}: the variables differ", case.name);
            for (name, expected) in &case.expected {
                let values = got[name.as_str()];
                assert_eq!(values.len(), case.n_rows, "{}: {name}", case.name);
                for (row, (&value, &want)) in values.iter().zip(&expected.0).enumerate() {
                    let where_ = format!("{} {name} row {row}", case.name);
                    if want.is_nan() {
                        assert!(value.is_nan(), "{where_}: {value} is no NaN");
                        continue;
                    }
                    assert!(value.is_finite(), "{where_}: {value} for {want}");
                    let tolerance = 1e-12 * want.abs();
                    assert!(
                        (value - want).abs() <= tolerance,
                        "{where_}: {value} against {want}"
                    );
                }
            }
            // the fixture really exercises the shapes it was built for
            assert!(case.n_boot > 1 && case.n_rows > 1, "{}", case.name);
        }
    }

    #[test]
    fn the_classification_is_pythons() {
        let oracle = oracle();
        assert_eq!(LOGNORMAL_PARAMETERS, oracle.lognormal, "log-normal");
        assert_eq!(DISCRETE_PARAMETERS, oracle.discrete, "discrete");
        for list in [LOGNORMAL_PARAMETERS, DISCRETE_PARAMETERS] {
            assert!(list.windows(2).all(|pair| pair[0] < pair[1]), "{list:?}");
        }
        // every parameter the kernel reports is classified at most one way, and the two lists
        // are reached by binary search, which the sortedness above makes valid
        for parameter in PARAMETERS {
            assert!(
                !(is_lognormal(parameter.name) && is_discrete(parameter.name)),
                "{} is both",
                parameter.name
            );
            assert!(
                !is_derived(parameter.name),
                "{} reads as a derived variable",
                parameter.name
            );
        }
        // a name of the lists is reached by the lookup it is written for
        assert!(is_lognormal("auc_inf_obs") && !is_lognormal("tmax"));
        assert!(is_discrete("tmax") && !is_discrete("auc_inf_obs"));
        assert!(is_derived("auc_inf_obs_se") && is_derived("auc_inf_obs_geocv"));
        assert!(!is_derived("auc_inf_obs"), "a parameter is not derived");
        // `_n` is a suffix, but a variable called `n` is the subject count itself
        assert!(!is_derived("_n") && !is_derived("n"));
    }

    /// A batch of one row, `n_boot` replicates of one log-normal parameter and one plain one.
    fn one_row(values: &[f64], estimate: f64) -> (Vec<Replicates>, Vec<ParameterColumn>) {
        (
            vec![
                Replicates {
                    name: "auc_inf_obs".to_owned(),
                    values: values.to_vec(),
                },
                Replicates {
                    name: "auc_extrap_fraction".to_owned(),
                    values: values.to_vec(),
                },
            ],
            vec![
                ParameterColumn {
                    name: "auc_inf_obs".to_owned(),
                    unit: UNIT.to_owned(),
                    values: vec![estimate],
                },
                ParameterColumn {
                    name: "auc_extrap_fraction".to_owned(),
                    unit: DIMENSIONLESS.to_owned(),
                    values: vec![estimate],
                },
            ],
        )
    }

    fn named<'a>(answer: &'a [ParameterColumn], name: &str) -> &'a [f64] {
        answer
            .iter()
            .find(|column| column.name == name)
            .unwrap_or_else(|| panic!("{name} is not in the answer"))
            .values
            .as_slice()
    }

    #[test]
    fn without_the_subject_count_the_between_subject_variables_are_nan() {
        let (replicates, point) = one_row(&[10.0, 11.0, 12.0, 13.0], 11.5);
        let answer = reduce_replicates(
            &replicates,
            &point,
            BootstrapSpread::Se,
            None,
            0.90,
            &[true],
        );
        // the scale of the replicates is the standard error, which is what they give
        assert!(named(&answer, "auc_inf_obs_se")[0].is_finite());
        assert!(named(&answer, "auc_inf_obs_ci_low")[0].is_finite());
        assert!(named(&answer, "auc_inf_obs_ci_high")[0].is_finite());
        // the other scale needs the subject count, so it is unknown and not an identity
        assert!(named(&answer, "auc_inf_obs_sd")[0].is_nan());
        assert!(named(&answer, "auc_inf_obs_geocv")[0].is_nan());
        // the geometric mean is a mean of the replicates and needs no count
        assert!(named(&answer, "auc_inf_obs_geomean")[0].is_finite());
        // with a count both scales are there, and they differ by its root
        let with_n = reduce_replicates(
            &replicates,
            &point,
            BootstrapSpread::Se,
            Some(&[9.0]),
            0.90,
            &[true],
        );
        let (se, sd) = (
            named(&with_n, "auc_inf_obs_se")[0],
            named(&with_n, "auc_inf_obs_sd")[0],
        );
        assert!((sd - 3.0 * se).abs() < 1e-12 * sd, "{sd} against {se}");
        assert!(named(&with_n, "auc_inf_obs_geocv")[0].is_finite());
        // under `se` draws the percentiles of the replicates are the interval of the estimate,
        // so there is no interval of the individuals to report
        assert!(!with_n.iter().any(|column| column.name.ends_with("_pi_low")));
    }

    #[test]
    fn a_replicate_column_of_one_finite_value_is_nan() {
        let nan = f64::NAN;
        let (replicates, point) = one_row(&[10.0, nan, nan, nan], 10.0);
        let answer = reduce_replicates(
            &replicates,
            &point,
            BootstrapSpread::Sd,
            Some(&[8.0]),
            0.90,
            &[true],
        );
        for suffix in ["_sd", "_se", "_ci_low", "_ci_high", "_pi_low", "_pi_high"] {
            let name = format!("auc_inf_obs{suffix}");
            assert!(named(&answer, &name)[0].is_nan(), "{name}");
        }
        // one replicate is a value and no spread, so the geometric statistics go as well
        assert!(named(&answer, "auc_inf_obs_geomean")[0].is_nan());
        assert!(named(&answer, "auc_inf_obs_geocv")[0].is_nan());
        // an infinite replicate is a hole, as `numpy.isfinite` has it, so this row has one value
        let (infinite, point) = one_row(&[10.0, f64::INFINITY, nan, nan], 10.0);
        let answer = reduce_replicates(
            &infinite,
            &point,
            BootstrapSpread::Sd,
            Some(&[8.0]),
            0.90,
            &[true],
        );
        assert!(named(&answer, "auc_inf_obs_sd")[0].is_nan());
        // and two finite replicates are the smallest row which carries a spread
        let (two, point) = one_row(&[10.0, 12.0, nan, nan], 11.0);
        let answer = reduce_replicates(
            &two,
            &point,
            BootstrapSpread::Sd,
            Some(&[8.0]),
            0.90,
            &[true],
        );
        assert!(named(&answer, "auc_inf_obs_sd")[0].is_finite());
    }

    #[test]
    fn the_log_variables_ignore_a_non_positive_replicate() {
        // three positive replicates, one zero and one negative: the logarithms are the three
        let (replicates, point) = one_row(&[10.0, 12.0, 14.0, 0.0, -3.0], 12.0);
        let answer = reduce_replicates(
            &replicates,
            &point,
            BootstrapSpread::Sd,
            Some(&[8.0]),
            0.90,
            &[true],
        );
        let geomean = named(&answer, "auc_inf_obs_geomean")[0];
        let expected = (10.0f64.ln() + 12.0f64.ln() + 14.0f64.ln()) / 3.0;
        assert!(
            (geomean - expected.exp()).abs() < 1e-12 * geomean,
            "{geomean} against {}",
            expected.exp()
        );
        assert!(named(&answer, "auc_inf_obs_geocv")[0].is_finite());
        // the spread and the interval see all five, which is what makes them differ from the
        // geometric statistics here
        assert!(named(&answer, "auc_inf_obs_sd")[0] > 5.0);

        // one positive replicate is no variance of the logarithms, so both are NaN while the
        // ordinary spread of the five values is still there
        let (one_positive, point) = one_row(&[10.0, 0.0, -3.0, -4.0, -5.0], 10.0);
        let answer = reduce_replicates(
            &one_positive,
            &point,
            BootstrapSpread::Sd,
            Some(&[8.0]),
            0.90,
            &[true],
        );
        assert!(named(&answer, "auc_inf_obs_geomean")[0].is_nan());
        assert!(named(&answer, "auc_inf_obs_geocv")[0].is_nan());
        assert!(named(&answer, "auc_inf_obs_sd")[0].is_finite());
        // a plain parameter never carries them, whatever its replicates look like
        assert!(!answer.iter().any(|column| {
            column.name.starts_with("auc_extrap_fraction") && column.name.ends_with("_geocv")
        }));
    }

    #[test]
    fn a_discrete_a_derived_and_an_estimateless_parameter_are_skipped() {
        let replicates = vec![
            Replicates {
                name: "tmax".to_owned(),
                values: vec![1.0, 1.5, 2.0],
            },
            Replicates {
                name: "auc_inf_obs_se".to_owned(),
                values: vec![1.0, 1.5, 2.0],
            },
            Replicates {
                name: "ghost".to_owned(),
                values: vec![1.0, 1.5, 2.0],
            },
        ];
        let point: Vec<ParameterColumn> = ["tmax", "auc_inf_obs_se"]
            .iter()
            .map(|name| ParameterColumn {
                name: (*name).to_owned(),
                unit: UNIT.to_owned(),
                values: vec![1.5],
            })
            .collect();
        let answer = reduce_replicates(
            &replicates,
            &point,
            BootstrapSpread::Se,
            Some(&[8.0]),
            0.90,
            &[true],
        );
        assert!(answer.is_empty(), "{answer:?}");
    }

    #[test]
    fn a_row_which_is_not_usable_has_no_uncertainty() {
        let values = [10.0, 12.0, 14.0, 11.0, 10.0, 12.0, 14.0, 11.0];
        let replicates = vec![Replicates {
            name: "auc_inf_obs".to_owned(),
            values: values.to_vec(),
        }];
        let point = vec![ParameterColumn {
            name: "auc_inf_obs".to_owned(),
            unit: UNIT.to_owned(),
            values: vec![11.5, 11.5],
        }];
        let answer = reduce_replicates(
            &replicates,
            &point,
            BootstrapSpread::Se,
            Some(&[8.0, 8.0]),
            0.90,
            &[true, false],
        );
        for column in &answer {
            assert!(column.values[0].is_finite(), "{}", column.name);
            assert!(column.values[1].is_nan(), "{}", column.name);
        }
        // and a row whose estimate is missing is not an estimate of anything either
        let point = vec![ParameterColumn {
            name: "auc_inf_obs".to_owned(),
            unit: UNIT.to_owned(),
            values: vec![11.5, f64::NAN],
        }];
        let answer = reduce_replicates(
            &replicates,
            &point,
            BootstrapSpread::Se,
            Some(&[8.0, 8.0]),
            0.90,
            &[true, true],
        );
        for column in &answer {
            assert!(column.values[1].is_nan(), "{}", column.name);
        }
    }
}
