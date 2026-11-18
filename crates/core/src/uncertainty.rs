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

use pkpdutils_abi::{
    Batch, BootstrapDistribution, BootstrapSpread, Kind, NcaOptions, ParameterColumn,
};

use crate::KernelError;
use crate::parameters::{PARAMETERS, RowParameters, Scratch};
use crate::percentile::nan_percentile_rows;
use crate::random::Draws;
use crate::shard::rows_of;
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

/// The draws one block of the bootstrap holds at once, 1 Mi values and so 8 MB.
///
/// The replicates of a whole batch are `n_rows * n_boot * n_time` values - 96 MB for a thousand
/// group curves of twelve points at a thousand replicates - so the rows are cut into blocks and
/// only the draws, the replicate rows and the parameters of one block exist at once, as python
/// does it (`bootstrap` of `uncertainty.py`). A block is a number of rows and never a part of a
/// row, so the draws of a block are the draws of those rows in an unblocked run: every draw is
/// asked for by its index in the whole batch ([`Draws`]), which is what makes the blocking
/// invisible in the answer rather than merely harmless.
const BLOCK_DRAWS: usize = 1 << 20;

/// How many rows one block holds, at least one however large a row is.
fn block_rows_of(n_boot: usize, n_time: usize) -> usize {
    let per_row = n_boot.saturating_mul(n_time).max(1);
    (BLOCK_DRAWS / per_row).max(1)
}

/// `numpy.logaddexp(0, l)`, the stable `ln(1 + exp(l))`.
///
/// The log-normal shaping needs `ln(1 + (s / mean)^2)` and computes it from the logarithms,
/// because `(s / mean)^2` overflows for a spread which is some `1e154` times its mean while the
/// logarithm of that square, and the draws which come out of it, are perfectly finite. The
/// branch is numpy's: the larger of the two arguments is factored out, so the exponential is
/// always of a non-positive number and cannot overflow.
fn log_add_exp_zero(l: f64) -> f64 {
    if l > 0.0 {
        l + (-l).exp().ln_1p()
    } else {
        l.exp().ln_1p()
    }
}

/// The spread every point of the batch is resampled with, `n_rows * n_time`.
///
/// The port of `resolve_spread` of `uncertainty.py`: the requested one where the batch carries
/// it, the other one converted with the subject count of the point, and an error naming what is
/// missing when neither is possible. `spread` overrides the option, which is what the delta
/// method needs: it always propagates `se`, whatever the bootstrap would resample with.
///
/// # Errors
///
/// [`KernelError::Batch`] when the requested spread is neither carried nor derivable.
fn resolve_spread(
    batch: &Batch,
    options: &NcaOptions,
    spread: Option<BootstrapSpread>,
) -> Result<Vec<f64>, KernelError> {
    let kind = spread.unwrap_or(options.bootstrap_spread);
    let n_rows = batch.n_samples();
    let n_time = batch.n_time;
    // one count per row or one per point: both convert the spread of a point, the second one
    // with the count of that point (`resolve_spread` reshapes `n` to `(n_rows, -1)`)
    let root_n = batch.n.as_ref().map(|n| {
        let per_point = n.len() == n_rows * n_time;
        move |row: usize, time: usize| {
            let count = if per_point {
                n[row * n_time + time]
            } else {
                n.get(row).copied().unwrap_or(f64::NAN)
            };
            if count > 0.0 { count.sqrt() } else { f64::NAN }
        }
    });
    let convert = |from: &[f64], to_se: bool| -> Vec<f64> {
        let root = root_n.as_ref().expect("checked below");
        (0..n_rows)
            .flat_map(|row| {
                (0..n_time).map(move |time| {
                    let value = from[row * n_time + time];
                    let r = root(row, time);
                    if to_se { value / r } else { value * r }
                })
            })
            .collect()
    };
    match kind {
        BootstrapSpread::Se => {
            if let Some(se) = batch.se.as_ref() {
                return Ok(se.clone());
            }
            match (batch.sd.as_ref(), root_n.is_some()) {
                (Some(sd), true) => Ok(convert(sd, true)),
                _ => Err(KernelError::Batch(
                    "The batch has no 'se' and it cannot be derived ('sd' and 'n' needed)"
                        .to_owned(),
                )),
            }
        }
        BootstrapSpread::Sd => {
            if let Some(sd) = batch.sd.as_ref() {
                return Ok(sd.clone());
            }
            match (batch.se.as_ref(), root_n.is_some()) {
                (Some(se), true) => Ok(convert(se, false)),
                _ => Err(KernelError::Batch(
                    "The batch has no 'sd' and it cannot be derived ('se' and 'n' needed)"
                        .to_owned(),
                )),
            }
        }
    }
}

/// What one call of [`resample_values`] draws and where in the stream it starts.
///
/// The fields are the batch's and not the block's, `first_index` apart: a block of rows draws
/// the numbers the whole batch would, and the index is what says which rows it holds.
struct Resampling {
    /// The points of a row.
    n_time: usize,
    /// The replicates of a row.
    n_boot: usize,
    /// The seed of the whole batch.
    seed: u64,
    /// The distribution the points are drawn from.
    distribution: BootstrapDistribution,
    /// Whether a normal draw below zero is held at zero.
    clip_at_zero: bool,
    /// The flat draw index of the first point of the first row of the block, in the C order of
    /// `(n_rows, n_boot, n_time)` of the whole batch.
    first_index: u64,
}

/// The replicates of every point of a block of rows, `(rows, n_boot, n_time)` row major.
///
/// The port of `resample_values` of `uncertainty.py`. Every point is drawn around its own mean
/// with its own spread, and a point whose spread is not usable - missing, zero, negative, or a
/// mean which is not finite - is its mean in every replicate. It **still consumes its draw**:
/// the draws are a function of the index and not of the data, so a point which is masked here
/// cannot shift the draws of the points after it.
///
/// `clip_at_zero` holds normal draws at zero, which is right for a concentration and wrong for
/// an effect; the log-normal shaping needs no clip, since it draws a positive value by
/// construction, and leaves a point whose mean is not positive at its mean.
///
/// # Arguments
///
/// * `values` - the means, `rows * n_time`
/// * `spread` - the spread of every mean, same layout
/// * `spec` - the shape of the draws and where in the stream they start
fn resample_values(values: &[f64], spread: &[f64], spec: &Resampling) -> Vec<f64> {
    let &Resampling {
        n_time,
        n_boot,
        seed,
        distribution,
        clip_at_zero,
        first_index,
    } = spec;
    let rows = values.len().checked_div(n_time).unwrap_or(0);
    let draws = Draws::new(seed);
    let mut out = vec![0.0f64; rows * n_boot * n_time];
    if out.is_empty() {
        return out;
    }
    let mut z = vec![0.0f64; n_boot * n_time];
    for row in 0..rows {
        let start = first_index + (row as u64) * (n_boot as u64) * (n_time as u64);
        draws.normals_into(start, &mut z);
        for time in 0..n_time {
            let mean = values[row * n_time + time];
            let s = spread[row * n_time + time];
            let usable = s.is_finite() && s > 0.0 && mean.is_finite();
            // the log-normal moments of the point, which are the same for all its replicates
            let (mu, sigma) = if usable && distribution == BootstrapDistribution::Lognormal {
                let log_mean = mean.ln();
                let sigma2 = log_add_exp_zero(2.0 * (s.ln() - log_mean));
                (log_mean - sigma2 / 2.0, sigma2.sqrt())
            } else {
                (f64::NAN, f64::NAN)
            };
            for replicate in 0..n_boot {
                let value = if !usable {
                    mean
                } else if distribution == BootstrapDistribution::Lognormal {
                    if mean > 0.0 {
                        (mu + sigma * z[replicate * n_time + time]).exp()
                    } else {
                        mean
                    }
                } else {
                    let normal = mean + s * z[replicate * n_time + time];
                    if clip_at_zero {
                        normal.max(0.0)
                    } else {
                        normal
                    }
                };
                out[(row * n_boot + replicate) * n_time + time] = value;
            }
        }
    }
    out
}

/// One parameter whose replicates are reduced: its column of the result and its field reader.
type Reducible<'a> = (&'a ParameterColumn, fn(&RowParameters) -> f64);

/// The parameters of a result which carry an uncertainty, with the reader of every one.
///
/// A parameter of the result which is discrete, is itself a derived variable or has no entry in
/// [`PARAMETERS`] is left out here rather than inside the reduction, so the replicates of a
/// parameter nothing will reduce are never collected.
fn reducible(point: &[ParameterColumn]) -> Vec<Reducible<'_>> {
    point
        .iter()
        .filter(|column| !is_discrete(&column.name) && !is_derived(&column.name))
        .filter_map(|column| {
            PARAMETERS
                .iter()
                .find(|parameter| parameter.name == column.name)
                .map(|parameter| (column, parameter.read))
        })
        .collect()
}

/// The uncertainty of the parameters of a batch from the parametric bootstrap.
///
/// Every point of every row is resampled around its own value with its own spread
/// ([`resample_values`]), the replicate curves run through the same analysis as the original
/// ones, and their parameters are reduced to the uncertainty variables
/// ([`reduce_replicates`]). The answer joins the parameters of the result.
///
/// The rows are cut into blocks of [`block_rows_of`] rows, or of `block_rows` where a caller
/// names one, and the replicate rows of a block are sharded like the rows of an analysis. The
/// block size changes no number, which `the_block_size_changes_no_number` asserts.
///
/// # Arguments
///
/// * `batch` - the batch, which carries the spread the points are resampled with
/// * `options` - `n_boot`, `seed`, `ci_level`, `bootstrap_spread` and `bootstrap_distribution`
/// * `point` - the parameters of the original curves, the estimates of the intervals
/// * `n_subjects` - the subjects of every row, the `n` column of the result
/// * `shards` - the shard plan of the replicate rows
/// * `block_rows` - the rows of one block, `None` for [`block_rows_of`]
/// * `analyse` - the analysis of one row with the values of a replicate in place of its own
///
/// # Errors
///
/// [`KernelError::Options`] for log-normal draws of an effect curve, which is python's own
/// refusal, and for a seed the host left open, which the kernel cannot draw: it reads no
/// entropy, having none in a wasm instance. [`KernelError::Batch`] when the batch carries no
/// spread the options can resample with.
pub(crate) fn bootstrap<F>(
    batch: &Batch,
    options: &NcaOptions,
    point: &[ParameterColumn],
    n_subjects: Option<&[f64]>,
    shards: usize,
    block_rows: Option<usize>,
    analyse: F,
) -> Result<Vec<ParameterColumn>, KernelError>
where
    F: Fn(usize, &[f64], &mut Scratch) -> RowParameters + Send + Sync,
{
    // python raises this from `bootstrap` and not from the options, so it is raised here as
    // well; `unsupported_reason` refuses an effect curve outright today, so the analysis cannot
    // reach it yet and `the_effect_kind_refuses_lognormal_draws` calls this function directly
    if options.kind == Kind::Effect
        && options.bootstrap_distribution == BootstrapDistribution::Lognormal
    {
        return Err(KernelError::Options(
            "log-normal draws need positive values, effect timecourses use NORMAL".to_owned(),
        ));
    }
    let seed = options.seed.ok_or_else(|| {
        KernelError::Options(
            "the bootstrap needs a seed: the kernel reads no entropy, so a host which leaves \
             `seed` open draws one itself and passes it"
                .to_owned(),
        )
    })?;
    let n_rows = batch.n_samples();
    let n_time = batch.n_time;
    let n_boot = options.n_boot as usize;
    let spread = resolve_spread(batch, options, None)?;
    let usable_rows: Vec<bool> = (0..n_rows)
        .map(|row| {
            spread[row * n_time..(row + 1) * n_time]
                .iter()
                .any(|s| s.is_finite() && *s > 0.0)
        })
        .collect();
    let columns = reducible(point);
    let block = block_rows
        .unwrap_or_else(|| block_rows_of(n_boot, n_time))
        .max(1);
    let mut parts: Vec<Vec<ParameterColumn>> = Vec::new();
    let mut start = 0usize;
    while start < n_rows {
        let stop = (start + block).min(n_rows);
        let rows = stop - start;
        let draws = resample_values(
            &batch.values[start * n_time..stop * n_time],
            &spread[start * n_time..stop * n_time],
            &Resampling {
                n_time,
                n_boot,
                seed,
                distribution: options.bootstrap_distribution,
                clip_at_zero: options.kind == Kind::Concentration,
                first_index: (start as u64) * (n_boot as u64) * (n_time as u64),
            },
        );
        let analysed: Vec<RowParameters> =
            rows_of(rows * n_boot, n_time, shards, |index, scratch| {
                let row = start + index / n_boot.max(1);
                let values = &draws[index * n_time..(index + 1) * n_time];
                analyse(row, values, scratch)
            });
        let replicates: Vec<Replicates> = columns
            .iter()
            .map(|(column, read)| Replicates {
                name: column.name.clone(),
                values: analysed.iter().map(read).collect(),
            })
            .collect();
        let block_point: Vec<ParameterColumn> = columns
            .iter()
            .map(|(column, _)| ParameterColumn {
                name: column.name.clone(),
                unit: column.unit.clone(),
                values: column.values[start..stop].to_vec(),
            })
            .collect();
        parts.push(reduce_replicates(
            &replicates,
            &block_point,
            options.bootstrap_spread,
            n_subjects.map(|values| &values[start..stop]),
            options.ci_level,
            &usable_rows[start..stop],
        ));
        start = stop;
    }
    Ok(join_blocks(parts, n_rows))
}

/// The parameters whose derivative may be taken at any point, whatever the terminal window does.
///
/// `TERMINAL_INDEPENDENT_PARAMETERS` of `uncertainty.py`, which holds for an analysis that reads
/// the dosing interval as it was measured. Above a `tau_tolerance` of 0 the exposure of a last
/// interval which falls short of its end is completed with the terminal regression
/// (`complete_last_interval`), so the five parameters of that interval depend on the window
/// after all and leave the set ([`terminal_independent`]).
const TERMINAL_INDEPENDENT_PARAMETERS: &[&str] = &[
    "auc_all",
    "auc_last",
    "auc_tau",
    "auec_baseline",
    "auec_last",
    "aumc_all",
    "aumc_last",
    "c0",
    "cavg",
    "clast",
    "cmax",
    "cmax_half",
    "cmax_ss",
    "cmin",
    "cmin_ss",
    "ctrough",
    "e0",
    "emax_baseline",
    "emax_obs",
    "time_above",
];

/// The parameters of a last interval which the terminal regression may complete.
///
/// `COMPLETABLE_PARAMETERS` of `uncertainty.py`.
const COMPLETABLE_PARAMETERS: &[&str] = &["auc_tau", "cavg", "cmax_ss", "cmin_ss", "ctrough"];

/// Whether the derivative of `name` may be taken at a point where the terminal window flipped.
fn terminal_independent(name: &str, options: &NcaOptions) -> bool {
    listed(TERMINAL_INDEPENDENT_PARAMETERS, name)
        && !(options.tau_tolerance > 0.0 && listed(COMPLETABLE_PARAMETERS, name))
}

/// The two variables whose change says the perturbation selected another terminal window.
///
/// `_window_changed` of `uncertainty.py` reads exactly these: the number of points of the
/// regression and where it starts. Two missing windows are equal, a missing and a present one
/// are not.
const WINDOW_VARIABLES: [&str; 2] = ["lambda_z_n_points", "lambda_z_t_first"];

/// One window variable of a result: its field reader and the column of the original curves.
type WindowVariable<'a> = (fn(&RowParameters) -> f64, &'a [f64]);

/// What the delta method propagated for one row.
struct DeltaRow {
    /// The standard error of every reducible parameter, in the order of the columns.
    standard_error: Vec<f64>,
    /// Whether a perturbation of this row selected another terminal window.
    window_changed: bool,
    /// Whether any point of the row carried a usable standard error.
    any_usable: bool,
}

/// The uncertainty of the parameters of a batch from the delta method, with the flags it raises.
pub(crate) struct DeltaAnswer {
    /// The uncertainty variables, the same family the bootstrap reports.
    pub columns: Vec<ParameterColumn>,
    /// The flag word of every row, to be combined with the flags of the analysis.
    pub flags: Vec<u32>,
}

/// The step of the forward difference of one point.
///
/// `h = max(delta_step * se, nextafter(0, 1), sqrt(eps) * |c|)` of `uncertainty.py`: the
/// relative step of the options, never zero, and never so small against the value that
/// `c + h == c`. The smallest positive double is a subnormal, `f64::from_bits(1)`, and not
/// `f64::MIN_POSITIVE`, which is the smallest *normal* one and would be a step `1e292` times
/// larger than numpy's.
fn delta_step(value: f64, se: f64, relative: f64) -> f64 {
    let smallest_positive = f64::from_bits(1);
    (relative * se)
        .max(smallest_positive)
        .max(f64::EPSILON.sqrt() * value.abs())
}

/// Propagate the standard errors of the points of a batch through the numerical Jacobian.
///
/// `var(x) = sum_i (dx/dC_i)^2 se_i^2` with `dx/dC_i` from a forward difference, which is the
/// delta method of `uncertainty.py` [^efron]. The norm is accumulated with `hypot` and never by
/// squaring first, since a finite standard error can have a variance outside the range of a
/// double. The method always propagates `se`, the uncertainty of the mean curve, so
/// `NcaOptions::bootstrap_spread` does not apply to it.
///
/// A perturbation which selects another terminal window makes the difference quotient a jump
/// between two regressions instead of a derivative. Such a point is skipped for every parameter
/// which depends on the window and the row is flagged [`NcaFlag::DELTA_WINDOW_CHANGE`], which
/// says that the uncertainty of its terminal parameters is incomplete; the bootstrap, which
/// follows the window, is the answer for those rows.
///
/// # Arguments
///
/// * `batch` - the batch, which carries `se` or `sd` and the subject count
/// * `options` - `delta_step`, `ci_level`, `tau_tolerance`
/// * `point` - the parameters of the original curves
/// * `n_subjects` - the subjects of every row, for the between subject scale
/// * `shards` - the shard plan of the rows
/// * `analyse` - the analysis of one row with the values of a perturbed curve
///
/// # Errors
///
/// [`KernelError::Batch`] when the batch has no `se` and none can be derived from its `sd`.
///
/// [^efron]: Efron & Tibshirani 1993, ch. 5.
pub(crate) fn delta<F>(
    batch: &Batch,
    options: &NcaOptions,
    point: &[ParameterColumn],
    n_subjects: Option<&[f64]>,
    shards: usize,
    analyse: F,
) -> Result<DeltaAnswer, KernelError>
where
    F: Fn(usize, &[f64], &mut Scratch) -> RowParameters + Send + Sync,
{
    let n_rows = batch.n_samples();
    let n_time = batch.n_time;
    // the delta method propagates the uncertainty of the mean curve whatever the bootstrap would
    // resample with, so the spread is `se` and not `options.bootstrap_spread`
    let se = resolve_spread(batch, options, Some(BootstrapSpread::Se))?;
    let columns = reducible(point);
    let window: Vec<Option<WindowVariable<'_>>> = WINDOW_VARIABLES
        .iter()
        .map(|name| {
            let read = PARAMETERS
                .iter()
                .find(|parameter| parameter.name == *name)
                .map(|parameter| parameter.read)?;
            let values = point
                .iter()
                .find(|column| column.name == *name)
                .map(|column| column.values.as_slice())?;
            Some((read, values))
        })
        .collect();
    let keep: Vec<bool> = columns
        .iter()
        .map(|(column, _)| terminal_independent(&column.name, options))
        .collect();

    let rows: Vec<DeltaRow> = rows_of(n_rows, n_time, shards, |row, scratch| {
        let values = &batch.values[row * n_time..(row + 1) * n_time];
        let spread = &se[row * n_time..(row + 1) * n_time];
        let mut standard_error = vec![0.0f64; columns.len()];
        let mut window_changed = false;
        let mut any_usable = false;
        let mut perturbed = values.to_vec();
        for time in 0..n_time {
            let value = values[time];
            let point_se = spread[time];
            if !(point_se.is_finite() && point_se > 0.0 && value.is_finite()) {
                continue;
            }
            any_usable = true;
            let h = delta_step(value, point_se, options.delta_step);
            let forward = value + h;
            // at the upper finite limit only a backward difference exists
            perturbed[time] = if forward.is_finite() {
                forward
            } else {
                value - h
            };
            let actual_step = perturbed[time] - value;
            let answer = analyse(row, &perturbed, scratch);
            perturbed[time] = value;
            let changed = window.iter().flatten().any(|(read, values)| {
                let base = values[row];
                let other = read(&answer);
                !((base.is_nan() && other.is_nan()) || base == other)
            });
            window_changed |= changed;
            for (index, ((column, read), independent)) in columns.iter().zip(&keep).enumerate() {
                if changed && !independent {
                    continue;
                }
                let derivative = (read(&answer) - column.values[row]) / actual_step;
                standard_error[index] = standard_error[index].hypot(derivative * point_se);
            }
        }
        DeltaRow {
            standard_error,
            window_changed,
            any_usable,
        }
    });

    let alpha = 1.0 - options.ci_level;
    let z = normal_isf(alpha / 2.0);
    let mut out = Vec::new();
    for (index, (column, _)) in columns.iter().enumerate() {
        let name = column.name.as_str();
        let lognormal = is_lognormal(name);
        let unit = column.unit.as_str();
        let mut x_sd = vec![f64::NAN; n_rows];
        let mut x_se = vec![f64::NAN; n_rows];
        let mut low = vec![f64::NAN; n_rows];
        let mut high = vec![f64::NAN; n_rows];
        let mut geomean = vec![f64::NAN; n_rows];
        let mut geocv = vec![f64::NAN; n_rows];
        for (row, answer) in rows.iter().enumerate() {
            let estimate = column.values[row];
            let standard_error = answer.standard_error[index];
            if !(estimate.is_finite() && standard_error.is_finite() && answer.any_usable) {
                continue;
            }
            x_se[row] = standard_error;
            x_sd[row] = n_subjects.map_or(f64::NAN, |counts| standard_error * counts[row].sqrt());
            if lognormal {
                let relative = standard_error / estimate;
                low[row] = estimate * (-z * relative).exp();
                high[row] = estimate * (z * relative).exp();
                geomean[row] = estimate;
                // the geometric CV is the spread over subjects, as in the bootstrap: with
                // `mu = x` and `sd = x_sd` the log-normal moment relation gives
                // `sigma_log^2 = ln(1 + (sd / mu)^2)`, so `geocv = sqrt(exp(sigma^2) - 1)` is
                // `|sd / mu|`, the arithmetic coefficient of variation
                geocv[row] = (x_sd[row] / estimate).abs();
            } else {
                low[row] = estimate - z * standard_error;
                high[row] = estimate + z * standard_error;
            }
        }
        // the order is python's and not an arrangement of its own: a log-normal parameter
        // carries its geometric statistics in front of the four variables every parameter has
        // (`delta` of `uncertainty.py`), and the order of the columns of a result is the column
        // order of every table rendered from it
        if lognormal {
            out.push(crate::uncertainty::column(name, "_geomean", unit, geomean));
            out.push(crate::uncertainty::column(
                name,
                "_geocv",
                DIMENSIONLESS,
                geocv,
            ));
        }
        out.push(crate::uncertainty::column(name, "_sd", unit, x_sd));
        out.push(crate::uncertainty::column(name, "_se", unit, x_se));
        out.push(crate::uncertainty::column(name, "_ci_low", unit, low));
        out.push(crate::uncertainty::column(name, "_ci_high", unit, high));
    }
    Ok(DeltaAnswer {
        columns: out,
        flags: rows
            .iter()
            .map(|row| {
                if row.window_changed {
                    pkpdutils_abi::NcaFlag::DELTA_WINDOW_CHANGE.bits()
                } else {
                    0
                }
            })
            .collect(),
    })
}

/// The columns of the blocks as one column per variable, in the order the first block has them.
///
/// Every block reduces the same variables, since the estimates and the classification are the
/// same for all of them, so this is a concatenation; a variable a later block does not carry
/// would be `NaN` in those rows rather than shifting the column, which cannot happen and is
/// written out because a silent shift is the one failure a concatenation can have.
fn join_blocks(parts: Vec<Vec<ParameterColumn>>, n_rows: usize) -> Vec<ParameterColumn> {
    let Some(first) = parts.first() else {
        return Vec::new();
    };
    let mut out: Vec<ParameterColumn> = first
        .iter()
        .map(|column| ParameterColumn {
            name: column.name.clone(),
            unit: column.unit.clone(),
            values: Vec::with_capacity(n_rows),
        })
        .collect();
    for part in &parts {
        for column in &mut out {
            match part.iter().find(|other| other.name == column.name) {
                Some(found) => column.values.extend_from_slice(&found.values),
                None => column.values.push(f64::NAN),
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {

    /// A batch of `rows` group curves with a spread, a subject count and one oral dose.
    fn group_batch(rows: usize) -> Batch {
        let times: Vec<f64> = vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
        let n_time = times.len();
        let mut all_times = Vec::new();
        let mut values = Vec::new();
        let mut se = Vec::new();
        for row in 0..rows {
            let clearance = 0.2 + 0.05 * row as f64;
            all_times.extend_from_slice(&times);
            for &t in &times {
                let value = 20.0 * (-clearance * t).exp() * (1.0 - (-2.0 * t).exp());
                values.push(value);
                se.push(0.12 * value);
            }
        }
        Batch {
            sample_dims: vec!["subject".to_owned()],
            sample_shape: vec![rows],
            n_time,
            times: all_times,
            values,
            se: Some(se),
            n: Some(vec![9.0; rows * n_time]),
            max_doses: 1,
            dose_amount: vec![100.0; rows],
            dose_time: vec![0.0; rows],
            dose_duration: vec![f64::NAN; rows],
            route: Some(pkpdutils_abi::Route::Oral),
            // the coordinates of the one row batch this is built from would be the wrong length
            coords: Vec::new(),
            ..crate::parameters::tests::oral_curve()
        }
    }

    /// The options of a bootstrap of `n_boot` replicates under `seed`.
    fn boot_options(n_boot: u32, seed: u64) -> NcaOptions {
        NcaOptions {
            n_boot,
            seed: Some(seed),
            ci_level: 0.90,
            ..crate::parameters::tests::linear()
        }
    }

    /// The point parameters of a batch, the estimates the uncertainty is built on.
    fn point_of(batch: &Batch, options: &NcaOptions) -> Vec<ParameterColumn> {
        let mut without = options.clone();
        without.uncertainty = Some(pkpdutils_abi::UncertaintyMethod::None);
        crate::analyze_batch(batch, &without)
            .expect("the point analysis")
            .parameters
    }

    /// The bootstrap of a batch at a chosen block size, through the analysis of the crate.
    fn bootstrap_at(
        batch: &Batch,
        options: &NcaOptions,
        block_rows: Option<usize>,
    ) -> Vec<ParameterColumn> {
        let point = point_of(batch, options);
        let n_subjects: Vec<f64> = (0..batch.n_samples()).map(|_| 9.0).collect();
        let rows = crate::Rows {
            batch,
            options,
            manual: None,
            windows: None,
            has_dose: batch.max_doses > 0,
        };
        bootstrap(
            batch,
            options,
            &point,
            Some(&n_subjects),
            1,
            block_rows,
            |row, values, scratch| rows.row_of(row, values, scratch, None).parameters,
        )
        .expect("the bootstrap")
    }

    /// The delta method of a batch through the analysis of the crate.
    fn delta_at(batch: &Batch, options: &NcaOptions) -> DeltaAnswer {
        let point = point_of(batch, options);
        let n_subjects: Vec<f64> = (0..batch.n_samples()).map(|_| 9.0).collect();
        let rows = crate::Rows {
            batch,
            options,
            manual: None,
            windows: None,
            has_dose: batch.max_doses > 0,
        };
        delta(
            batch,
            options,
            &point,
            Some(&n_subjects),
            1,
            |row, values, scratch| rows.row_of(row, values, scratch, None).parameters,
        )
        .expect("the delta method")
    }

    /// The options of a delta method of `step`.
    fn delta_options(step: f64) -> NcaOptions {
        NcaOptions {
            uncertainty: Some(pkpdutils_abi::UncertaintyMethod::Delta),
            delta_step: step,
            ci_level: 0.90,
            ..crate::parameters::tests::linear()
        }
    }

    #[test]
    fn a_perturbation_which_moves_the_window_is_skipped_and_flagged() {
        // a spread of 30 per cent and a step of a fifth of it move the terminal window, which a
        // step of a thousandth never does
        let mut batch = group_batch(1);
        let n_time = batch.n_time;
        let se: Vec<f64> = batch.values.iter().map(|value| 0.3 * value).collect();
        batch.se = Some(se);
        batch.n = Some(vec![9.0; n_time]);

        let moved = delta_at(&batch, &delta_options(0.2));
        assert_eq!(
            moved.flags,
            vec![pkpdutils_abi::NcaFlag::DELTA_WINDOW_CHANGE.bits()],
            "a step which moves the window raises the flag"
        );
        let fine = delta_at(&batch, &delta_options(1e-3));
        assert_eq!(fine.flags, vec![0], "a small step moves no window");

        // The area of the measured points is linear in those points, so its difference quotient
        // is the derivative itself at every step, and it depends on no window: the two steps
        // agree on it to the last bits whatever the window did in between.
        let auc_moved = named(&moved.columns, "auc_last_se")[0];
        let auc_fine = named(&fine.columns, "auc_last_se")[0];
        assert!(auc_moved > 0.0, "{auc_moved}");
        assert!(
            (auc_moved - auc_fine).abs() <= 1e-9 * auc_fine,
            "a window independent parameter is untouched: {auc_moved} against {auc_fine}"
        );

        // A parameter which reads the terminal window loses the points at which the window moved,
        // so its propagated error comes out *below* the one of a step which moved nothing: a
        // dropped point contributes nothing to the norm. Were the point kept, the difference
        // quotient would be the jump between two regressions, which is far larger than the
        // derivative and would push the error above the one of the fine step instead.
        for name in ["auc_inf_obs_se", "cl_f_se"] {
            let with_jumps = named(&fine.columns, name)[0];
            let skipped = named(&moved.columns, name)[0];
            assert!(
                skipped > 0.0 && skipped < 0.8 * with_jumps,
                "{name}: {skipped} against {with_jumps} of the fine step"
            );
        }
    }

    #[test]
    fn the_increment_is_the_representable_one() {
        // a value whose standard error is far below its own floating point resolution: the step
        // of the options would be lost in the addition, so the floor of `sqrt(eps) * |c|` is what
        // makes the difference quotient a derivative instead of a division by zero
        let value = 1e8;
        let tiny = 1e-20;
        assert_eq!(
            value + 1e-3 * tiny,
            value,
            "the step of the options is lost in the value"
        );
        let step = delta_step(value, tiny, 1e-3);
        assert!(step > 0.0, "the step is positive");
        assert_ne!(value + step, value, "the step changes the value");
        assert_eq!(
            step,
            f64::EPSILON.sqrt() * value,
            "the floor of the value's own scale is the step"
        );

        // the smallest positive double is the other floor, which a zero of both other terms falls
        // back to; it is the subnormal and not `f64::MIN_POSITIVE`
        assert_eq!(delta_step(0.0, 0.0, 1e-3), f64::from_bits(1));

        // and the relative step is the one the options ask for wherever it is representable
        assert_eq!(delta_step(10.0, 2.0, 1e-3), 2e-3);
    }

    #[test]
    fn the_delta_ignores_the_bootstrap_spread_option() {
        // the delta method propagates the uncertainty of the mean curve, so it reads `se` whatever
        // the bootstrap would resample with: the same batch gives the same answer under both
        // spread options, and a batch which carries `sd` alone is an error of the batch
        let batch = group_batch(2);
        let with_se = delta_at(
            &batch,
            &NcaOptions {
                bootstrap_spread: pkpdutils_abi::BootstrapSpread::Se,
                ..delta_options(1e-3)
            },
        );
        let with_sd = delta_at(
            &batch,
            &NcaOptions {
                bootstrap_spread: pkpdutils_abi::BootstrapSpread::Sd,
                ..delta_options(1e-3)
            },
        );
        assert_eq!(
            named(&with_se.columns, "auc_last_se"),
            named(&with_sd.columns, "auc_last_se"),
            "the spread option moves no number of the delta method"
        );
        assert!(named(&with_se.columns, "auc_last_se")[0].is_finite());
    }

    #[test]
    fn the_block_size_changes_no_number() {
        // the draws are asked for by their index in the whole batch, so a run which materializes
        // one row at a time and a run which materializes all five give the same bits; a
        // generator which was advanced instead would give the same numbers only by luck of the
        // block boundaries falling where they did
        let batch = group_batch(5);
        let options = boot_options(16, 99);
        let whole = bootstrap_at(&batch, &options, Some(5));
        assert!(!whole.is_empty(), "the bootstrap reported nothing");
        for block in [1usize, 2, 3, 4, 5, 6, 50] {
            let cut = bootstrap_at(&batch, &options, Some(block));
            assert_eq!(cut.len(), whole.len(), "block {block}");
            for (a, b) in whole.iter().zip(&cut) {
                assert_eq!(a.name, b.name, "block {block}");
                assert_eq!(a.values.len(), 5, "{}", a.name);
                for (row, (x, y)) in a.values.iter().zip(&b.values).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "block {block}, {} row {row}: {x} against {y}",
                        a.name
                    );
                }
            }
        }
        // and the automatic size is a size of its own, which the same batch has to agree with
        let automatic = bootstrap_at(&batch, &options, None);
        for (a, b) in whole.iter().zip(&automatic) {
            assert_eq!(
                a.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{}",
                a.name
            );
        }
        // the test is not comparing columns of NaN
        let finite = whole
            .iter()
            .flat_map(|column| column.values.iter())
            .filter(|value| value.is_finite())
            .count();
        assert!(finite > 100, "only {finite} finite values");
    }

    #[test]
    fn a_bootstrap_below_two_replicates_is_refused_by_the_options() {
        // python refuses it in the options as well (`n_boot: int = Field(default=1000, ge=2)`),
        // so the kernel never sees a bootstrap which cannot have a spread
        for n_boot in [0u32, 1] {
            let error = boot_options(n_boot, 1).validate().expect_err("refused");
            assert!(error.contains("n_boot"), "{error}");
        }
        // two is the smallest bootstrap there is, and it carries a spread
        let batch = group_batch(2);
        let options = boot_options(2, 3);
        let answer = bootstrap_at(&batch, &options, None);
        let sd = answer
            .iter()
            .find(|column| column.name == "auc_inf_obs_se")
            .expect("the standard error");
        assert!(sd.values.iter().all(|value| value.is_finite()), "{sd:?}");

        // a caller of the crate which asks for no replicate at all gets NaN and no panic: the
        // reduction sees no finite value and `count > 1` is its guard
        let point = point_of(&batch, &options);
        let empty = reduce_replicates(
            &[Replicates {
                name: "auc_inf_obs".to_owned(),
                values: Vec::new(),
            }],
            &point,
            BootstrapSpread::Se,
            None,
            0.90,
            &[true, true],
        );
        assert!(!empty.is_empty(), "the variables are reported");
        for column in &empty {
            assert!(
                column.values.iter().all(|value| value.is_nan()),
                "{}",
                column.name
            );
        }
    }

    #[test]
    fn a_row_without_a_usable_spread_is_nan_and_the_rows_before_it_are_untouched() {
        let mut batch = group_batch(3);
        let n_time = batch.n_time;
        // the last row carries no usable spread at all
        if let Some(se) = batch.se.as_mut() {
            for value in se[2 * n_time..3 * n_time].iter_mut() {
                *value = f64::NAN;
            }
        }
        let options = boot_options(16, 7);
        let three = bootstrap_at(&batch, &options, None);

        let two = Batch {
            sample_shape: vec![2],
            times: batch.times[..2 * n_time].to_vec(),
            values: batch.values[..2 * n_time].to_vec(),
            se: batch.se.as_ref().map(|se| se[..2 * n_time].to_vec()),
            n: batch.n.as_ref().map(|n| n[..2 * n_time].to_vec()),
            dose_amount: batch.dose_amount[..2].to_vec(),
            dose_time: batch.dose_time[..2].to_vec(),
            dose_duration: batch.dose_duration[..2].to_vec(),
            ..batch.clone()
        };
        let pair = bootstrap_at(&two, &options, None);
        for (column, same) in three.iter().zip(&pair) {
            assert_eq!(column.name, same.name);
            assert!(
                column.values[2].is_nan(),
                "{}: the row has no spread",
                column.name
            );
            // the rows before it are the same bits as in a batch which does not carry it: a row
            // draws from its own window of the stream, which its position in the batch fixes
            for row in 0..2 {
                assert_eq!(
                    column.values[row].to_bits(),
                    same.values[row].to_bits(),
                    "{} row {row}",
                    column.name
                );
            }
        }
        let finite = pair
            .iter()
            .flat_map(|column| column.values.iter())
            .filter(|value| value.is_finite())
            .count();
        assert!(finite > 40, "only {finite} finite values");
    }

    #[test]
    fn a_lognormal_point_without_a_positive_mean_is_copied() {
        // the log-normal shaping needs a positive mean: a zero and a negative point are their
        // mean in every replicate rather than NaN, which is what `np.where(positive, ...)` of
        // `resample_values` does with them
        let values = [5.0, 0.0, -2.0, 7.0];
        let spread = [0.5, 0.5, 0.5, 0.5];
        let draws = resample_values(
            &values,
            &spread,
            &Resampling {
                n_time: 4,
                n_boot: 8,
                seed: 11,
                distribution: BootstrapDistribution::Lognormal,
                clip_at_zero: true,
                first_index: 0,
            },
        );
        for replicate in 0..8 {
            assert_eq!(draws[replicate * 4 + 1], 0.0, "the zero point");
            assert_eq!(draws[replicate * 4 + 2], -2.0, "the negative point");
            assert!(draws[replicate * 4] > 0.0, "a log-normal draw is positive");
            assert_ne!(draws[replicate * 4], 5.0, "and is not the mean");
        }
        // a normal draw of the same points is shaped around them, negative mean and all
        let normal = resample_values(
            &values,
            &spread,
            &Resampling {
                n_time: 4,
                n_boot: 8,
                seed: 11,
                distribution: BootstrapDistribution::Normal,
                clip_at_zero: false,
                first_index: 0,
            },
        );
        assert!(normal.iter().any(|value| *value < 0.0), "{normal:?}");
        // and a concentration clips at zero instead
        let clipped = resample_values(
            &values,
            &spread,
            &Resampling {
                n_time: 4,
                n_boot: 8,
                seed: 11,
                distribution: BootstrapDistribution::Normal,
                clip_at_zero: true,
                first_index: 0,
            },
        );
        assert!(clipped.iter().all(|value| *value >= -2.0));
        assert!(
            clipped
                .iter()
                .enumerate()
                .all(|(i, value)| i % 4 == 2 || *value >= 0.0),
            "only the copied mean is negative"
        );
    }

    #[test]
    fn a_spread_which_overflows_the_square_still_draws() {
        // `(s / mean)^2` is `inf` here while its logarithm is about 1017, which is why the
        // shaping works on the log scale: a draw of such a point is enormous and finite
        let mean: f64 = 1e-10;
        let s: f64 = 1e211;
        assert!(
            (s / mean).powi(2).is_infinite(),
            "the square has to overflow"
        );
        let draws = resample_values(
            &[mean],
            &[s],
            &Resampling {
                n_time: 1,
                n_boot: 16,
                seed: 5,
                distribution: BootstrapDistribution::Lognormal,
                clip_at_zero: true,
                first_index: 0,
            },
        );
        assert!(
            draws.iter().all(|value| value.is_finite() && *value > 0.0),
            "{draws:?}"
        );
        // the logarithm of the relative spread is what the shaping uses, and `ln(1 + x)` of an
        // enormous `x` is that logarithm
        let sigma2 = log_add_exp_zero(2.0 * (s.ln() - mean.ln()));
        assert!(
            (sigma2 - 2.0 * (s.ln() - mean.ln())).abs() < 1e-9,
            "{sigma2}"
        );
        // and the other end of the branch is the stable one, where the square is tiny
        let small = log_add_exp_zero(-1500.0);
        assert_eq!(small, 0.0, "exp(-1500) is below the resolution of 1");
        assert!((log_add_exp_zero(0.0) - std::f64::consts::LN_2).abs() < 1e-15);
    }

    #[test]
    fn the_effect_kind_refuses_lognormal_draws() {
        // python raises `ValueError` from `bootstrap` for this pair, so the kernel answers with
        // the class which is a `ValueError` on the host and with python's own sentence.
        // `unsupported_reason` refuses an effect curve outright today, so the analysis cannot
        // reach this and the driver is called directly
        let batch = group_batch(2);
        let options = NcaOptions {
            kind: Kind::Effect,
            bootstrap_distribution: BootstrapDistribution::Lognormal,
            ..boot_options(8, 1)
        };
        let point = point_of(&batch, &boot_options(8, 1));
        let error = bootstrap(&batch, &options, &point, None, 1, None, |_, _, _| {
            RowParameters::EMPTY
        })
        .expect_err("the pair is refused");
        assert!(
            matches!(&error, KernelError::Options(what)
                if what == "log-normal draws need positive values, effect timecourses use NORMAL"),
            "{error}"
        );
        assert_eq!(error.class(), crate::ErrorClass::Malformed);
        // the normal distribution of an effect curve is no error of the uncertainty
        let normal = NcaOptions {
            bootstrap_distribution: BootstrapDistribution::Normal,
            ..options
        };
        assert!(
            bootstrap(&batch, &normal, &point, None, 1, None, |_, _, _| {
                RowParameters::EMPTY
            })
            .is_ok()
        );
    }

    #[test]
    fn a_bootstrap_without_a_seed_is_refused_rather_than_seeded_here() {
        // the kernel reads no entropy, so a seed is the host's to draw; `unsupported_reason`
        // names the same refusal, which is what keeps `is_supported` and the analysis in step
        let batch = group_batch(2);
        let options = NcaOptions {
            seed: None,
            ..boot_options(8, 1)
        };
        let point = point_of(&batch, &boot_options(8, 1));
        let error = bootstrap(&batch, &options, &point, None, 1, None, |_, _, _| {
            RowParameters::EMPTY
        })
        .expect_err("a seedless bootstrap is refused");
        assert!(matches!(&error, KernelError::Options(what) if what.contains("reads no entropy")));
        assert!(
            crate::unsupported_reason(&batch, &options)
                .is_some_and(|reason| reason.contains("seed")),
            "the predicate refuses it as well"
        );
    }
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
