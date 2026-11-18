//! The sparse (destructive and batch design) analysis, the port of `pkpdutils.nca.sparse`.
//!
//! A preclinical study rarely samples one animal repeatedly, so there is no curve per animal,
//! only a mean curve over the animals of every nominal time. Bailer (1988) gives the variance of
//! the area under that mean curve; Nedelman, Gibiansky and Lau (1995) give the Satterthwaite
//! degrees of freedom; Nedelman and Jia (1998) extend the estimator to a batch design and
//! Holder (2001) writes its covariance. All three come out of one identity: the area is
//! written per animal, `A_i = sum_{j in T_i} (w_j / n_j) y_ij`, the animals are independent
//! whatever the design, and the variance is `sum_b m_b s2_{A,b}` over the batches `b` of animals
//! with the same sampling times.
//!
//! The order of the floating point operations is that of numpy, whose `sum` is pairwise, so that
//! the result agrees with the host bit for bit where it can.

use crate::KernelError;
use pkpdutils_abi::{
    AnalysisResult, Attributes, NcaFlag, NcaOptions, ParameterColumn, PointVariable, SparseAnswer,
    SparseRequest,
};
use pkpdutils_units::{UnitContext, parameter_unit};
use std::collections::BTreeMap;

/// The unit expression of every parameter of a sparse result, `PARAMETER_UNITS` of the host.
const SPARSE_VARIABLES: [(&str, &str); 7] = [
    ("auc_last", "({unit}) * ({time})"),
    ("auc_last_se", "({unit}) * ({time})"),
    ("auc_last_df", "dimensionless"),
    ("auc_all", "({unit}) * ({time})"),
    ("cmax", "{unit}"),
    ("cmax_se", "{unit}"),
    ("tmax", "{time}"),
];

/// The rows of a validated row-major `(n_rows, n_cols)` matrix; `n_cols` may be 0, which gives
/// `n_rows` empty rows.
fn rows(values: &[f64], n_rows: usize, n_cols: usize) -> impl Iterator<Item = &[f64]> {
    (0..n_rows).map(move |i| &values[i * n_cols..(i + 1) * n_cols])
}

/// The block of numpy's pairwise summation, `PW_BLOCKSIZE`.
const PAIRWISE_BLOCK: usize = 128;

/// The sum of numpy's `np.sum` over a contiguous array: pairwise, eight accumulators per block.
///
/// Plain left to right summation differs from it in the last bits above eight terms.
pub(crate) fn numpy_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut result = 0.0;
        for x in a {
            result += x;
        }
        return result;
    }
    if n <= PAIRWISE_BLOCK {
        let mut r = [0.0; 8];
        r.copy_from_slice(&a[..8]);
        let body = n - n % 8;
        let mut i = 8;
        while i < body {
            for (j, slot) in r.iter_mut().enumerate() {
                *slot += a[i + j];
            }
            i += 8;
        }
        let mut result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for x in &a[body..] {
            result += x;
        }
        return result;
    }
    let mut half = n / 2;
    half -= half % 8;
    numpy_sum(&a[..half]) + numpy_sum(&a[half..])
}

/// The sample variance (`ddof = 1`) of at least two values, as `ndarray.var(ddof=1)`.
fn numpy_var1(a: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean = numpy_sum(a) / n;
    let squares: Vec<f64> = a
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .collect();
    numpy_sum(&squares) / (n - 1.0)
}

/// Weights of the linear trapezoid rule over a grid of times.
///
/// The area under the polygon through `(t_j, y_j)` is `sum_j w_j y_j` with
/// `w_1 = (t_2 - t_1) / 2`, `w_j = (t_{j+1} - t_{j-1}) / 2` and `w_J = (t_J - t_{J-1}) / 2`,
/// the form Bailer (1988) needs because the area is linear in the values. The logarithmic rules
/// are not and have no such weights. Fewer than two times give all zero weights.
pub(crate) fn trapezoid_weights(times: &[f64]) -> Vec<f64> {
    let n = times.len();
    let mut weights = vec![0.0; n];
    if n < 2 {
        return weights;
    }
    weights[0] = 0.5 * (times[1] - times[0]);
    weights[n - 1] = 0.5 * (times[n - 1] - times[n - 2]);
    for j in 1..n - 1 {
        weights[j] = 0.5 * (times[j + 1] - times[j - 1]);
    }
    weights
}

/// The mean, the standard deviation (`ddof = 1`) and the count of every time of a sparse design.
///
/// `values` is row-major `(n_rows, n_cols)`, non-finite where an animal has no sample. A time
/// nobody was sampled at has a `NaN` mean and a count of 0, a time with a single animal has a
/// `NaN` standard deviation. The sums run down the rows in order, as numpy's `nanmean` and
/// `nanstd` along axis 0 do.
pub(crate) fn point_statistics(
    values: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> (Vec<f64>, Vec<f64>, Vec<i64>) {
    let mut count = vec![0_i64; n_cols];
    let mut total = vec![0.0; n_cols];
    for row in rows(values, n_rows, n_cols) {
        for (j, &v) in row.iter().enumerate() {
            if v.is_finite() {
                count[j] += 1;
                total[j] += v;
            }
        }
    }
    let mean_of: Vec<f64> = (0..n_cols).map(|j| total[j] / count[j] as f64).collect();
    let mut squares = vec![0.0; n_cols];
    for row in rows(values, n_rows, n_cols) {
        for (j, &v) in row.iter().enumerate() {
            if v.is_finite() {
                let d = v - mean_of[j];
                squares[j] += d * d;
            }
        }
    }
    let mean = (0..n_cols)
        .map(|j| if count[j] > 0 { mean_of[j] } else { f64::NAN })
        .collect();
    let sd = (0..n_cols)
        .map(|j| {
            if count[j] > 1 {
                (squares[j] / (count[j] - 1) as f64).sqrt()
            } else {
                f64::NAN
            }
        })
        .collect();
    (mean, sd, count)
}

/// The nominal times `auc_last` covers: the observed ones up to the last whose mean is positive.
///
/// All `false` when no mean is positive.
pub(crate) fn area_window(observed: &[bool], mean: &[f64]) -> Vec<bool> {
    let mut window = observed.to_vec();
    let last = observed
        .iter()
        .zip(mean)
        .rposition(|(&seen, &m)| seen && m > 0.0);
    match last {
        None => vec![false; observed.len()],
        Some(last) => {
            for slot in window.iter_mut().skip(last + 1) {
                *slot = false;
            }
            window
        }
    }
}

/// The variance of the sparse area estimate, its Satterthwaite degrees of freedom and the number
/// of batches.
///
/// With `A_i = sum_{j in T_i} (w_j / n_j) y_ij` and `n_j` the animals sampled at time `j`,
///
/// `Var = sum_b m_b s2_{A,b}`, `nu = (sum_b c_b)^2 / sum_b c_b^2 / (m_b - 1)`,
/// `c_b = m_b s2_{A,b}`,
///
/// where a batch `b` is the group of the `m_b` animals with the same sampling times (Bailer 1988;
/// Nedelman, Gibiansky and Lau 1995; Nedelman and Jia 1998; Holder 2001). The batches are
/// visited in the lexicographic order of the distinct rows of the usage matrix with `false`
/// before `true`, the order of `np.unique(axis=0)`, because the summation order decides the last
/// bits. An animal without a usable sample is no batch. The variance and the degrees of freedom
/// are `NaN` when any batch holds a single animal, whose contribution cannot be estimated, and
/// when there is no batch at all (then with a count of 0).
#[cfg(test)]
pub(crate) fn bailer_variance(
    weights: &[f64],
    values: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> (f64, f64, usize) {
    let (variance, df, sizes) = bailer_batches(weights, values, n_rows, n_cols);
    (variance, df, sizes.len())
}

/// [`bailer_variance`] with the size of every batch instead of their number.
fn bailer_batches(
    weights: &[f64],
    values: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> (f64, f64, Vec<usize>) {
    let used_at = |v: f64, j: usize| v.is_finite() && weights[j] != 0.0;
    let mut n_per_time = vec![0_usize; n_cols];
    for row in rows(values, n_rows, n_cols) {
        for (j, &v) in row.iter().enumerate() {
            if used_at(v, j) {
                n_per_time[j] += 1;
            }
        }
    }
    let per_value: Vec<f64> = (0..n_cols)
        .map(|j| {
            if n_per_time[j] > 0 {
                weights[j] / n_per_time[j] as f64
            } else {
                0.0
            }
        })
        .collect();
    // the pattern of a row is the key, the animals of that pattern in order of appearance
    let mut batches: BTreeMap<Vec<bool>, Vec<f64>> = BTreeMap::new();
    for row in rows(values, n_rows, n_cols) {
        let pattern: Vec<bool> = row
            .iter()
            .enumerate()
            .map(|(j, &v)| used_at(v, j))
            .collect();
        let contributions: Vec<f64> = row
            .iter()
            .enumerate()
            .map(|(j, &v)| if pattern[j] { v * per_value[j] } else { 0.0 })
            .collect();
        batches
            .entry(pattern)
            .or_default()
            .push(numpy_sum(&contributions));
    }
    let mut terms: Vec<f64> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
    for (pattern, members) in &batches {
        if !pattern.iter().any(|&u| u) {
            continue;
        }
        sizes.push(members.len());
        terms.push(if members.len() > 1 {
            members.len() as f64 * numpy_var1(members)
        } else {
            f64::NAN
        });
    }
    if terms.is_empty() {
        return (f64::NAN, f64::NAN, sizes);
    }
    if sizes.iter().any(|&size| size < 2) {
        return (f64::NAN, f64::NAN, sizes);
    }
    let variance = numpy_sum(&terms);
    let parts: Vec<f64> = terms
        .iter()
        .zip(&sizes)
        .map(|(&term, &size)| term * term / (size - 1) as f64)
        .collect();
    let denominator = numpy_sum(&parts);
    let df = if denominator > 0.0 {
        variance * variance / denominator
    } else {
        f64::NAN
    };
    (variance, df, sizes)
}

/// The area `sum(w * mean)` over the non-`NaN` means, `NaN` when no weight is non-zero.
fn weighted_area(weights: &[f64], mean: &[f64]) -> f64 {
    if weights.iter().all(|&w| w == 0.0) {
        return f64::NAN;
    }
    let products: Vec<f64> = weights
        .iter()
        .zip(mean)
        .map(|(&w, &m)| {
            let p = w * if m.is_nan() { 0.0 } else { m };
            if p.is_nan() { 0.0 } else { p }
        })
        .collect();
    numpy_sum(&products)
}

/// Non-compartmental analysis of a sparse or destructive sampling design.
///
/// The area under the mean curve with the standard error of Bailer (1988), the degrees of
/// freedom of Nedelman, Gibiansky and Lau (1995) and, for the batch design of Nedelman and Jia
/// (1998), the covariance of Holder (2001):
///
/// `AUC = sum_j w_j mean_j`, `se = sqrt(Var[AUC])`, `CI = AUC +- t(1 - alpha/2, nu) se`,
///
/// with the weights `w` of the linear trapezoid rule over the nominal times. `auc_last` covers
/// the observed times up to the last positive mean (`area_window`), `auc_all` every observed
/// time; nothing is inserted at time 0. The peak is the first maximum of the observed means,
/// `cmax_se = sd / sqrt(n)` there for more than one animal. Fewer than two observed times flag
/// `NO_DATA` and leave both areas `NaN`.
///
/// The result has no sample dimensions: the seven parameters of one value each, the point
/// variable `n_animals` over `time` and the flags.
///
/// # Errors
///
/// [`KernelError::Sparse`] when [`SparseRequest::validate`] refuses the request, and
/// [`KernelError::Unsupported`] or [`KernelError::Batch`] for a unit which cannot be derived.
pub fn nca_sparse(request: &SparseRequest) -> Result<SparseAnswer, KernelError> {
    request.validate().map_err(KernelError::Sparse)?;
    let (n_rows, n_cols) = (request.n_rows, request.n_cols);
    let t = &request.times;
    let y = &request.values;
    let (mean, sd, n) = point_statistics(y, n_rows, n_cols);
    let observed: Vec<bool> = n.iter().map(|&c| c > 0).collect();
    let n_observed = observed.iter().filter(|&&o| o).count();
    let mut flags = NcaFlag::empty();
    if n_observed < 2 {
        flags |= NcaFlag::NO_DATA;
    }

    let mut all_weights = vec![0.0; n_cols];
    if n_observed > 1 {
        let grid: Vec<f64> = (0..n_cols).filter(|&j| observed[j]).map(|j| t[j]).collect();
        let weights = trapezoid_weights(&grid);
        for (slot, w) in (0..n_cols).filter(|&j| observed[j]).zip(weights) {
            all_weights[slot] = w;
        }
    }
    let through_last = area_window(&observed, &mean);
    let mut last_weights = vec![0.0; n_cols];
    if through_last.iter().filter(|&&o| o).count() > 1 {
        let slots: Vec<usize> = (0..n_cols).filter(|&j| through_last[j]).collect();
        let grid: Vec<f64> = slots.iter().map(|&j| t[j]).collect();
        for (slot, w) in slots.into_iter().zip(trapezoid_weights(&grid)) {
            last_weights[slot] = w;
        }
    }

    let auc_last = weighted_area(&last_weights, &mean);
    let auc_all = weighted_area(&all_weights, &mean);
    let (variance, df, sizes) = bailer_batches(&last_weights, y, n_rows, n_cols);
    let n_batches = sizes.len();
    let singletons = sizes.iter().filter(|&&size| size < 2).count();

    let mut peak: Option<usize> = None;
    for j in (0..n_cols).filter(|&j| observed[j]) {
        if peak.is_none_or(|p| mean[j] > mean[p]) {
            peak = Some(j);
        }
    }
    let (cmax, tmax, cmax_se) = match peak {
        None => (f64::NAN, f64::NAN, f64::NAN),
        Some(p) => (
            mean[p],
            t[p],
            if n[p] > 1 {
                sd[p] / (n[p] as f64).sqrt()
            } else {
                f64::NAN
            },
        ),
    };

    let magnitudes = [auc_last, variance.sqrt(), df, auc_all, cmax, cmax_se, tmax];
    let context = UnitContext {
        unit: &request.unit,
        time: &request.time_unit,
        dose: request.dose_unit.as_deref(),
        amount: None,
        volume: None,
    };
    let mut result = AnalysisResult::new(
        Vec::new(),
        Vec::new(),
        Attributes {
            substance: None,
            route: None,
            tissue: None,
            options: NcaOptions::default(),
        },
    );
    for ((name, expression), magnitude) in SPARSE_VARIABLES.iter().zip(magnitudes) {
        let (unit, factor) = parameter_unit(expression, &context).map_err(|e| {
            if e.is_unsupported() {
                KernelError::Unsupported {
                    what: format!("a unit which the registry of this build does not read ({e})"),
                }
            } else {
                KernelError::Batch(format!("the unit of {name} cannot be derived: {e}"))
            }
        })?;
        result.parameters.push(ParameterColumn {
            name: (*name).to_string(),
            unit,
            values: vec![magnitude * factor],
        });
    }
    result.point_variables.push(PointVariable {
        name: "n_animals".to_string(),
        unit: "dimensionless".to_string(),
        dim: "time".to_string(),
        size: n_cols,
        values: n.iter().map(|&c| c as f64).collect(),
    });
    result.flags = vec![flags.bits()];
    Ok(SparseAnswer {
        result,
        n_batches,
        singletons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pkpdutils_abi::SparseDesign;

    const NAN: f64 = f64::NAN;

    fn request(times: &[f64], values: &[f64], design: SparseDesign) -> SparseRequest {
        SparseRequest {
            times: times.to_vec(),
            values: values.to_vec(),
            n_rows: values.len() / times.len(),
            n_cols: times.len(),
            design,
            unit: "mg/l".to_string(),
            time_unit: "h".to_string(),
            dose_unit: None,
        }
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} != {b}");
    }

    /// Seven animals at three times: 2 at t=0, 3 at t=1, 2 at t=3.
    fn serial() -> (Vec<f64>, Vec<f64>) {
        (
            vec![0.0, 1.0, 3.0],
            vec![
                2.0, NAN, NAN, //
                4.0, NAN, NAN, //
                NAN, 10.0, NAN, //
                NAN, 12.0, NAN, //
                NAN, 14.0, NAN, //
                NAN, NAN, 6.0, //
                NAN, NAN, 8.0,
            ],
        )
    }

    #[test]
    fn the_trapezoid_weights_of_three_times_are_half_the_neighbour_gaps() {
        // by hand for t = 0, 1, 3: (1 - 0) / 2, (3 - 0) / 2 and (3 - 1) / 2
        assert_eq!(trapezoid_weights(&[0.0, 1.0, 3.0]), vec![0.5, 1.5, 1.0]);
        // two times: both weights half the gap
        assert_eq!(trapezoid_weights(&[1.0, 5.0]), vec![2.0, 2.0]);
    }

    #[test]
    fn the_weights_of_one_time_are_all_zero() {
        assert_eq!(trapezoid_weights(&[2.0]), vec![0.0]);
        assert!(trapezoid_weights(&[]).is_empty());
    }

    #[test]
    fn the_bailer_variance_of_a_serial_design_is_the_sum_of_w2_s2_over_n() {
        // by hand: means 3, 12, 7; s2 = 2, 4, 2; n = 2, 3, 2; w = 1/2, 3/2, 1, so
        // sum w2 s2 / n = 1/4 + 9/4 * 4 / 3 + 1 = 0.25 + 3 + 1
        let (t, y) = serial();
        let (variance, _, batches) = bailer_variance(&trapezoid_weights(&t), &y, 7, 3);
        close(variance, 4.25);
        assert_eq!(batches, 3);
    }

    #[test]
    fn the_degrees_of_freedom_of_a_serial_design_are_the_satterthwaite_value() {
        // by hand: terms 1/4, 3, 1 with n - 1 = 1, 2, 1, so
        // df = 4.25^2 / (0.25^2 / 1 + 3^2 / 2 + 1^2 / 1) = (289 / 16) / (89 / 16) = 289 / 89
        let (t, y) = serial();
        let (_, df, _) = bailer_variance(&trapezoid_weights(&t), &y, 7, 3);
        close(df, 289.0 / 89.0);
        let result = nca_sparse(&request(&t, &y, SparseDesign::Serial))
            .unwrap()
            .result;
        let value = |name: &str| {
            result
                .parameters
                .iter()
                .find(|p| p.name == name)
                .unwrap()
                .values[0]
        };
        close(value("auc_last"), 26.5);
        close(value("auc_last_se"), 4.25_f64.sqrt());
        close(value("auc_last_df"), 289.0 / 89.0);
        close(value("cmax_se"), 2.0 / 3.0_f64.sqrt());
        assert_eq!(value("tmax"), 1.0);
    }

    #[test]
    fn a_batch_design_carries_the_covariance_between_the_times_of_one_animal() {
        // three animals sampled at t = 0, 1, 2 (w = 1/2, 1, 1/2, n = 3 so w / n = 1/6, 1/3, 1/6):
        // A_i = (y0 + 2 y1 + y2) / 6 = 8/3, 4, 19/3 for (2, 4, 6), (4, 6, 8), (6, 11, 10);
        // the mean is 13/3, the squared deviations sum to 62/9, so s2 = 31/9, and one batch of
        // three animals gives Var = 3 * 31/9 = 31/3 and nu = m - 1 = 2. Treating the times as
        // independent would give 5, so the covariance is what the 31/3 carries.
        let y = [2.0, 4.0, 6.0, 4.0, 6.0, 8.0, 6.0, 11.0, 10.0];
        let (variance, df, batches) = bailer_variance(&[0.5, 1.0, 0.5], &y, 3, 3);
        close(variance, 31.0 / 3.0);
        close(df, 2.0);
        assert_eq!(batches, 1);

        // two batches: (2, 4, -), (4, 8, -) at t = 0, 1 and (-, 6, 2), (-, 10, 6) at t = 1, 2.
        // n = 2, 4, 2 so w / n = 1/4 for every time; A = 1.5, 3 and 2, 4;
        // batch 1: 2 * var(1.5, 3) = 2.25, batch 2: 2 * var(2, 4) = 4, Var = 6.25 and
        // nu = 6.25^2 / (2.25^2 / 1 + 4^2 / 1) = 625 / 337
        let y = [
            2.0, 4.0, NAN, //
            4.0, 8.0, NAN, //
            NAN, 6.0, 2.0, //
            NAN, 10.0, 6.0,
        ];
        let (variance, df, batches) = bailer_variance(&[0.5, 1.0, 0.5], &y, 4, 3);
        close(variance, 6.25);
        close(df, 625.0 / 337.0);
        assert_eq!(batches, 2);
    }

    #[test]
    fn a_pattern_of_one_animal_gives_no_variance_and_no_degrees_of_freedom() {
        // the two batches above plus a fifth animal alone at t = 0: its pattern holds one
        // member, so the variance of the whole area cannot be estimated
        let y = [
            2.0, 4.0, NAN, //
            4.0, 8.0, NAN, //
            NAN, 6.0, 2.0, //
            NAN, 10.0, 6.0, //
            3.0, NAN, NAN,
        ];
        let (variance, df, batches) = bailer_variance(&[0.5, 1.0, 0.5], &y, 5, 3);
        assert!(variance.is_nan() && df.is_nan());
        assert_eq!(batches, 3);
        // and through the front end the area is still there, only its error is not
        let answer = nca_sparse(&request(&[0.0, 1.0, 2.0], &y, SparseDesign::Batch)).unwrap();
        assert_eq!((answer.n_batches, answer.singletons), (3, 1));
        let result = answer.result;
        let get = |name: &str| {
            result
                .parameters
                .iter()
                .find(|p| p.name == name)
                .unwrap()
                .values[0]
        };
        assert!(get("auc_last").is_finite());
        assert!(get("auc_last_se").is_nan() && get("auc_last_df").is_nan());
    }

    #[test]
    fn a_time_nobody_was_sampled_at_has_no_mean_and_a_count_of_zero() {
        // column 1 is empty; column 0 has one animal (no sd), column 2 has two: 4 and 8
        let y = [1.0, NAN, 4.0, NAN, NAN, 8.0];
        let (mean, sd, n) = point_statistics(&y, 2, 3);
        assert_eq!(n, vec![1, 0, 2]);
        assert!(mean[1].is_nan() && sd[1].is_nan());
        assert_eq!(mean[0], 1.0);
        assert!(sd[0].is_nan());
        assert_eq!(mean[2], 6.0);
        // s2 = ((4 - 6)^2 + (8 - 6)^2) / 1 = 8
        close(sd[2], 8.0_f64.sqrt());
        // the observed window skips the empty time
        let window = area_window(&[true, false, true], &mean);
        assert_eq!(window, vec![true, false, true]);
    }

    #[test]
    fn one_sampling_time_gives_a_nan_area() {
        let y = [1.0, NAN, 3.0, NAN];
        let result = nca_sparse(&request(&[0.0, 1.0], &y, SparseDesign::Serial))
            .unwrap()
            .result;
        let get = |name: &str| {
            result
                .parameters
                .iter()
                .find(|p| p.name == name)
                .unwrap()
                .values[0]
        };
        assert!(get("auc_last").is_nan() && get("auc_all").is_nan());
        assert!(get("auc_last_se").is_nan() && get("auc_last_df").is_nan());
        // the peak of the single observed time is still there: mean 2 of two animals
        assert_eq!(get("cmax"), 2.0);
        close(get("cmax_se"), 1.0);
        assert_eq!(get("tmax"), 0.0);
        assert_eq!(result.flags, vec![NcaFlag::NO_DATA.bits()]);
        assert_eq!(result.point_variables[0].values, vec![2.0, 0.0]);
        assert!(result.validate().is_ok());
    }

    #[test]
    fn the_area_window_stops_after_the_last_positive_mean() {
        let all = [true; 4];
        assert_eq!(
            area_window(&all, &[1.0, 5.0, 0.0, 0.0]),
            vec![true, true, false, false]
        );
        // an unobserved time after the last positive mean is dropped too, one before is kept
        assert_eq!(
            area_window(&[true, false, true, true], &[1.0, NAN, 2.0, 0.0]),
            vec![true, false, true, false]
        );
        // nothing positive: no window at all
        assert_eq!(area_window(&all, &[0.0; 4]), vec![false; 4]);
    }

    #[test]
    fn a_malformed_request_is_an_error_and_not_a_panic() {
        let mut r = request(&[0.0, 1.0], &[1.0, 2.0], SparseDesign::Serial);
        r.n_rows = 3;
        assert!(matches!(nca_sparse(&r), Err(KernelError::Sparse(_))));
    }

    #[test]
    fn the_pairwise_summation_matches_numpy_at_every_block_boundary() {
        // `np.sum` of a[i] = (-1)^i (i + 1) 1e-3 + 1e8, as the bit pattern numpy produced; every
        // one differs from a left to right sum, so each runs the pairwise path: 8 is one block,
        // 9 a block and a tail, 16 two blocks, 128 the last single block, 129 and 200 the
        // recursion
        let expected: [(usize, u64); 6] = [
            (8, 0x41c7_d783_ffff_7cee),
            (9, 0x41ca_d274_8000_a3d8),
            (16, 0x41d7_d783_ffff_7cee),
            (128, 0x4207_d783_ffff_7cee),
            (129, 0x4208_0733_0800_851f),
            (200, 0x4212_a05f_1fff_999a),
        ];
        for (n, bits) in expected {
            let a: Vec<f64> = (0..n)
                .map(|i| (if i % 2 == 0 { 1.0 } else { -1.0 }) * (i + 1) as f64 * 1e-3 + 1e8)
                .collect();
            assert_eq!(numpy_sum(&a).to_bits(), bits, "n = {n}");
        }
    }

    #[test]
    fn twelve_sampling_times_reach_the_pairwise_sums_through_the_front_end() {
        // 40 animals at 12 times in several batches; the seven parameters as the bit patterns
        // of `nca_sparse` of the python host (numpy sums) on the same values
        let (r, c) = (40_usize, 12_usize);
        let mut y = vec![NAN; r * c];
        for i in 0..r {
            for j in 0..c {
                if (i * 5 + j * 3) % 4 != 0 && (i % 3 != 0 || j % 2 == 0) {
                    y[i * c + j] = ((i * 7 + j * 13) % 17) as f64 * 0.37
                        + 0.11 * j as f64
                        + 1.0 / (1 + i) as f64;
                }
            }
        }
        let t: Vec<f64> = (0..c).map(|j| (j as f64).powf(1.3)).collect();
        let answer = nca_sparse(&request(&t, &y, SparseDesign::Batch)).unwrap();
        let expected: [(&str, u64); 7] = [
            ("auc_last", 4635719407022746458),
            ("auc_last_se", 4612750861153895999),
            ("auc_last_df", 4624867174836361197),
            ("auc_all", 4635719407022746458),
            ("cmax", 4616649554149344940),
            ("cmax_se", 4599778889821543669),
            ("tmax", 4626309381818504046),
        ];
        for (name, bits) in expected {
            let column = answer.result.parameters.iter().find(|p| p.name == name);
            assert_eq!(column.unwrap().values[0].to_bits(), bits, "{name}");
        }
        assert_eq!(answer.singletons, 0);
    }

    #[test]
    fn an_empty_shape_gives_the_no_data_result_of_python() {
        for (n_rows, n_cols) in [(0_usize, 3_usize), (4, 0), (0, 0)] {
            let times: Vec<f64> = (0..n_cols).map(|j| j as f64).collect();
            let r = SparseRequest {
                times,
                values: vec![NAN; n_rows * n_cols],
                n_rows,
                n_cols,
                design: SparseDesign::Serial,
                unit: "mg/l".to_string(),
                time_unit: "h".to_string(),
                dose_unit: None,
            };
            let answer = nca_sparse(&r).unwrap();
            assert!(answer.result.validate().is_ok());
            assert_eq!(answer.result.flags, vec![NcaFlag::NO_DATA.bits()]);
            assert_eq!(answer.result.parameters.len(), 7);
            for p in &answer.result.parameters {
                assert!(p.values[0].is_nan(), "{} for {n_rows}x{n_cols}", p.name);
            }
            assert_eq!(answer.result.point_variables[0].size, n_cols);
            assert!(
                answer.result.point_variables[0]
                    .values
                    .iter()
                    .all(|&v| v == 0.0)
            );
            assert_eq!((answer.n_batches, answer.singletons), (0, 0));
        }
    }
}
