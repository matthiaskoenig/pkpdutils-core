//! The summary statistics of a parameter table.
//!
//! The port of the arithmetic of `ParameterResult.summarize` (`result.py`): the values of every
//! group are reduced over their samples to the mean, the standard deviation, the standard error,
//! the coefficient of variation, the t interval of the mean, the quartiles, the extremes, the
//! count and the geometric mean and coefficient of variation.
//!
//! Every mask is on the finite count, not on the number of samples: a value which is not finite
//! is a missing one and is left out of every statistic, `sd`, `se`, `cv` and the interval are
//! `NaN` where fewer than two values remain (the t quantile at one degree of freedom is finite
//! there, and is not used), and `mean`, `median`, the quartiles and the extremes are `NaN` where
//! none remains. The geometric statistics count the positive values separately: a zero or a
//! negative value leaves them without leaving the row, `geomean` needs one positive value and
//! `geocv` two.
//!
//! Sums go through [`pairwise_sum`], numpy's summation order, so that a mean and a variance agree
//! with `numpy.nanmean` and `numpy.nanvar` in the last bits, and `geocv` is
//! `sqrt(expm1(variance))`, which keeps its digits for the small variance of a tight group.
//! The flags `lognormal` and `discrete` of the request are the host's classification and are
//! not read: every statistic is computed for every variable, and the host drops the ones which
//! do not apply to it.

use crate::percentile::nan_percentile_rows;
use crate::stats::student_t_isf;
use pkpdutils_abi::{SummaryAnswer, SummaryRequest};

/// The block size below which numpy's pairwise summation is a plain unrolled loop.
const PAIRWISE_BLOCK: usize = 128;

/// The sum of `a` in the order of numpy's `add.reduce` on a contiguous axis: a plain loop below
/// eight values, eight interleaved accumulators up to 128, halves (cut at a multiple of eight)
/// above.
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= PAIRWISE_BLOCK {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - n % 8 {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut half = n / 2;
        half -= half % 8;
        pairwise_sum(&a[..half]) + pairwise_sum(&a[half..])
    }
}

/// `numpy.nanmean` of a row which has `count` values besides its `NaN`: `zeroed` is the row with
/// every missing value as 0. `NaN` for a count of 0.
fn mean_of(zeroed: &[f64], count: usize) -> f64 {
    pairwise_sum(zeroed) / count as f64
}

/// `numpy.nanvar(ddof=1)` of the same row, `NaN` below two values; `present` flags the values.
fn variance_of(zeroed: &[f64], present: impl Fn(usize) -> bool, count: usize) -> f64 {
    if count < 2 {
        return f64::NAN;
    }
    let mean = mean_of(zeroed, count);
    let squares: Vec<f64> = zeroed
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            if present(i) {
                let d = x - mean;
                d * d
            } else {
                0.0
            }
        })
        .collect();
    pairwise_sum(&squares) / (count - 1) as f64
}

/// All statistics of one group of values.
struct Group {
    mean: f64,
    sd: f64,
    se: f64,
    cv: f64,
    ci_low: f64,
    ci_high: f64,
    min: f64,
    max: f64,
    count: f64,
    geomean: f64,
    geocv: f64,
}

fn summarize_group(values: &[f64], alpha: f64) -> Group {
    let finite = |i: usize| values[i].is_finite();
    let zeroed: Vec<f64> = values
        .iter()
        .map(|&x| if x.is_finite() { x } else { 0.0 })
        .collect();
    let count = values.iter().filter(|x| x.is_finite()).count();
    let mean = if count > 0 {
        mean_of(&zeroed, count)
    } else {
        f64::NAN
    };
    let sd = variance_of(&zeroed, finite, count).sqrt();
    let n = count as f64;
    let se = sd / n.sqrt();
    let tq = student_t_isf(alpha / 2.0, (n - 1.0).max(1.0));
    let (ci_low, ci_high) = if count > 1 {
        (mean - tq * se, mean + tq * se)
    } else {
        (f64::NAN, f64::NAN)
    };
    let (min, max) =
        values
            .iter()
            .filter(|x| x.is_finite())
            .fold((f64::NAN, f64::NAN), |(lo, hi), &x| {
                (
                    if lo.is_nan() || x < lo { x } else { lo },
                    if hi.is_nan() || x > hi { x } else { hi },
                )
            });
    // the logarithms of the finite positive values, a zero elsewhere
    let positive = |i: usize| values[i].is_finite() && values[i] > 0.0;
    let logs: Vec<f64> = values
        .iter()
        .enumerate()
        .map(|(i, &x)| if positive(i) { x.ln() } else { 0.0 })
        .collect();
    let n_pos = (0..values.len()).filter(|&i| positive(i)).count();
    let geomean = if n_pos > 0 {
        mean_of(&logs, n_pos).exp()
    } else {
        f64::NAN
    };
    let geocv = if n_pos > 1 {
        variance_of(&logs, positive, n_pos).exp_m1().sqrt()
    } else {
        f64::NAN
    };
    Group {
        mean,
        sd,
        se,
        cv: sd / mean.abs(),
        ci_low,
        ci_high,
        min,
        max,
        count: n,
        geomean,
        geocv,
    }
}

/// Summarize the groups of a request, every statistic of every variable.
///
/// A request whose lengths disagree with its dimensions (`SummaryRequest::is_consistent`) gives
/// an answer of empty vectors, and a `ci_level` which is not strictly between 0 and 1 gives a
/// `NaN` interval, where python raises, since a kernel does not panic.
pub fn summarize(request: &SummaryRequest) -> SummaryAnswer {
    let rows = request.n_vars.saturating_mul(request.n_groups);
    let consistent = request.is_consistent();
    let rows = if consistent { rows } else { 0 };
    let n_samples = request.n_samples;
    let alpha = if request.ci_level > 0.0 && request.ci_level < 1.0 {
        1.0 - request.ci_level
    } else {
        f64::NAN
    };
    let mut answer = SummaryAnswer {
        mean: Vec::with_capacity(rows),
        sd: Vec::with_capacity(rows),
        se: Vec::with_capacity(rows),
        cv: Vec::with_capacity(rows),
        ci_low: Vec::with_capacity(rows),
        ci_high: Vec::with_capacity(rows),
        median: Vec::new(),
        q25: Vec::new(),
        q75: Vec::new(),
        min: Vec::with_capacity(rows),
        max: Vec::with_capacity(rows),
        count: Vec::with_capacity(rows),
        geomean: Vec::with_capacity(rows),
        geocv: Vec::with_capacity(rows),
    };
    // the rows with every missing value as NaN, which the percentile reads
    let filled: Vec<f64> = request
        .values
        .iter()
        .map(|&x| if x.is_finite() { x } else { f64::NAN })
        .collect();
    for r in 0..rows {
        let group = summarize_group(&request.values[r * n_samples..(r + 1) * n_samples], alpha);
        answer.mean.push(group.mean);
        answer.sd.push(group.sd);
        answer.se.push(group.se);
        answer.cv.push(group.cv);
        answer.ci_low.push(group.ci_low);
        answer.ci_high.push(group.ci_high);
        answer.min.push(group.min);
        answer.max.push(group.max);
        answer.count.push(group.count);
        answer.geomean.push(group.geomean);
        answer.geocv.push(group.geocv);
    }
    if n_samples == 0 || rows == 0 {
        answer.median = vec![f64::NAN; rows];
        answer.q25 = vec![f64::NAN; rows];
        answer.q75 = vec![f64::NAN; rows];
    } else {
        answer.median = nan_percentile_rows(&filled, n_samples, 50.0);
        answer.q25 = nan_percentile_rows(&filled, n_samples, 25.0);
        answer.q75 = nan_percentile_rows(&filled, n_samples, 75.0);
    }
    answer
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAN: f64 = f64::NAN;

    /// One variable, one group.
    fn one(values: &[f64], ci_level: f64) -> SummaryAnswer {
        summarize(&SummaryRequest {
            values: values.to_vec(),
            n_vars: 1,
            n_groups: 1,
            n_samples: values.len(),
            ci_level,
            lognormal: vec![true],
            discrete: vec![false],
        })
    }

    fn close(got: f64, want: f64, rel: f64) {
        assert!(
            (got - want).abs() <= rel * want.abs(),
            "got {got:e}, want {want:e}"
        );
    }

    // The references below were computed by running the formulas of `result.py:1290-1361`
    // (numpy `nanmean`, `nanstd(ddof=1)`, `scipy.stats.t.isf`, `nanmedian`, `nanpercentile`,
    // `exp(nanmean(log))`, `sqrt(expm1(nanvar(log, ddof=1)))`) on the same values.

    #[test]
    fn the_mean_and_spread_of_a_group_match_the_python_formulas() {
        let a = one(&[2.0, 4.0, 9.0], 0.95);
        // mean 5, deviations -3, -1, 4: variance 26 / 2 = 13, sd = sqrt(13)
        assert_eq!(a.mean, [5.0]);
        assert_eq!(a.sd, [13.0_f64.sqrt()]);
        assert_eq!(a.se, [13.0_f64.sqrt() / 3.0_f64.sqrt()]);
        assert_eq!(a.cv, [13.0_f64.sqrt() / 5.0]);
        assert_eq!(a.count, [3.0]);
        assert_eq!((a.median[0], a.q25[0], a.q75[0]), (4.0, 3.0, 6.5));
        assert_eq!((a.min[0], a.max[0]), (2.0, 9.0));
        close(a.sd[0], 3.605551275463989, 1e-15);
        close(a.se[0], 2.0816659994661326, 1e-15);
    }

    #[test]
    fn the_coefficient_of_variation_uses_the_absolute_mean() {
        let a = one(&[-2.0, -4.0, -9.0], 0.95);
        assert_eq!(a.mean, [-5.0]);
        assert_eq!(a.cv, [13.0_f64.sqrt() / 5.0]);
    }

    #[test]
    fn a_group_of_one_sample_has_no_spread_and_no_interval() {
        let a = one(&[7.5], 0.95);
        for spread in [&a.sd, &a.se, &a.cv, &a.ci_low, &a.ci_high, &a.geocv] {
            assert!(spread[0].is_nan());
        }
        assert_eq!(a.mean, [7.5]);
        assert_eq!(a.median, [7.5]);
        assert_eq!((a.q25[0], a.q75[0]), (7.5, 7.5));
        assert_eq!((a.min[0], a.max[0]), (7.5, 7.5));
        assert_eq!(a.count, [1.0]);
        close(a.geomean[0], 7.5, 1e-15);
    }

    #[test]
    fn a_single_finite_value_among_missing_ones_is_a_group_of_one() {
        // the mask is on the finite count (1), not the sample count (4)
        let a = one(&[NAN, 3.0, f64::INFINITY, NAN], 0.95);
        assert_eq!(a.count, [1.0]);
        assert_eq!(a.mean, [3.0]);
        for spread in [&a.sd, &a.se, &a.cv, &a.ci_low, &a.ci_high] {
            assert!(spread[0].is_nan());
        }
    }

    #[test]
    fn a_variable_of_only_nan_is_nan_everywhere_and_counts_zero() {
        let a = one(&[NAN, NAN, f64::INFINITY], 0.95);
        assert_eq!(a.count, [0.0]);
        for s in [
            &a.mean, &a.sd, &a.se, &a.cv, &a.ci_low, &a.ci_high, &a.median, &a.q25, &a.q75, &a.min,
            &a.max, &a.geomean, &a.geocv,
        ] {
            assert!(s[0].is_nan());
        }
    }

    #[test]
    fn a_group_without_samples_is_missing_and_does_not_panic() {
        let a = one(&[], 0.95);
        assert_eq!(a.count, [0.0]);
        assert!(a.mean[0].is_nan() && a.median[0].is_nan() && a.q75[0].is_nan());
    }

    #[test]
    fn the_interval_is_the_t_interval_at_the_finite_count() {
        // t(0.975, 2) = 4.302652729749462 and the interval is mean +- t * se
        let a = one(&[2.0, 4.0, 9.0], 0.95);
        close(a.ci_low[0], -3.9566858950295973, 1e-13);
        close(a.ci_high[0], 13.956685895029597, 1e-13);
        // the degrees of freedom follow the finite count (12 values, 11 df) and not the 13
        // samples: t(0.975, 11) = 2.200985160082949
        let v = [
            2.7202889111845105,
            3.2519256302754624,
            2.3060066065514233,
            NAN,
            2.0692733953078184,
            1.4993205922523765,
            2.818165556958034,
            6.074678993975963,
            2.02319124461201,
            f64::INFINITY,
            3.6470011607840322,
            3.3673706082975468,
            2.8957627545570936,
        ];
        let a = one(&v, 0.95);
        assert_eq!(a.count, [11.0]);
        close(a.mean[0], 2.970271404977843, 1e-15);
        close(a.sd[0], 1.213969017545109, 1e-14);
        close(a.ci_low[0], 2.1547159218958947, 1e-13);
        close(a.ci_high[0], 3.7858268880597916, 1e-13);
        close(a.median[0], 2.818165556958034, 1e-15);
        close(a.q25[0], 2.187640000929621, 1e-14);
        close(a.q75[0], 3.3096481192865044, 1e-14);
        close(a.geomean[0], 2.78428183407289, 1e-14);
        close(a.geocv[0], 0.3805803591882349, 1e-13);
    }

    #[test]
    fn the_confidence_level_sets_the_interval() {
        // the same group at 90 %: t(0.95, 2) = 2.919985580353725, a narrower interval
        let a = one(&[2.0, 4.0, 9.0], 0.90);
        let half = 2.919985580353725 * a.se[0];
        close(a.ci_low[0], 5.0 - half, 1e-12);
        close(a.ci_high[0], 5.0 + half, 1e-12);
        // an invalid level is a NaN interval and nothing else
        let bad = one(&[2.0, 4.0, 9.0], 1.0);
        assert!(bad.ci_low[0].is_nan() && bad.ci_high[0].is_nan());
        assert_eq!(bad.mean, [5.0]);
    }

    #[test]
    fn the_geometric_statistics_use_the_positive_values_alone() {
        let a = one(&[1.0, 4.0, 0.0, -2.0], 0.95);
        // exp(mean(ln 1, ln 4)) = 2, and the variance of (0, ln 4) with ddof 1 is
        // ln(4)^2 / 2, so geocv = sqrt(expm1(ln(4)^2 / 2))
        close(a.geomean[0], 2.0, 1e-15);
        close(a.geocv[0], 1.2704581124166188, 1e-14);
        // the zero and the negative value stay in the arithmetic statistics
        assert_eq!(a.count, [4.0]);
        assert_eq!(a.mean, [0.75]);
        close(a.sd[0], 2.5, 1e-15);
        assert_eq!((a.min[0], a.max[0]), (-2.0, 4.0));
    }

    #[test]
    fn one_positive_value_has_a_geometric_mean_but_no_geometric_coefficient_of_variation() {
        let a = one(&[0.0, 5.0, -1.0, NAN], 0.95);
        close(a.geomean[0], 5.0, 1e-15);
        assert!(a.geocv[0].is_nan());
        assert!(a.sd[0].is_finite());
        // no positive value at all: neither geometric statistic
        let a = one(&[0.0, -5.0], 0.95);
        assert!(a.geomean[0].is_nan() && a.geocv[0].is_nan());
        assert!(a.sd[0].is_finite());
    }

    #[test]
    fn the_geometric_coefficient_of_variation_keeps_its_digits_for_a_tight_group() {
        // exp(x) - 1 loses about seven digits at a variance of 3.5e-10
        let c: Vec<f64> = (1..=6).map(|i| 1.0 + 1e-5 * i as f64).collect();
        let a = one(&c, 0.95);
        close(a.geocv[0], 1.870763217203228e-05, 1e-13);
        close(a.geomean[0], 1.0000349998541718, 1e-15);
        close(a.sd[0], 1.870828693387358e-05, 1e-12);
    }

    #[test]
    fn a_large_group_sums_in_the_order_of_numpy() {
        // 200 values, 12 of them missing, so the sums cross the 128 block of the pairwise
        // summation; the references are numpy's own results on the same values
        let v: Vec<f64> = (0..200)
            .map(|i| {
                if i % 17 == 0 {
                    NAN
                } else {
                    ((i * 37) % 101) as f64 / 10.0 + 0.5
                }
            })
            .collect();
        let a = one(&v, 0.90);
        assert_eq!(a.count, [188.0]);
        assert_eq!(a.mean, [5.553723404255319]);
        assert_eq!(a.sd, [2.928622710748041]);
        assert_eq!(a.se, [0.21359176340190197]);
        close(a.ci_low[0], 5.200647058626233, 1e-13);
        close(a.ci_high[0], 5.906799749884405, 1e-13);
        assert_eq!(a.median, [5.55]);
        assert_eq!((a.q25[0], a.q75[0]), (3.0, 8.1));
        assert_eq!(a.geomean, [4.5289617108750395]);
        assert_eq!(a.geocv, [0.8322538125092614]);
    }

    #[test]
    fn variables_and_groups_are_laid_out_row_major() {
        // (2 variables, 2 groups, 3 samples), the answers are (2, 2)
        let request = SummaryRequest {
            values: vec![
                1.0, 2.0, 3.0, //
                10.0, 20.0, 60.0, //
                NAN, 5.0, NAN, //
                4.0, 4.0, 4.0,
            ],
            n_vars: 2,
            n_groups: 2,
            n_samples: 3,
            ci_level: 0.95,
            lognormal: vec![true, false],
            discrete: vec![false, true],
        };
        let a = summarize(&request);
        assert_eq!(a.mean, [2.0, 30.0, 5.0, 4.0]);
        assert_eq!(a.count, [3.0, 3.0, 1.0, 3.0]);
        assert_eq!(a.median, [2.0, 20.0, 5.0, 4.0]);
        assert_eq!(a.min, [1.0, 10.0, 5.0, 4.0]);
        assert_eq!(a.max, [3.0, 60.0, 5.0, 4.0]);
        // a discrete variable is computed too, the host drops it
        assert_eq!(a.sd[3], 0.0);
        assert_eq!(a.sd[0], 1.0);
        assert!(a.sd[2].is_nan());
    }

    #[test]
    fn an_inconsistent_request_answers_with_nothing() {
        let a = summarize(&SummaryRequest {
            values: vec![1.0, 2.0, 3.0],
            n_vars: 2,
            n_groups: 1,
            n_samples: 2,
            ci_level: 0.95,
            lognormal: vec![true, true],
            discrete: vec![false, false],
        });
        assert!(a.mean.is_empty() && a.median.is_empty() && a.geocv.is_empty());
    }

    #[test]
    fn the_pairwise_sum_splits_where_numpy_splits() {
        // numpy.sum of 1e16 followed by fifteen ones is 1e16 + 14 (eight interleaved
        // accumulators), where a left to right sum absorbs every one and gives 1e16
        let mut a = vec![1.0; 16];
        a[0] = 1e16;
        assert_eq!(pairwise_sum(&a), 10_000_000_000_000_014.0);
        // below eight values the order is left to right
        assert_eq!(pairwise_sum(&[1e16, 1.0, 1.0, 1.0]), 1e16);
        // above 128 values the halves are cut at a multiple of eight: numpy gives 1e16 + 188
        let mut b = vec![1.0; 200];
        b[0] = 1e16;
        assert_eq!(pairwise_sum(&b), 10_000_000_000_000_188.0);
        assert_eq!(pairwise_sum(&[]), 0.0);
    }
}
