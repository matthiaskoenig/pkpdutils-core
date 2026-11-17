//! The monoexponential time to steady state, the port of `_monoexponential_row` of
//! `pkpdutils.nca.tss`.
//!
//! The troughs of the dosing intervals approach a plateau as `C(t) = C_ss * (1 - exp(-k t))`. For
//! a given `k` the plateau is linear in the data, `C_ss = sum(y g) / sum(g g)` with
//! `g = 1 - exp(-k t)`, so only `k` is searched and the plateau is profiled out at every step.
//!
//! Python runs Levenberg-Marquardt from three starts. Here the same profiled cost is minimized
//! over `log10(k)` by a golden-section search on a bracket around each start, and the lowest cost
//! of the three wins, so the two agree on the minimum and not on the iterates.

use crate::KernelError;
use crate::stats::student_t_sf;
use crate::terminal::{WindowStats, window_statistics};
use pkpdutils_abi::{TssAnswer, TssMethod, TssRequest};

/// The starting values of `k`, as multiples of `1 / t_last` (`_STARTS` of `tss.py`).
const STARTS: [f64; 3] = [0.5, 2.0, 8.0];

/// The half width, in decades of `k`, of the bracket searched around a start.
const BRACKET_DECADES: f64 = 2.0;

/// Golden-section iterations; the bracket shrinks by 0.618 each, so 120 reach the resolution of
/// a double far below the noise of the cost.
const ITERATIONS: usize = 120;

/// A minimum closer than this, in decades, to a bracket edge is an edge: the cost still falls
/// towards it, so the series has no plateau of its own inside the bracket.
const EDGE_TOLERANCE: f64 = 1e-6;

/// The least squares plateau for a rate constant, `0` when the basis `g` is all zero as
/// `tss.py:134` does.
fn profiled_plateau(k: f64, time: &[f64], value: &[f64]) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (&t, &y) in time.iter().zip(value) {
        let g = 1.0 - (-k * t).exp();
        numerator += y * g;
        denominator += g * g;
    }
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// The residual sum of squares of the profiled model at `k = 10 ** x`.
fn profiled_cost(x: f64, time: &[f64], value: &[f64]) -> f64 {
    let k = 10f64.powf(x);
    let plateau = profiled_plateau(k, time, value);
    time.iter()
        .zip(value)
        .map(|(&t, &y)| {
            let r = y - plateau * (1.0 - (-k * t).exp());
            r * r
        })
        .sum()
}

/// Golden-section minimum of `f` on `[low, high]`: the abscissa and its value.
fn golden_section(f: impl Fn(f64) -> f64, low: f64, high: f64) -> (f64, f64) {
    let ratio = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (low, high);
    let mut c = b - ratio * (b - a);
    let mut d = a + ratio * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..ITERATIONS {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - ratio * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + ratio * (b - a);
            fd = f(d);
        }
    }
    if fc < fd { (c, fc) } else { (d, fd) }
}

/// The time to `fraction` of the plateau and the plateau of troughs `value` taken at `time`.
///
/// `-ln(1 - fraction) / k` and the plateau at the best `k` of the three starts; `(NaN, NaN)` for
/// fewer than three troughs, a trough which is not positive, a time which is not finite or a last
/// time which is not positive (the guard of `tss.py`), and wherever the search ends on the edge of
/// its bracket, i.e. the troughs show no plateau of their own (flat, falling or still linear).
pub(crate) fn monoexponential(time: &[f64], value: &[f64], fraction: f64) -> (f64, f64) {
    let nan = (f64::NAN, f64::NAN);
    let Some(&last) = time.last() else { return nan };
    if time.len() < 3
        || time.len() != value.len()
        || !value.iter().all(|&y| y > 0.0 && y.is_finite())
        || !time.iter().all(|t| t.is_finite())
        || last <= 0.0
    {
        return nan;
    }
    let mut best: Option<(f64, f64, f64, f64)> = None; // x, cost, bracket low, bracket high
    for factor in STARTS {
        let start = (factor / last).log10();
        let (low, high) = (start - BRACKET_DECADES, start + BRACKET_DECADES);
        let (x, cost) = golden_section(|x| profiled_cost(x, time, value), low, high);
        if !cost.is_finite() {
            continue;
        }
        if best.is_none_or(|(_, c, _, _)| cost < c) {
            best = Some((x, cost, low, high));
        }
    }
    let Some((x, _, low, high)) = best else {
        return nan;
    };
    if x - low < EDGE_TOLERANCE || high - x < EDGE_TOLERANCE {
        return nan;
    }
    let k = 10f64.powf(x);
    if !k.is_finite() || k <= 0.0 {
        return nan;
    }
    (-(-fraction).ln_1p() / k, profiled_plateau(k, time, value))
}

/// Whether the troughs of one window no longer trend: the two-sided t test of its slope against
/// zero with `n - 2` degrees of freedom does not reject at `alpha`.
///
/// A window which is no window (`n` is `NaN`) has no answer. An exact straight line has no
/// residual scatter, `se_slope` is 0 and the t statistic is `|slope| / 0`: the p value is not
/// read there, a slope of exactly 0 is no trend and any other slope is one (`_stepwise` of
/// `tss.py`).
fn no_trend(window: &WindowStats, alpha: f64) -> bool {
    if !window.n.is_finite() {
        return false;
    }
    if window.se_slope == 0.0 {
        return window.slope == 0.0;
    }
    let t_statistic = window.slope.abs() / window.se_slope;
    2.0 * student_t_sf(t_statistic, window.n - 2.0) > alpha
}

/// The first interval from which the troughs no longer trend, per row: the port of `_stepwise`
/// of `tss.py` (PKNCA `pk.tss.stepwise.linear`).
///
/// `time`, `value`, `start` and `valid` are `n_rows * n_cols` row-major arrays: the troughs of a
/// row, the start of the interval of each and which entries are troughs. Every window (a suffix
/// of the troughs) is regressed at once by [`window_statistics`] and the answer of a row is the
/// `start` of the first window whose slope test does not reject at `alpha`, `NaN` when every
/// window of at least three troughs still trends. Arrays shorter than the shape give all `NaN`.
pub(crate) fn stepwise(
    time: &[f64],
    value: &[f64],
    start: &[f64],
    valid: &[bool],
    n_rows: usize,
    n_cols: usize,
    alpha: f64,
) -> Vec<f64> {
    let Some(size) = n_rows.checked_mul(n_cols) else {
        return vec![f64::NAN; n_rows];
    };
    if time.len() < size || value.len() < size || start.len() < size || valid.len() < size {
        return vec![f64::NAN; n_rows];
    }
    let mut windows = vec![WindowStats::NONE; n_cols];
    (0..n_rows)
        .map(|r| {
            let row = r * n_cols..(r + 1) * n_cols;
            window_statistics(
                &time[row.clone()],
                &value[row.clone()],
                &valid[row.clone()],
                &mut windows,
            );
            windows
                .iter()
                .position(|w| no_trend(w, alpha))
                .map_or(f64::NAN, |s| start[row.start + s])
        })
        .collect()
}

/// The time to steady state of every sample of a request.
///
/// The monoexponential estimate fits the troughs of a row read from the start of its first
/// interval and reports the start back added, as `time_to_steady_state` of the host does; a row
/// without a trough is `NaN`. The stepwise estimate is [`stepwise`] and has no plateau.
///
/// # Errors
///
/// [`KernelError::Tss`] when `TssRequest::validate` refuses the request.
pub fn time_to_steady_state(request: &TssRequest) -> Result<TssAnswer, KernelError> {
    request.validate().map_err(KernelError::Tss)?;
    let (n_rows, n_cols) = (request.n_rows, request.n_cols);
    if request.method == TssMethod::Stepwise {
        return Ok(TssAnswer {
            tss: stepwise(
                &request.time,
                &request.value,
                &request.start,
                &request.valid,
                n_rows,
                n_cols,
                request.alpha,
            ),
            c_ss: Vec::new(),
        });
    }
    let mut tss = Vec::with_capacity(n_rows);
    let mut c_ss = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let row = r * n_cols..(r + 1) * n_cols;
        let count = request.valid[row.clone()].iter().filter(|&&v| v).count();
        if count == 0 {
            tss.push(f64::NAN);
            c_ss.push(f64::NAN);
            continue;
        }
        let first = request.start[row.start];
        let time: Vec<f64> = request.time[row.start..row.start + count]
            .iter()
            .map(|t| t - first)
            .collect();
        let (estimate, plateau) = monoexponential(
            &time,
            &request.value[row.start..row.start + count],
            request.fraction,
        );
        tss.push(estimate + first);
        c_ss.push(plateau);
    }
    Ok(TssAnswer { tss, c_ss })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rises(plateau: f64, k: f64, times: &[f64]) -> Vec<f64> {
        times
            .iter()
            .map(|t| plateau * (1.0 - (-k * t).exp()))
            .collect()
    }

    fn grid() -> Vec<f64> {
        (1..=12).map(f64::from).collect()
    }

    #[test]
    fn a_clean_monoexponential_approach_recovers_its_rate_and_plateau() {
        let t = grid();
        let (tss, plateau) = monoexponential(&t, &rises(10.0, 0.3, &t), 0.9);
        assert!((plateau - 10.0).abs() < 1e-9, "{plateau}");
        // python: 7.67528364331349 = ln(10) / 0.3
        assert!((tss - 10f64.ln() / 0.3).abs() < 1e-8, "{tss}");
        // the fraction enters as -ln(1 - f): 0.5 gives ln(2) / 0.3
        let (half, _) = monoexponential(&t, &rises(10.0, 0.3, &t), 0.5);
        assert!((half - 2f64.ln() / 0.3).abs() < 1e-8, "{half}");
    }

    #[test]
    fn a_noisy_series_matches_the_python_minimum() {
        let t = grid();
        let v: Vec<f64> = t
            .iter()
            .map(|t| 10.0 * (1.0 - (-0.3 * t).exp()) * (1.0 + 0.05 * (7.0 * t).sin()))
            .collect();
        let (tss, plateau) = monoexponential(&t, &v, 0.9);
        // the exact minimum of the profiled cost: the root of its derivative computed at 40
        // decimal digits with mpmath, not any optimizer's output. Python's default LM tolerances
        // stop at 8.054337149, 2.9e-7 short of it, which is a looseness this test must not pin
        let (ref_tss, ref_plateau) = (8.054339489045725, 10.210161963657178);
        assert!((tss / ref_tss - 1.0).abs() < 1e-7, "{tss}");
        assert!((plateau / ref_plateau - 1.0).abs() < 1e-7, "{plateau}");
    }

    #[test]
    fn a_slow_and_a_fast_approach_are_found_by_the_start_whose_window_holds_them() {
        // measured relative errors: 9e-12, 3e-12, 2e-13, 4e-12 and, where the cost surface nearly
        // flattens against the plateau (250), 2.7e-9, so that case allows 1e-7, loose because the NaN of the mutant it kills is no small error, and the rest 1e-9
        // noise free, k * t_last = 0.01: only the window of the 0.5 start (0.005 and up) holds
        // it, the windows of the 2 and 8 starts begin at 0.02 and 0.08 and end on their edge
        let t = grid();
        for (k_t_last, rel) in [
            (0.006, 1e-9),
            (0.01, 1e-9),
            (0.015, 1e-9),
            (150.0, 1e-9),
            (250.0, 1e-7),
        ] {
            let k = k_t_last / 12.0;
            let (tss, plateau) = monoexponential(&t, &rises(10.0, k, &t), 0.9);
            assert!(
                (tss / (10f64.ln() / k) - 1.0).abs() < rel,
                "{k_t_last}: {tss}"
            );
            assert!((plateau / 10.0 - 1.0).abs() < rel, "{k_t_last}: {plateau}");
        }
    }

    #[test]
    fn a_flat_series_has_no_rate_and_returns_nan() {
        let t = grid();
        let (tss, plateau) = monoexponential(&t, &[5.0; 12], 0.9);
        assert!(tss.is_nan() && plateau.is_nan());
    }

    #[test]
    fn a_series_that_never_plateaus_returns_nan() {
        let t = grid();
        let line: Vec<f64> = t.clone();
        let falling: Vec<f64> = t.iter().map(|t| 12.0 - t).collect();
        for v in [line, falling] {
            let (tss, plateau) = monoexponential(&t, &v, 0.9);
            assert!(tss.is_nan() && plateau.is_nan(), "{v:?}");
        }
    }

    #[test]
    fn the_three_starts_find_the_same_optimum_on_a_well_posed_series() {
        let t = grid();
        let v = rises(10.0, 0.3, &t);
        let (tss, _) = monoexponential(&t, &v, 0.9);
        let k = -(0.1f64).ln() / tss;
        let at = |k: f64| profiled_cost(k.log10(), &t, &v);
        for factor in STARTS {
            assert!(at(k) <= at(factor / 12.0), "start {factor}");
        }
        // and each start alone reaches that k, not only the best of them
        for factor in STARTS {
            let s = (factor / 12.0f64).log10();
            let (x, _) = golden_section(|x| profiled_cost(x, &t, &v), s - 2.0, s + 2.0);
            assert!(
                (10f64.powf(x) / 0.3 - 1.0).abs() < 1e-6,
                "start {factor}: {x}"
            );
        }
    }

    #[test]
    fn the_plateau_is_zero_when_the_basis_is_degenerate() {
        let zeros = [0.0; 4];
        assert_eq!(profiled_plateau(0.7, &zeros, &[1.0, 2.0, 3.0, 4.0]), 0.0);
        // a non-degenerate basis gives the least squares plateau: g = 1 - e^-1 at t = 1 for k = 1
        let g = 1.0 - (-1.0f64).exp();
        let p = profiled_plateau(1.0, &[1.0, 1.0, 1.0], &[g, g, g]);
        assert!((p - 1.0).abs() < 1e-15, "{p}");
    }

    #[test]
    fn the_guard_of_python_answers_nan() {
        let nan = |t: &[f64], v: &[f64]| {
            let (a, b) = monoexponential(t, v, 0.9);
            a.is_nan() && b.is_nan()
        };
        assert!(nan(&[1.0, 2.0], &[1.0, 2.0])); // fewer than three
        assert!(nan(&[1.0, 2.0, 3.0], &[1.0, 0.0, 2.0])); // a trough of 0
        assert!(nan(&[1.0, 2.0, 3.0], &[1.0, -1.0, 2.0]));
        assert!(nan(&[1.0, 2.0, 3.0], &[1.0, f64::NAN, 2.0]));
        assert!(nan(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0])); // last time 0, python guards before the basis
        assert!(nan(&[1.0, f64::NAN, 3.0], &[1.0, 2.0, 3.0]));
        assert!(nan(&[1.0, 2.0, 3.0], &[1.0, 2.0])); // length mismatch
        assert!(nan(&[], &[]));
    }

    /// One row padded to six columns: starts are `10 * (time - 1)`, so a start is not a time.
    fn padded(rows: &[(&[f64], &[f64])]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>) {
        let (mut t, mut v, mut s, mut ok) = (vec![], vec![], vec![], vec![]);
        for (time, value) in rows {
            for c in 0..6 {
                let here = c < time.len();
                t.push(if here { time[c] } else { 0.0 });
                v.push(if here { value[c] } else { 0.0 });
                s.push(if here { time[c] * 10.0 - 10.0 } else { 0.0 });
                ok.push(here);
            }
        }
        (t, v, s, ok)
    }

    fn walk(rows: &[(&[f64], &[f64])], alpha: f64) -> Vec<f64> {
        let (t, v, s, ok) = padded(rows);
        stepwise(&t, &v, &s, &ok, rows.len(), 6, alpha)
    }

    const T6: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    #[test]
    fn a_series_which_stops_trending_returns_the_start_of_the_first_flat_window() {
        // python p values of the windows from index 0: 0.03501005, 0.11054577, 0.19824607,
        // 0.78770438 (scipy t.sf, df 4, 3, 2, 1)
        let v = [1.0, 1.8, 2.3, 2.4, 2.38, 2.41];
        assert_eq!(walk(&[(&T6, &v)], 0.05), [10.0]);
        // the p value of the first window sits 1e-5 either side of alpha
        assert_eq!(walk(&[(&T6, &v)], 0.0350), [0.0]);
        assert_eq!(walk(&[(&T6, &v)], 0.0351), [10.0]);
    }

    #[test]
    fn an_exact_straight_line_is_a_trend_unless_its_slope_is_zero() {
        // a line of slope 1 has se_slope 0 and is a trend in every window: no answer
        assert!(walk(&[(&T6[..5], &[1.0, 2.0, 3.0, 4.0, 5.0])], 0.05)[0].is_nan());
        let line = |slope: f64, se_slope: f64| WindowStats {
            n: 5.0,
            slope,
            se_slope,
            ..WindowStats::NONE
        };
        assert!(!no_trend(&line(1.0, 0.0), 0.05));
        assert!(!no_trend(&line(-1.0, 0.0), 0.05));
        assert!(no_trend(&line(0.0, 0.0), 0.05));
        // alpha at its extremes does not change the verdict: the p value is never read
        assert!(!no_trend(&line(1.0, 0.0), 0.0));
        assert!(no_trend(&line(0.0, 0.0), 1.0));
    }

    #[test]
    fn a_p_value_either_side_of_alpha_decides_on_n_minus_two_degrees_of_freedom() {
        let w = |n: f64| WindowStats {
            n,
            slope: 1.0,
            se_slope: 1.0,
            ..WindowStats::NONE
        };
        // df 1, t = 1: p = 1 - 2 atan(1) / pi = 0.5 exactly
        assert!(no_trend(&w(3.0), 0.5 - 1e-6));
        assert!(!no_trend(&w(3.0), 0.5 + 1e-6));
        // df 2, t = 1: p = 1 - 1 / sqrt(3) = 0.42264973081
        assert!(no_trend(&w(4.0), 0.42264973 - 1e-6));
        assert!(!no_trend(&w(4.0), 0.42264973 + 1e-6));
        // the sign of the slope does not matter
        let neg = WindowStats {
            slope: -1.0,
            ..w(3.0)
        };
        assert!(no_trend(&neg, 0.5 - 1e-6) && !no_trend(&neg, 0.5 + 1e-6));
        // a window which is no window has no answer, whatever alpha
        assert!(!no_trend(&WindowStats::NONE, 0.0));
        assert!(!no_trend(&WindowStats::NONE, 1.0));
        // a zero slope of no window is not the exact line of a window
        let hollow = WindowStats {
            n: f64::NAN,
            slope: 0.0,
            se_slope: 0.0,
            ..WindowStats::NONE
        };
        assert!(!no_trend(&hollow, 0.05));
    }

    #[test]
    fn a_series_which_trends_throughout_is_nan() {
        // python: every window p < 0.05, the last of three points included
        let v = [1.0, 1.5, 2.2, 3.1, 4.2, 5.4];
        assert!(walk(&[(&T6, &v)], 0.05)[0].is_nan());
    }

    #[test]
    fn a_window_of_fewer_than_three_troughs_is_not_an_answer() {
        // two troughs: no window at all
        assert!(walk(&[(&T6[..2], &[1.0, 2.0])], 0.05)[0].is_nan());
        // three troughs are the smallest window: python p = 0.12103772 > 0.05 at index 0
        assert_eq!(walk(&[(&T6[..3], &[1.0, 2.0, 2.5])], 0.05), [0.0]);
        // constant troughs have no spread: no window either, not a flat one
        assert!(walk(&[(&T6[..4], &[2.0, 2.0, 2.0, 2.0])], 0.05)[0].is_nan());
    }

    #[test]
    fn rows_are_independent_and_an_invalid_entry_is_skipped() {
        let v0 = [1.0, 1.8, 2.3, 2.4, 2.38, 2.41];
        let v1 = [1.0, 1.5, 2.2, 3.1, 4.2, 5.4];
        let v2 = [1.0, 2.0, 2.5];
        let (t, v, mut s, ok) = padded(&[(&T6, &v0), (&T6, &v1), (&T6[..3], &v2), (&T6, &v0)]);
        for start in &mut s[18..] {
            *start += 100.0;
        }
        let got = stepwise(&t, &v, &s, &ok, 4, 6, 0.05);
        // the last row reads its own starts, not those of the first row
        assert_eq!(got[3], 110.0);
        assert_eq!(got[0], 10.0);
        assert!(got[1].is_nan());
        assert_eq!(got[2], 0.0);
        // python: a 9.0 at index 2 marked invalid; p = 0.03234161, 0.07684945 -> start of index 1
        let w = [1.0, 1.8, 9.0, 2.3, 2.4, 2.38];
        let (t, v, s, mut ok) = padded(&[(&T6, &w)]);
        ok[2] = false;
        assert_eq!(stepwise(&t, &v, &s, &ok, 1, 6, 0.05), [10.0]);
        // short arrays and a zero shape are all NaN / empty, never a panic
        assert!(stepwise(&t[..5], &v, &s, &ok, 1, 6, 0.05)[0].is_nan());
        assert!(stepwise(&t, &v, &s, &ok, 0, 6, 0.05).is_empty());
    }

    fn request(method: TssMethod) -> TssRequest {
        let t = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let a = rises(5.0, 0.05, &t);
        let b = rises(8.0, 0.03, &t);
        let mut time = Vec::new();
        let mut value = Vec::new();
        let mut start = Vec::new();
        let mut valid = Vec::new();
        // row 0 has all seven troughs, row 1 four, row 2 none; the first interval of row 0
        // starts at 100, so its troughs are read from there and the start is added back
        for (r, (n, offset, v)) in [(7, 100.0, &a), (4, 0.0, &b), (0, 0.0, &b)]
            .iter()
            .enumerate()
        {
            for c in 0..7 {
                time.push(t[c] + offset);
                value.push(v[c]);
                start.push(if c == 0 { *offset } else { t[c - 1] + offset });
                valid.push(c < *n);
            }
            let _ = r;
        }
        TssRequest {
            time,
            value,
            start,
            valid,
            n_rows: 3,
            n_cols: 7,
            method,
            fraction: 0.9,
            alpha: 0.05,
        }
    }

    #[test]
    fn the_monoexponential_request_reads_each_row_from_its_own_start() {
        let answer = time_to_steady_state(&request(TssMethod::Monoexponential)).unwrap();
        let (t0, c0) = monoexponential(
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            &rises(5.0, 0.05, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]),
            0.9,
        );
        assert!((answer.tss[0] - (t0 + 100.0)).abs() < 1e-9);
        assert!((answer.c_ss[0] - c0).abs() < 1e-9);
        assert!((answer.c_ss[0] - 5.0).abs() < 1e-6);
        // four troughs only: the padding is not read
        assert!(answer.tss[1].is_finite());
        assert!((answer.c_ss[1] - 8.0).abs() < 1e-3);
        // a row without a trough carries no estimate
        assert!(answer.tss[2].is_nan() && answer.c_ss[2].is_nan());
    }

    #[test]
    fn the_stepwise_request_has_no_plateau() {
        let answer = time_to_steady_state(&request(TssMethod::Stepwise)).unwrap();
        assert_eq!(answer.tss.len(), 3);
        assert!(answer.c_ss.is_empty());
    }

    #[test]
    fn a_malformed_request_is_an_error_and_not_a_nan() {
        let mut r = request(TssMethod::Monoexponential);
        r.value.pop();
        assert!(matches!(
            time_to_steady_state(&r),
            Err(KernelError::Tss(message)) if message.contains("value")
        ));
    }
}
