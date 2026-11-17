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
// no caller yet: the time to steady state front end of the next task
#[cfg_attr(not(test), expect(dead_code))]
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
        assert!((tss / ref_tss - 1.0).abs() < 1e-6, "{tss}");
        assert!((plateau / ref_plateau - 1.0).abs() < 1e-6, "{plateau}");
    }

    #[test]
    fn a_slow_and_a_fast_approach_are_found_by_the_start_whose_window_holds_them() {
        // noise free, k * t_last = 0.01: only the window of the 0.5 start (0.005 and up) holds
        // it, the windows of the 2 and 8 starts begin at 0.02 and 0.08 and end on their edge
        let t = grid();
        for (k_t_last, rel) in [(0.01, 1e-6), (0.006, 1e-6), (0.015, 1e-6), (150.0, 1e-6)] {
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
}
