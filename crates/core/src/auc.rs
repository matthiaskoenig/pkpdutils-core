//! The trapezoid rules over the segments of one packed curve.
//!
//! On a segment from `(t1, c1)` to `(t2, c2)` with `dt = t2 - t1` the linear rule gives the
//! area `dt (c1 + c2) / 2` and the first moment `dt (t1 c1 + t2 c2) / 2`; the logarithmic rule,
//! which is exact for a mono-exponential decline, gives `dt (c1 - c2) / L` and
//! `dt (t1 c1 - t2 c2) / L + dt² (c1 - c2) / L²` with `L = ln(c1 / c2)`. They are the rules of
//! Gabrielsson & Weiner (2016, ch. 2.8) and of the Phoenix WinNonlin NCA.
//!
//! `segment_areas` of `src/pkpdutils/nca/auc.py` evaluates both rules for every segment and
//! picks one with `numpy.where`, which is why a `NaN` out of the logarithm never reaches the
//! result. Here the logarithmic rule is evaluated only where it is chosen, which is the same
//! answer for less work, and the order of the operations inside each formula is the order of
//! the python expression, down to the `ln(c1 / c2)` of one ratio instead of a difference of two
//! logarithms: the benchmark gate compares with the python package at a relative tolerance of
//! 1e-8 and a rearranged expression drifts.
//!
//! The rule is chosen per segment. A segment is a candidate for the logarithmic rule only when
//! both values are strictly positive and differ, so a zero, a negative value or a flat segment
//! is linear under every method and the logarithm is never taken of something undefined.
//!
//! The one place this is not the python arithmetic is the summation: the segments are added
//! left to right here, while `numpy.sum` adds pairwise above eight of them. The two agree to
//! rounding and not bit for bit, which is what `areas.md` trap T14 already says of the
//! regression itself and why the benchmark gate carries a tolerance at all.

use pkpdutils_abi::AucMethod;

/// The area under a curve and under its first moment curve, over the same segments.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Areas {
    /// The area under the curve, `AUC`.
    pub auc: f64,
    /// The area under the first moment curve, `AUMC`.
    pub aumc: f64,
}

/// Sum the trapezoids between the consecutive points of one packed curve.
///
/// `t` and `c` are the packed points of a row, cut to the valid ones: the padding `pack_valid`
/// leaves behind must not be passed, since a `NaN` time would poison the sum. Fewer than two
/// points have no segment and give zero, which is what the `in_curve` mask of the python
/// `segment_areas` produces for such a row; the caller turns that into `NaN` through its own
/// `has_data`. A row whose times and values differ in length is cut to the shorter, so no
/// shape from a host can index past the end.
///
/// Mirrors `segment_areas` and the unmasked `auc_aumc` of `src/pkpdutils/nca/auc.py`.
pub(crate) fn trapezoid(t: &[f64], c: &[f64], method: AucMethod) -> Areas {
    sum_segments(t, c, method, |_| true)
}

/// Sum the trapezoids of the segments which end at or before `t_end`.
///
/// This is `auc_aumc(..., t_end=...)` of `src/pkpdutils/nca/auc.py`, which masks **whole
/// segments** by the time of their end point rather than cutting the row at an index. The two
/// are the same thing only while the times increase, and nothing in this kernel sorts a row
/// (`areas.md` trap T1), so the mask is what `auc_last` needs: a row whose times fall back on
/// themselves keeps the segments which end in the window wherever they sit, and drops the
/// others.
///
/// A `NaN` `t_end` keeps no segment at all and gives zero, since a comparison against a `NaN`
/// is false; that is the row which has no last measurable value, whose area the caller
/// overwrites with `NaN` all the same.
pub(crate) fn trapezoid_until(t: &[f64], c: &[f64], t_end: f64, method: AucMethod) -> Areas {
    sum_segments(t, c, method, |end| end <= t_end)
}

/// Sum the trapezoids of the segments `keep` accepts, by the time of their end point.
fn sum_segments(t: &[f64], c: &[f64], method: AucMethod, keep: impl Fn(f64) -> bool) -> Areas {
    let mut auc = 0.0;
    let mut aumc = 0.0;
    for (times, values) in t.windows(2).zip(c.windows(2)) {
        let (&[t1, t2], &[c1, c2]) = (times, values) else {
            break;
        };
        if !keep(t2) {
            continue;
        }
        let dt = t2 - t1;
        // `c1 > 0.0` is false for a NaN, so neither value can be NaN at the comparison.
        let positive = c1 > 0.0 && c2 > 0.0 && c1 != c2;
        let use_log = match method {
            AucMethod::Linear => false,
            // "linear up, log down": down is strictly the later value below the earlier one
            AucMethod::LinearLog => positive && c2 < c1,
            AucMethod::Log => positive,
        };
        if use_log {
            let log_ratio = (c1 / c2).ln();
            auc += (c1 - c2) / log_ratio * dt;
            aumc += dt * (t1 * c1 - t2 * c2) / log_ratio
                + dt * dt * (c1 - c2) / (log_ratio * log_ratio);
        } else {
            auc += midpoint(c1, c2) * dt;
            aumc += midpoint(t1 * c1, t2 * c2) * dt;
        }
    }
    Areas { auc, aumc }
}

/// Half a sum, avoiding an overflow while preserving the rounding of a subnormal.
///
/// Two values near the top of the range sum to an infinity, and halving that infinity keeps it;
/// halving each of them first is finite but rounds a subnormal twice. `_midpoint` of
/// `src/pkpdutils/nca/auc.py` therefore sums first and only falls back where the sum ran away.
fn midpoint(first: f64, last: f64) -> f64 {
    let midpoint = (first + last) * 0.5;
    if midpoint.is_infinite() {
        0.5 * first + 0.5 * last
    } else {
        midpoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance against the values the python kernel prints for the same row,
    /// four orders tighter than the 1e-8 of the benchmark gate: one segment is one
    /// expression, so only the last place or two is at stake.
    const TOL: f64 = 1e-12;

    #[test]
    fn the_linear_rule_sums_hand_computed_trapezoids() {
        // (0,2),(1,4),(3,4): (2+4)/2*1 = 3.0 and (4+4)/2*2 = 8.0
        let a = trapezoid(&[0.0, 1.0, 3.0], &[2.0, 4.0, 4.0], AucMethod::Linear);
        assert_eq!(a.auc, 11.0);
        // the moments: (0*2 + 1*4)/2*1 = 2.0 and (1*4 + 3*4)/2*2 = 16.0
        assert_eq!(a.aumc, 18.0);
    }

    #[test]
    fn a_falling_segment_is_logarithmic_under_linear_log() {
        // (0,4),(2,2): (4-2)/ln(4/2)*2, then a flat segment stays linear
        let a = trapezoid(&[0.0, 2.0, 4.0], &[4.0, 2.0, 2.0], AucMethod::LinearLog);
        let expected = 2.0 / std::f64::consts::LN_2 * 2.0 + 4.0;
        assert!((a.auc - expected).abs() < TOL, "{}", a.auc);
        assert!(
            (a.auc - 9.770780163555854).abs() < TOL,
            "the python kernel on the same row: {}",
            a.auc
        );
    }

    #[test]
    fn a_log_segment_falls_back_to_linear_where_the_log_is_undefined() {
        // areas.md section 2: `positive = (c1 > 0) & (c2 > 0) & (c1 != c2)` is the
        // guard, so every one of these segments takes the linear rule under LOG,
        // where every positive segment would otherwise be logarithmic.
        let equal = trapezoid(&[0.0, 1.0], &[2.0, 2.0], AucMethod::Log);
        assert_eq!((equal.auc, equal.aumc), (2.0, 1.0), "ln(c1/c2) would be 0");

        let last_zero = trapezoid(&[0.0, 1.0], &[2.0, 0.0], AucMethod::Log);
        assert_eq!(
            (last_zero.auc, last_zero.aumc),
            (1.0, 0.0),
            "ln(2/0) is inf"
        );

        let first_zero = trapezoid(&[0.0, 1.0], &[0.0, 2.0], AucMethod::Log);
        assert_eq!(
            (first_zero.auc, first_zero.aumc),
            (1.0, 1.0),
            "ln(0/2) is -inf"
        );

        let negative = trapezoid(&[0.0, 1.0], &[-2.0, 2.0], AucMethod::Log);
        assert_eq!((negative.auc, negative.aumc), (0.0, 1.0), "ln(-1) is NaN");
    }

    #[test]
    fn the_rising_side_stays_linear_under_linear_log() {
        // (0,2),(1,4) rises, so "log down" does not apply: (2+4)/2*1
        let up = trapezoid(&[0.0, 1.0], &[2.0, 4.0], AucMethod::LinearLog);
        assert_eq!(up.auc, 3.0);
        assert_eq!(up.aumc, 2.0);

        // the same segment under LOG, which takes every positive segment
        let log = trapezoid(&[0.0, 1.0], &[2.0, 4.0], AucMethod::Log);
        assert!(
            (log.auc - 2.8853900817779268).abs() < TOL,
            "the direction test has to separate the two: {}",
            log.auc
        );
    }

    #[test]
    fn the_moment_curve_uses_the_same_rule_as_the_area() {
        // The same falling segment (0,4),(2,2) under the two rules. The moment
        // follows the rule the area took, so switching the method has to move both.
        let linear = trapezoid(&[0.0, 2.0], &[4.0, 2.0], AucMethod::Linear);
        assert_eq!((linear.auc, linear.aumc), (6.0, 4.0));

        let log = trapezoid(&[0.0, 2.0], &[4.0, 2.0], AucMethod::LinearLog);
        // dt (c1 - c2) / L = 4 / ln 2
        assert!((log.auc - 5.7707801635558535).abs() < TOL, "{}", log.auc);
        // dt (t1 c1 - t2 c2) / L + dt² (c1 - c2) / L² = -8/ln 2 + 8/ln² 2
        assert!((log.aumc - 5.109391520933155).abs() < TOL, "{}", log.aumc);
    }

    #[test]
    fn a_row_too_short_for_a_segment_has_zero_area() {
        // areas.md section 1: a row of nought or one valid point has every segment
        // at 0.0, never NaN. The caller turns that into NaN through `has_data`.
        for (t, c) in [(&[][..], &[][..]), (&[1.0][..], &[2.0][..])] {
            for method in [AucMethod::Linear, AucMethod::LinearLog, AucMethod::Log] {
                let a = trapezoid(t, c, method);
                assert_eq!((a.auc, a.aumc), (0.0, 0.0), "{t:?} under {method:?}");
            }
        }
        // a value without a time is no segment either
        let ragged = trapezoid(&[0.0, 1.0], &[2.0], AucMethod::Linear);
        assert_eq!((ragged.auc, ragged.aumc), (0.0, 0.0));
    }

    #[test]
    fn a_midpoint_which_would_overflow_is_halved_first() {
        // areas.md section 2 (A:103-111) mirrors an upstream guard: `c1 + c2`
        // overflows to inf for two values near the top of the range, and python then
        // falls back to `0.5 c1 + 0.5 c2`, which keeps the area finite.
        let a = trapezoid(&[0.0, 1.0], &[f64::MAX, f64::MAX], AucMethod::Linear);
        assert_eq!(a.auc, f64::MAX, "the area stays finite");
        assert_eq!(a.aumc, f64::MAX * 0.5, "and so does the moment");
    }
}
