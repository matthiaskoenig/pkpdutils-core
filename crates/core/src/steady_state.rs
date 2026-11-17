//! The multiple dose analysis of a last dosing interval.
//!
//! Mirrors `src/pkpdutils/nca/steady_state.py`. This module holds the completion of a last
//! interval which falls a little short of its end.

use pkpdutils_abi::{Kind, NcaOptions, Route};

use crate::intervals::{Interval, analyze_intervals, count_points, interval_bounds};

/// Complete a last dosing interval whose last sample lies a little before its end, with the
/// terminal regression.
///
/// `t`, `c` and `n` are the packed valid points of the row in the frame of the interval, `tlast`
/// is the last measurable time relative to the last dose, `clast` the value there and `lambda_z`
/// the terminal rate. The shortfall is `end - (start + tlast)`, the tail the area of
/// `clast * exp(-lambda_z * (t - tlast))` over it, `clast / lambda_z * (1 - exp(-lambda_z *
/// shortfall))`, and the trough the same regression at the end, `clast * exp(-lambda_z *
/// shortfall)`. The interval is re-analysed up to its last sample (`start` to `start + tlast`,
/// one dose, [`analyze_intervals`]) and its exposure, minimum, trough, average, fluctuation and
/// swing are patched in place; the minimum is the smaller of the observed one and the trough.
///
/// Two guards come first and return without looking at the row: only a concentration analysis
/// is completed, and `tau_tolerance <= 0` disables the completion altogether, so a tolerance of
/// zero is "never complete", not "the interval must end exactly on `tau`". Then seven conditions
/// must all hold: the area of the interval is not finite, its start and its end are finite, the
/// shortfall is positive and at most `tau_tolerance * tau` (`tau` being the length of this
/// interval, `end - start`, not `NcaOptions::tau`), `clast` is finite and `lambda_z` is positive.
///
/// Returns the extrapolated fraction of the exposure `tail / area`, or `NaN` when the row is not
/// completed, in which case the interval is untouched. A row whose observed part is not an
/// interval (no sample inside it) is not completed either. The caller raises
/// `NcaFlag::INCOMPLETE_INTERVAL` for a row which stays incomplete.
///
/// Mirrors `complete_last_interval` of `src/pkpdutils/nca/steady_state.py`
/// (Phoenix WinNonlin `AUC_TAU_%Extrap`).
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "reached by the multiple dose analysis, which a later task wires in"
    )
)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn complete_last_interval(
    t: &[f64],
    c: &[f64],
    n: usize,
    interval: &mut Interval,
    tlast: f64,
    clast: f64,
    lambda_z: f64,
    route: Option<Route>,
    options: &NcaOptions,
) -> f64 {
    // the guards: an effect has no terminal regression, and zero disables the completion
    if options.kind != Kind::Concentration || options.tau_tolerance <= 0.0 {
        return f64::NAN;
    }
    let (t_start, t_end) = (interval.start, interval.end);
    let tau = t_end - t_start;
    let shortfall = t_end - (t_start + tlast);
    let completes = !interval.auc.is_finite()
        && t_start.is_finite()
        && t_end.is_finite()
        && shortfall > 0.0
        && shortfall <= options.tau_tolerance * tau
        && clast.is_finite()
        && lambda_z > 0.0;
    if !completes {
        return f64::NAN;
    }
    // the same interval, ending at the last measurable sample: what was observed of it
    let n = n.min(t.len()).min(c.len());
    let mut observed = Vec::with_capacity(1);
    interval_bounds(&[t_start], 1, Some(tlast), &mut observed);
    count_points(
        &t[..n],
        &c[..n],
        route == Some(Route::IvBolus),
        1,
        &mut observed,
    );
    let (mut curve_t, mut curve_c) = (Vec::new(), Vec::new());
    analyze_intervals(
        t,
        c,
        n,
        1,
        route,
        options,
        &mut observed,
        &mut curve_t,
        &mut curve_c,
    );
    let Some(observed) = observed.first() else {
        return f64::NAN;
    };
    let decay = (-lambda_z * shortfall).exp();
    let tail = clast / lambda_z * (1.0 - decay);
    let trough = clast * decay;
    let area = observed.auc + tail;
    // an observed part which is not an interval at all keeps its incomplete flag
    if !area.is_finite() {
        return f64::NAN;
    }
    // `np.minimum` carries a NaN, `f64::min` would drop it
    let cmin = if observed.cmin.is_nan() || trough.is_nan() {
        f64::NAN
    } else {
        observed.cmin.min(trough)
    };
    let cmax = observed.cmax;
    let average = area / tau;
    interval.n_points = observed.n_points;
    interval.auc = area;
    interval.cmax = cmax;
    interval.tmax = observed.tmax;
    interval.cmin = cmin;
    interval.ctrough = trough;
    interval.c_start = observed.c_start;
    interval.cavg = average;
    interval.fluctuation = (cmax - cmin) / average;
    interval.swing = (cmax - cmin) / cmin;
    tail / area
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: [f64; 7] = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 11.5];
    const C: [f64; 7] = [2.0, 10.0, 8.0, 5.0, 3.0, 2.0, 1.1];

    fn open(start: f64, end: f64) -> Interval {
        Interval {
            start,
            end,
            ..Interval::EMPTY
        }
    }

    /// Complete the interval `[start, end]` of the row; `tlast` and `clast` are given, not read.
    #[allow(clippy::too_many_arguments)]
    fn run(
        t: &[f64],
        c: &[f64],
        mut interval: Interval,
        tlast: f64,
        clast: f64,
        lambda_z: f64,
        route: Option<Route>,
        options: &NcaOptions,
    ) -> (f64, Interval) {
        let f = complete_last_interval(
            t,
            c,
            t.len(),
            &mut interval,
            tlast,
            clast,
            lambda_z,
            route,
            options,
        );
        (f, interval)
    }

    fn oral(t: &[f64], c: &[f64], ts: f64, te: f64, tlast: f64) -> (f64, Interval) {
        let options = NcaOptions::default();
        run(
            t,
            c,
            open(ts, te),
            tlast,
            c[c.len() - 1],
            0.2,
            Some(Route::Oral),
            &options,
        )
    }

    fn untouched(i: &Interval, ts: f64, te: f64) {
        assert_eq!((i.start, i.end), (ts, te));
        for v in [
            i.n_points,
            i.auc,
            i.cmax,
            i.tmax,
            i.cmin,
            i.ctrough,
            i.c_start,
            i.cavg,
            i.fluctuation,
            i.swing,
        ] {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn a_last_interval_short_of_its_end_within_the_tolerance_is_completed() {
        // numbers from python `complete_last_interval`
        let (f, i) = oral(&T, &C, 0.0, 12.0, 11.5);
        assert_eq!(f, 0.011308280183579463);
        assert_eq!(i.n_points, 7.0);
        assert_eq!(i.auc, 46.284155707623285);
        assert_eq!((i.cmax, i.tmax, i.c_start), (10.0, 1.0, 2.0));
        assert_eq!(i.ctrough, 0.9953211598395556);
        // the trough 1.1 * exp(-0.2 * 0.5) is below the observed minimum 1.1
        assert_eq!(i.cmin, 0.9953211598395556);
        assert_eq!(i.cavg, 3.8570129756352736);
        assert_eq!(i.fluctuation, 2.3346249797558225);
        assert_eq!(i.swing, 9.04700834614225);
        // the bounds stay what they were
        assert_eq!((i.start, i.end), (0.0, 12.0));
        // the tail is `clast / lambda_z * (1 - exp(-lambda_z * shortfall))` and the fraction its share
        let tail = 1.1 / 0.2 * (1.0 - (-0.2f64 * 0.5).exp());
        assert_eq!(f, tail / i.auc);
    }

    #[test]
    fn the_minimum_is_the_smaller_of_the_observed_one_and_the_trough() {
        // an observed 0.5 below the trough 0.9953: python gives 0.5, so a port which always
        // takes the trough as the minimum is wrong here, and one which always keeps the observed
        // minimum is wrong in the test above
        let c = [2.0, 10.0, 8.0, 5.0, 3.0, 0.5, 1.1];
        let (f, i) = oral(&T, &c, 0.0, 12.0, 11.5);
        assert_eq!(f, 0.012559519460279275);
        assert_eq!(i.cmin, 0.5);
        assert_eq!(i.ctrough, 0.9953211598395556);
        assert_eq!(i.auc, 41.673107196299085);
        assert_eq!(i.swing, 19.0);
        assert_eq!(i.fluctuation, 2.735577154422604);
    }

    #[test]
    fn a_last_interval_short_by_more_than_the_tolerance_is_flagged_incomplete() {
        // 2 hours short of 12 against 0.1 * 12 = 1.2: nothing is completed, so the interval keeps
        // its NaN area, which is what the caller flags as INCOMPLETE_INTERVAL
        let mut t = T;
        t[6] = 10.0;
        let (f, i) = oral(&t, &C, 0.0, 12.0, 10.0);
        assert!(f.is_nan());
        untouched(&i, 0.0, 12.0);
        // the boundary: 1.2 itself (12 - 10.8) completes, 1.21 does not
        t[6] = 10.8;
        let (f, i) = oral(&t, &C, 0.0, 12.0, 10.8);
        assert_eq!(f, 0.025578329040934416);
        assert_eq!(i.auc, 45.880509327089506);
        t[6] = 10.79;
        assert!(oral(&t, &C, 0.0, 12.0, 10.79).0.is_nan());
    }

    #[test]
    fn a_tau_tolerance_of_zero_disables_completion_entirely() {
        // Review Focus. Zero is a guard: the interval stays incomplete and is flagged, for the
        // short interval of the first test and for one which ends exactly on its last sample
        let zero = NcaOptions {
            tau_tolerance: 0.0,
            ..NcaOptions::default()
        };
        let route = Some(Route::Oral);
        let (f, i) = run(&T, &C, open(0.0, 12.0), 11.5, 1.1, 0.2, route, &zero);
        assert!(f.is_nan());
        untouched(&i, 0.0, 12.0);
        // "exact" reading of zero: a shortfall of exactly nothing. The row has one sample, at the
        // end, so its observed part has a single point and no area; python gives NaN under zero
        // and under the default tolerance alike, a port which reads zero as "shortfall <= 0" and
        // lets `shortfall >= 0` through would complete it
        let (f, i) = run(
            &[12.0],
            &[1.0],
            open(0.0, 12.0),
            12.0,
            1.0,
            0.2,
            route,
            &zero,
        );
        assert!(f.is_nan());
        untouched(&i, 0.0, 12.0);
        let default = NcaOptions::default();
        let (f, i) = run(
            &[12.0],
            &[1.0],
            open(0.0, 12.0),
            12.0,
            1.0,
            0.2,
            route,
            &default,
        );
        assert!(f.is_nan());
        untouched(&i, 0.0, 12.0);
    }

    #[test]
    fn a_shortfall_of_nothing_is_not_a_shortfall() {
        // condition 4. The last sample sits on the end of an interval which was not analysed
        // (NaN area): the observed part is the whole interval and has a finite area, so only
        // `shortfall > 0` keeps it out, under the default tolerance and under zero (python: NaN
        // for both). A port which reads a tolerance of zero as "exact" lets it through
        let t = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0];
        let c = [2.0, 10.0, 8.0, 5.0, 3.0, 2.0, 1.1];
        for tolerance in [0.1, 0.0] {
            let options = NcaOptions {
                tau_tolerance: tolerance,
                ..NcaOptions::default()
            };
            let (f, i) = run(
                &t,
                &c,
                open(0.0, 12.0),
                12.0,
                1.1,
                0.2,
                Some(Route::Oral),
                &options,
            );
            assert!(f.is_nan());
            untouched(&i, 0.0, 12.0);
        }
    }

    #[test]
    fn a_shortfall_equal_to_the_tolerance_times_tau_is_completed() {
        // condition 5, `<=` and not `<`: 12 - 6 is exactly 0.5 * 12 (python: completed; a hair
        // more is not)
        let options = NcaOptions {
            tau_tolerance: 0.5,
            ..NcaOptions::default()
        };
        let (t, c) = ([0.0, 1.0, 2.0, 4.0, 6.0], [2.0, 10.0, 8.0, 5.0, 3.0]);
        let (f, i) = run(
            &t,
            &c,
            open(0.0, 12.0),
            6.0,
            3.0,
            0.2,
            Some(Route::Oral),
            &options,
        );
        assert_eq!(f, 0.22766731086251266);
        assert_eq!(i.auc, 46.0412466840576);
        assert_eq!(i.cmin, 0.9035826357366061);
        let t = [0.0, 1.0, 2.0, 4.0, 5.999];
        let (f, _) = run(
            &t,
            &c,
            open(0.0, 12.0),
            5.999,
            3.0,
            0.2,
            Some(Route::Oral),
            &options,
        );
        assert!(f.is_nan());
    }

    #[test]
    fn the_shortfall_is_measured_against_the_intervals_own_length() {
        let route = Some(Route::Oral);
        // NcaOptions::tau of 1 would allow only 0.1 of shortfall; the interval is 12 long, so the
        // 0.5 shortfall completes (python: same numbers as the plain case)
        let short = NcaOptions {
            tau: Some(1.0),
            ..NcaOptions::default()
        };
        let (f, i) = run(&T, &C, open(0.0, 12.0), 11.5, 1.1, 0.2, route, &short);
        assert_eq!(f, 0.011308280183579463);
        assert_eq!(i.auc, 46.284155707623285);
        // NcaOptions::tau of 100 would allow 10; the 2 hour shortfall is still more than 1.2
        let mut t = T;
        t[6] = 10.0;
        let long = NcaOptions {
            tau: Some(100.0),
            ..NcaOptions::default()
        };
        let (f, i) = run(&t, &C, open(0.0, 12.0), 10.0, 1.1, 0.2, route, &long);
        assert!(f.is_nan());
        untouched(&i, 0.0, 12.0);
        // a longer interval allows a longer shortfall: 2.4 of 24
        let t = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 21.6];
        let (f, i) = oral(&t, &C, 0.0, 24.0, 21.6);
        assert_eq!(f, 0.0332479543142625);
        assert_eq!(i.cmin, 0.6806617309867551);
        // and `tmax` stays relative to the start while the times are absolute
        let shifted: Vec<f64> = T.iter().map(|x| x + 100.0).collect();
        let (f, i) = oral(&shifted, &C, 100.0, 112.0, 11.5);
        assert_eq!(f, 0.011308280183579463);
        assert_eq!((i.tmax, i.auc), (1.0, 46.284155707623285));
    }

    #[test]
    fn a_row_whose_observed_part_is_not_a_number_is_left_uncompleted() {
        // no sample inside [0, 12): python keeps the interval incomplete
        let (f, i) = oral(&[20.0], &[1.0], 0.0, 12.0, 11.5);
        assert!(f.is_nan());
        untouched(&i, 0.0, 12.0);
    }

    #[test]
    fn an_interval_which_already_has_an_area_is_not_touched() {
        // condition 1: the area is finite (here forced), so nothing is completed
        let mut done = open(0.0, 12.0);
        done.auc = 5.0;
        let options = NcaOptions::default();
        let (f, i) = run(&T, &C, done, 11.5, 1.1, 0.2, Some(Route::Oral), &options);
        assert!(f.is_nan());
        assert_eq!(i.auc, 5.0);
        assert!(i.cmax.is_nan() && i.cavg.is_nan());
    }

    #[test]
    fn an_effect_is_never_completed() {
        let effect = NcaOptions {
            kind: Kind::Effect,
            ..NcaOptions::default()
        };
        let (f, i) = run(
            &T,
            &C,
            open(0.0, 12.0),
            11.5,
            1.1,
            0.2,
            Some(Route::Oral),
            &effect,
        );
        assert!(f.is_nan());
        untouched(&i, 0.0, 12.0);
    }

    #[test]
    fn a_terminal_rate_which_is_not_positive_does_not_complete() {
        // condition 7. A negative rate gives a finite tail in the arithmetic, so only the
        // condition keeps it out (python: not completed); zero and NaN are out as well
        let options = NcaOptions::default();
        for lambda in [-0.2, 0.0, f64::NAN] {
            let (f, i) = run(
                &T,
                &C,
                open(0.0, 12.0),
                11.5,
                1.1,
                lambda,
                Some(Route::Oral),
                &options,
            );
            assert!(f.is_nan());
            untouched(&i, 0.0, 12.0);
        }
    }

    #[test]
    fn a_last_value_which_is_not_finite_does_not_complete() {
        let options = NcaOptions::default();
        for clast in [f64::NAN, f64::INFINITY] {
            let (f, i) = run(
                &T,
                &C,
                open(0.0, 12.0),
                11.5,
                clast,
                0.2,
                Some(Route::Oral),
                &options,
            );
            assert!(f.is_nan());
            untouched(&i, 0.0, 12.0);
        }
    }

    #[test]
    fn an_interval_which_is_not_bounded_is_not_completed() {
        // conditions 2 and 3: python gives NaN for each
        let options = NcaOptions::default();
        for (ts, te) in [
            (0.0, f64::INFINITY),
            (f64::NEG_INFINITY, 12.0),
            (f64::NAN, 12.0),
            (0.0, f64::NAN),
        ] {
            let (f, i) = run(
                &T,
                &C,
                open(ts, te),
                11.5,
                1.1,
                0.2,
                Some(Route::Oral),
                &options,
            );
            assert!(f.is_nan());
            assert!(i.auc.is_nan() && i.cavg.is_nan());
        }
    }

    #[test]
    fn the_completion_follows_the_route_and_the_trapezoid_rule() {
        // an intravenous bolus back extrapolates the start (numbers from python)
        let options = NcaOptions::default();
        let (f, i) = run(
            &T[1..],
            &C[1..],
            open(0.0, 12.0),
            11.5,
            1.1,
            0.2,
            Some(Route::IvBolus),
            &options,
        );
        assert_eq!(f, 0.010165420863430117);
        assert_eq!(i.n_points, 6.0);
        assert_eq!(i.auc, 51.48770600193466);
        assert_eq!(
            (i.cmax, i.tmax, i.c_start),
            (12.500000000000007, 0.0, 12.500000000000007)
        );
        assert_eq!(i.fluctuation, 2.6813419513531627);
        assert_eq!(i.swing, 11.55876043267782);
        // a start which cannot be interpolated stays NaN for an oral dose
        let (f, i) = oral(&[1.0, 11.5], &[10.0, 1.1], 0.0, 12.0, 11.5);
        assert_eq!(f, 0.01221152721656908);
        assert_eq!((i.n_points, i.auc), (2.0, 42.86066693542318));
        assert!(i.c_start.is_nan());
        // a wider tolerance
        let wide = NcaOptions {
            tau_tolerance: 0.5,
            ..NcaOptions::default()
        };
        let t = [0.0, 1.0, 2.0, 4.0, 6.0, 7.0];
        let c = [2.0, 10.0, 8.0, 5.0, 3.0, 1.1];
        let (f, i) = run(
            &t,
            &c,
            open(0.0, 12.0),
            7.0,
            1.1,
            0.2,
            Some(Route::Oral),
            &wide,
        );
        assert_eq!(f, 0.08494257597481082);
        assert_eq!(i.n_points, 6.0);
    }
}
