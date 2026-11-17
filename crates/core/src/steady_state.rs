//! The multiple dose analysis of a last dosing interval.
//!
//! Mirrors `src/pkpdutils/nca/steady_state.py`. This module holds the driver of a multiple dose
//! row ([`compute_steady_state`]) and the completion of a last interval which falls a little
//! short of its end ([`complete_last_interval`]).

use pkpdutils_abi::{Kind, NcaFlag, NcaOptions, Route};

use crate::intervals::{
    Interval, analyze_intervals, count_points, fill_doses, interval_bounds, last_interval_length,
};
use crate::pack::pack_valid;
use crate::parameters::{RowInput, RowParameters, Scratch, compute_row};
use crate::terminal::Candidate;

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

/// Whether a value is before the last dose of the row and therefore dropped from the point
/// parameters: the dose time is a number and the sample is strictly before it.
fn before_dose(time: Option<&f64>, dose_time: f64) -> bool {
    dose_time.is_finite() && time.is_some_and(|&t| t < dose_time)
}

/// The point, per-interval and steady state parameters of one multiple dose row.
///
/// `input.dose` is the last dose of the protocol (`reference_dose(.., last: true)`), so the point
/// parameters (`cmax`, `tmax`, `clast`, the terminal phase, ...) are those of the curve from the
/// last dose on, in times relative to it: the values before it are dropped from the row
/// `compute_row` reads, and the row is analysed with `single_dose: false`. The thirteen
/// single dose quantities of `SINGLE_DOSE_PARAMETERS` (the clearances, volumes and dose
/// normalized exposures, which divide the dose by an exposure the earlier doses contribute to)
/// are `NaN`; the clearance of the analysis is `cl_ss`.
///
/// The dosing intervals are always computed, whether `NcaOptions::intervals` asks for them or not,
/// since `NcaFlag::EXTRAPOLATED_TROUGH` and `NcaFlag::INCOMPLETE_INTERVAL` are raised from them;
/// the caller decides whether they are emitted. They run over the protocol `input.dose_times` on
/// the raw samples in the frame of the host, with the length of the last one `tau`: the option
/// where it is set, otherwise the last gap of the protocol, `t_K - t_{K-1}`, with no check that
/// the gaps are regular and no flag (an irregular protocol takes its last gap), and `NaN` for
/// one dose, which then has no last interval. A batch without a protocol but with `tau` is one
/// interval from time 0. The last interval is the steady state interval; one which falls short
/// of its end by at most `NcaOptions::tau_tolerance` is completed
/// ([`complete_last_interval`]), one which is not covered by the data stays `NaN` and is flagged.
///
/// The accumulation ratios compare the last interval with the first of a protocol of at least two
/// doses and are `NaN` for one. `accumulation_ratio` is the predicted one,
/// `1 / (1 - exp(-lambda_z tau))`.
///
/// `candidates` is passed on to `compute_row`. An effect analysis is out of scope here, as
/// everywhere in this build, and reports no steady state parameter.
///
/// Mirrors `compute_steady_state` of `src/pkpdutils/nca/steady_state.py` for one row.
pub(crate) fn compute_steady_state(
    input: &RowInput<'_>,
    options: &NcaOptions,
    scratch: &mut Scratch,
    candidates: Option<&mut Vec<Candidate>>,
    intervals: &mut Vec<Interval>,
) -> RowParameters {
    // the point parameters describe the curve from the last dose on
    let mut masked = std::mem::take(&mut scratch.masked);
    masked.clear();
    masked.extend(input.values.iter().enumerate().map(|(j, &value)| {
        if before_dose(input.times.get(j), input.dose.time) {
            f64::NAN
        } else {
            value
        }
    }));
    let point_input = RowInput {
        values: &masked,
        ..*input
    };
    let mut out = compute_row(&point_input, options, scratch, candidates);
    scratch.masked = masked;
    out.multiple = true;

    // the single dose quantities do not describe a multiple dose curve
    for field in [
        &mut out.cl,
        &mut out.cl_pred,
        &mut out.vz,
        &mut out.vz_pred,
        &mut out.vss,
        &mut out.vss_pred,
        &mut out.auc_inf_dn,
        &mut out.auc_inf_pred_dn,
        &mut out.cmax_dn,
    ] {
        *field = f64::NAN;
    }

    // a batch without a protocol but with `tau` is one interval from time 0
    let own = [0.0];
    let dose_times = if input.dose_times.is_empty() && options.tau.is_some() {
        &own[..]
    } else {
        input.dose_times
    };
    let n_doses = dose_times.iter().filter(|t| t.is_finite()).count();
    let tau = options
        .tau
        .or_else(|| last_interval_length(dose_times, n_doses));
    let tau_value = tau.unwrap_or(f64::NAN);

    // the intervals run on the raw samples, not on the row the point parameters read
    interval_bounds(dose_times, n_doses, tau, intervals);
    count_points(
        input.times,
        input.values,
        input.route == Some(Route::IvBolus),
        n_doses,
        intervals,
    );
    let columns = input
        .times
        .len()
        .min(input.values.len())
        .min(scratch.host_t.len())
        .min(scratch.host_c.len());
    let n_host = pack_valid(
        &input.times[..columns],
        &input.values[..columns],
        &mut scratch.host_t,
        &mut scratch.host_c,
    )
    .min(columns);
    let extrapolated = analyze_intervals(
        &scratch.host_t,
        &scratch.host_c,
        n_host,
        n_doses,
        input.route,
        options,
        intervals,
        &mut scratch.curve_t,
        &mut scratch.curve_c,
    );
    if input.has_dose {
        fill_doses(intervals, input.dose_amounts);
    }

    // a last interval which falls short of its end by at most the tolerance is completed with the
    // terminal regression instead of being given up as incomplete
    let has_interval = n_doses > 0;
    let last_index = n_doses.saturating_sub(1);
    let mut completed = false;
    let mut extrap_fraction = f64::NAN;
    if has_interval && let Some(last) = intervals.get_mut(last_index) {
        let area_before = last.auc;
        extrap_fraction = complete_last_interval(
            &scratch.host_t,
            &scratch.host_c,
            n_host,
            last,
            out.tlast,
            out.clast,
            out.lambda_z,
            input.route,
            options,
        );
        // completing writes a finite area over a `NaN` one, and nothing else changes it
        completed = !area_before.is_finite() && last.auc.is_finite();
    }

    let nan = Interval::EMPTY;
    let last = if has_interval {
        intervals.get(last_index).copied().unwrap_or(nan)
    } else {
        nan
    };
    let first = if n_doses >= 2 {
        intervals.first().copied().unwrap_or(nan)
    } else {
        nan
    };
    out.n_doses = input.dose_times.iter().filter(|t| t.is_finite()).count() as f64;
    out.tau = if has_interval { tau_value } else { f64::NAN };

    if options.kind == Kind::Concentration {
        let (area, cmax_ss, cmin_ss, ctrough, cavg) =
            (last.auc, last.cmax, last.cmin, last.ctrough, last.cavg);
        out.auc_tau = area;
        // 0 where the interval was covered by the data, the extrapolated share where it was
        // completed, `NaN` where it stays incomplete
        out.auc_tau_extrap_fraction = if completed {
            extrap_fraction
        } else if area.is_finite() {
            0.0
        } else {
            f64::NAN
        };
        out.cmin_ss = cmin_ss;
        out.cmax_ss = cmax_ss;
        out.ctrough = ctrough;
        out.cavg = cavg;
        out.fluctuation = last.fluctuation;
        out.swing = last.swing;
        // the trough variants: the value at the end of the interval in place of its smallest
        // observed value (Phoenix `Fluctuation%_Tau`, `Swing_Tau`; PKNCA `pk.calc.ptr`)
        out.fluctuation_tau = (cmax_ss - ctrough) / cavg;
        out.swing_tau = (cmax_ss - ctrough) / ctrough;
        out.ptr = cmax_ss / ctrough;
        out.accumulation_ratio = 1.0 / (1.0 - (-out.lambda_z * tau_value).exp());
        out.accumulation_ratio_obs = area / first.auc;
        // the observed accumulation of the peak, the minimum and the trough, the last over the
        // first dosing interval of the protocol (CDISC `ARCMAX`, `ARCMIN`, `ARCTROUG`)
        out.accumulation_ratio_cmax_obs = cmax_ss / first.cmax;
        out.accumulation_ratio_cmin_obs = cmin_ss / first.cmin;
        out.accumulation_ratio_ctrough_obs = ctrough / first.ctrough;
        if input.has_dose && input.route.is_some() {
            // without the fraction absorbed the clearance of an extravascular dose is `CL/F`
            let dose = if input.dose.amount > 0.0 {
                input.dose.amount
            } else {
                f64::NAN
            };
            out.cl_ss = dose / area;
        }
        let incomplete = has_interval && tau_value.is_finite() && !area.is_finite();
        if incomplete {
            out.flags |= NcaFlag::INCOMPLETE_INTERVAL;
        }
    }
    if extrapolated {
        out.flags |= NcaFlag::EXTRAPOLATED_TROUGH;
    }
    out
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
    fn the_shortfall_is_a_difference_of_the_end_and_the_sum_of_start_and_tlast() {
        // a cancelling subtraction at a large absolute time: `end - (start + tlast)` is
        // 2.525497620808892e-06 and `(end - start) - tlast` 2.5254968374355258e-06, 3e-7 apart in
        // relative terms. Python's association gives the fraction below (numbers from python)
        let start = 10000.058535104792;
        let t = [
            start,
            10001.058535104792,
            10002.058535104792,
            10004.058535104792,
            10006.058532579294,
        ];
        let c = [2.0, 10.0, 8.0, 5.0, 3.0];
        let options = NcaOptions::default();
        let route = Some(Route::Oral);
        let (f, i) = run(
            &t,
            &c,
            open(start, 10006.058535104792),
            5.999997474503163,
            3.0,
            0.2,
            route,
            &options,
        );
        assert_eq!(f, 2.1306722294214696e-07);
        assert_eq!(i.ctrough, 2.99999848470181);
        assert_eq!(i.auc, 35.55915755132657);
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

// the fixtures are concentrations from python, a few of which happen to be near 3.14 and 0.318
#[cfg(test)]
#[allow(clippy::approx_constant)]
mod driver_tests {
    use super::*;
    use crate::dose::reference_dose;

    /// The profile of three oral doses at 0, 12 and 24, sampled over 36 hours (python
    /// `curve([0, 12, 24], times)`).
    const T: [f64; 16] = [
        0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 13.0, 14.0, 16.0, 20.0, 24.0, 25.0, 26.0, 28.0, 32.0, 36.0,
    ];
    const C: [f64; 16] = [
        0.0, 6.872, 7.16, 5.156, 2.33, 1.047, 7.729, 7.862, 5.626, 2.541, 1.142, 7.807, 7.925,
        5.669, 2.56, 1.15,
    ];

    /// One multiple dose row through `compute_steady_state`, the way `Rows::row` builds it: the
    /// reference dose is the last of the protocol.
    fn steady(
        times: &[f64],
        values: &[f64],
        amounts: &[f64],
        dose_times: &[f64],
        route: Route,
        options: &NcaOptions,
    ) -> (RowParameters, Vec<Interval>) {
        let durations = vec![f64::NAN; amounts.len()];
        let input = RowInput {
            times,
            values,
            dose: reference_dose(amounts, dose_times, Some(&durations), true),
            route: Some(route),
            has_dose: !amounts.is_empty(),
            lloq: None,
            window: None,
            manual: None,
            single_dose: false,
            dose_times,
            dose_amounts: amounts,
        };
        let mut intervals = Vec::new();
        let row = compute_steady_state(
            &input,
            options,
            &mut Scratch::new(times.len()),
            None,
            &mut intervals,
        );
        (row, intervals)
    }

    fn oral(options: &NcaOptions) -> (RowParameters, Vec<Interval>) {
        steady(
            &T,
            &C,
            &[10.0, 20.0, 30.0],
            &[0.0, 12.0, 24.0],
            Route::Oral,
            options,
        )
    }

    #[test]
    fn the_steady_state_interval_is_always_the_last_dose_interval() {
        // numbers from python `compute_steady_state`. No search and no scoring: the last dose's
        // interval, whatever its quality, and the dose of the clearance is the last one (30, not
        // the first 10 and not the mean)
        let (p, i) = oral(&NcaOptions::default());
        assert_eq!(p.auc_tau, 48.49926973218703);
        assert_eq!(p.auc_tau_extrap_fraction, 0.0);
        assert_eq!((p.cmin_ss, p.cmax_ss, p.ctrough), (1.142, 7.925, 1.15));
        assert_eq!(p.cavg, 4.041605811015586);
        assert_eq!(p.fluctuation, 1.6782933114141452);
        assert_eq!(p.swing, 5.939579684763573);
        assert_eq!(p.fluctuation_tau, 1.676313900166716);
        assert_eq!(p.swing_tau, 5.891304347826088);
        assert_eq!(p.ptr, 6.891304347826087);
        assert_eq!(p.cl_ss, 0.6185660148216664);
        assert_eq!((p.n_doses, p.tau), (3.0, 12.0));
        assert_eq!(p.flags, NcaFlag::NONE);
        // the intervals, and their doses (`fill_doses` is called, the amounts are the row's own)
        let column = |read: fn(&Interval) -> f64| i.iter().map(read).collect::<Vec<_>>();
        assert_eq!(column(|i| i.start), [0.0, 12.0, 24.0]);
        assert_eq!(column(|i| i.end), [12.0, 24.0, 36.0]);
        assert_eq!(column(|i| i.dose), [10.0, 20.0, 30.0]);
        assert_eq!(column(|i| i.n_points), [6.0, 6.0, 6.0]);
        assert_eq!(
            column(|i| i.auc),
            [43.305541919868325, 48.06909355615606, 48.49926973218703]
        );
        assert_eq!(column(|i| i.cmax), [7.16, 7.862, 7.925]);
        assert_eq!(column(|i| i.cmin), [0.0, 1.047, 1.142]);
        assert_eq!(column(|i| i.ctrough), [1.047, 1.142, 1.15]);
        assert_eq!(column(|i| i.c_start), [0.0, 1.047, 1.142]);
        assert_eq!(
            column(|i| i.cavg),
            [3.608795159989027, 4.005757796346338, 4.041605811015586]
        );
        assert_eq!(
            column(|i| i.swing),
            [f64::INFINITY, 6.5090735434574984, 5.939579684763573]
        );
    }

    #[test]
    fn the_predicted_accumulation_ratio_is_one_over_one_minus_exp_minus_lambda_z_tau() {
        let (p, _) = oral(&NcaOptions::default());
        assert_eq!(p.lambda_z, 0.19940634912087601);
        assert_eq!(p.accumulation_ratio, 1.1005537656433146);
        assert_eq!(
            p.accumulation_ratio,
            1.0 / (1.0 - (-p.lambda_z * p.tau).exp())
        );
        // the observed ratios are the last interval over the first, and a first interval whose
        // minimum is zero divides to an infinity as numpy does
        assert_eq!(p.accumulation_ratio_obs, 1.1199321745454445);
        assert_eq!(p.accumulation_ratio_cmax_obs, 1.1068435754189945);
        assert_eq!(p.accumulation_ratio_cmin_obs, f64::INFINITY);
        assert_eq!(p.accumulation_ratio_ctrough_obs, 1.0983763132760267);
    }

    fn column(intervals: &[Interval], read: fn(&Interval) -> f64) -> Vec<f64> {
        intervals.iter().map(read).collect()
    }

    /// Whether two slices are equal with `NaN` equal to `NaN`.
    fn same(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
    }

    #[test]
    fn the_point_parameters_are_computed_from_the_last_dose_with_times_relative_to_it() {
        // a loading dose of 30 and then two of 10, sampled on other offsets in every interval.
        // python: the curve before the last dose is dropped, so cmax is the 8.318 of the last
        // dose and not the 21.995 of the loading one, and every time is relative to the dose at 24
        // (tlast 12 is the absolute 36, the terminal window starts at the relative 4 = 28).
        // A port which analysed the whole curve gives cmax 21.995
        let t = [
            0.0, 1.5, 3.0, 6.0, 12.0, 13.5, 15.0, 16.0, 18.0, 24.0, 25.5, 28.0, 30.0, 34.0, 36.0,
        ];
        let c = [
            0.0, 21.995, 18.613, 10.422, 3.14, 9.658, 7.928, 6.567, 4.42, 1.332, 8.318, 5.754,
            3.875, 1.742, 1.168,
        ];
        let (p, _) = steady(
            &t,
            &c,
            &[30.0, 10.0, 10.0],
            &[0.0, 12.0, 24.0],
            Route::Oral,
            &NcaOptions::default(),
        );
        assert_eq!((p.cmax, p.tmax), (8.318, 1.5));
        assert_eq!((p.cmin, p.tmin), (1.168, 12.0));
        assert_eq!((p.clast, p.tlast), (1.168, 12.0));
        assert_eq!(p.lambda_z_t_first, 4.0);
        assert_eq!(p.lambda_z, 0.19943582729108203);
        assert_eq!(p.auc_last, 47.67997262443133);
        assert_eq!(p.auc_all, 47.67997262443133);
        assert_eq!(p.auc_tau, 47.67997262443133);
        assert_eq!(p.cl_ss, 0.2097316640420212);
    }

    #[test]
    fn a_multiple_dose_row_reports_no_single_dose_quantities() {
        use crate::parameters::PARAMETERS;
        // the thirteen names of `SINGLE_DOSE_PARAMETERS`, which the nine fields carry
        const BLANKED: [&str; 13] = [
            "cl",
            "cl_f",
            "cl_pred",
            "cl_f_pred",
            "vz",
            "vz_f",
            "vz_pred",
            "vz_f_pred",
            "vss",
            "vss_pred",
            "auc_inf_dn",
            "auc_inf_pred_dn",
            "cmax_dn",
        ];
        for route in [Route::Oral, Route::IvBolus] {
            let (p, _) = steady(
                &T,
                &C,
                &[10.0, 20.0, 30.0],
                &[0.0, 12.0, 24.0],
                route,
                &NcaOptions::default(),
            );
            let mut seen = 0;
            for parameter in PARAMETERS {
                if BLANKED.contains(&parameter.name) && parameter.reported_by(Some(route), true) {
                    assert!((parameter.read)(&p).is_nan(), "{}", parameter.name);
                    seen += 1;
                }
            }
            // an intravenous row carries nine of the thirteen names (`cl`, `vz`, their
            // predicted forms, `vss`, `vss_pred` and the three normalized ones), an
            // extravascular one seven (the `_f` forms and the three normalized ones)
            assert_eq!(seen, if route.is_iv() { 9 } else { 7 }, "{route:?}");
            // what is not single dose stays: the areas after the last dose are reported
            assert_eq!(p.auc_inf_obs, 54.26638800635439);
            // one unit in the last place off numpy's (a point parameter of `compute_row`, which
            // this task does not touch), so compared to the relative tolerance of the engine
            crate::parameters::tests::close(p.aumc_inf, 311.88487000802655);
            crate::parameters::tests::close(p.mrt, 5.747293701793936);
        }
    }

    #[test]
    fn the_clearance_of_the_analysis_is_cl_ss_for_iv_and_cl_ss_f_otherwise() {
        use crate::parameters::PARAMETERS;
        let names = |route: Route| {
            PARAMETERS
                .iter()
                .filter(|p| p.name.starts_with("cl_ss") && p.reported_by(Some(route), true))
                .map(|p| p.name)
                .collect::<Vec<_>>()
        };
        assert_eq!(names(Route::IvBolus), ["cl_ss"]);
        assert_eq!(names(Route::Oral), ["cl_ss_f"]);
        // the dose is the last one, 30, over the area of the interval; python gives 0.61856...
        // for the bolus as for the oral dose
        let (p, _) = steady(
            &T,
            &C,
            &[10.0, 20.0, 30.0],
            &[0.0, 12.0, 24.0],
            Route::IvBolus,
            &NcaOptions::default(),
        );
        assert_eq!(p.cl_ss, 0.6185660148216664);
        assert_eq!(p.auc_tau, 48.49926973218703);
        // a last dose of zero is no dose: python gives NaN, not an infinity
        let (p, _) = steady(
            &T,
            &C,
            &[10.0, 20.0, 0.0],
            &[0.0, 12.0, 24.0],
            Route::IvBolus,
            &NcaOptions::default(),
        );
        assert!(p.cl_ss.is_nan());
    }

    #[test]
    fn an_irregular_protocol_takes_tau_from_its_last_gap_without_complaint() {
        // Review Focus. Doses at 0, 12 and 36: tau is 36 - 12 = 24, with no regularity check, no
        // averaging of the gaps (18) and no flag. python: auc_tau 48.5747..., tau 24, flags 0, the
        // intervals [0, 12], [12, 36], [36, 60]. A port which validated the regularity would
        // give NaN, one which averaged would end the last interval at 54, one which took the
        // first gap at 48
        let t = [
            0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 13.0, 14.0, 16.0, 20.0, 24.0, 30.0, 36.0, 37.0, 38.0,
            40.0, 44.0, 48.0, 54.0, 60.0,
        ];
        let c = [
            0.0, 6.872, 7.16, 5.156, 2.33, 1.047, 7.729, 7.862, 5.626, 2.541, 1.142, 0.344, 0.104,
            6.957, 7.229, 5.203, 2.35, 1.056, 0.318, 0.096,
        ];
        let (p, i) = steady(
            &t,
            &c,
            &[10.0, 10.0, 10.0],
            &[0.0, 12.0, 36.0],
            Route::Oral,
            &NcaOptions::default(),
        );
        assert_eq!(p.tau, 24.0);
        assert_eq!(p.flags, NcaFlag::NONE);
        assert_eq!(p.auc_tau, 48.5747470074613);
        assert_eq!(p.cavg, 2.023947791977554);
        assert_eq!(p.n_doses, 3.0);
        assert_eq!(p.accumulation_ratio_obs, 1.121675075613717);
        assert_eq!(column(&i, |i| i.start), [0.0, 12.0, 36.0]);
        assert_eq!(column(&i, |i| i.end), [12.0, 36.0, 60.0]);
        assert_eq!(
            column(&i, |i| i.auc),
            [43.305541919868325, 53.26320459265443, 48.5747470074613]
        );
    }

    #[test]
    fn an_explicit_tau_applies_only_to_the_last_interval() {
        // doses 0, 24 and 36 with tau 10: the interior intervals end at the next dose (24 and
        // 36), only the last is 10 long, although the last gap is 12. python: tau 10, intervals
        // [0, 24], [24, 36], [36, 46], auc_tau 45.2973..., and SPAN_LOW (the terminal phase of
        // 10 hours)
        let t = [
            0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 25.0, 26.0, 28.0, 30.0, 36.0, 37.0, 38.0, 40.0,
            44.0, 46.0,
        ];
        let c = [
            0.0, 6.872, 7.16, 5.156, 2.33, 1.047, 0.095, 6.95, 7.224, 5.199, 3.502, 1.055, 7.736,
            7.867, 5.63, 2.543, 1.704,
        ];
        let options = NcaOptions {
            tau: Some(10.0),
            ..NcaOptions::default()
        };
        let (p, i) = steady(
            &t,
            &c,
            &[10.0, 10.0, 10.0],
            &[0.0, 24.0, 36.0],
            Route::Oral,
            &options,
        );
        assert_eq!(p.tau, 10.0);
        assert_eq!(column(&i, |i| i.start), [0.0, 24.0, 36.0]);
        assert_eq!(column(&i, |i| i.end), [24.0, 36.0, 46.0]);
        assert_eq!(
            column(&i, |i| i.auc),
            [48.0659241006657, 43.748320945761606, 45.29732764345983]
        );
        assert_eq!(p.auc_tau, 45.29732764345983);
        assert_eq!(p.cavg, 4.529732764345983);
        assert_eq!(p.accumulation_ratio_obs, 0.9424000160403132);
        assert_eq!(p.flags, NcaFlag::SPAN_LOW);
    }

    #[test]
    fn a_batch_without_a_protocol_but_with_tau_is_one_interval_from_time_zero() {
        // python: dose arrays None, tau 12: n_doses 0 (the count of the protocol), tau 12, one
        // interval [0, 12], and no clearance since there is no dose
        let t = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0];
        let c = [0.0, 6.872, 7.16, 5.156, 2.33, 1.047];
        let options = NcaOptions {
            tau: Some(12.0),
            ..NcaOptions::default()
        };
        let input = RowInput {
            times: &t,
            values: &c,
            dose: reference_dose(&[], &[], None, true),
            route: Some(Route::Oral),
            has_dose: false,
            lloq: None,
            window: None,
            manual: None,
            single_dose: false,
            dose_times: &[],
            dose_amounts: &[],
        };
        let mut intervals = Vec::new();
        let p = compute_steady_state(&input, &options, &mut Scratch::new(6), None, &mut intervals);
        assert_eq!((p.n_doses, p.tau), (0.0, 12.0));
        assert_eq!((p.cmax, p.tmax), (7.16, 2.0));
        assert_eq!(p.auc_tau, 43.305541919868325);
        assert_eq!(p.ctrough, 1.047);
        assert_eq!(p.cavg, 3.608795159989027);
        assert!(p.cl_ss.is_nan());
        assert!(p.accumulation_ratio_obs.is_nan());
        assert_eq!(column(&intervals, |i| i.start), [0.0]);
        assert_eq!(column(&intervals, |i| i.end), [12.0]);
        assert_eq!(column(&intervals, |i| i.n_points), [6.0]);
        // there are no amounts, so the dose of the interval stays `NaN`
        assert!(intervals[0].dose.is_nan());
    }

    #[test]
    fn the_three_meanings_of_not_a_number_stay_distinguishable() {
        // padding of a dose array, an interval which does not exist, and an interval whose area
        // could not be computed are all NaN in the output; the flag separates the last
        let o = NcaOptions::default();
        // (1) two doses in a row of four slots: the two padded intervals are empty (NaN bounds,
        // count, dose), the two real ones are complete, and nothing is flagged
        let t = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 13.0, 14.0, 16.0, 20.0, 24.0];
        let c = [
            0.0, 6.872, 7.16, 5.156, 2.33, 1.047, 7.729, 7.862, 5.626, 2.541, 1.142,
        ];
        let n = f64::NAN;
        let (p, i) = steady(
            &t,
            &c,
            &[10.0, 10.0, n, n],
            &[0.0, 12.0, n, n],
            Route::Oral,
            &o,
        );
        assert_eq!(i.len(), 4);
        assert!(same(&column(&i, |i| i.start), &[0.0, 12.0, n, n]));
        assert!(same(&column(&i, |i| i.dose), &[10.0, 10.0, n, n]));
        assert!(same(&column(&i, |i| i.n_points), &[6.0, 6.0, n, n]));
        assert!(same(
            &column(&i, |i| i.auc),
            &[43.305541919868325, 48.06909355615606, n, n]
        ));
        assert!(same(&column(&i, |i| i.c_start), &[0.0, 1.047, n, n]));
        assert_eq!((p.n_doses, p.tau), (2.0, 12.0));
        assert_eq!(p.auc_tau, 48.06909355615606);
        assert_eq!(p.accumulation_ratio, 1.1006589341008814);
        assert_eq!(p.accumulation_ratio_obs, 1.109998661259155);
        assert_eq!(p.cl_ss, 0.20803387915600358);
        assert_eq!(p.flags, NcaFlag::NONE);
        // (2) one dose and no tau: the last interval does not exist, its area is NaN and the row
        // is not flagged, since there is nothing which could be incomplete (python flags 0)
        let (p, i) = steady(&t[..6], &c[..6], &[10.0], &[0.0], Route::Oral, &o);
        assert_eq!(i.len(), 1);
        assert!(i[0].start.is_nan() && i[0].end.is_nan() && i[0].auc.is_nan());
        assert!(p.auc_tau.is_nan() && p.tau.is_nan() && p.accumulation_ratio.is_nan());
        assert_eq!(p.n_doses, 1.0);
        assert_eq!(p.flags, NcaFlag::NONE);
        // (3) an interval which exists and which the data do not cover: NaN area, flagged
        // INCOMPLETE_INTERVAL (python 266 = incomplete, no maximum, too few points) and its
        // bounds, count and start value present
        let (p, i) = steady(
            &T[..12],
            &C[..12],
            &[10.0, 10.0, 10.0],
            &[0.0, 12.0, 24.0],
            Route::Oral,
            &o,
        );
        assert_eq!((i[2].start, i[2].end, i[2].n_points), (24.0, 36.0, 2.0));
        assert_eq!(i[2].c_start, 1.142);
        assert!(i[2].auc.is_nan() && i[2].cmax.is_nan());
        assert_eq!(
            p.flags,
            NcaFlag::INCOMPLETE_INTERVAL | NcaFlag::NO_MAX | NcaFlag::TOO_FEW_POINTS
        );
        // every steady state parameter is NaN, none of them fell back to another interval: the
        // last interval is the steady state interval, whatever its quality
        for v in [
            p.auc_tau,
            p.auc_tau_extrap_fraction,
            p.cmin_ss,
            p.cmax_ss,
            p.ctrough,
            p.cavg,
            p.accumulation_ratio_obs,
            p.cl_ss,
        ] {
            assert!(v.is_nan());
        }
        assert_eq!(p.tau, 12.0);
    }

    #[test]
    fn a_short_last_interval_is_completed_and_a_shorter_one_is_flagged() {
        // python, last sample at 35.5 of 36: completed (the area, the trough 1.1504 of the
        // regression and its extrapolated share 0.01247), no flag
        let mut t = T;
        t[15] = 35.5;
        let mut c = C;
        c[15] = 1.271;
        let (p, i) = steady(
            &t,
            &c,
            &[10.0, 10.0, 10.0],
            &[0.0, 12.0, 24.0],
            Route::Oral,
            &NcaOptions::default(),
        );
        assert_eq!(p.auc_tau, 48.49941529953174);
        assert_eq!(p.auc_tau_extrap_fraction, 0.012471394479248284);
        assert_eq!(p.ctrough, 1.150424064174049);
        assert_eq!(p.cmin_ss, 1.142);
        assert_eq!(p.cavg, 4.041617941627645);
        assert_eq!(p.fluctuation, 1.678288274143088);
        assert_eq!(p.accumulation_ratio_obs, 1.1199355359476635);
        assert_eq!(p.cl_ss, 0.20618805274744312);
        assert_eq!(p.flags, NcaFlag::NONE);
        assert_eq!(i[2].auc, 48.49941529953174);
        // the tolerance of zero disables it: the same row stays incomplete and is flagged, and
        // the predicted accumulation, which needs no interval, is still reported
        let zero = NcaOptions {
            tau_tolerance: 0.0,
            ..NcaOptions::default()
        };
        let (p, _) = steady(
            &t,
            &c,
            &[10.0, 10.0, 10.0],
            &[0.0, 12.0, 24.0],
            Route::Oral,
            &zero,
        );
        assert!(p.auc_tau.is_nan() && p.auc_tau_extrap_fraction.is_nan());
        assert_eq!(p.accumulation_ratio, 1.1006329689619072);
        assert_eq!(p.flags, NcaFlag::INCOMPLETE_INTERVAL);
        // a completed row whose interval was covered by the data reports 0, not NaN, as the
        // extrapolated share (`auc_tau_extrap_fraction` of the first test)
    }
}
