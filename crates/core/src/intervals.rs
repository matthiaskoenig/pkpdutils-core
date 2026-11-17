//! The dosing intervals of a multiple dose curve.
//!
//! Mirrors `src/pkpdutils/nca/intervals.py`: interval `k` of a row runs from the dose time `t_k`
//! to the next dose time `t_{k+1}`, the last one from `t_K` to `t_K + tau`. This module holds the
//! bounds of the intervals and the number of samples each one uses, which need no concentration
//! maths, and the table of the variables the result reports over the dimension `interval`.

/// One dosing interval of one row, in the time frame of the curve as the host sent it.
///
/// An interval which does not exist (no dose there, no end, or an end which is not after the
/// start) is [`Interval::EMPTY`], all `NaN`, which is the `NaN` padding of the python arrays.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Interval {
    /// The dose time the interval starts at (`interval_start`).
    pub start: f64,
    /// The time it ends at (`interval_end`).
    pub end: f64,
    /// The number of samples the interval uses (`interval_n_points`).
    pub n_points: f64,
}

impl Interval {
    /// The interval which does not exist.
    pub(crate) const EMPTY: Interval = Interval {
        start: f64::NAN,
        end: f64::NAN,
        n_points: f64::NAN,
    };
}

/// One per-interval variable of the result: its name, its unit expression and the field it reads.
///
/// The same shape as `Parameter` of `parameters.rs`; python's `INTERVAL_UNITS` is the counterpart.
pub(crate) struct IntervalVariable {
    /// The name of the variable, exactly the python one.
    pub name: &'static str,
    /// The unit expression, resolved like every parameter's.
    pub unit: &'static str,
    /// The field of the interval it reports.
    pub read: fn(&Interval) -> f64,
}

/// The per-interval variables this build reports, in the order python writes them.
pub(crate) const INTERVAL_VARIABLES: &[IntervalVariable] = &[
    IntervalVariable {
        name: "interval_start",
        unit: "{time}",
        read: |i| i.start,
    },
    IntervalVariable {
        name: "interval_end",
        unit: "{time}",
        read: |i| i.end,
    },
    IntervalVariable {
        name: "interval_n_points",
        unit: "dimensionless",
        read: |i| i.n_points,
    },
];

/// The relative tolerance of a sample at the end of an interval (`_interval_column`).
const END_TOLERANCE: f64 = 1e-9;

/// The bounds of every interval of a row, written to `out` (cleared first, one entry per dose
/// column).
///
/// `n_doses` is the number of finite dose times, which python counts over the whole row; the last
/// interval is the one at `n_doses - 1`. The end of interval `k` is `dose_times[k + 1]` and of the
/// last one `dose_times[n_doses - 1] + tau`; an interval is [`Interval::EMPTY`] unless its start
/// and end are both finite and `end > start`. `tau` of `None` is python's `NaN`: the last
/// interval then does not exist. `n_points` is left `NaN` here (see [`count_points`]).
///
/// Mirrors the bounds block of `compute_intervals`.
pub(crate) fn interval_bounds(
    dose_times: &[f64],
    n_doses: usize,
    tau: Option<f64>,
    out: &mut Vec<Interval>,
) {
    out.clear();
    let tau = tau.unwrap_or(f64::NAN);
    for (k, &start) in dose_times.iter().enumerate() {
        let end = if k + 1 == n_doses {
            start + tau
        } else {
            dose_times.get(k + 1).copied().unwrap_or(f64::NAN)
        };
        out.push(if start.is_finite() && end.is_finite() && end > start {
            Interval {
                start,
                end,
                n_points: f64::NAN,
            }
        } else {
            Interval::EMPTY
        });
    }
}

/// The length of the last interval of a protocol, `t_K - t_{K-1}`, `None` for fewer than two doses.
///
/// Mirrors `_interval_length` of `steady_state.py`, which is `tau` when `NCAOptions.tau` is unset.
pub(crate) fn last_interval_length(dose_times: &[f64], n_doses: usize) -> Option<f64> {
    let last = dose_times.get(n_doses.checked_sub(1)?)?;
    let previous = dose_times.get(n_doses.checked_sub(2)?)?;
    Some(last - previous)
}

/// Fill `n_points` of every existing interval of a row from its samples.
///
/// The samples are the pairs of `times` and `values` which are both finite, in order, in the same
/// frame as the interval bounds. An interval uses the samples inside `[start, end)` plus the one
/// at its end, unless that one is a post-dose sample of the next dose (a bolus whose sample at the
/// next dose lies above the last sample inside); a sample at a boundary is used by both
/// neighbours. An interval which does not exist keeps `NaN`.
///
/// Mirrors `n_points` of `_interval_column`.
pub(crate) fn count_points(
    times: &[f64],
    values: &[f64],
    bolus: bool,
    n_doses: usize,
    intervals: &mut [Interval],
) {
    let valid = || {
        times
            .iter()
            .zip(values)
            .filter(|(t, c)| t.is_finite() && c.is_finite())
            .map(|(&t, &c)| (t, c))
    };
    for (k, interval) in intervals.iter_mut().enumerate() {
        let (start, end) = (interval.start, interval.end);
        if !(start.is_finite() && end.is_finite()) {
            continue;
        }
        let tolerance = END_TOLERANCE * end.abs().max(1.0);
        let (mut before, mut inside) = (0usize, 0usize);
        let mut at_end: Option<f64> = None;
        for (t, c) in valid() {
            if t < start {
                before += 1;
            } else if t < end {
                inside += 1;
            }
            if (t - end).abs() <= tolerance {
                at_end = Some(at_end.map_or(c, |m| m.max(c)));
            }
        }
        // the samples of the interval are a block of the packed row, so the last one inside is
        // `before + inside - 1` of it. A position past the row reads as `NaN` here (python clips the
        // index instead); the two differ only for an unsorted row, which is out of scope and untested
        let last_inside = inside
            .checked_sub(1)
            .and_then(|i| valid().nth(before + i))
            .map_or(f64::NAN, |(_, c)| c);
        let ends_at_dose = k + 1 != n_doses;
        let post_dose = bolus && ends_at_dose && at_end.is_some_and(|c| c > last_inside);
        let counted_end = usize::from(at_end.is_some() && !post_dose);
        interval.n_points = (inside + counted_end) as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(times: &[f64], tau: Option<f64>) -> Vec<Interval> {
        let n = times.iter().filter(|t| t.is_finite()).count();
        let mut out = Vec::new();
        interval_bounds(times, n, tau, &mut out);
        out
    }

    #[test]
    fn the_interval_bounds_run_from_each_dose_to_the_next_and_the_last_adds_tau() {
        let got = bounds(&[0.0, 24.0, 48.0], Some(12.0));
        let pairs: Vec<_> = got.iter().map(|i| (i.start, i.end)).collect();
        assert_eq!(pairs, [(0.0, 24.0), (24.0, 48.0), (48.0, 60.0)]);
        assert!(got.iter().all(|i| i.n_points.is_nan()));
        // a trailing NaN padding column is an empty interval, and the last is the last dose
        let padded = bounds(&[0.0, 24.0, f64::NAN], Some(12.0));
        assert_eq!((padded[1].start, padded[1].end), (24.0, 36.0));
        assert!(padded[2].start.is_nan() && padded[2].end.is_nan());
        assert_eq!(last_interval_length(&[0.0, 24.0, 48.0], 3), Some(24.0));
        assert_eq!(last_interval_length(&[5.0], 1), None);
    }

    #[test]
    fn an_interval_whose_end_is_not_after_its_start_is_left_empty() {
        // no tau and one dose: the last interval has no end
        assert!(bounds(&[0.0], None)[0].start.is_nan());
        // equal and decreasing dose times
        let got = bounds(&[0.0, 0.0, -1.0], Some(5.0));
        assert!(got[0].start.is_nan() && got[0].end.is_nan());
        assert!(got[1].start.is_nan(), "0 -> -1 goes backwards");
        assert_eq!((got[2].start, got[2].end), (-1.0, 4.0));
        // a zero or negative tau
        assert!(bounds(&[0.0], Some(0.0))[0].end.is_nan());
        assert!(bounds(&[0.0], Some(-3.0))[0].end.is_nan());
    }

    #[test]
    fn the_samples_of_an_interval_are_counted_with_the_one_at_its_end() {
        let t = [0.0, 1.0, 2.0, 3.0, 4.0];
        let c = [5.0, 4.0, 3.0, 2.0, 1.0];
        let mut out = Vec::new();
        interval_bounds(&[0.0, 2.0], 2, Some(2.0), &mut out);
        count_points(&t, &c, false, 2, &mut out);
        // [0,2): 0,1 plus the one at 2; [2,4): 2,3 plus the one at 4
        assert_eq!((out[0].n_points, out[1].n_points), (3.0, 3.0));
        // a bolus whose sample at the next dose is above the last one inside belongs to the next dose
        let c = [5.0, 4.0, 9.0, 2.0, 1.0];
        count_points(&t, &c, true, 2, &mut out);
        assert_eq!((out[0].n_points, out[1].n_points), (2.0, 3.0));
        // a bolus whose sample at the next dose is below the last one inside is no post-dose
        // sample, so the interval counts it
        let c = [5.0, 4.0, 1.5, 2.0, 1.0];
        count_points(&t, &c, true, 2, &mut out);
        assert_eq!((out[0].n_points, out[1].n_points), (3.0, 3.0));
    }

    #[test]
    fn the_interval_variables_are_laid_out_sample_major() {
        use crate::assemble::result_of;
        use crate::parameters::tests::{many, regimen};
        use pkpdutils_abi::{NcaOptions, Route};

        let times = [0.0, 1.0, 2.0, 3.0];
        let batch = regimen(
            many(
                &[&times, &times],
                &[&[4.0, 3.0, 2.0, 1.0], &[4.0, 3.0, 2.0, 1.0]],
            ),
            Route::Oral,
            10.0,
            &[0.0, 2.0, 4.0],
        );
        // the protocol of each row is built here, not taken from the batch (`regimen` writes the same
        // dose times to every sample): sample 1 is shifted by 1, sample 0 keeps it
        let rows: Vec<_> = (0..2)
            .map(|i| {
                let doses = [f64::from(i), 2.0 + f64::from(i), 4.0 + f64::from(i)];
                crate::assemble::RowResult {
                    route: Some(Route::Oral),
                    dose_amount: 10.0,
                    parameters: crate::parameters::RowParameters::EMPTY,
                    intervals: {
                        let mut v = Vec::new();
                        interval_bounds(&doses, 3, Some(2.0), &mut v);
                        v
                    },
                }
            })
            .collect();
        let result = result_of(&batch, &NcaOptions::default(), &rows, &[]).unwrap();
        let start = result
            .point_variables
            .iter()
            .find(|v| v.name == "interval_start")
            .expect("the variable is emitted");
        assert_eq!((start.dim.as_str(), start.size), ("interval", 3));
        assert_eq!(start.values, [0.0, 2.0, 4.0, 1.0, 3.0, 5.0]);
        assert_eq!(start.values[3], 1.0, "sample 1 interval 0 sits at index 3");
        let end = result
            .point_variables
            .iter()
            .find(|v| v.name == "interval_end")
            .unwrap();
        assert_eq!(end.values[5], 7.0);
        assert_eq!(end.unit, "hour");
        let n = result
            .point_variables
            .iter()
            .find(|v| v.name == "interval_n_points")
            .unwrap();
        assert_eq!(n.unit, "dimensionless");
    }

    #[test]
    fn a_row_of_a_single_time_point_is_no_data_and_nothing_panics() {
        use crate::parameters::tests::linear;
        use crate::parameters::{RowInput, Scratch, compute_row};
        use pkpdutils_abi::{NcaFlag, Route};

        let (times, values) = ([1.0], [5.0]);
        let doses = [0.0, 2.0, 4.0];
        let input = RowInput {
            times: &times,
            values: &values,
            dose: crate::dose::ReferenceDose {
                amount: 10.0,
                time: 0.0,
                duration: Some(f64::NAN),
            },
            route: Some(Route::IvBolus),
            has_dose: true,
            lloq: None,
            window: None,
            manual: None,
            single_dose: false,
            dose_times: &doses,
        };
        let mut scratch = Scratch::new(1);
        let mut intervals = Vec::new();
        let row = compute_row(&input, &linear(), &mut scratch, None, Some(&mut intervals));
        assert!(row.flags.contains(NcaFlag::NO_DATA));
        assert_eq!(intervals.len(), 3);
        // the bounds exist and the one sample is counted, with no index past the row
        assert!(intervals.iter().all(|i| i.n_points <= 1.0));
        assert_eq!(intervals[0].n_points, 1.0);
        assert_eq!(intervals[1].n_points, 0.0);
    }

    #[test]
    fn the_limit_of_quantification_does_not_change_the_samples_an_interval_counts() {
        // python hands `compute_intervals` the raw times and values (`compute_steady_state`
        // BLQ-treats only the copy the point parameters read), verified by running it:
        // lloq 0.5 on this row gives interval_n_points [4, 3], the same as without a limit
        use crate::parameters::tests::linear;
        use crate::parameters::{RowInput, Scratch, compute_row};
        use pkpdutils_abi::{NcaOptions, Route};

        let times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let values = [5.0, 4.0, 0.1, 2.0, 1.0, 0.5];
        let doses = [0.0, 3.0];
        let counts = |lloq: Option<f64>| {
            let input = RowInput {
                times: &times,
                values: &values,
                dose: crate::dose::ReferenceDose {
                    amount: 10.0,
                    time: 0.0,
                    duration: Some(f64::NAN),
                },
                route: Some(Route::Oral),
                has_dose: true,
                lloq,
                window: None,
                manual: None,
                single_dose: false,
                dose_times: &doses,
            };
            let options = NcaOptions {
                tau: Some(3.0),
                ..linear()
            };
            let mut intervals = Vec::new();
            compute_row(
                &input,
                &options,
                &mut Scratch::new(6),
                None,
                Some(&mut intervals),
            );
            intervals.iter().map(|i| i.n_points).collect::<Vec<_>>()
        };
        assert_eq!(counts(None), [4.0, 3.0]);
        assert_eq!(counts(Some(0.5)), [4.0, 3.0]);
    }
}
