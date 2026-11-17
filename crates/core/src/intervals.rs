//! The dosing intervals of a multiple dose curve.
//!
//! Mirrors `src/pkpdutils/nca/intervals.py`: interval `k` of a row runs from the dose time `t_k`
//! to the next dose time `t_{k+1}`, the last one from `t_K` to `t_K + tau`. This module holds the
//! bounds of the intervals and the number of samples each one uses, which need no concentration
//! maths, the table of the variables the result reports over the dimension `interval`, the
//! interpolation of a curve at a time and the curve of one interval.

use pkpdutils_abi::{AucMethod, Batch, C0Method, NcaOptions, Route};

use crate::auc::trapezoid;
use crate::parameters::{first_maximum, first_minimum};

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
    /// The dose amount of the interval (`interval_dose`), reported only for a batch with doses.
    pub dose: f64,
    /// The number of samples the interval uses (`interval_n_points`).
    pub n_points: f64,
    /// The area under the curve of the interval (`interval_auc`).
    pub auc: f64,
    /// The largest value of the interval curve (`interval_cmax`).
    pub cmax: f64,
    /// The time of the first largest value, after the start of the interval (`interval_tmax`).
    pub tmax: f64,
    /// The smallest value of the interval curve (`interval_cmin`).
    pub cmin: f64,
    /// The value at the end of the interval (`interval_ctrough`).
    pub ctrough: f64,
    /// The value at the start of the interval (`interval_c_start`).
    pub c_start: f64,
    /// The average over the interval, area over length (`interval_cavg`).
    pub cavg: f64,
    /// `(cmax - cmin) / cavg` (`interval_fluctuation`).
    pub fluctuation: f64,
    /// `(cmax - cmin) / cmin` (`interval_swing`).
    pub swing: f64,
}

impl Interval {
    /// The interval which does not exist.
    pub(crate) const EMPTY: Interval = Interval {
        start: f64::NAN,
        end: f64::NAN,
        dose: f64::NAN,
        n_points: f64::NAN,
        auc: f64::NAN,
        cmax: f64::NAN,
        tmax: f64::NAN,
        cmin: f64::NAN,
        ctrough: f64::NAN,
        c_start: f64::NAN,
        cavg: f64::NAN,
        fluctuation: f64::NAN,
        swing: f64::NAN,
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
    /// Whether the batch reports the variable: `interval_dose` only for a batch which carries
    /// dose amounts (`has_dose` of `interval_variables` in `intervals.py`), every other one always.
    pub present: fn(&Batch) -> bool,
}

/// The variables a batch reports whatever its protocol.
fn always(_: &Batch) -> bool {
    true
}

/// A batch which carries dose amounts, which `max_doses > 0` says (as `has_dose` of `assemble.rs`).
fn has_amounts(batch: &Batch) -> bool {
    batch.max_doses > 0
}

/// The per-interval variables this build reports, in the order python writes them.
pub(crate) const INTERVAL_VARIABLES: &[IntervalVariable] = &[
    IntervalVariable {
        name: "interval_start",
        unit: "{time}",
        read: |i| i.start,
        present: always,
    },
    IntervalVariable {
        name: "interval_end",
        unit: "{time}",
        read: |i| i.end,
        present: always,
    },
    IntervalVariable {
        name: "interval_dose",
        unit: "{dose}",
        read: |i| i.dose,
        present: has_amounts,
    },
    IntervalVariable {
        name: "interval_n_points",
        unit: "dimensionless",
        read: |i| i.n_points,
        present: always,
    },
    IntervalVariable {
        name: "interval_auc",
        unit: "({unit}) * ({time})",
        read: |i| i.auc,
        present: always,
    },
    IntervalVariable {
        name: "interval_cmax",
        unit: "{unit}",
        read: |i| i.cmax,
        present: always,
    },
    IntervalVariable {
        name: "interval_tmax",
        unit: "{time}",
        read: |i| i.tmax,
        present: always,
    },
    IntervalVariable {
        name: "interval_cmin",
        unit: "{unit}",
        read: |i| i.cmin,
        present: always,
    },
    IntervalVariable {
        name: "interval_ctrough",
        unit: "{unit}",
        read: |i| i.ctrough,
        present: always,
    },
    IntervalVariable {
        name: "interval_c_start",
        unit: "{unit}",
        read: |i| i.c_start,
        present: always,
    },
    IntervalVariable {
        name: "interval_cavg",
        unit: "{unit}",
        read: |i| i.cavg,
        present: always,
    },
    IntervalVariable {
        name: "interval_fluctuation",
        unit: "dimensionless",
        read: |i| i.fluctuation,
        present: always,
    },
    IntervalVariable {
        name: "interval_swing",
        unit: "dimensionless",
        read: |i| i.swing,
        present: always,
    },
];

/// The relative tolerance of a sample at the end of an interval (`_interval_column`).
const END_TOLERANCE: f64 = 1e-9;

/// The largest value among the samples at the end of an interval, `None` with no sample there.
///
/// A sample is at the end when its time is within `END_TOLERANCE * max(1, |end|)` of `end`, a
/// tolerance relative to the time frame of the interval bounds (absolute host times). Shared by
/// [`count_points`] and [`interval_curve`] so the two cannot drift; each applies its own rule on
/// top of it (`n_points` against the post-dose sample, `covered` against the next dose).
///
/// Mirrors `tolerance`, `at_end` and `c_sample_at_end` of `_interval_column` of
/// `src/pkpdutils/nca/intervals.py`.
fn largest_at_end(samples: impl Iterator<Item = (f64, f64)>, end: f64) -> Option<f64> {
    let tolerance = END_TOLERANCE * end.abs().max(1.0);
    samples
        .filter(|&(t, _)| (t - end).abs() <= tolerance)
        .map(|(_, c)| c)
        .reduce(f64::max)
}

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
                ..Interval::EMPTY
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
        let (mut before, mut inside) = (0usize, 0usize);
        for (t, _) in valid() {
            if t < start {
                before += 1;
            } else if t < end {
                inside += 1;
            }
        }
        let at_end = largest_at_end(valid(), end);
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

/// The value of a packed curve at a query time, by interpolation between the bracketing samples.
///
/// `t` and `c` are the packed points of a row and `n_valid` how many of them are valid, so the
/// padding behind them is never read. Between two samples the value is linear, or log-linear
/// `exp(ln c1 + f (ln c2 - ln c1))` on a segment the trapezoid `method` treats logarithmically
/// (`Log` where both values are positive and differ, `LinearLog` where it also falls). A query
/// exactly at the last time is that sample, one before the first or after the last time, or a
/// `NaN`, is `NaN`. A query at an inner sample takes the segment which starts there, so the log
/// rule may return it up to one rounding off.
///
/// Mirrors `interpolate_at` of `src/pkpdutils/nca/auc.py`, with the same expressions.
pub(crate) fn interpolate_at(
    t: &[f64],
    c: &[f64],
    n_valid: usize,
    t_query: f64,
    method: AucMethod,
) -> f64 {
    let n_valid = n_valid.min(t.len()).min(c.len());
    if n_valid == 0 {
        return f64::NAN;
    }
    if t_query == t[n_valid - 1] {
        return c[n_valid - 1];
    }
    // the number of valid points at or before the query time; `count < n_valid` also keeps `i2`
    // inside the row
    let count = t[..n_valid].iter().filter(|&&x| x <= t_query).count();
    if count == 0 || count >= n_valid {
        return f64::NAN;
    }
    let (t1, t2, c1, c2) = (t[count - 1], t[count], c[count - 1], c[count]);
    let fraction = (t_query - t1) / (t2 - t1);
    let positive = c1 > 0.0 && c2 > 0.0 && c1 != c2;
    let use_log = match method {
        AucMethod::Linear => false,
        AucMethod::LinearLog => positive && c2 < c1,
        AucMethod::Log => positive,
    };
    if use_log {
        (c1.ln() + fraction * (c2.ln() - c1.ln())).exp()
    } else {
        c1 + fraction * (c2 - c1)
    }
}

/// The value at the start of an interval of an intravenous bolus whose first sample comes later.
///
/// The log-linear back-extrapolation of the first two samples after `start`,
/// `exp(ln c1 - (ln c2 - ln c1) / (t2 - t1) (t1 - start))`; the first sample when there is only
/// one, when either is not positive, when the pair does not decline, or when `c0_method` is not
/// `LogBackExtrapolation` (`FirstValue`, and `None`, which says that the `C0` of a dose is not
/// estimated while an interval still needs a value at its start). `NaN` with no sample after
/// `start`.
///
/// Mirrors `_back_extrapolate` of `src/pkpdutils/nca/intervals.py`.
fn back_extrapolate(t: &[f64], c: &[f64], n_valid: usize, start: f64, method: C0Method) -> f64 {
    let first = t[..n_valid].iter().filter(|&&x| x <= start).count();
    let n_after = n_valid - first;
    if n_after == 0 {
        return f64::NAN;
    }
    let first_value = c[first];
    if method != C0Method::LogBackExtrapolation || n_after < 2 {
        return first_value;
    }
    let (t1, c1, t2, c2) = (t[first], c[first], t[first + 1], c[first + 1]);
    if c1 > 0.0 && c2 > 0.0 && c2 < c1 && t2 > t1 {
        (c1.ln() - (c2.ln() - c1.ln()) / (t2 - t1) * (t1 - start)).exp()
    } else {
        first_value
    }
}

/// The number of positive samples the regression of [`extrapolate_end`] uses at most
/// (`TROUGH_POINTS` of `src/pkpdutils/nca/intervals.py`).
pub(crate) const TROUGH_POINTS: usize = 3;

/// Value at the end of an interval from the log-linear regression of its last samples.
///
/// The least squares line through `(t_i, ln C_i)` of the last `max_points` positive samples with
/// `inside[i]` is evaluated at `t_end`: the estimate of a trough which was not observed
/// (Gabrielsson & Weiner 2016, ch. 2.8). The line is fitted about the centroid, `t` and `ln C`
/// both centred, `exp(y_bar + slope (t_end - t_bar))` with `slope = sum(dt dy) / sum(dt dt)`,
/// because the normal equations on absolute times subtract two nearly equal sums and lose their
/// digits once the times are large relative to their spread. The sums run in the order of the
/// samples, as the cumulative sums of python's `_ordered_sum` do.
///
/// `NaN` with fewer than two positive samples inside, or when `sum(dt dt)` is not positive (all
/// selected samples at one time). `t`, `c` and `inside` are read up to the shortest of the three.
///
/// Mirrors `_extrapolate_end` of `src/pkpdutils/nca/intervals.py`.
pub(crate) fn extrapolate_end(
    t: &[f64],
    c: &[f64],
    inside: &[bool],
    t_end: f64,
    max_points: usize,
) -> f64 {
    let n = t.len().min(c.len()).min(inside.len());
    let positive = |j: usize| inside[j] && c[j] > 0.0;
    let total = (0..n).filter(|&j| positive(j)).count();
    // the last `max_points` positive samples: those with at most `max_points - 1` after them
    let mut seen = 0usize;
    let selected = |j: usize, seen: &mut usize| {
        if !positive(j) {
            return false;
        }
        *seen += 1;
        total - *seen < max_points
    };
    let chosen: Vec<usize> = (0..n).filter(|&j| selected(j, &mut seen)).collect();
    let count = chosen.len() as f64;
    let t_bar = chosen.iter().fold(0.0, |s, &j| s + t[j]) / count;
    let y_bar = chosen.iter().fold(0.0, |s, &j| s + c[j].ln()) / count;
    let denominator = chosen.iter().fold(0.0, |s, &j| {
        let dt = t[j] - t_bar;
        s + dt * dt
    });
    // centring y as well keeps the slope exact when t_bar carries a rounding error
    let numerator = chosen
        .iter()
        .fold(0.0, |s, &j| s + (t[j] - t_bar) * (c[j].ln() - y_bar));
    if chosen.len() >= 2 && denominator > 0.0 {
        (y_bar + numerator / denominator * (t_end - t_bar)).exp()
    } else {
        f64::NAN
    }
}

/// What [`interval_curve`] reports besides the curve it wrote.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IntervalCurve {
    /// The number of points of the curve.
    pub n: usize,
    /// The value at the start of the interval as interpolated (and, for a bolus, back
    /// extrapolated): not the first point of the curve, which on a tie is the observed sample.
    pub c_start: f64,
    /// Whether the interval has an observation of its own: a sample inside it, or one at its
    /// end when no dose follows. Python's `covering >= 1`, taken from the raw row.
    pub covered: bool,
    /// Whether the trough was replaced by the regression of [`extrapolate_end`], a finite one
    /// (`NCAFlag.EXTRAPOLATED_TROUGH`).
    pub extrapolated: bool,
}

/// The curve of one interval: the samples inside it plus both bounds, into `out_t` and `out_c`
/// (cleared first).
///
/// The samples are those with `start <= t < end` of the packed row (`t`, `c`, `n_valid`), the
/// contiguous block behind the ones before the start. The value at the start is interpolated
/// by `options.auc_method`, and for [`Route::IvBolus`] only, where that is `NaN` (the start lies
/// before the first sample) it is [`back_extrapolate`]d. The value at the end is interpolated,
/// except for the trough of a bolus: when `ends_at_dose` (a dose follows the interval), a sample
/// lies within `1e-9 * max(1, |end|)` of the end and the largest such sample exceeds the last
/// sample inside, that sample is the post-dose value of the next bolus and the end value is the
/// regression of [`extrapolate_end`] instead. It is assigned **even where the regression is
/// `NaN`**, so a failed regression does not fall back to the interpolated value: the end is then
/// not finite, the end bound is not inserted and the interval is not complete. Only the flag
/// `extrapolated` needs a finite trough. A bound whose time or value is not finite is not
/// inserted, so an interval beyond the data or an [`Interval::EMPTY`] gives a shorter curve. A
/// sample recorded exactly at the start (compared with `==`, no tolerance) and the bound carry
/// the same time, and their two values are swapped so the observed one stays in front, which
/// makes a zero-width step of the trapezoid rule.
///
/// The row is in the frame of the interval bounds (the host frame, absolute times).
///
/// Mirrors the block, the start and end values, the post-dose rule and `_with_bounds` of
/// `_interval_column` of `src/pkpdutils/nca/intervals.py`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn interval_curve(
    t: &[f64],
    c: &[f64],
    n_valid: usize,
    interval: &Interval,
    ends_at_dose: bool,
    route: Option<Route>,
    options: &NcaOptions,
    out_t: &mut Vec<f64>,
    out_c: &mut Vec<f64>,
) -> IntervalCurve {
    out_t.clear();
    out_c.clear();
    let n_valid = n_valid.min(t.len()).min(c.len());
    let (start, end) = (interval.start, interval.end);
    let (t, c) = (&t[..n_valid], &c[..n_valid]);
    let before = t.iter().filter(|&&x| x < start).count();
    let inside = t.iter().filter(|&&x| x >= start && x < end).count();
    let mut c_start = interpolate_at(t, c, n_valid, start, options.auc_method);
    if route == Some(Route::IvBolus) && c_start.is_nan() {
        c_start = back_extrapolate(t, c, n_valid, start, options.c0_method);
    }
    let mut c_end = interpolate_at(t, c, n_valid, end, options.auc_method);

    // a sample at the end of the interval, and the largest of them
    let sample_at_end = largest_at_end(t.iter().copied().zip(c.iter().copied()), end);
    let last_inside = inside.checked_sub(1).map_or(f64::NAN, |i| c[before + i]);
    let post_dose = route == Some(Route::IvBolus)
        && ends_at_dose
        && sample_at_end.is_some_and(|v| v > last_inside);
    let mut extrapolated = false;
    if post_dose {
        let mask: Vec<bool> = t.iter().map(|&x| x >= start && x < end).collect();
        let trough = extrapolate_end(t, c, &mask, end, TROUGH_POINTS);
        // assigned whatever the regression gave, a NaN included
        c_end = trough;
        extrapolated = trough.is_finite();
    }
    let covered = inside + usize::from(sample_at_end.is_some() && !ends_at_dose) >= 1;

    let has_start = start.is_finite() && c_start.is_finite();
    let tie = has_start && inside >= 1 && t[before] == start;
    if has_start {
        // on a tie the observed value goes in front and the interpolated one takes its place
        out_t.push(start);
        out_c.push(if tie { c[before] } else { c_start });
    }
    for (k, (&time, &value)) in t[before..before + inside]
        .iter()
        .zip(&c[before..before + inside])
        .enumerate()
    {
        out_t.push(time);
        out_c.push(if tie && k == 0 { c_start } else { value });
    }
    if end.is_finite() && c_end.is_finite() {
        out_t.push(end);
        out_c.push(c_end);
    }
    IntervalCurve {
        n: out_t.len(),
        c_start,
        covered,
        extrapolated,
    }
}

/// The parameters of every interval of a row, written into `intervals` (the bounds and the
/// counts of [`interval_bounds`] and [`count_points`] already in place).
///
/// `t` and `c` are the packed valid points of the row in the frame of the interval bounds, the
/// raw times and values (no limit of quantification treatment, no shift by the dose time), and
/// `n_valid` how many there are. Every interval builds its curve ([`interval_curve`]; `ends_at_dose`
/// is true for every interval but the last of the protocol of `n_doses` doses) and takes its
/// parameters from it ([`interval_parameters`]). `curve_t` and `curve_c` are scratch buffers.
/// Returns whether the trough of any interval was regressed (`NCAFlag.EXTRAPOLATED_TROUGH`).
///
/// Mirrors the loop over the intervals of `compute_intervals` in
/// `src/pkpdutils/nca/intervals.py`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn analyze_intervals(
    t: &[f64],
    c: &[f64],
    n_valid: usize,
    n_doses: usize,
    route: Option<Route>,
    options: &NcaOptions,
    intervals: &mut [Interval],
    curve_t: &mut Vec<f64>,
    curve_c: &mut Vec<f64>,
) -> bool {
    let mut extrapolated = false;
    for (k, interval) in intervals.iter_mut().enumerate() {
        let curve = interval_curve(
            t,
            c,
            n_valid,
            interval,
            k + 1 != n_doses,
            route,
            options,
            curve_t,
            curve_c,
        );
        interval_parameters(
            curve_t,
            curve_c,
            curve.n,
            curve.c_start,
            curve.covered,
            interval,
            options,
        );
        extrapolated |= curve.extrapolated;
    }
    extrapolated
}

/// Write the dose amount of every interval which exists (`interval_dose`).
///
/// `amounts` are the dose amounts of the row, in the order of its dose times; interval `k` takes
/// `amounts[k]`, and an interval which does not exist, or has no amount, keeps `NaN`.
///
/// Mirrors `np.where(usable, amounts, np.nan)` in `compute_intervals`.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "reached by the multiple dose analysis, which a later task wires in"
    )
)]
pub(crate) fn fill_doses(intervals: &mut [Interval], amounts: &[f64]) {
    for (k, interval) in intervals.iter_mut().enumerate() {
        interval.dose = if interval.start.is_finite() {
            amounts.get(k).copied().unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };
    }
}

/// The nine concentration variables of one interval, from the curve of the interval.
///
/// `t` and `c` are the first `n` points of the curve [`interval_curve`] built: the samples inside
/// plus both bounds. `c_start` is the value at the start as [`interval_curve`] computed it (not
/// the first point of the curve, which is the observed sample on a tie), and `covered` whether
/// the interval has an observation of its own: a sample inside it, or at its end when no dose
/// follows (`covering >= 1` of `_interval_column`, which the caller knows from the raw row).
///
/// An interval is complete when it exists, its curve reaches its end and has two points, and it is
/// covered. Only a complete interval reports the area, the extremes, the trough and what is built
/// from them; `c_start` is reported for every interval which exists. `tmax` is the time of the
/// first largest value after the start, `fluctuation` is `(cmax - cmin) / cavg` and `swing`
/// `(cmax - cmin) / cmin`, with the IEEE division of numpy where a denominator is zero.
///
/// Mirrors the concentration branch of `_interval_column` of `src/pkpdutils/nca/intervals.py`
/// (Gabrielsson & Weiner 2016, ch. 2.8 for the fluctuation and the swing).
pub(crate) fn interval_parameters(
    t: &[f64],
    c: &[f64],
    n: usize,
    c_start: f64,
    covered: bool,
    interval: &mut Interval,
    options: &NcaOptions,
) {
    let nan = f64::NAN;
    let exists = interval.start.is_finite() && interval.end.is_finite();
    interval.c_start = if exists { c_start } else { nan };
    let n = n.min(t.len()).min(c.len());
    let (t, c) = (&t[..n], &c[..n]);
    // the curve reaches the end of the interval only when a finite value at the end was inserted
    let complete = exists && covered && n >= 2 && t[n - 1] == interval.end;
    if !complete {
        (
            interval.auc,
            interval.cmax,
            interval.tmax,
            interval.cmin,
            interval.ctrough,
            interval.cavg,
            interval.fluctuation,
            interval.swing,
        ) = (nan, nan, nan, nan, nan, nan, nan, nan);
        return;
    }
    let area = trapezoid(t, c, options.auc_method).auc;
    let imax = first_maximum(c, n);
    interval.auc = area;
    interval.cmax = c[imax];
    interval.tmax = t[imax] - interval.start;
    interval.cmin = c[first_minimum(c, n)];
    interval.ctrough = c[n - 1];
    interval.cavg = area / (interval.end - interval.start);
    interval.fluctuation = (interval.cmax - interval.cmin) / interval.cavg;
    interval.swing = (interval.cmax - interval.cmin) / interval.cmin;
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

    fn eval(t: &[f64], c: &[f64], q: f64, m: AucMethod) -> f64 {
        interpolate_at(t, c, t.len(), q, m)
    }

    #[test]
    fn interpolation_is_linear_or_log_linear_by_the_area_method() {
        let (t, c) = ([0.0, 2.0, 4.0], [8.0, 4.0, 6.0]);
        // the falling segment, query at its middle: linear 6, log-linear the geometric mean
        let geo = 32.0_f64.sqrt();
        assert_eq!(eval(&t, &c, 1.0, AucMethod::Linear), 6.0);
        assert!((eval(&t, &c, 1.0, AucMethod::LinearLog) - geo).abs() < 1e-14);
        assert!((eval(&t, &c, 1.0, AucMethod::Log) - geo).abs() < 1e-14);
        // the rising segment: only `Log` is logarithmic, linear-up/log-down stays linear
        assert_eq!(eval(&t, &c, 3.0, AucMethod::Linear), 5.0);
        assert_eq!(eval(&t, &c, 3.0, AucMethod::LinearLog), 5.0);
        assert!((eval(&t, &c, 3.0, AucMethod::Log) - 24.0_f64.sqrt()).abs() < 1e-14);
        // a segment the logarithm is undefined on is linear under every method: a zero, a
        // negative value and a flat segment (`positive` of `interpolate_at`)
        for (pair, expected) in [([4.0, 0.0], 2.0), ([4.0, -2.0], 1.0), ([3.0, 3.0], 3.0)] {
            for m in [AucMethod::Linear, AucMethod::LinearLog, AucMethod::Log] {
                assert_eq!(eval(&[0.0, 2.0], &pair, 1.0, m), expected);
            }
        }
        // a query exactly on an inner sample takes the segment which starts there
        assert_eq!(eval(&t, &c, 2.0, AucMethod::Linear), 4.0);
        // the interpolation reads the first `n_valid` points only
        let t = [0.0, 2.0, 4.0, 5.0];
        let c = [8.0, 4.0, 6.0, 99.0];
        assert_eq!(interpolate_at(&t, &c, 3, 1.0, AucMethod::Linear), 6.0);
    }

    #[test]
    fn interpolation_beyond_the_data_is_not_a_number() {
        let (t, c) = ([1.0, 2.0, 4.0], [8.0, 4.0, 2.0]);
        for m in [AucMethod::Linear, AucMethod::LinearLog, AucMethod::Log] {
            assert!(eval(&t, &c, 0.5, m).is_nan(), "before the first sample");
            assert!(eval(&t, &c, 4.5, m).is_nan(), "after the last sample");
            assert!(eval(&t, &c, f64::NAN, m).is_nan(), "a query of NaN");
            // the first sample is inside (the log rule returns it to a rounding) and the last
            // one is the sample itself, exactly
            assert!((eval(&t, &c, 1.0, m) - 8.0).abs() < 1e-14);
            assert_eq!(eval(&t, &c, 4.0, m), 2.0);
        }
        // the last time is the last VALID point, not the end of the slice, and a row of no valid
        // point has no value at all
        let (t, c) = ([1.0, 2.0, 4.0, 6.0], [8.0, 4.0, 2.0, 1.0]);
        let m = AucMethod::Linear;
        assert!(interpolate_at(&t, &c, 3, 5.0, m).is_nan(), "past n_valid");
        assert!(
            interpolate_at(&t, &c, 3, 6.0, m).is_nan(),
            "padding is no sample"
        );
        assert_eq!(interpolate_at(&t, &c, 3, 4.0, m), 2.0);
        assert!(interpolate_at(&t, &c, 0, 4.0, m).is_nan());
        assert!(interpolate_at(&[], &[], 0, 4.0, m).is_nan());
        assert!(interpolate_at(&[3.0], &[5.0], 1, 2.0, m).is_nan());
        assert_eq!(interpolate_at(&[3.0], &[5.0], 1, 3.0, m), 5.0);
    }

    fn curve(
        t: &[f64],
        c: &[f64],
        (start, end): (f64, f64),
        route: Option<Route>,
        options: &NcaOptions,
    ) -> (Vec<f64>, Vec<f64>, usize) {
        let interval = Interval {
            start,
            end,
            ..Interval::EMPTY
        };
        let (mut out_t, mut out_c) = (vec![9.0], vec![9.0]);
        let n = interval_curve(
            t,
            c,
            t.len(),
            &interval,
            false,
            route,
            options,
            &mut out_t,
            &mut out_c,
        )
        .n;
        (out_t, out_c, n)
    }

    fn with(method: AucMethod, c0: C0Method) -> NcaOptions {
        NcaOptions {
            auc_method: method,
            c0_method: c0,
            ..NcaOptions::default()
        }
    }

    #[test]
    fn a_sample_exactly_at_the_interval_start_keeps_the_observed_value_first() {
        // Log-linear interpolation at an inner sample is exp(ln 3), which is 3 + 1 ulp: the
        // observed value and the interpolated one are told apart, and the observed one goes first
        // (numbers from python `_with_bounds`)
        let log = with(AucMethod::Log, C0Method::LogBackExtrapolation);
        let (t, c) = ([0.0, 1.0, 2.0, 3.0], [5.0, 3.0, 2.0, 1.0]);
        let (ot, oc, n) = curve(&t, &c, (1.0, 3.0), Some(Route::Oral), &log);
        assert_eq!((ot, n), (vec![1.0, 1.0, 2.0, 3.0], 4));
        let interpolated = 3.0 + f64::EPSILON * 2.0;
        assert_eq!(interpolate_at(&t, &c, 4, 1.0, AucMethod::Log), interpolated);
        assert_eq!(oc, [3.0, interpolated, 2.0, 1.0]);
    }

    #[test]
    fn samples_coincident_with_an_interval_bound_keep_their_input_order() {
        let lin = with(AucMethod::Linear, C0Method::LogBackExtrapolation);
        // two samples at the start: python gives [5, 6, 6, 4, 3] (the value at the start is the
        // second duplicate, swapped behind the first, which stays in front)
        let (t, c) = ([0.0, 1.0, 1.0, 2.0, 3.0], [9.0, 5.0, 6.0, 4.0, 3.0]);
        let (ot, oc, n) = curve(&t, &c, (1.0, 3.0), Some(Route::Oral), &lin);
        assert_eq!(ot, [1.0, 1.0, 1.0, 2.0, 3.0]);
        assert_eq!(oc, [5.0, 6.0, 6.0, 4.0, 3.0]);
        assert_eq!(n, 5);
        // the comparison with the start is exact, not by a tolerance: a sample 1e-12 after it is
        // no tie, so the bound goes first with its interpolated value (python: 4.000000000001622)
        let log = with(AucMethod::Log, C0Method::LogBackExtrapolation);
        let (t, c) = ([0.0, 1.0, 2.0 + 1e-12, 4.0], [8.0, 6.0, 4.0, 2.0]);
        let (ot, oc, _) = curve(&t, &c, (2.0, 4.0), Some(Route::Oral), &log);
        assert_eq!(ot, [2.0, 2.0 + 1e-12, 4.0]);
        assert_eq!(oc[0], 4.000000000001622);
        assert_eq!(&oc[1..], [4.0, 2.0]);
        // and one 1e-12 before it is no sample of the interval at all
        let (t, c) = ([0.0, 2.0 - 1e-12, 3.0, 4.0], [8.0, 4.0, 3.0, 2.0]);
        let (ot, _, n) = curve(&t, &c, (2.0, 4.0), Some(Route::Oral), &lin);
        assert_eq!((ot, n), (vec![2.0, 3.0, 4.0], 3));
        // a sample exactly at the end is not inside, the end is inserted from the interpolation
        let (t, c) = ([0.0, 1.0, 2.0, 3.0], [8.0, 6.0, 4.0, 2.0]);
        let (ot, oc, _) = curve(&t, &c, (0.0, 2.0), Some(Route::Oral), &lin);
        assert_eq!(
            (ot, oc),
            (vec![0.0, 0.0, 1.0, 2.0], vec![8.0, 8.0, 6.0, 4.0])
        );
    }

    #[test]
    fn a_sample_at_the_start_which_is_the_last_sample_is_a_tie_too() {
        // the value at the start is the `on_last` branch of `interpolate_at`, not a segment
        let lin = with(AucMethod::Linear, C0Method::LogBackExtrapolation);
        let (t, c) = ([0.0, 1.0, 2.0], [5.0, 3.0, 2.0]);
        let (ot, oc, n) = curve(&t, &c, (2.0, 4.0), Some(Route::Oral), &lin);
        assert_eq!((ot, oc, n), (vec![2.0, 2.0], vec![2.0, 2.0], 2));
    }

    #[test]
    fn a_bolus_interval_back_extrapolates_a_start_the_data_does_not_reach() {
        let t = [1.0, 2.0, 4.0, 6.0];
        let c = [8.0, 4.0, 2.0, 1.0];
        let log = with(AucMethod::Linear, C0Method::LogBackExtrapolation);
        let (ot, oc, n) = curve(&t, &c, (0.0, 4.0), Some(Route::IvBolus), &log);
        assert_eq!((ot, n), (vec![0.0, 1.0, 2.0, 4.0], 4));
        assert_eq!(
            oc,
            [15.999999999999991, 8.0, 4.0, 2.0],
            "bit for bit with python"
        );
        // the line runs from the sample, so a start away from zero is told from the dose at zero
        let (t2, c2) = ([2.0, 3.0, 5.0], [8.0, 4.0, 2.0]);
        let (_, oc, _) = curve(&t2, &c2, (0.5, 3.5), Some(Route::IvBolus), &log);
        assert_eq!(oc, [22.627416997969508, 8.0, 4.0, 3.5]);
        // the other routes and the other methods do not extrapolate
        let (ot, oc, n) = curve(&t, &c, (0.0, 4.0), Some(Route::Oral), &log);
        assert_eq!((ot, oc, n), (vec![1.0, 2.0, 4.0], vec![8.0, 4.0, 2.0], 3));
        let (_, oc, _) = curve(&t, &c, (0.0, 4.0), None, &log);
        assert_eq!(oc.len(), 3);
        let (_, oc, _) = curve(&t, &c, (0.0, 4.0), Some(Route::IvInfusion), &log);
        assert_eq!(oc.len(), 3);
        // a start after the last sample has no value, a bolus included, so no start is inserted
        let (ot, oc, n) = curve(&t, &c, (7.0, 9.0), Some(Route::IvBolus), &log);
        assert_eq!((ot.len(), oc.len(), n), (0, 0, 0));
        // an interval which does not exist is an empty curve
        let empty = Interval::EMPTY;
        let (mut a, mut b) = (vec![1.0], vec![1.0]);
        let n = interval_curve(
            &t,
            &c,
            4,
            &empty,
            false,
            Some(Route::IvBolus),
            &log,
            &mut a,
            &mut b,
        )
        .n;
        assert_eq!((n, a.len(), b.len()), (0, 0, 0));
        // only the first n_valid points are read
        let interval = Interval {
            start: 0.0,
            end: 4.0,
            ..Interval::EMPTY
        };
        let n = interval_curve(
            &t,
            &c,
            2,
            &interval,
            false,
            Some(Route::IvBolus),
            &log,
            &mut a,
            &mut b,
        )
        .n;
        assert_eq!(
            (n, a.clone()),
            (3, vec![0.0, 1.0, 2.0]),
            "no end beyond the data"
        );
    }

    #[test]
    fn a_bolus_start_falls_back_to_the_first_sample_when_the_pair_does_not_decline() {
        let t = [1.0, 2.0, 4.0, 6.0];
        let bolus = Some(Route::IvBolus);
        let first = |c: &[f64], options: &NcaOptions| curve(&t, c, (0.0, 4.0), bolus, options).1[0];
        let on = with(AucMethod::Linear, C0Method::LogBackExtrapolation);
        // rising, a zero second value and a zero first value (python: 8, 8, 0)
        assert_eq!(first(&[8.0, 9.0, 2.0, 1.0], &on), 8.0);
        assert_eq!(first(&[8.0, 0.0, 2.0, 1.0], &on), 8.0);
        assert_eq!(first(&[0.0, 0.0, 2.0, 1.0], &on), 0.0);
        // equal values do not decline either
        assert_eq!(first(&[8.0, 8.0, 2.0, 1.0], &on), 8.0);
        // two samples at the same time are no pair to extrapolate with (`t2 > t1`): the first
        // value, not the infinity a division by the zero width gives
        let (_, oc, _) = curve(&[1.0, 1.0], &[8.0, 4.0], (0.0, 4.0), bolus, &on);
        assert_eq!(oc, [8.0, 8.0, 4.0]);
        // one sample after the start: nothing to extrapolate with
        let (_, oc, _) = curve(&[1.0], &[8.0], (0.0, 4.0), bolus, &on);
        assert_eq!(oc, [8.0, 8.0]);
        // the option: the first value and no estimate both start at the first sample, which a
        // dosing interval needs anyway
        let c = [8.0, 4.0, 2.0, 1.0];
        for method in [C0Method::FirstValue, C0Method::None] {
            let o = with(AucMethod::Linear, method);
            assert_eq!(first(&c, &o), 8.0, "{method:?}");
        }
        assert!((first(&c, &on) - 16.0).abs() < 1e-12);
    }

    // ---- the nine concentration variables ----

    /// The variables of one interval, from the curve `interval_curve` builds for it.
    fn parameters_of(
        t: &[f64],
        c: &[f64],
        (start, end): (f64, f64),
        method: AucMethod,
        route: Route,
    ) -> Interval {
        let options = with(method, C0Method::LogBackExtrapolation);
        let mut interval = Interval {
            start,
            end,
            ..Interval::EMPTY
        };
        let (mut ot, mut oc) = (Vec::new(), Vec::new());
        let n = interval_curve(
            t,
            c,
            t.len(),
            &interval,
            false,
            Some(route),
            &options,
            &mut ot,
            &mut oc,
        )
        .n;
        let c_start = interpolate_at(t, c, t.len(), start, options.auc_method);
        interval_parameters(&ot, &oc, n, c_start, true, &mut interval, &options);
        interval
    }

    fn near(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn the_parameters_of_one_interval_match_the_closed_form_of_a_mono_exponential() {
        // noise free c0*exp(-k t): with the log rule the trapezoid is the integral, so the area of
        // [start, end] is c0/k*(exp(-k*start) - exp(-k*end)); the peak is the first sample and the
        // trough the last. The second interval starts at 6 (not 0), so a tmax which forgot to
        // subtract the start would be 6, not 0
        let (c0, k) = (10.0_f64, 0.3_f64);
        let t: Vec<f64> = (0..=12).map(f64::from).collect();
        let c: Vec<f64> = t.iter().map(|&x| c0 * (-k * x).exp()).collect();
        let f = |x: f64| c0 * (-k * x).exp();
        for (start, end) in [(0.0, 6.0), (6.0, 12.0)] {
            let i = parameters_of(&t, &c, (start, end), AucMethod::Log, Route::Oral);
            near(i.auc, c0 / k * ((-k * start).exp() - (-k * end).exp()));
            near(i.cmax, f(start));
            assert_eq!(i.tmax, 0.0);
            near(i.cmin, f(end));
            near(i.ctrough, f(end));
            near(i.c_start, f(start));
            near(i.cavg, i.auc / (end - start));
            near(i.fluctuation, (f(start) - f(end)) / i.cavg);
            near(i.swing, (f(start) - f(end)) / f(end));
        }
    }

    #[test]
    fn a_hump_in_the_interval_separates_the_extremes_from_the_ends() {
        // samples 4..8 = 3, 1, 9, 6, 2 (hand computed, linear): the minimum 1 is inside, not the
        // trough 2; the peak 9 is at 6, 2 after the start of the interval. Area 2 + 5 + 7.5 + 4.
        // swing uses cmin (8 / 1), not the trough (8 / 2); the fluctuation the mean (8 / 4.625)
        let t = [4.0, 5.0, 6.0, 7.0, 8.0];
        let c = [3.0, 1.0, 9.0, 6.0, 2.0];
        let i = parameters_of(&t, &c, (4.0, 8.0), AucMethod::Linear, Route::Oral);
        assert_eq!(
            (i.auc, i.cmax, i.tmax, i.cmin, i.ctrough, i.c_start),
            (18.5, 9.0, 2.0, 1.0, 2.0, 3.0)
        );
        assert_eq!(i.cavg, 4.625);
        assert_eq!(i.swing, 8.0);
        near(i.fluctuation, 8.0 / 4.625);
    }

    #[test]
    fn a_tied_maximum_reports_the_first_time() {
        // 9 at 6 and at 7: tmax is 2, not 3
        let t = [4.0, 5.0, 6.0, 7.0, 8.0];
        let c = [3.0, 1.0, 9.0, 9.0, 2.0];
        let i = parameters_of(&t, &c, (4.0, 8.0), AucMethod::Linear, Route::Oral);
        assert_eq!((i.tmax, i.auc), (2.0, 21.5));
    }

    #[test]
    fn a_trough_which_is_the_minimum_and_a_peak_at_the_end_are_found() {
        // the extremes are searched over the whole curve, the end included: a rising interval
        // peaks at its end (tmax is the interval length) and its minimum is the start
        let t = [0.0, 1.0, 2.0];
        let c = [1.0, 2.0, 5.0];
        let i = parameters_of(&t, &c, (0.0, 2.0), AucMethod::Linear, Route::Oral);
        assert_eq!((i.cmax, i.tmax, i.cmin, i.ctrough), (5.0, 2.0, 1.0, 5.0));
    }

    #[test]
    fn the_start_value_is_the_one_given_not_the_first_point_of_the_curve() {
        // on a tie the curve starts with the observed value; the variable is the interpolated one
        let options = NcaOptions::default();
        let mut i = Interval {
            start: 0.0,
            end: 2.0,
            ..Interval::EMPTY
        };
        interval_parameters(
            &[0.0, 0.0, 2.0],
            &[5.0, 4.0, 1.0],
            3,
            4.0,
            true,
            &mut i,
            &options,
        );
        assert_eq!((i.c_start, i.cmax), (4.0, 5.0));
    }

    #[test]
    fn an_interval_without_an_observation_of_its_own_reports_only_its_start_value() {
        let options = with(AucMethod::Linear, C0Method::LogBackExtrapolation);
        let mut i = Interval {
            start: 0.0,
            end: 2.0,
            ..Interval::EMPTY
        };
        interval_parameters(&[0.0, 2.0], &[5.0, 1.0], 2, 5.0, false, &mut i, &options);
        assert_eq!(i.c_start, 5.0);
        for v in [
            i.auc,
            i.cmax,
            i.tmax,
            i.cmin,
            i.ctrough,
            i.cavg,
            i.fluctuation,
            i.swing,
        ] {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn a_curve_which_is_short_or_does_not_reach_its_end_is_incomplete() {
        let options = with(AucMethod::Linear, C0Method::LogBackExtrapolation);
        let run = |t: &[f64], c: &[f64]| {
            let mut i = Interval {
                start: 0.0,
                end: 2.0,
                ..Interval::EMPTY
            };
            interval_parameters(t, c, t.len(), 5.0, true, &mut i, &options);
            i
        };
        // one point: no area
        assert!(run(&[0.0], &[5.0]).auc.is_nan());
        // a lone point at the end is no interval either
        let lone = run(&[2.0], &[1.0]);
        assert!(lone.auc.is_nan() && lone.cmax.is_nan());
        // two points which stop before the end (no value at the end): not complete
        let short = run(&[0.0, 1.0], &[5.0, 3.0]);
        assert!(short.auc.is_nan() && short.ctrough.is_nan());
        assert_eq!(short.c_start, 5.0);
        // n smaller than the slices: only the first n points are read
        let mut i = Interval {
            start: 0.0,
            end: 2.0,
            ..Interval::EMPTY
        };
        interval_parameters(
            &[0.0, 1.0, 2.0, 3.0],
            &[5.0, 3.0, 1.0, 99.0],
            3,
            5.0,
            true,
            &mut i,
            &options,
        );
        assert_eq!((i.cmax, i.cmin), (5.0, 1.0));
    }

    #[test]
    fn an_interval_which_does_not_exist_is_all_not_a_number() {
        let options = NcaOptions::default();
        let mut i = Interval::EMPTY;
        interval_parameters(&[0.0, 2.0], &[5.0, 1.0], 2, 5.0, true, &mut i, &options);
        assert!(i.c_start.is_nan() && i.auc.is_nan() && i.cmax.is_nan());
    }

    #[test]
    fn a_zero_trough_gives_the_infinite_swing_of_numpy() {
        let options = with(AucMethod::Linear, C0Method::LogBackExtrapolation);
        let mut i = Interval {
            start: 0.0,
            end: 2.0,
            ..Interval::EMPTY
        };
        interval_parameters(
            &[0.0, 1.0, 2.0],
            &[4.0, 2.0, 0.0],
            3,
            4.0,
            true,
            &mut i,
            &options,
        );
        assert_eq!((i.cmin, i.swing), (0.0, f64::INFINITY));
        // python: area 4, cavg 2, fluctuation (4 - 0) / 2
        assert_eq!((i.cavg, i.fluctuation), (2.0, 2.0));
    }

    #[test]
    fn an_all_zero_curve_has_a_not_a_number_fluctuation_and_swing() {
        // python (measured): cavg 0, fluctuation 0 / 0 = NaN, swing 0 / 0 = NaN, not infinity
        let options = with(AucMethod::Linear, C0Method::LogBackExtrapolation);
        let mut i = Interval {
            start: 0.0,
            end: 2.0,
            ..Interval::EMPTY
        };
        interval_parameters(&[0.0, 1.0, 2.0], &[0.0; 3], 3, 0.0, true, &mut i, &options);
        assert_eq!((i.auc, i.cavg, i.cmax, i.cmin), (0.0, 0.0, 0.0, 0.0));
        assert!(i.fluctuation.is_nan() && i.swing.is_nan());
    }

    #[test]
    fn the_dose_of_an_interval_follows_its_dose_and_only_an_existing_interval_has_one() {
        let mut v = Vec::new();
        interval_bounds(&[0.0, 2.0, f64::NAN], 2, Some(2.0), &mut v);
        v.push(Interval::EMPTY);
        // amounts for the empty intervals too: they must not be written
        fill_doses(&mut v, &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!((v[0].dose, v[1].dose), (10.0, 20.0));
        assert!(v[2].dose.is_nan() && v[3].dose.is_nan());
        // a shorter amount list leaves the rest not a number
        fill_doses(&mut v, &[7.0]);
        assert_eq!(v[0].dose, 7.0);
        assert!(v[1].dose.is_nan());
    }

    #[test]
    fn the_interval_variables_are_the_names_python_emits_in_its_order() {
        // copied from `interval_variables` and `CONCENTRATION_VARIABLES` of intervals.py
        let names: Vec<&str> = INTERVAL_VARIABLES.iter().map(|v| v.name).collect();
        assert_eq!(
            names,
            [
                "interval_start",
                "interval_end",
                "interval_dose",
                "interval_n_points",
                "interval_auc",
                "interval_cmax",
                "interval_tmax",
                "interval_cmin",
                "interval_ctrough",
                "interval_c_start",
                "interval_cavg",
                "interval_fluctuation",
                "interval_swing",
            ]
        );
        let units: Vec<&str> = INTERVAL_VARIABLES.iter().map(|v| v.unit).collect();
        assert_eq!(
            units,
            [
                "{time}",
                "{time}",
                "{dose}",
                "dimensionless",
                "({unit}) * ({time})",
                "{unit}",
                "{time}",
                "{unit}",
                "{unit}",
                "{unit}",
                "{unit}",
                "dimensionless",
                "dimensionless",
            ]
        );
    }

    #[test]
    fn interval_dose_is_emitted_only_for_a_batch_with_amounts() {
        use crate::assemble::{RowResult, result_of};
        use crate::parameters::tests::{many, regimen};

        let times = [0.0, 1.0, 2.0, 3.0];
        let plain = many(&[&times], &[&[4.0, 3.0, 2.0, 1.0]]);
        let dosed = regimen(plain.clone(), Route::Oral, 10.0, &[0.0, 2.0]);
        let row = |dose: f64| {
            let mut intervals = Vec::new();
            interval_bounds(&[0.0, 2.0], 2, Some(2.0), &mut intervals);
            fill_doses(&mut intervals, &[dose, dose]);
            RowResult {
                route: Some(Route::Oral),
                dose_amount: dose,
                parameters: crate::parameters::RowParameters::EMPTY,
                intervals,
            }
        };
        let emitted = |batch: &Batch| -> Vec<String> {
            let rows = [row(10.0)];
            let result = result_of(batch, &NcaOptions::default(), &rows, &[]).unwrap();
            result
                .point_variables
                .iter()
                .map(|v| v.name.clone())
                .collect()
        };
        let with_dose = emitted(&dosed);
        let without = emitted(&Batch {
            max_doses: 0,
            ..plain
        });
        assert!(with_dose.contains(&"interval_dose".to_string()));
        assert!(!without.contains(&"interval_dose".to_string()));
        // everything else is in both, in one order
        let others: Vec<_> = with_dose.iter().filter(|n| *n != "interval_dose").collect();
        assert_eq!(others, without.iter().collect::<Vec<_>>());
        assert!(without.contains(&"interval_swing".to_string()));
    }

    // ---- the trough regression of a bolus interval, through compute_row ----

    /// The intervals and the flags of one row, analysed by `compute_row` over the protocol `doses`
    /// (absolute times, the frame of the host), as the python reference of the same name does.
    fn row_of(
        times: &[f64],
        values: &[f64],
        doses: &[f64],
        route: Route,
        method: AucMethod,
        lloq: Option<f64>,
    ) -> (Vec<Interval>, NcaFlag) {
        use crate::parameters::tests::linear;
        use crate::parameters::{RowInput, Scratch, compute_row};

        let input = RowInput {
            times,
            values,
            dose: crate::dose::ReferenceDose {
                amount: 10.0,
                time: *doses.last().unwrap(),
                duration: Some(if route == Route::IvInfusion {
                    1.0
                } else {
                    f64::NAN
                }),
            },
            route: Some(route),
            has_dose: true,
            lloq,
            window: None,
            manual: None,
            single_dose: false,
            dose_times: doses,
        };
        let tau = doses[1] - doses[0];
        let options = NcaOptions {
            auc_method: method,
            tau: Some(tau),
            ..linear()
        };
        let mut intervals = Vec::new();
        let row = compute_row(
            &input,
            &options,
            &mut Scratch::new(times.len()),
            None,
            Some(&mut intervals),
        );
        (intervals, row.flags)
    }

    use pkpdutils_abi::NcaFlag;

    const T: [f64; 9] = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 13.0, 16.0, 24.0];
    // the trough of the first interval of `T` with a post-dose 9.0 at 12, from python
    const TROUGH: f64 = 0.7022990702519408;

    fn bolus(values: &[f64]) -> (Vec<Interval>, NcaFlag) {
        row_of(
            &T,
            values,
            &[0.0, 12.0],
            Route::IvBolus,
            AucMethod::Linear,
            None,
        )
    }

    #[test]
    fn a_post_dose_sample_at_a_bolus_interval_end_is_replaced_by_the_regression() {
        // all four conditions hold: IvBolus, the interval ends at a dose, a sample sits at that end
        // and exceeds the last sample inside (9.0 > 1.7). The trough is the regression of the last
        // THREE of the five samples inside (a fit of all five gives another value), and every value
        // below is from `compute_intervals` of the python source
        let (i, flags) = bolus(&[10.0, 8.0, 6.4, 4.1, 1.7, 9.0, 7.0, 5.0, 2.0]);
        assert_eq!(i[0].ctrough, TROUGH);
        assert_eq!(
            i[0].cmin, TROUGH,
            "the trough is the end of the curve, so also its minimum"
        );
        assert_eq!(i[0].auc, 43.10459814050388);
        assert_eq!(i[0].cavg, 3.59204984504199);
        assert_eq!(i[0].c_start, 10.0);
        assert!(flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
        // the last interval has no dose after it: its trough is the sample at the end, observed
        assert_eq!(i[1].ctrough, 2.0);
        assert_eq!(i[1].auc, 54.0);
        assert_eq!(
            i[1].c_start, 9.0,
            "the post-dose sample starts the next interval"
        );
        // and the python log rule agrees on the trough, the area follows the rule
        let (log, flags) = row_of(
            &T,
            &[10.0, 8.0, 6.4, 4.1, 1.7, 9.0, 7.0, 5.0, 2.0],
            &[0.0, 12.0],
            Route::IvBolus,
            AucMethod::Log,
            None,
        );
        assert_eq!(log[0].ctrough, TROUGH);
        assert_eq!(log[0].auc, 41.881977449437684);
        assert!(flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
    }

    #[test]
    fn the_substituted_trough_is_not_counted_as_a_point_of_the_interval() {
        let values = [10.0, 8.0, 6.4, 4.1, 1.7, 9.0, 7.0, 5.0, 2.0];
        let (i, _) = bolus(&values);
        assert_eq!((i[0].n_points, i[1].n_points), (5.0, 4.0));
        // the same samples as an oral dose count the one at the end
        let (oral, _) = row_of(
            &T,
            &values,
            &[0.0, 12.0],
            Route::Oral,
            AucMethod::Linear,
            None,
        );
        assert_eq!((oral[0].n_points, oral[1].n_points), (6.0, 4.0));
    }

    #[test]
    fn an_observed_trough_at_the_dose_time_is_used_as_it_is() {
        // 1.2 < 1.7: the sample at 12 is the pre-dose trough. No regression, no flag
        let (i, flags) = bolus(&[10.0, 8.0, 6.4, 4.1, 1.7, 1.2, 7.0, 5.0, 2.0]);
        assert_eq!(i[0].ctrough, 1.2);
        assert_eq!(i[0].auc, 44.099999999999994);
        assert_eq!(i[0].n_points, 6.0);
        assert!(!flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
        // equal to the last sample inside is no post-dose sample either: the comparison is strict
        let (i, flags) = bolus(&[10.0, 8.0, 6.4, 4.1, 1.7, 1.7, 7.0, 5.0, 2.0]);
        assert_eq!((i[0].ctrough, i[0].n_points), (1.7, 6.0));
        assert!(!flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
    }

    #[test]
    fn only_a_bolus_substitutes_its_trough() {
        // the data of the first test: 9.0 at the next dose is above the last sample inside
        let values = [10.0, 8.0, 6.4, 4.1, 1.7, 9.0, 7.0, 5.0, 2.0];
        for route in [Route::Oral, Route::IvInfusion] {
            let (i, flags) = row_of(&T, &values, &[0.0, 12.0], route, AucMethod::Linear, None);
            assert_eq!(i[0].ctrough, 9.0, "{route:?}");
            assert_eq!(i[0].auc, 59.699999999999996, "{route:?}");
            assert!(!flags.contains(NcaFlag::EXTRAPOLATED_TROUGH), "{route:?}");
        }
    }

    #[test]
    fn the_last_interval_of_a_bolus_has_no_dose_after_it_and_keeps_its_sample() {
        // the sample at 24 (20.0) is above the last one inside (5.0), but no dose follows
        let (i, flags) = bolus(&[10.0, 8.0, 6.4, 4.1, 1.7, 9.0, 7.0, 5.0, 20.0]);
        assert_eq!((i[1].ctrough, i[1].auc), (20.0, 126.0));
        assert_eq!(i[1].n_points, 4.0);
        // only the first interval was regressed
        assert!(flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
        assert_eq!(i[0].ctrough, TROUGH);
    }

    #[test]
    fn a_failed_regression_overwrites_the_interpolated_end_with_nan() {
        // post-dose, but the regression has nothing to fit: the end value is NaN, NOT the
        // interpolated 9.0. The end bound is then not inserted, the interval is incomplete and
        // the flag (which needs a finite trough) stays down. python: auc, cmin, ctrough nan,
        // c_start 10, n_points 3
        let times = [0.0, 1.0, 2.0, 12.0, 13.0, 24.0];
        let (i, flags) = row_of(
            &times,
            &[10.0, 0.0, 0.0, 9.0, 7.0, 2.0],
            &[0.0, 12.0],
            Route::IvBolus,
            AucMethod::Linear,
            None,
        );
        assert!(i[0].ctrough.is_nan() && i[0].auc.is_nan() && i[0].cmin.is_nan());
        assert!(i[0].cmax.is_nan() && i[0].cavg.is_nan());
        assert_eq!((i[0].c_start, i[0].n_points), (10.0, 3.0));
        assert!(
            !flags.contains(NcaFlag::EXTRAPOLATED_TROUGH),
            "a NaN trough raises no flag"
        );
        // the next interval does not care
        assert_eq!((i[1].ctrough, i[1].auc), (2.0, 57.5));

        // a single sample inside is too few as well (python: nan, c_start 5, n_points 1)
        let (i, flags) = row_of(
            &[0.0, 12.0, 13.0, 24.0],
            &[5.0, 9.0, 7.0, 2.0],
            &[0.0, 12.0],
            Route::IvBolus,
            AucMethod::Linear,
            None,
        );
        assert!(i[0].ctrough.is_nan() && i[0].auc.is_nan());
        assert_eq!((i[0].c_start, i[0].n_points), (5.0, 1.0));
        assert!(!flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
    }

    #[test]
    fn a_sample_at_the_end_is_matched_within_a_relative_tolerance() {
        let values = [10.0, 8.0, 6.4, 4.1, 1.7, 9.0, 7.0, 5.0, 2.0];
        let at = |end: f64| {
            let mut times = T;
            times[5] = end;
            row_of(
                &times,
                &values,
                &[0.0, 12.0],
                Route::IvBolus,
                AucMethod::Linear,
                None,
            )
        };
        // 1e-10 away is within 1e-9 * 12; the sample counts as the one at the end
        let (i, flags) = at(12.0 + 1e-10);
        assert_eq!((i[0].ctrough, i[0].n_points), (TROUGH, 5.0));
        assert!(flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
        // 1e-3 away is not: the end is interpolated and nothing is regressed
        let (i, flags) = at(12.0 + 1e-3);
        assert_eq!(i[0].ctrough, 8.998175456135966);
        assert!(!flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
    }

    #[test]
    fn the_end_tolerance_scales_with_the_time_and_is_floored_at_one() {
        // `1e-9 * max(1, |end|)`, values from python. Late: the end is 1e5 + 12, so the tolerance
        // is 1e-4 and a sample 5e-5 (5e-10 relative) away is the one at the end; a flat 1e-9
        // would miss it. 5e-4 is out of both
        let t0 = 1e5;
        let v = [10.0, 8.0, 6.4, 4.1, 1.7, 9.0, 7.0, 5.0, 2.0];
        let late = |offset: f64| {
            let times: Vec<f64> = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0 + offset, 13.0, 16.0, 24.0]
                .iter()
                .map(|x| t0 + x)
                .collect();
            row_of(
                &times,
                &v,
                &[t0, t0 + 12.0],
                Route::IvBolus,
                AucMethod::Linear,
                None,
            )
        };
        let (i, flags) = late(5e-5);
        assert_eq!(i[0].ctrough, 0.7022990702526931);
        assert!(flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
        let (i, flags) = late(5e-4);
        assert_eq!(i[0].ctrough, 8.999087614058014);
        assert!(!flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
        // Early: the end is 0.5, a tolerance of 1e-9 from the floor (5e-10 without it). A sample
        // 8e-10 away is at the end; 3e-9 away is not
        let small = |offset: f64| {
            let times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5 + offset, 0.6, 0.8, 1.0];
            row_of(
                &times,
                &v,
                &[0.0, 0.5],
                Route::IvBolus,
                AucMethod::Linear,
                None,
            )
        };
        let (i, flags) = small(8e-10);
        assert_eq!(i[0].ctrough, 0.9420483605210609);
        assert!(flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
        let (i, flags) = small(3e-9);
        assert_eq!(i[0].ctrough, 8.999999781000005);
        assert!(!flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
    }

    #[test]
    fn the_largest_of_several_samples_at_the_end_decides() {
        // two samples within the tolerance of 12, the higher (9.0) first. The largest is above
        // the last one inside (1.7), so the interval is post-dose; the last of them (1.0) is
        // not. Python: ctrough 0.7022990702519408, flagged, n_points 5
        let times = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 12.0 + 5e-10, 16.0, 24.0];
        let (i, flags) = row_of(
            &times,
            &[10.0, 8.0, 6.4, 4.1, 1.7, 9.0, 1.0, 5.0, 2.0],
            &[0.0, 12.0],
            Route::IvBolus,
            AucMethod::Linear,
            None,
        );
        assert_eq!((i[0].ctrough, i[0].n_points), (TROUGH, 5.0));
        assert!(flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
        // the same with the lower one first: the maximum still is 9.0
        let (i, flags) = row_of(
            &times,
            &[10.0, 8.0, 6.4, 4.1, 1.7, 1.0, 9.0, 5.0, 2.0],
            &[0.0, 12.0],
            Route::IvBolus,
            AucMethod::Linear,
            None,
        );
        assert_eq!((i[0].ctrough, i[0].n_points), (TROUGH, 5.0));
        assert!(flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
    }

    #[test]
    fn the_regression_agrees_with_the_centred_python_fit_on_late_sample_times() {
        // t ~ 1e5 with a 0.2 spread: the centred reference (python, Task 1) is 5.248632296621781,
        // the textbook normal equations on the absolute times lose about three digits here
        let t0 = 1e5;
        let times = [t0, t0 + 0.1, t0 + 0.2, t0 + 0.3, t0 + 0.4, t0 + 0.6];
        let values = [10.0, 8.1, 6.5, 9.0, 7.0, 5.0];
        let (i, flags) = row_of(
            &times,
            &values,
            &[t0, t0 + 0.3],
            Route::IvBolus,
            AucMethod::Linear,
            None,
        );
        assert!(flags.contains(NcaFlag::EXTRAPOLATED_TROUGH));
        let reference = 5.248632296621781;
        assert!(
            (i[0].ctrough / reference - 1.0).abs() < 1e-12,
            "{}",
            i[0].ctrough
        );
        assert_eq!(i[0].ctrough, reference);
        assert_eq!(i[0].auc, 2.222431614854223);
        // the textbook form on the same samples is visibly off, so this row separates the two
        let (x, y): (Vec<f64>, Vec<f64>) = (
            times[..3].to_vec(),
            values[..3].iter().map(|v| v.ln()).collect(),
        );
        let n = 3.0;
        let (sx, sy) = (x.iter().sum::<f64>(), y.iter().sum::<f64>());
        let sxx: f64 = x.iter().map(|v| v * v).sum();
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let textbook = ((sy - slope * sx) / n + slope * (t0 + 0.3)).exp();
        assert!((textbook / reference - 1.0).abs() > 1e-6, "{textbook}");
    }

    #[test]
    fn the_regression_is_exact_for_a_mono_exponential_and_selects_its_samples() {
        let t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let c: Vec<f64> = t.iter().map(|x| 8.0 * (-0.4_f64 * x).exp()).collect();
        let all = [true; 6];
        let truth = 8.0 * (-0.4_f64 * 6.0).exp();
        let got = extrapolate_end(&t, &c, &all, 6.0, 3);
        assert!((got / truth - 1.0).abs() < 1e-12, "{got} vs {truth}");
        // the last `max_points` positive samples: perturb the first two, nothing moves
        let mut noisy = c.clone();
        noisy[0] = 100.0;
        noisy[1] = 0.001;
        assert_eq!(extrapolate_end(&t, &noisy, &all, 6.0, 3), got);
        // with more points allowed the perturbed ones count and the value changes
        assert_ne!(extrapolate_end(&t, &noisy, &all, 6.0, 5), got);
        // a sample outside the interval is not used, a zero is skipped and the next positive
        // one before it takes its place
        let mut mask = all;
        mask[5] = false;
        let mut zero = c.clone();
        zero[3] = 0.0;
        let got = extrapolate_end(&t, &zero, &mask, 6.0, 3);
        let expected = extrapolate_end(&[1.0, 2.0, 4.0], &[c[1], c[2], c[4]], &[true; 3], 6.0, 3);
        assert_eq!(got, expected);
        // fewer than two positive samples, and no spread of the times
        assert!(
            extrapolate_end(&t, &c, &[false, false, false, false, false, true], 6.0, 3).is_nan()
        );
        assert!(extrapolate_end(&t, &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], &all, 6.0, 3).is_nan());
        assert!(extrapolate_end(&[2.0, 2.0], &[3.0, 2.0], &[true; 2], 6.0, 3).is_nan());
        assert!(extrapolate_end(&[], &[], &[], 6.0, 3).is_nan());
        // times so close that the squared spread underflows to zero while the cross product does
        // not: the slope would be infinite, python's `denominator > 0` makes it NaN
        assert!(extrapolate_end(&[0.0, 1e-200], &[3.0, 2.0], &[true; 2], 6.0, 3).is_nan());
    }

    #[test]
    fn the_limit_of_quantification_changes_neither_the_trough_nor_the_flag() {
        let values = [10.0, 8.0, 6.4, 4.1, 1.7, 9.0, 7.0, 5.0, 2.0];
        let plain = bolus(&values);
        let limited = row_of(
            &T,
            &values,
            &[0.0, 12.0],
            Route::IvBolus,
            AucMethod::Linear,
            Some(3.0),
        );
        assert_eq!(
            limited.1.contains(NcaFlag::EXTRAPOLATED_TROUGH),
            plain.1.contains(NcaFlag::EXTRAPOLATED_TROUGH)
        );
        assert_eq!(limited.0[0].ctrough, plain.0[0].ctrough);
        assert_eq!(limited.0[0].auc, plain.0[0].auc);
        assert_eq!(limited.0[0].n_points, plain.0[0].n_points);
    }

    // ---- covered and the start value, derived by the caller ----

    #[test]
    fn an_interval_without_a_sample_of_its_own_is_not_complete_unless_it_is_the_last() {
        let times = [0.0, 1.0, 5.0, 24.0];
        let values = [10.0, 8.0, 5.0, 1.0];
        // doses 0, 12, 24: the interval [12, 24) has no sample inside and the one at 24 belongs to
        // the next. Its bounds are interpolated and finite, so only `covered` makes it incomplete
        // (python: auc nan, c_start 3.526..., n_points 1)
        let (i, _) = row_of(
            &times,
            &values,
            &[0.0, 12.0, 24.0],
            Route::Oral,
            AucMethod::Linear,
            None,
        );
        assert!(i[1].auc.is_nan() && i[1].ctrough.is_nan() && i[1].cmax.is_nan());
        assert_eq!((i[1].c_start, i[1].n_points), (3.526315789473684, 1.0));
        // the same samples, doses 0 and 12: the interval is the last, the sample at its end is
        // its own, and it is complete
        let (i, _) = row_of(
            &times,
            &values,
            &[0.0, 12.0],
            Route::Oral,
            AucMethod::Linear,
            None,
        );
        assert_eq!(i[1].auc, 27.157894736842103);
        assert_eq!(i[1].ctrough, 1.0);
        assert_eq!(i[1].n_points, 1.0);
        // and the first interval is covered by its own samples in both
        assert_eq!(i[0].auc, 64.84210526315789);
    }

    #[test]
    fn the_start_value_of_an_interval_is_the_interpolated_one_even_on_a_tie() {
        // a sample sits exactly at the start of [2, 4). The log rule interpolates exp(ln 3), which
        // is 3 + 1 ulp: the reported c_start is that value and the observed 3.0 is only the first
        // point of the curve (python: 3.0000000000000004, and 4.999999999999999 at the first start)
        let times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let values = [5.0, 4.0, 3.0, 2.5, 1.2, 0.6, 0.3];
        let (i, _) = row_of(
            &times,
            &values,
            &[0.0, 2.0, 4.0],
            Route::Oral,
            AucMethod::Log,
            None,
        );
        assert_eq!(i[1].c_start, 3.000_000_000_000_000_4);
        assert_ne!(i[1].c_start, 3.0);
        assert_eq!(i[0].c_start, 4.999_999_999_999_999);
        // the same value leads the curve's extremes, as the curve holds [3, 3 + ulp, 2.5, 1.2]
        assert_eq!(i[1].cmax, 3.000_000_000_000_000_4);
        assert_eq!(i[1].auc, 4.513599022698392);
    }
}
