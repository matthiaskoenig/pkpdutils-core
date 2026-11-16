//! The dose of a row, the concentration at the dose time and the point the areas start from.
//!
//! The analysis of a curve is run against one dose of its protocol: the first of them for a
//! single dose analysis and the last at steady state ([`reference_dose`]). The caller subtracts
//! the time of that dose from the times of the row, so everything here reads a time axis whose
//! origin is the dose.
//!
//! A curve whose first sample comes after that origin says nothing about the area between the
//! dose and the sample, and the routes fill the gap differently, as Phoenix WinNonlin does and
//! `src/pkpdutils/nca/nca.py` after it: an intravenous bolus starts at `c0`, the concentration
//! back-extrapolated from its first two samples ([`bolus_c0`]), an infusion and an extravascular
//! single dose start at zero, and a row without a route starts at its first sample, the area
//! before it simply not counted.
//!
//! **The inserted point reaches the areas and nothing else.** [`insert_dose_value`] writes a
//! second, wider row and leaves the packed row it reads untouched, because `cmax`, `tmax`,
//! `clast`, `tlast` and above all the terminal regression are computed from the row as it was
//! measured (`areas.md` section 4 and trap T4). An estimate which entered the regression would
//! make `lambda_z` a function of itself: `c0` is extrapolated from the first two samples with a
//! slope of its own, and the regression would then fit that slope as well as the data.

use pkpdutils_abi::{C0Method, Route};

/// The one dose of the protocol of a row which the analysis is run against.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReferenceDose {
    /// The amount, `NaN` for a row without a dose.
    pub amount: f64,
    /// The time, `NaN` for a row without a dose.
    pub time: f64,
    /// The infusion duration, `None` when the protocol carries none at all.
    pub duration: Option<f64>,
}

/// The dose of a protocol the parameters of its row are computed against.
///
/// `last` picks the last dose instead of the first, which is what the steady state analysis
/// reads; the single dose analysis takes the first.
///
/// A dose counts when its amount and its time are both finite, and a protocol without one gives
/// `NaN` for all three: a subject whose dose records are missing is still analysed, its
/// dose-independent parameters and all, and only the parameters which divide by the dose come
/// out `NaN`. The duration is `None` only where the protocol carries no durations at all, which
/// is every route but an infusion, and `Some(NaN)` where it carries one which is not a number:
/// the two are different downstream, since a route without a duration excludes no sample from
/// the terminal regression while a `NaN` duration excludes the samples at or before `t = 0`.
///
/// The count is the number of dose columns which are a dose, not the position of the last one,
/// so a protocol with a hole in it - a `NaN` amount with doses behind it, which the batch of a
/// host can carry although `Timecourses` packs its doses to the front - picks a column which is
/// not its last dose. That is what the python kernel picks, mirrored rather than corrected.
///
/// The three arrays are the dose columns of the row as the batch carries them, `NaN` padding and
/// all: the count is taken over every column, which is what makes a protocol with a hole in it
/// read as it does. `durations` is `Some` for a batch which carries durations at all, which is
/// every batch with doses, and `None` for one without.
///
/// Mirrors `reference_dose` of `src/pkpdutils/nca/nca.py`, for one row.
pub(crate) fn reference_dose(
    amounts: &[f64],
    times: &[f64],
    durations: Option<&[f64]>,
    last: bool,
) -> ReferenceDose {
    let columns = amounts.len().min(times.len());
    let count = (0..columns)
        .filter(|&j| amounts[j].is_finite() && times[j].is_finite())
        .count();
    let index = if last { count.saturating_sub(1) } else { 0 };
    let pick = |values: &[f64]| {
        if count == 0 {
            f64::NAN
        } else {
            values.get(index).copied().unwrap_or(f64::NAN)
        }
    };
    ReferenceDose {
        amount: pick(amounts),
        time: pick(times),
        duration: durations.map(pick),
    }
}

/// The concentration at the dose time of an intravenous bolus, and the rule which produced it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct C0 {
    /// The estimate, `NaN` where there is none.
    pub value: f64,
    /// The rule, one of the `C0Method::CODE_*` codes, reported as `c0_method`.
    pub method: i32,
}

/// Estimate the concentration at the dose time of an intravenous bolus.
///
/// With `C0Method::LogBackExtrapolation` the first two packed points `(t1, c1)` and `(t2, c2)`
/// are extrapolated back to the dose,
///
/// ```text
/// c0 = exp(ln c1 - (ln c2 - ln c1) / (t2 - t1) * t1)
/// ```
///
/// the value at `t = 0` of the line through them in the logarithm (Gabrielsson & Weiner 2016,
/// ch. 2.8). The expression keeps the two logarithms apart, where the logarithmic trapezoid of
/// [`crate::auc`] takes the logarithm of one ratio; the two differ in the last places and
/// `areas.md` trap T7 asks for each form as it is written.
///
/// The back-extrapolation needs two samples which decline, so it is used only where the row
/// holds a point, both values are positive, the second is below the first and the second time is
/// after the first. Everything else falls back to the first observed value, which is the
/// documented fallback chain of Phoenix WinNonlin. `C0Method::FirstValue` takes that value
/// straight away and `C0Method::None` estimates nothing.
///
/// `t` and `c` are the packed points of the row and `n_valid` how many of them are points; a row
/// of none has no estimate and the code `C0Method::CODE_NONE`. The estimate is computed from the
/// times as they are, so a row whose first sample lies at or before the dose gets one as well
/// although no point is inserted for it (`areas.md` trap T3): at `t1 = 0` the formula returns
/// `c1` and at a negative `t1` it extrapolates forward.
///
/// Mirrors `bolus_c0` of `src/pkpdutils/nca/nca.py`.
pub(crate) fn bolus_c0(t: &[f64], c: &[f64], n_valid: usize, method: C0Method) -> C0 {
    if method == C0Method::None {
        return C0 {
            value: f64::NAN,
            method: C0Method::CODE_NONE,
        };
    }
    let has_point = n_valid >= 1;
    let t1 = t.first().copied().unwrap_or(f64::NAN);
    let c1 = c.first().copied().unwrap_or(f64::NAN);
    let first = C0 {
        value: if has_point { c1 } else { f64::NAN },
        method: if has_point {
            C0Method::CODE_FIRST_VALUE
        } else {
            C0Method::CODE_NONE
        },
    };
    if method != C0Method::LogBackExtrapolation {
        return first;
    }
    let (t2, c2) = if n_valid > 1 {
        (
            t.get(1).copied().unwrap_or(f64::NAN),
            c.get(1).copied().unwrap_or(f64::NAN),
        )
    } else {
        (f64::NAN, f64::NAN)
    };
    // every comparison is false for a NaN, so a row of fewer than two points, which carries the
    // NaN padding here, is not usable and falls back to the first value
    let usable = has_point && c1 > 0.0 && c2 > 0.0 && c2 < c1 && t2 > t1;
    if usable {
        C0 {
            value: (c1.ln() - (c2.ln() - c1.ln()) / (t2 - t1) * t1).exp(),
            method: C0Method::CODE_BACK_EXTRAPOLATION,
        }
    } else {
        first
    }
}

/// The value the route of a row puts at the dose time, `None` for a route which puts none.
///
/// An intravenous bolus puts `c0` there, whether the analysis is of a single dose or not, and a
/// `NaN` estimate - the row of `C0Method::None` - is still that route: nothing is inserted, but
/// the row the areas run on is repacked all the same, which [`insert_dose_value`] mirrors. An
/// infusion and an extravascular dose put a zero there, and only for a single dose: a steady
/// state interval starts at its trough and inserts nothing. A row without a route puts nothing
/// at the dose time at all, so the area before its first sample is not counted.
///
/// Mirrors the route branch of `compute_parameters`, `src/pkpdutils/nca/nca.py`.
pub(crate) fn dose_value(route: Option<Route>, single_dose: bool, c0: f64) -> Option<f64> {
    match route {
        Some(Route::IvBolus) => Some(c0),
        Some(Route::IvInfusion | Route::Oral) if single_dose => Some(0.0),
        _ => None,
    }
}

/// The point the analysis added at the dose time, and the row the areas run on.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Inserted {
    /// Whether a point was added at the dose time.
    pub inserted: bool,
    /// The points of the row the areas run on.
    pub n: usize,
}

/// Write the row the areas run on, the value at the dose time in its place.
///
/// `t` and `c` are the packed points of the row and `n_valid` how many of them are points;
/// `value` is what [`dose_value`] made of the route, `None` for a route which inserts nothing.
/// The row is written into `t_area` and `c_area`, which have to hold one point more than the
/// row; a shorter pair is cut to what fits, as [`crate::pack::pack_valid`] cuts, and the slots
/// behind the points written are filled with `NaN`.
///
/// A point goes in only where the row has two points at all, its first sample lies strictly
/// after the dose, since a row which starts at or before the dose needs no estimate of what
/// happened in between, and the buffers have room for it. A `NaN` value - a bolus whose `c0` is
/// not a number - inserts nothing either. [`Inserted::inserted`] says whether a point really
/// went in, and the caller reads it rather than the value, because the fractions which are
/// back-extrapolated are `0.0` and not `NaN` for a row which had nothing inserted (`areas.md`
/// section 3.8 and trap T3).
///
/// The python `insert_point` concatenates the new point to the row and stable-sorts the whole of
/// it by time, which reorders a row that arrived out of time order although `pack_valid` left it
/// alone, and reorders it whenever the route inserts at all, even where the value is `NaN` and
/// nothing goes in. Both are mirrored: a row in time order is written through with the new point
/// in the one place the sort would put it, which is before every later point and behind an
/// existing point of the same time, and only a row out of order is really sorted.
///
/// Mirrors `insert_point` of `src/pkpdutils/nca/auc.py` as `compute_parameters` calls it.
pub(crate) fn insert_dose_value(
    t: &[f64],
    c: &[f64],
    n_valid: usize,
    value: Option<f64>,
    t_area: &mut [f64],
    c_area: &mut [f64],
) -> Inserted {
    let n_valid = n_valid.min(t.len()).min(c.len());
    let width = t_area.len().min(c_area.len());
    let first_time = t.first().copied().unwrap_or(f64::NAN);
    // `n_valid >= 2` is `has_data` and a NaN first time fails the comparison; `n_valid < width`
    // is the room for one point more than the row, which the caller leaves and python always
    // has. Without it the row alone is written, so `inserted` never claims a point the area row
    // does not carry.
    let inserted =
        value.is_some_and(f64::is_finite) && n_valid >= 2 && first_time > 0.0 && n_valid < width;
    let mut n = 0usize;
    while n < n_valid && n < width {
        t_area[n] = t[n];
        c_area[n] = c[n];
        n += 1;
    }
    let ordered = t_area[..n].windows(2).all(|pair| pair[0] <= pair[1]);
    if inserted && let Some(value) = value {
        // the stable position of a point at the dose time: behind every point whose time is not
        // later, which is the front of a row in order. A row out of order is appended to and
        // sorted below, which is what python does to every row it inserts into.
        let position = if ordered {
            (0..n).find(|&j| t_area[j] > 0.0).unwrap_or(n)
        } else {
            n
        };
        // `n` is `n_valid`, which is below `width`, so the row grows into room it has
        n += 1;
        for j in (position + 1..n).rev() {
            t_area[j] = t_area[j - 1];
            c_area[j] = c_area[j - 1];
        }
        t_area[position] = 0.0;
        c_area[position] = value;
    }
    if value.is_some() && !ordered {
        sort_by_time(&mut t_area[..n], &mut c_area[..n]);
    }
    for slot in t_area.iter_mut().skip(n) {
        *slot = f64::NAN;
    }
    for slot in c_area.iter_mut().skip(n) {
        *slot = f64::NAN;
    }
    Inserted { inserted, n }
}

/// Sort the points of a row by time, keeping the order of two points of the same time.
///
/// The row holds the points of a packed row and at most one inserted value, so every time in it
/// is finite and the comparison is total; a `NaN` which cannot be there would compare equal and
/// stay where it is, where the python sort moves it to the end of the row, out of the points.
fn sort_by_time(t: &mut [f64], c: &mut [f64]) {
    let mut points: Vec<(f64, f64)> = t.iter().copied().zip(c.iter().copied()).collect();
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for (slot, point) in t.iter_mut().zip(&points) {
        *slot = point.0;
    }
    for (slot, point) in c.iter_mut().zip(&points) {
        *slot = point.1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use pkpdutils_abi::{AucMethod, NcaFlag, TerminalMethod, TerminalPhase};

    use crate::auc::trapezoid;
    use crate::terminal::terminal_fit;

    /// A tolerance of a last place or two, four orders tighter than the 1e-8 of the
    /// benchmark gate: the expressions and their order are reproduced exactly, only `ln`
    /// and `exp` may round differently in two libms.
    #[track_caller]
    fn close(actual: f64, expected: f64) {
        let tolerance = 1e-12 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} is not {expected}"
        );
    }

    /// The dose columns of a row: the amounts, the times and the durations of the batch.
    type Protocol<'a> = (&'a [f64], &'a [f64], Option<&'a [f64]>);

    /// The reference dose of one protocol.
    fn picked(protocol: Protocol<'_>, last: bool) -> ReferenceDose {
        reference_dose(protocol.0, protocol.1, protocol.2, last)
    }

    /// The area row of one insertion, with the one column of room a caller leaves.
    fn area_row(
        t: &[f64],
        c: &[f64],
        n_valid: usize,
        value: Option<f64>,
    ) -> (Vec<f64>, Vec<f64>, Inserted) {
        let mut t_area = vec![0.0; t.len() + 1];
        let mut c_area = vec![0.0; c.len() + 1];
        let inserted = insert_dose_value(t, c, n_valid, value, &mut t_area, &mut c_area);
        (t_area, c_area, inserted)
    }

    #[test]
    fn the_reference_dose_is_the_first_of_the_protocol_and_the_last_at_steady_state() {
        let protocol: Protocol<'_> = (&[100.0, 50.0, 25.0], &[0.0, 12.0, 24.0], None);
        let first = picked(protocol, false);
        assert_eq!((first.amount, first.time), (100.0, 0.0));
        assert_eq!(first.duration, None, "no durations without an infusion");

        let last = picked(protocol, true);
        assert_eq!((last.amount, last.time), (25.0, 24.0));

        // an infusion carries a duration per dose, which travels with the dose picked
        let infusion: Protocol<'_> = (&[10.0, 10.0], &[0.0, 8.0], Some(&[0.25, 0.5]));
        assert_eq!(picked(infusion, false).duration, Some(0.25));
        assert_eq!(picked(infusion, true).duration, Some(0.5));
    }

    #[test]
    fn a_row_without_a_dose_has_a_reference_dose_of_nan() {
        // a dose counts only where its amount and its time are both finite, so a subject
        // whose dose records are missing has no reference dose at all and the parameters
        // which divide by it come out NaN
        let missing: [Protocol<'_>; 4] = [
            (&[], &[], None),
            (&[f64::NAN], &[0.0], None),
            (&[100.0], &[f64::NAN], None),
            (&[100.0], &[f64::INFINITY], None),
        ];
        for protocol in missing {
            for last in [false, true] {
                let reference = picked(protocol, last);
                assert!(reference.amount.is_nan(), "{protocol:?} last {last}");
                assert!(reference.time.is_nan(), "{protocol:?} last {last}");
            }
        }
        // the durations stay an array, with NaN in it, where the protocol has one
        let duration = picked((&[], &[], Some(&[])), false).duration;
        assert_eq!(duration.map(f64::is_nan), Some(true));
    }

    #[test]
    fn a_protocol_with_a_hole_in_it_picks_the_column_the_python_kernel_picks() {
        // An upstream oddity, mirrored rather than corrected and raised with the
        // maintainer: the last dose is picked by the NUMBER of doses of the protocol and
        // not by the position of the last one, so a protocol whose doses are not packed to
        // the front - which `Timecourses` never builds and the batch of a host can carry -
        // picks a column which is no dose at all. Two of these three columns are a dose,
        // so the steady state analysis reads column 1: a NaN amount at t = 12.
        let holed: Protocol<'_> = (&[100.0, f64::NAN, 25.0], &[0.0, 12.0, 24.0], None);
        let last = picked(holed, true);
        assert!(last.amount.is_nan(), "not the 25.0 of the last column");
        assert_eq!(last.time, 12.0);

        // the first dose is the first column either way
        let first = picked(holed, false);
        assert_eq!((first.amount, first.time), (100.0, 0.0));
    }

    #[test]
    fn a_bolus_inserts_the_back_extrapolated_value_at_the_dose_time() {
        // areas.md section 4: `c0 = exp(ln c1 - (ln c2 - ln c1) / (t2 - t1) * t1)`, the value
        // at t = 0 of the line through the first two points in the logarithm. With
        // t1 = t2 - t1 = 1 the exponent is `2 ln c1 - ln c2`, so c0 = c1 * c1 / c2 = 64 / 4.
        let t = [1.0, 2.0, 4.0];
        let c = [8.0, 4.0, 1.0];
        let c0 = bolus_c0(&t, &c, 3, C0Method::LogBackExtrapolation);
        close(c0.value, 16.0);
        assert_eq!(c0.method, C0Method::CODE_BACK_EXTRAPOLATION);

        // areas.md trap T7: the expression keeps `ln(c1)` and `ln(c2)` apart, where the
        // logarithmic trapezoid takes the logarithm of one ratio, and the two are not the
        // same double. The python kernel gives 15.999999999999991 for this row and the
        // ratio form gives 15.999999999999998, which is the only thing that holds the two
        // forms apart: they differ by a last place or two and the benchmark gate compares
        // at a relative 1e-8.
        //
        // Neither `f64::ln` nor `f64::exp` is correctly rounded or specified bit for bit,
        // so the exact doubles hold where the standard library calls the same glibc as
        // CPython does. A build on macOS, on Windows or against musl may round the last
        // place differently with the expression right, so the pin is taken there only.
        // The gate goes away the day the two are correctly rounded, or the day the kernel
        // carries a logarithm of its own.
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        {
            assert_eq!(c0.value, 15.999_999_999_999_991);
            let ratio_form = (c[0].ln() + (c[0] / c[1]).ln() / (t[1] - t[0]) * t[0]).exp();
            assert_ne!(c0.value, ratio_form, "the ratio form is a different double");
        }

        let (t_area, c_area, inserted) =
            area_row(&t, &c, 3, dose_value(Some(Route::IvBolus), true, c0.value));
        assert!(inserted.inserted);
        assert_eq!(inserted.n, 4);
        assert_eq!(&t_area[..4], &[0.0, 1.0, 2.0, 4.0]);
        close(c_area[0], 16.0);
        assert_eq!(&c_area[1..4], &[8.0, 4.0, 1.0]);
    }

    #[test]
    fn the_back_extrapolation_falls_back_to_the_first_observed_value() {
        // areas.md section 4 and section 8: the Phoenix fallback chain. Two samples which
        // do not decline, a zero among the first two, a second time which is not later and
        // a row of one point all report the first value instead, with its own rule code.
        let first_value = [
            ("rising", [1.0, 2.0], [4.0, 8.0], 2, 4.0),
            ("equal", [1.0, 2.0], [4.0, 4.0], 2, 4.0),
            ("the first value is zero", [1.0, 2.0], [0.0, 8.0], 2, 0.0),
            ("the second value is zero", [1.0, 2.0], [8.0, 0.0], 2, 8.0),
            ("the second time is earlier", [2.0, 1.0], [8.0, 4.0], 2, 8.0),
            ("one point", [1.0, f64::NAN], [8.0, f64::NAN], 1, 8.0),
        ];
        for (name, t, c, n_valid, expected) in first_value {
            let c0 = bolus_c0(&t, &c, n_valid, C0Method::LogBackExtrapolation);
            assert_eq!(c0.value, expected, "{name}");
            assert_eq!(c0.method, C0Method::CODE_FIRST_VALUE, "{name}");
        }

        // a row of no point at all has no estimate and no rule
        let empty = bolus_c0(
            &[f64::NAN, f64::NAN],
            &[f64::NAN, f64::NAN],
            0,
            C0Method::LogBackExtrapolation,
        );
        assert!(empty.value.is_nan());
        assert_eq!(empty.method, C0Method::CODE_NONE);

        // and the two other methods: the first value without an extrapolation, and none
        let t = [1.0, 2.0];
        let c = [8.0, 4.0];
        let straight = bolus_c0(&t, &c, 2, C0Method::FirstValue);
        assert_eq!(straight.value, 8.0);
        assert_eq!(straight.method, C0Method::CODE_FIRST_VALUE);
        let none = bolus_c0(&t, &c, 2, C0Method::None);
        assert!(none.value.is_nan());
        assert_eq!(none.method, C0Method::CODE_NONE);
    }

    #[test]
    fn the_estimate_is_reported_for_a_row_which_starts_at_or_before_the_dose() {
        // areas.md trap T3: `c0` is estimated from the times as they are, although nothing
        // is inserted for such a row. At t1 = 0 the formula returns c1 (up to the rounding
        // of exp(ln c1)), and at a negative t1 it extrapolates forward: the line through
        // (-1, ln 8) and (1, ln 4) is at t = 0 the geometric mean, sqrt(32) = 8 / sqrt(2).
        let at_the_dose = bolus_c0(&[0.0, 2.0], &[8.0, 4.0], 2, C0Method::LogBackExtrapolation);
        close(at_the_dose.value, 8.0);
        assert_eq!(at_the_dose.method, C0Method::CODE_BACK_EXTRAPOLATION);

        let before = bolus_c0(&[-1.0, 1.0], &[8.0, 4.0], 2, C0Method::LogBackExtrapolation);
        close(before.value, 32.0_f64.sqrt());
        assert_eq!(before.method, C0Method::CODE_BACK_EXTRAPOLATION);
    }

    #[test]
    fn an_infusion_and_an_extravascular_dose_insert_a_zero() {
        // areas.md section 4: an infusion and an extravascular dose start at the dose with
        // a zero, and only for a single dose; a steady state interval starts at its trough.
        let t = [1.0, 2.0, 4.0];
        let c = [8.0, 4.0, 1.0];
        for route in [Route::IvInfusion, Route::Oral] {
            let (t_area, c_area, inserted) =
                area_row(&t, &c, 3, dose_value(Some(route), true, f64::NAN));
            assert!(inserted.inserted, "{route:?}");
            assert_eq!(inserted.n, 4);
            assert_eq!(&t_area[..4], &[0.0, 1.0, 2.0, 4.0], "{route:?}");
            assert_eq!(&c_area[..4], &[0.0, 8.0, 4.0, 1.0], "{route:?}");

            // the same row at steady state: nothing is inserted
            let (t_area, c_area, inserted) =
                area_row(&t, &c, 3, dose_value(Some(route), false, f64::NAN));
            assert!(!inserted.inserted, "{route:?} at steady state");
            assert_eq!(inserted.n, 3);
            assert_eq!(&t_area[..3], &t);
            assert_eq!(&c_area[..3], &c);
            assert!(
                t_area[3].is_nan() && c_area[3].is_nan(),
                "the room the caller left is padded with NaN"
            );
        }

        // a row without a route inserts nothing either, whatever its first sample is:
        // the area before that sample is simply not counted
        let (t_area, _, inserted) = area_row(&t, &c, 3, dose_value(None, true, 16.0));
        assert!(!inserted.inserted);
        assert_eq!(inserted.n, 3);
        assert_eq!(&t_area[..3], &t);
    }

    #[test]
    fn nothing_is_inserted_when_the_first_sample_is_at_or_before_the_dose() {
        // areas.md section 4 and trap T2: a point goes in only where the first sample lies
        // strictly after the dose. A sample at the dose is the value at the dose already,
        // and a pre-dose sample is kept as it is, negative time and all.
        let bolus = dose_value(Some(Route::IvBolus), true, 16.0);
        for (name, t, c) in [
            ("at the dose", [0.0, 2.0, 4.0], [8.0, 4.0, 1.0]),
            ("before the dose", [-1.0, 2.0, 4.0], [8.0, 4.0, 1.0]),
        ] {
            let (t_area, c_area, inserted) = area_row(&t, &c, 3, bolus);
            assert!(!inserted.inserted, "{name}");
            assert_eq!(inserted.n, 3, "{name}");
            assert_eq!(&t_area[..3], &t, "{name}");
            assert_eq!(&c_area[..3], &c, "{name}");
        }

        // nor into a row of fewer than two points, nor where the estimate is not a number
        let (_, _, one_point) = area_row(&[1.0, f64::NAN], &[8.0, f64::NAN], 1, bolus);
        assert!(!one_point.inserted);
        assert_eq!(one_point.n, 1);
        let no_estimate = dose_value(Some(Route::IvBolus), true, f64::NAN);
        let (t_area, _, inserted) = area_row(&[1.0, 2.0], &[8.0, 4.0], 2, no_estimate);
        assert!(!inserted.inserted, "C0Method::None estimates nothing");
        assert_eq!(inserted.n, 2);
        assert_eq!(&t_area[..2], &[1.0, 2.0]);

        // areas.md 3.8 and trap T3: the back-extrapolated fraction of such a row is the
        // area of the inserted segment over AUC(0-inf), and 0.0 where nothing was
        // inserted - the numerator is a zero and not the NaN of an absent segment
        let (t_area, c_area, inserted) = area_row(&[0.0, 2.0, 4.0], &[8.0, 4.0, 1.0], 3, bolus);
        let segment = trapezoid(&t_area[..2], &c_area[..2], AucMethod::Linear).auc;
        let numerator = if inserted.inserted { segment } else { 0.0 };
        assert_eq!(
            segment, 12.0,
            "the first segment of the row is a real segment"
        );
        assert_eq!(
            numerator, 0.0,
            "and the fraction reads `inserted`, not the area"
        );
        assert_eq!(numerator / 42.0, 0.0, "a fraction of 0.0, never NaN");
    }

    #[test]
    fn the_inserted_point_reaches_the_areas_and_never_the_regression() {
        // The single most important property of the insertion (areas.md section 4 and trap
        // T4). The curve declines steeply at first and settles on the terminal line
        // c = 16 * 2^(-t/2) from t = 2 on; the back-extrapolation of its first two samples
        // is c0 = 40 * 40 / 20 = 80, far above everything the terminal phase would predict.
        let t = [0.5, 1.0, 2.0, 4.0, 8.0];
        let c = [40.0, 20.0, 8.0, 4.0, 1.0];
        let c0 = bolus_c0(&t, &c, 5, C0Method::LogBackExtrapolation);
        close(c0.value, 80.0);

        let (t_area, c_area, inserted) =
            area_row(&t, &c, 5, dose_value(Some(Route::IvBolus), true, c0.value));
        assert!(inserted.inserted);
        assert_eq!(inserted.n, 6);
        assert_eq!(&t_area[..6], &[0.0, 0.5, 1.0, 2.0, 4.0, 8.0]);

        // the areas see it: the linear trapezoids of the row are
        // 15 + 14 + 12 + 10 = 51 and the inserted segment adds (80 + 40) / 2 * 0.5 = 30
        let row = trapezoid(&t, &c, AucMethod::Linear);
        let area = trapezoid(&t_area[..6], &c_area[..6], AucMethod::Linear);
        assert_eq!(row.auc, 51.0);
        close(area.auc, 81.0);

        // and the regression does not: it is handed the packed row, which this call never
        // touched, so its window is the four points after the maximum and not the five of
        // the area row. A leak would report t_first = 0.5 over five points and a slope an
        // eighth steeper.
        let phase = TerminalPhase {
            method: TerminalMethod::AllAfterTmax,
            ..TerminalPhase::default()
        };
        let permitted = [true; 6];
        let fit = terminal_fit(&t, &c, 5, &permitted[..5], 0, &phase);
        assert_eq!(fit.n_points, 4.0);
        assert_eq!(fit.t_first, 1.0);
        assert_eq!(fit.t_last, 8.0);
        assert_eq!(fit.flags, NcaFlag::NONE);

        let leaked = terminal_fit(&t_area, &c_area, 6, &permitted, 0, &phase);
        assert_eq!(leaked.n_points, 5.0, "what a leak would regress");
        assert_eq!(leaked.t_first, 0.5);
        assert!(
            (fit.slope - leaked.slope).abs() > 1e-3,
            "the two windows have to differ: {} and {}",
            fit.slope,
            leaked.slope
        );
    }

    #[test]
    fn a_row_out_of_time_order_is_sorted_where_python_sorts_it() {
        // `insert_point` concatenates the new point and stable-sorts the whole row, so a
        // row which arrives out of time order is reordered for the areas although
        // `pack_valid` left it alone (areas.md section 2, trap T1). It is sorted whenever
        // the route inserts at all, even where no point goes in, and not at all for a row
        // without a route.
        let t = [4.0, 1.0, 2.0];
        let c = [1.0, 8.0, 4.0];
        let bolus = dose_value(Some(Route::IvBolus), true, 1.0);
        let (t_area, c_area, inserted) = area_row(&t, &c, 3, bolus);
        assert!(inserted.inserted);
        assert_eq!(inserted.n, 4);
        assert_eq!(&t_area[..4], &[0.0, 1.0, 2.0, 4.0]);
        assert_eq!(&c_area[..4], &[1.0, 8.0, 4.0, 1.0]);

        // the same row with a first sample at the dose: nothing is inserted and it is
        // sorted all the same
        let t = [0.0, 4.0, 2.0];
        let c = [1.0, 8.0, 4.0];
        let (t_area, c_area, inserted) = area_row(&t, &c, 3, bolus);
        assert!(!inserted.inserted);
        assert_eq!(inserted.n, 3);
        assert_eq!(&t_area[..3], &[0.0, 2.0, 4.0]);
        assert_eq!(&c_area[..3], &[1.0, 4.0, 8.0]);

        // while a row without a route is written through as it is
        let (t_area, c_area, inserted) = area_row(&t, &c, 3, dose_value(None, true, 1.0));
        assert!(!inserted.inserted);
        assert_eq!(&t_area[..3], &t);
        assert_eq!(&c_area[..3], &c);
    }

    #[test]
    fn a_point_at_the_dose_time_goes_behind_a_sample_of_the_same_time() {
        // The new point is the last column of the concatenated row, so the stable sort
        // puts it behind an existing sample of the same time. Only a row out of time order
        // reaches the tie, since a row in order whose first sample is at the dose inserts
        // nothing at all.
        let (t_area, c_area, inserted) = area_row(&[2.0, 0.0], &[4.0, 5.0], 2, Some(7.0));
        assert!(inserted.inserted);
        assert_eq!(inserted.n, 3);
        assert_eq!(&t_area[..3], &[0.0, 0.0, 2.0]);
        assert_eq!(
            &c_area[..3],
            &[5.0, 7.0, 4.0],
            "the sample first, the inserted point behind it"
        );
    }

    #[test]
    fn a_short_buffer_is_cut_and_never_indexed_past_its_end() {
        // The shapes come from a host: a row longer than the buffer it is written into is
        // cut like `pack_valid` cuts, and a count which claims more points than the row
        // has reads only the row. A buffer without room for the point at the dose time
        // takes the row alone, and says so: `inserted` means a point really went in, so a
        // caller which reads it for the back-extrapolated fraction is never told of a
        // segment the area row does not carry.
        let t = [1.0, 2.0, 4.0];
        let c = [8.0, 4.0, 1.0];
        let mut t_area = [0.0; 2];
        let mut c_area = [0.0; 2];
        let cut = insert_dose_value(&t, &c, 3, Some(16.0), &mut t_area, &mut c_area);
        assert!(!cut.inserted, "the buffer has no room for a fourth point");
        assert_eq!(cut.n, 2, "the buffer holds two of the three points");
        assert_eq!(t_area, [1.0, 2.0], "the row alone, and no point at t = 0");
        assert_eq!(c_area, [8.0, 4.0]);

        // a buffer of exactly the row is one short as well
        let mut t_area = [0.0; 3];
        let mut c_area = [0.0; 3];
        let exact = insert_dose_value(&t, &c, 3, Some(16.0), &mut t_area, &mut c_area);
        assert!(!exact.inserted);
        assert_eq!(exact.n, 3);
        assert_eq!(t_area, t);
        assert_eq!(c_area, c);

        // while the one column of room a caller leaves is enough
        let (t_area, _, room) = area_row(&t, &c, 3, Some(16.0));
        assert!(room.inserted);
        assert_eq!(room.n, 4);
        assert_eq!(&t_area[..4], &[0.0, 1.0, 2.0, 4.0]);

        let (_, _, over) = area_row(&t, &c, 9, Some(16.0));
        assert_eq!(over.n, 4, "three points and the inserted one");
        assert!(over.inserted);

        let mut empty: [f64; 0] = [];
        let mut empty_values: [f64; 0] = [];
        let none = insert_dose_value(&t, &c, 3, Some(16.0), &mut empty, &mut empty_values);
        assert!(!none.inserted);
        assert_eq!(none.n, 0, "a buffer of no room writes no point");
    }
}
