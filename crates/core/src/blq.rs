//! The values below the limit of quantification.
//!
//! A bioanalytical assay reports nothing below its lower limit of quantification, and what the
//! analysis does with such a sample is a choice with a name: drop it, keep the number the assay
//! printed, or impute a zero, the limit or half of it. The choice is made per position in the
//! curve, either by where the sample lies between the measurable values (`first`, `middle`,
//! `last`) or by whether it lies before the maximum (`before_tmax`, `after_tmax`), the two axes
//! of `BlqRules`, whose presets are the rules of PKNCA, PKanalix, Pumas and ICH M13A. A position
//! without a rule drops its values, which is the default of the python package.
//!
//! What follows from a rule is not only the value. A point which was imputed or kept is part of
//! the curve - it reaches the areas, `cmax` and `cmin` - and is still no quantified measurement:
//! it is not the last measurable value, so `clast` and `tlast` step over it, it is kept out of
//! the terminal regression unless `BlqRules::terminal_regression` asks otherwise, and it is not
//! a measurable value for `tlag`. [`apply_blq`] reports those points in `in_curve` and the
//! caller holds them out of the three; [`packed_mask`] carries the mask onto the packed row.
//!
//! The order is the order of `compute_parameters`: the times are shifted by the dose, the rules
//! run on the values as they arrived, and only then is the row packed, so a dropped value is
//! simply no point.

use pkpdutils_abi::{BlqAction, BlqRules, BlqSetting, NcaOptions};

/// The limit of quantification of one row, `None` where neither the options nor the row name one.
///
/// `NcaOptions::lloq` is the limit of the whole analysis and wins over the limit of the row.
/// With `NcaOptions::lloq_relative` the larger of the absolute limit and that fraction of the
/// peak of the row is used, where the peak is the largest finite positive value of the row; a
/// row without one has no relative cutoff, which is a `NaN` limit and therefore no limit, since
/// nothing is below a limit which is not a number.
///
/// `values` is the row as the rules will read it, before packing.
///
/// Mirrors `resolve_lloq` of `src/pkpdutils/nca/nca.py`, for one row.
pub(crate) fn resolve_lloq(options: &NcaOptions, lloq: Option<f64>, values: &[f64]) -> Option<f64> {
    let absolute = options.lloq.or(lloq);
    let Some(fraction) = options.lloq_relative else {
        return absolute;
    };
    let mut peak = 0.0_f64;
    for &value in values {
        if value.is_finite() && value > peak {
            peak = value;
        }
    }
    let relative = if peak > 0.0 {
        fraction * peak
    } else {
        f64::NAN
    };
    // `numpy.fmax` and `f64::max` agree: a NaN loses against a number
    Some(match absolute {
        Some(absolute) => absolute.max(relative),
        None => relative,
    })
}

/// Apply the rules for the values below the limit of quantification to one row.
///
/// `values` is the row, which is changed in place: a dropped value becomes `NaN` and so no point
/// at all once the row is packed, an imputed one becomes what its rule writes. `in_curve` is
/// filled with the points which are below the limit and still part of the curve, the imputed and
/// the kept ones, over the columns of the row. The two slices are read to the length they share,
/// which is how every row of this crate is read.
///
/// The return is whether a value was dropped or imputed, the `BLQ_TRUNCATED` flag of the row.
/// Keeping a value raises it no more than having no value below the limit does: nothing was
/// changed, so nothing was truncated.
///
/// The groups are read from the values as they arrived and are disjoint, so no rule sees what
/// another rule wrote. A row without a single measurable value has no first and no last one, so
/// every value of it is `first` on the positional axis and `after_tmax` on the tmax axis.
///
/// Mirrors `apply_blq` of `src/pkpdutils/nca/nca.py`.
pub(crate) fn apply_blq(
    values: &mut [f64],
    lloq: Option<f64>,
    rules: &BlqRules,
    in_curve: &mut [bool],
) -> bool {
    for slot in in_curve.iter_mut() {
        *slot = false;
    }
    let Some(limit) = lloq else {
        return false;
    };
    let columns = values.len().min(in_curve.len());
    // a limit which is not a number is no limit: nothing is below it
    let below = |value: f64| value.is_finite() && limit.is_finite() && value < limit;
    let (mut first_idx, mut last_idx, mut tmax_idx) = (columns, columns, 0usize);
    let mut peak = f64::NEG_INFINITY;
    for (j, &value) in values.iter().enumerate().take(columns) {
        if !value.is_finite() || below(value) {
            continue;
        }
        if first_idx == columns {
            first_idx = j;
        }
        last_idx = j;
        if value > peak {
            peak = value;
            tmax_idx = j;
        }
    }
    let by_tmax = rules.by_tmax();
    let mut truncated = false;
    for j in 0..columns {
        if !below(values[j]) {
            continue;
        }
        let rule = if by_tmax {
            if j < tmax_idx {
                rules.before_tmax
            } else {
                rules.after_tmax
            }
        } else if j < first_idx {
            rules.first
        } else if j > last_idx {
            rules.last
        } else {
            // the first and the last measurable value are measurable, so a value below the
            // limit which is neither before the one nor behind the other is between them
            rules.middle
        };
        // a position without a rule drops its values; `Drop` writes a NaN and keeps the point
        // out of the curve, `Keep` writes nothing at all, and everything else imputes a value
        // which stays part of the curve
        let (written, in_the_curve) = match rule.unwrap_or(BlqSetting::Action(BlqAction::Drop)) {
            BlqSetting::Action(BlqAction::Drop) => (Some(f64::NAN), false),
            BlqSetting::Action(BlqAction::Keep) => (None, true),
            BlqSetting::Action(BlqAction::Zero) => (Some(0.0), true),
            BlqSetting::Action(BlqAction::Lloq) => (Some(limit), true),
            BlqSetting::Action(BlqAction::HalfLloq) => (Some(0.5 * limit), true),
            BlqSetting::Value(value) => (Some(value), true),
        };
        if let Some(written) = written {
            values[j] = written;
            truncated = true;
        }
        in_curve[j] = in_the_curve;
    }
    truncated
}

/// Carry a mask over the columns of a row onto the positions of the packed row.
///
/// [`crate::pack::pack_valid`] moves the points whose time and value are both finite to the
/// front of the row, so the mask of a point moves with it; a column which is no point carries no
/// mask, and the positions behind the points are `false`. `times` and `values` are the row as
/// the packing reads it, the values after the rules of [`apply_blq`] ran, so a value a rule
/// dropped is no point and its mask falls away with it.
///
/// Mirrors `packed_mask` of `src/pkpdutils/nca/nca.py`.
pub(crate) fn packed_mask(times: &[f64], values: &[f64], mask: &[bool], out: &mut [bool]) {
    let columns = times.len().min(values.len());
    let mut packed = 0usize;
    for j in 0..columns {
        if !times[j].is_finite() || !values[j].is_finite() {
            continue;
        }
        let Some(slot) = out.get_mut(packed) else {
            break;
        };
        *slot = mask.get(j).copied().unwrap_or(false);
        packed += 1;
    }
    for slot in out.iter_mut().skip(packed) {
        *slot = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use pkpdutils_abi::{AucMethod, BlqHandling, NcaFlag, TerminalPhase};

    use crate::auc::trapezoid;
    use crate::pack::pack_valid;
    use crate::terminal::terminal_fit;

    /// Equality which holds on a `NaN`, which a dropped value is.
    #[track_caller]
    fn same(actual: &[f64], expected: &[f64]) {
        let equal = actual.len() == expected.len()
            && actual
                .iter()
                .zip(expected)
                .all(|(a, e)| a == e || (a.is_nan() && e.is_nan()));
        assert!(equal, "{actual:?} is not {expected:?}");
    }

    /// One of the named actions as a rule.
    fn action(rule: BlqAction) -> Option<BlqSetting> {
        Some(BlqSetting::Action(rule))
    }

    /// Rules by position.
    fn positional(first: BlqAction, middle: BlqAction, last: BlqAction) -> BlqRules {
        BlqRules {
            first: action(first),
            middle: action(middle),
            last: action(last),
            ..BlqRules::default()
        }
    }

    /// Rules against the maximum.
    fn by_tmax(before: Option<BlqSetting>, after: Option<BlqSetting>) -> BlqRules {
        BlqRules {
            before_tmax: before,
            after_tmax: after,
            ..BlqRules::default()
        }
    }

    /// The row after the rules ran, whether anything was truncated, and the points of the
    /// curve which are below the limit.
    fn applied(values: &[f64], lloq: Option<f64>, rules: &BlqRules) -> (Vec<f64>, bool, Vec<bool>) {
        let mut row = values.to_vec();
        let mut in_curve = vec![false; values.len()];
        let truncated = apply_blq(&mut row, lloq, rules, &mut in_curve);
        (row, truncated, in_curve)
    }

    #[test]
    fn each_blq_action_does_what_python_does_before_and_after_tmax() {
        // areas.md section 5: the tmax axis splits the values below the limit at the first
        // maximum of the measurable ones, `before_tmax` strictly before it and
        // `after_tmax` at it and behind it. The row measures 1.0 and 5.0, so the maximum
        // sits at index 2: the value at index 0 is before it, those at 3 and 4 after it.
        let row = [0.2, 1.0, 5.0, 0.3, 0.1];
        let (values, truncated, in_curve) = applied(
            &row,
            Some(0.5),
            &BlqRules::from_handling(BlqHandling::ZeroBeforeTmax),
        );
        same(&values, &[0.0, 1.0, 5.0, f64::NAN, f64::NAN]);
        assert!(truncated);
        assert_eq!(in_curve, [true, false, false, false, false]);

        // every action, on both groups at once, so that the table is read in full
        let both = [
            (BlqAction::Drop, f64::NAN, false),
            (BlqAction::Keep, f64::NAN, true),
            (BlqAction::Zero, 0.0, true),
            (BlqAction::Lloq, 0.5, true),
            (BlqAction::HalfLloq, 0.25, true),
        ];
        for (rule, imputed, in_the_curve) in both {
            let kept = rule == BlqAction::Keep;
            let expected = |original: f64| if kept { original } else { imputed };
            let (values, truncated, in_curve) =
                applied(&row, Some(0.5), &by_tmax(action(rule), action(rule)));
            same(
                &values,
                &[expected(0.2), 1.0, 5.0, expected(0.3), expected(0.1)],
            );
            assert_eq!(truncated, !kept, "{rule:?}");
            assert_eq!(
                in_curve,
                [in_the_curve, false, false, in_the_curve, in_the_curve],
                "{rule:?}"
            );
        }

        // and a number imputed as it is, which the rules take in place of an action
        let value = Some(BlqSetting::Value(0.123));
        let (values, truncated, in_curve) = applied(&row, Some(0.5), &by_tmax(value, value));
        same(&values, &[0.123, 1.0, 5.0, 0.123, 0.123]);
        assert!(truncated);
        assert_eq!(in_curve, [true, false, false, true, true]);

        // a row without a measurable value has no maximum, so every value of it is after
        // the maximum
        let (values, _, _) = applied(
            &[0.2, 0.3],
            Some(0.5),
            &by_tmax(action(BlqAction::Zero), action(BlqAction::Lloq)),
        );
        same(&values, &[0.5, 0.5]);
    }

    #[test]
    fn the_positional_axis_splits_at_the_first_and_the_last_measurable_value() {
        // areas.md section 5: the row measures 1.0 at index 1 and 5.0 at index 3, so the
        // value at index 0 is `first`, the one at index 2 `middle` and the one at index 4
        // `last`.
        let row = [0.2, 1.0, 0.3, 5.0, 0.1];
        let (values, truncated, in_curve) = applied(&row, Some(0.5), &BlqRules::ich_m13a());
        same(&values, &[0.0, 1.0, f64::NAN, 5.0, 0.0]);
        assert!(truncated);
        assert_eq!(in_curve, [true, false, false, false, true]);

        let (values, truncated, in_curve) = applied(&row, Some(0.5), &BlqRules::pumas());
        same(&values, &[0.2, 1.0, f64::NAN, 5.0, 0.1]);
        assert!(truncated, "the middle value was dropped");
        assert_eq!(in_curve, [true, false, false, false, true]);

        // the default of the python package is no preset at all: dropped everywhere
        let (values, truncated, in_curve) =
            applied(&row, Some(0.5), &BlqRules::from_handling(BlqHandling::Nan));
        same(&values, &[f64::NAN, 1.0, f64::NAN, 5.0, f64::NAN]);
        assert!(truncated);
        assert_eq!(in_curve, [false; 5]);

        // a row without a measurable value has no first and no last one, so all of it is
        // `first` and neither the middle nor the last rule is read
        let (values, truncated, _) = applied(
            &[0.2, 0.3],
            Some(0.5),
            &positional(BlqAction::Zero, BlqAction::Drop, BlqAction::Drop),
        );
        same(&values, &[0.0, 0.0]);
        assert!(truncated);
    }

    #[test]
    fn keeping_blq_values_does_not_set_the_truncated_flag() {
        // areas.md section 5: `KEEP` changes no value, so the row was not truncated,
        // although the points it kept are marked as below the limit. Every other action
        // changes something and raises the flag.
        let row = [0.2, 1.0, 0.3, 5.0, 0.1];
        let (values, truncated, in_curve) = applied(
            &row,
            Some(0.5),
            &positional(BlqAction::Keep, BlqAction::Keep, BlqAction::Keep),
        );
        same(&values, &row);
        assert!(!truncated, "nothing was changed, so nothing was truncated");
        assert_eq!(in_curve, [true, false, true, false, true]);

        for rule in [
            BlqAction::Drop,
            BlqAction::Zero,
            BlqAction::Lloq,
            BlqAction::HalfLloq,
        ] {
            let (_, truncated, _) = applied(&row, Some(0.5), &positional(rule, rule, rule));
            assert!(truncated, "{rule:?}");
        }

        // and a row with no limit at all is no truncation either
        let (values, truncated, in_curve) = applied(&row, None, &BlqRules::ich_m13a());
        same(&values, &row);
        assert!(!truncated);
        assert_eq!(in_curve, [false; 5]);

        // nor is a limit which is not a number, which `lloq_relative` gives a row without
        // a positive value
        let (values, truncated, _) = applied(&row, Some(f64::NAN), &BlqRules::ich_m13a());
        same(&values, &row);
        assert!(!truncated);
    }

    #[test]
    fn a_blq_value_counts_in_cmax_and_the_areas_but_not_in_clast_or_the_regression() {
        // areas.md section 5, the consequences: a kept value is part of the curve and no
        // measurement. Every value of this row is below the limit, so the row has no
        // measurable value at all and the whole of it is `first`, which `KEEP` keeps.
        let t = [1.0, 2.0, 4.0, 8.0, 12.0];
        let row = [2.0, 8.0, 4.0, 2.0, 1.0];
        let rules = positional(BlqAction::Keep, BlqAction::Keep, BlqAction::Keep);
        let (values, truncated, in_curve) = applied(&row, Some(10.0), &rules);
        same(&values, &row);
        assert!(!truncated);
        assert_eq!(in_curve, [true; 5]);

        let mut tp = [0.0; 5];
        let mut cp = [0.0; 5];
        let n_valid = pack_valid(&t, &values, &mut tp, &mut cp);
        assert_eq!(n_valid, 5);
        let mut blq = [false; 5];
        packed_mask(&t, &values, &in_curve, &mut blq);
        assert_eq!(blq, [true; 5]);

        // it counts in cmax, `imax = argmax(where(in_row, cp, -inf))` over every point
        let imax = (0..n_valid).fold(0, |best, j| if cp[j] > cp[best] { j } else { best });
        assert_eq!(
            (cp[imax], tp[imax]),
            (8.0, 2.0),
            "a BLQ value is the maximum"
        );

        // and in the areas, 5 + 12 + 12 + 6 under the linear rule
        assert_eq!(trapezoid(&tp, &cp, AucMethod::Linear).auc, 35.0);

        // but it is no measurable value, so the row has no last one: `positive` is
        // `in_row & cp > 0 & ~blq_packed` and clast and tlast are NaN with it
        let has_positive = (0..n_valid).any(|j| cp[j] > 0.0 && !blq[j]);
        assert!(!has_positive, "no value of the row is measurable");

        // and none of it is regressed, since the caller excludes the kept points
        let permitted: Vec<bool> = blq.iter().map(|&point| !point).collect();
        let phase = TerminalPhase::default();
        let fit = terminal_fit(&tp, &cp, n_valid, &permitted, imax, &phase);
        assert_eq!(fit.start, -1);
        assert!(fit.slope.is_nan());
        assert_eq!(fit.flags, NcaFlag::TOO_FEW_POINTS);

        // unless the rules let them in: `BlqRules::terminal_regression` is the one switch
        // which hands the same points to the regression, and the three from t = 4 on lie
        // on the line c = 8 * 2^(-t/4), whose rate is ln(2) / 4
        let regressed = terminal_fit(&tp, &cp, n_valid, &[true; 5], imax, &phase);
        assert_eq!(regressed.n_points, 3.0);
        assert_eq!(regressed.t_first, 4.0);
        assert_eq!(regressed.t_last, 12.0);
        let lambda_z = -regressed.slope;
        assert!(
            (lambda_z - std::f64::consts::LN_2 / 4.0).abs() < 1e-12,
            "lambda_z {lambda_z}"
        );
    }

    #[test]
    fn the_limit_is_the_larger_of_the_absolute_and_the_relative_one() {
        // areas.md section 5: `NCAOptions.lloq` wins over the limit of the row, and with
        // `lloq_relative` the larger of the absolute limit and that fraction of the peak
        // of the row is taken.
        let row = [40.0, 20.0, 8.0];
        let absolute = NcaOptions {
            lloq: Some(0.5),
            ..NcaOptions::default()
        };
        assert_eq!(resolve_lloq(&absolute, None, &row), Some(0.5));
        assert_eq!(
            resolve_lloq(&absolute, Some(0.1), &row),
            Some(0.5),
            "the limit of the options wins over the limit of the row"
        );

        let plain = NcaOptions::default();
        assert_eq!(resolve_lloq(&plain, Some(0.1), &row), Some(0.1));
        assert_eq!(resolve_lloq(&plain, None, &row), None, "no limit at all");

        let relative = NcaOptions {
            lloq_relative: Some(0.01),
            ..NcaOptions::default()
        };
        assert_eq!(resolve_lloq(&relative, None, &row), Some(0.4), "1 % of 40");
        assert_eq!(
            resolve_lloq(&relative, Some(0.1), &row),
            Some(0.4),
            "the relative cutoff is the larger one here"
        );
        let both = NcaOptions {
            lloq: Some(0.5),
            lloq_relative: Some(0.01),
            ..NcaOptions::default()
        };
        assert_eq!(resolve_lloq(&both, None, &row), Some(0.5));

        // a row without a finite positive value has no peak and so no relative cutoff
        let empty = [0.0, -2.0, f64::NAN, f64::INFINITY];
        assert_eq!(
            resolve_lloq(&relative, None, &empty).map(f64::is_nan),
            Some(true)
        );
        assert_eq!(
            resolve_lloq(&both, None, &empty),
            Some(0.5),
            "the absolute limit stands on its own"
        );
    }

    #[test]
    fn the_mask_of_a_value_moves_with_it_into_the_packed_row() {
        // `pack_valid` moves the points to the front of the row, so the mask follows them
        // there; a column which is no point carries nothing into the packed row.
        let times = [0.0, 1.0, f64::NAN, 3.0];
        let values = [1.0, f64::NAN, 9.0, 7.0];
        let mut out = [false; 4];
        packed_mask(&times, &values, &[true, true, true, false], &mut out);
        assert_eq!(out, [true, false, false, false]);

        packed_mask(&times, &values, &[false, true, true, true], &mut out);
        assert_eq!(out, [false, true, false, false], "the point at t = 3");

        // a value a rule dropped is no point, so its mask never arrives
        let dropped = [1.0, f64::NAN, 3.0];
        let mut out = [true; 3];
        packed_mask(&[0.0, 1.0, 2.0], &dropped, &[false, true, false], &mut out);
        assert_eq!(out, [false, false, false]);
    }
}
