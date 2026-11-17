//! The terminal elimination phase of one packed curve.
//!
//! The elimination rate constant `lambda_z` is the negative slope of the regression of `ln c`
//! on `t` over the points of the terminal phase (Gabrielsson & Weiner 2016, ch. 2.8; Phoenix
//! WinNonlin NCA). Which points those are is what [`TerminalPhase`] decides, and all four rules
//! read the same table: a window is a suffix of the packed row, from a start index to the last
//! point of the row which may be regressed, and [`window_statistics`] fills in the regression of
//! every one of them in a single backward pass over the suffix sums.
//!
//! The arithmetic is `src/pkpdutils/nca/terminal.py` expression by expression, down to the order
//! of the sums: they are accumulated from the last column towards the first, as the
//! `numpy.cumsum` of the reversed row is, and they are taken around the first valid point of the
//! row, which keeps the raw moments of a curve sampled at large times from cancelling. A
//! rearranged formula drifts in the last places and the benchmark gate compares at a relative
//! 1e-8.
//!
//! The rule which decides a window is where a port silently diverges, so three of them are
//! spelled out here and each is pinned by a test. `TerminalMethod::BestFit` collects every
//! window within an **absolute** tolerance of the row's best adjusted R squared and takes the
//! **earliest** start of that band, which is the window with the most points and not the window
//! with the best adjusted R squared. `TerminalMethod::Manual` applies no test on the sign of the
//! slope at all, so a rising window reports a negative `lambda_z` and raises no flag, where
//! `BestFit`, `LastN` and `AllAfterTmax` do test it. And `TerminalPhase::min_adj_r2` takes no
//! part in the selection either: it only rejects the window the selection chose, as
//! `NcaFlag::TOO_FEW_POINTS`, and never falls back to another window which would pass. The last
//! two are upstream oddities, mirrored rather than corrected and raised with the maintainer.

use pkpdutils_abi::{NcaFlag, TerminalMethod, TerminalPhase};

/// The half-lives a terminal window should cover at least, `SPAN_MINIMUM` of `nca.py`.
pub(crate) const SPAN_MINIMUM: f64 = 2.0;

/// Multiple of `n * eps * syy` below which a residual sum of squares is zero.
///
/// The residual of a window is `syy_c - slope * sxy_c`, two quantities which
/// are equal for a perfect fit, so what is left there is the rounding of the
/// sums themselves and its sign is arbitrary. Clamping it at zero reported
/// either 0 or the square root of that noise, around 1e-9, and which one
/// depended on the platform: this kernel and the python core disagreed on a
/// noise free curve, which the engine gate of `pkpdutils` caught.
///
/// The floor is taken against the uncentred `syy` and not against `syy_c`,
/// because every window of a row shares one origin, the row's first sample,
/// so a short window late in the curve has `syy` and `sy * sy / n` both of
/// the order of the whole drop of the curve while their difference is small.
/// The multiple is 16 from measurement over 120 000 windows: the ratio
/// `|ss_res| / (n * eps * syy)` reaches 1.20 for a noise free curve and stays
/// above 1.9e+05 for a curve of 1 % noise from four samples up. It must stay
/// identical to `pkpdutils.nca.terminal.RESIDUAL_FLOOR`.
pub(crate) const RESIDUAL_FLOOR: f64 = 16.0 * f64::EPSILON;

/// The regression statistics of one window.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WindowStats {
    /// The points of the window.
    pub n: f64,
    /// The slope of `ln c` against `t`.
    pub slope: f64,
    /// The intercept at `t = 0`.
    pub intercept: f64,
    /// The coefficient of determination.
    pub r2: f64,
    /// The adjusted coefficient of determination.
    pub r2_adj: f64,
    /// The standard error of the slope.
    pub se_slope: f64,
}

impl WindowStats {
    /// Every statistic `NaN`: a window which is no window.
    pub(crate) const NONE: Self = Self {
        n: f64::NAN,
        slope: f64::NAN,
        intercept: f64::NAN,
        r2: f64::NAN,
        r2_adj: f64::NAN,
        se_slope: f64::NAN,
    };
}

/// The suffix sums one window's statistics are closed-form expressions of.
#[derive(Debug, Clone, Copy)]
struct Sums {
    /// The points of the window, as a count of ones.
    n: f64,
    /// The sum of the shifted times.
    sx: f64,
    /// The sum of the shifted logarithms.
    sy: f64,
    /// The sum of the squared shifted times.
    sxx: f64,
    /// The sum of the products.
    sxy: f64,
    /// The sum of the squared shifted logarithms.
    syy: f64,
}

/// Fill `out` with the regression of every window of one row.
///
/// Column `s` of `out` describes the window which starts at index `s` and runs to the last
/// regressable point of the row: a window is always a suffix, so `n` never grows with `s`. A
/// start which is itself not regressable carries the statistics of the next point which is,
/// which is what lets a selection count a window it may not start at and then snap the start
/// forward onto a real point.
///
/// Every statistic, the count `n` among them, is `NaN` for a window of fewer than three points,
/// of no spread in the times (one repeated time) or of no spread in the logarithms (constant
/// values), so such a window answers no `n >= k` test and is no window at all.
///
/// The shapes come from a host, so none of them is trusted: the columns computed are the ones
/// all four slices have and the rest of `out` is filled with [`WindowStats::NONE`].
///
/// Mirrors `window_statistics` of `src/pkpdutils/nca/terminal.py`.
pub(crate) fn window_statistics(
    t: &[f64],
    y: &[f64],
    regressable: &[bool],
    out: &mut [WindowStats],
) {
    let columns = out.len().min(t.len()).min(y.len()).min(regressable.len());
    for slot in out.iter_mut().skip(columns) {
        *slot = WindowStats::NONE;
    }
    if columns == 0 {
        return;
    }
    // The origin is the first valid point of the ROW, not of the window: the sums of a curve
    // sampled around t = 1e8 cancel catastrophically without it. A row with no valid point at
    // all takes its padding, which never reaches a result since every window of it is NaN.
    let first = regressable[..columns].iter().position(|&p| p).unwrap_or(0);
    let x_origin = t[first];
    let y_origin = y[first];
    let mut sums = Sums {
        n: 0.0,
        sx: 0.0,
        sy: 0.0,
        sxx: 0.0,
        sxy: 0.0,
        syy: 0.0,
    };
    // backwards, which is the summation order of `cumsum(a[::-1])[::-1]`
    for s in (0..columns).rev() {
        let (x0, y0) = if regressable[s] {
            sums.n += 1.0;
            (t[s] - x_origin, y[s] - y_origin)
        } else {
            (0.0, 0.0)
        };
        sums.sx += x0;
        sums.sy += y0;
        sums.sxx += x0 * x0;
        sums.sxy += x0 * y0;
        sums.syy += y0 * y0;
        out[s] = window_of(sums, x_origin, y_origin);
    }
}

/// The regression of the window the suffix sums describe.
fn window_of(sums: Sums, x_origin: f64, y_origin: f64) -> WindowStats {
    let Sums {
        n,
        sx,
        sy,
        sxx,
        sxy,
        syy,
    } = sums;
    let sxx_c = sxx - sx * sx / n;
    let sxy_c = sxy - sx * sy / n;
    let syy_c = syy - sy * sy / n;
    if n >= 3.0 && sxx_c > 0.0 && syy_c > 0.0 {
        let slope = sxy_c / sxx_c;
        let raw = syy_c - slope * sxy_c;
        // the residual subtracts two quantities which are equal for a perfect
        // fit, so below the rounding of its own operands it is not a small
        // number but no number. A NaN fails this comparison and is kept,
        // which is what `numpy.where` does on the python side as well, and a
        // negative value is below the floor, which subsumes the `numpy.maximum`
        // this replaces.
        let residual = if raw < RESIDUAL_FLOOR * n * syy {
            0.0
        } else {
            raw
        };
        // `numpy.clip` lets a NaN through, which `f64::clamp` does as well
        let r2 = (1.0 - residual / syy_c).clamp(0.0, 1.0);
        WindowStats {
            n,
            slope,
            // the intercept at the original t = 0, not at the origin of the sums
            intercept: (sy - slope * sx) / n + y_origin - slope * x_origin,
            r2,
            // the degrees of freedom are n - 2 and the adjusted value is not clipped,
            // so it may well be negative
            r2_adj: 1.0 - (1.0 - r2) * (n - 1.0) / (n - 2.0),
            se_slope: (residual / (n - 2.0) / sxx_c).sqrt(),
        }
    } else {
        WindowStats::NONE
    }
}

/// The terminal regression of one row.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TerminalFit {
    /// The slope of `ln c` against `t`, the negative of `lambda_z`.
    pub slope: f64,
    /// The intercept at `t = 0`.
    pub intercept: f64,
    /// The coefficient of determination.
    pub r2: f64,
    /// The adjusted coefficient of determination.
    pub r2_adj: f64,
    /// The standard error of the slope.
    pub se_slope: f64,
    /// The points of the regression.
    pub n_points: f64,
    /// The time of the first point.
    pub t_first: f64,
    /// The time of the last point.
    pub t_last: f64,
    /// The flags the regression raised.
    pub flags: NcaFlag,
}

impl TerminalFit {
    /// The fit of a row without a window.
    pub(crate) const fn no_fit(flags: NcaFlag) -> Self {
        Self {
            slope: f64::NAN,
            intercept: f64::NAN,
            r2: f64::NAN,
            r2_adj: f64::NAN,
            se_slope: f64::NAN,
            n_points: f64::NAN,
            t_first: f64::NAN,
            t_last: f64::NAN,
            flags,
        }
    }
}

/// The terminal regression of one packed row.
///
/// `t` and `c` are the packed points of the row and `n_valid` is how many of them are points at
/// all; the slots behind them are the `NaN` padding of [`crate::pack::pack_valid`] and are read
/// by nothing. `tmax_idx` is the packed index of the first maximum over those points, which the
/// caller computes over every valid point and not only over the regressable ones.
///
/// `regressable` says which packed points the caller permits in the regression: the values a BLQ
/// rule kept or imputed, and the samples at or before the end of an infusion, are the ones it
/// keeps out. It is intersected here with the points of the row and with a finite `ln c`, which
/// is how python composes `regressable = in_row & isfinite(y) & ~exclude`, so a caller which
/// hands over the finished mask gets the same answer as one which hands over only its own part.
/// A zero, a negative value and a point behind `n_valid` are therefore never regressed, and a
/// zero in the middle of a window is skipped rather than ending it.
///
/// `TerminalMethod::Manual` regresses exactly the points of that mask, and nothing here reads
/// `TerminalPhase::points`: its indices are indices into the curve as the host sent it, and only
/// the caller can map them onto the packed row of a sample, so **the caller resolves them and
/// hands over `regressable` with the manual mask already folded in**. `TerminalPhase::windows`,
/// the window of a single row, this function does not take at all: a row which has one goes to
/// [`fit_in_window`] instead, since its window replaces the rule rather than narrowing it.
/// The obligation is worth stating because failing it is silent: a caller which passes only
/// its own exclusions under `TerminalMethod::Manual` regresses the whole row, and a manual
/// window carries no test on the sign of the slope and so raises no flag to show for it.
///
/// With `TerminalPhase::relative_warning` the fit also carries `TERMINAL_LOW_CONCENTRATION`
/// where a regressed point lies far below the peak of the row, which never changes the window.
/// A row of no columns at all, and `TerminalMethod::LastN` without `n_points`, which
/// `TerminalPhase::validate` rejects, are rows without a window rather than a panic.
///
/// Mirrors `terminal_fit`, `_fit_by_method`, `_fit_selected` and `_collect` of
/// `src/pkpdutils/nca/terminal.py`.
pub(crate) fn terminal_fit(
    t: &[f64],
    c: &[f64],
    n_valid: usize,
    regressable: &[bool],
    tmax_idx: usize,
    phase: &TerminalPhase,
) -> TerminalFit {
    let columns = t.len().min(c.len());
    if columns == 0 {
        return TerminalFit::no_fit(NcaFlag::TOO_FEW_POINTS);
    }
    let (t, c) = (&t[..columns], &c[..columns]);
    let (points, stats) = row_statistics(t, c, n_valid, regressable);
    let first_allowed = if phase.exclude_cmax {
        tmax_idx.saturating_add(1)
    } else {
        0
    };
    let last = columns - 1;
    let chosen = match phase.method {
        // No slope test, no tie rule, no `first_allowed`: the mask is the window. The start
        // mirrors `_fit_selected`, where it cannot be told from a plain 0, since `collect`
        // snaps forward over the very mask this reads; the mirror stays so the two agree
        // whatever `collect` does later.
        TerminalMethod::Manual => chosen_from(&points),
        TerminalMethod::BestFit => best_fit(&stats, first_allowed, phase),
        TerminalMethod::LastN => last_n(&stats, first_allowed, last, phase),
        // always after the maximum, whatever `exclude_cmax` says
        TerminalMethod::AllAfterTmax => {
            let after = tmax_idx.saturating_add(1);
            with_slope_test(&stats, after.min(last), after < n_valid, phase)
        }
    };
    let mut fit = collect(t, &stats, &points, chosen, phase);
    if let Some(threshold) = phase.relative_warning {
        fit.flags |= low_concentration(t, c, n_valid, &points, &fit, threshold);
    }
    fit
}

/// The terminal regression of one packed row over the window the row was given.
///
/// `TerminalPhase::windows` names a window `(t_first, t_last)` per sample, and a row which has
/// one regresses the points of that window and nothing else: the rule of
/// [`TerminalMethod`] decides nothing for it, `exclude_cmax` does not apply, and neither the
/// sign of the slope nor the tie rule is tested, exactly as `TerminalMethod::Manual` is not
/// tested. The window is inclusive at both ends.
///
/// The arguments are those of [`terminal_fit`] and mean the same, `tmax_idx` excepted, which no
/// rule here reads. `regressable` carries the caller's own exclusions and **not** the manual
/// mask: a row with a window ignores `TerminalPhase::points` as it ignores the method.
///
/// The narrowing happens before the statistics, which is the whole of the function: python's
/// `_fit_selected` calls `window_statistics` again with `valid = selected`, so the origin of the
/// sums and the suffix sums themselves are over the selected points. A port which read the
/// statistics of the row and only moved the start would count the points behind the window and
/// report a window which runs to the last regressable point of the row instead of to `t_last`.
///
/// Mirrors `_fit_selected` of `src/pkpdutils/nca/terminal.py` as `terminal_fit` calls it for a
/// row whose window is given.
pub(crate) fn fit_in_window(
    t: &[f64],
    c: &[f64],
    n_valid: usize,
    regressable: &[bool],
    window: (f64, f64),
    phase: &TerminalPhase,
) -> TerminalFit {
    let columns = t.len().min(c.len());
    if columns == 0 {
        return TerminalFit::no_fit(NcaFlag::TOO_FEW_POINTS);
    }
    let (t, c) = (&t[..columns], &c[..columns]);
    // the selection, before any statistic is taken: a NaN time fails both comparisons
    let selected: Vec<bool> = (0..columns)
        .map(|j| {
            regressable.get(j).copied().unwrap_or(false) && t[j] >= window.0 && t[j] <= window.1
        })
        .collect();
    let (points, stats) = row_statistics(t, c, n_valid, &selected);
    let mut fit = collect(t, &stats, &points, chosen_from(&points), phase);
    if let Some(threshold) = phase.relative_warning {
        fit.flags |= low_concentration(t, c, n_valid, &points, &fit, threshold);
    }
    fit
}

/// The window of a mask: the first point it holds, and whether it holds one at all.
///
/// The rule of a mask, which is what `TerminalMethod::Manual` and a given window share: no
/// `first_allowed`, no test on the sign of the slope and no tie band, so the mask is the window.
fn chosen_from(points: &[bool]) -> Chosen {
    Chosen {
        start: points.iter().position(|&p| p).unwrap_or(0),
        has_fit: points.iter().any(|&p| p),
        flags: NcaFlag::NONE,
    }
}

/// The points of a row which may be regressed and every window of them.
///
/// `t` and `c` are the row cut to its common width. A point is regressed when it is a point of
/// the row, carries a positive value whose logarithm is finite, and the caller permits it.
fn row_statistics(
    t: &[f64],
    c: &[f64],
    n_valid: usize,
    regressable: &[bool],
) -> (Vec<bool>, Vec<WindowStats>) {
    let mut y = Vec::with_capacity(c.len());
    let mut points = Vec::with_capacity(c.len());
    for (j, &value) in c.iter().enumerate() {
        let logarithm = if j < n_valid && value > 0.0 {
            value.ln()
        } else {
            f64::NAN
        };
        points.push(logarithm.is_finite() && regressable.get(j).copied().unwrap_or(false));
        y.push(logarithm);
    }
    let mut stats = vec![WindowStats::NONE; c.len()];
    window_statistics(t, &y, &points, &mut stats);
    (points, stats)
}

/// What a selection rule decides: where the window starts, whether there is one, and the flag.
#[derive(Debug, Clone, Copy)]
struct Chosen {
    /// The index the window starts at, before it is snapped onto a regressable point.
    start: usize,
    /// Whether the rule found a window at all.
    has_fit: bool,
    /// `POSITIVE_SLOPE` where the rule found only rising windows.
    flags: NcaFlag,
}

/// The window of the largest adjusted R squared, preferring more points within the tolerance.
///
/// The band is **absolute**: every window within `tie_tolerance` below the row's best adjusted R
/// squared is in it, and the **earliest** start of the band wins, which is the window with the
/// most points. The best adjusted R squared only wins when it is also the earliest, so a strict
/// greater-than picks a different, shorter window on real data. `min_adj_r2` is not consulted
/// here, only in [`collect`].
fn best_fit(stats: &[WindowStats], first_allowed: usize, phase: &TerminalPhase) -> Chosen {
    let min_points = f64::from(phase.min_points);
    let mut best = f64::NEG_INFINITY;
    let mut any_window = false;
    let mut any_candidate = false;
    for window in stats.iter().skip(first_allowed) {
        // a NaN count fails the comparison, so a degenerate window is no candidate
        if window.n >= min_points {
            any_window = true;
            if window.slope < 0.0 {
                any_candidate = true;
                if window.r2_adj > best {
                    best = window.r2_adj;
                }
            }
        }
    }
    let edge = best - phase.tie_tolerance;
    let start = stats
        .iter()
        .enumerate()
        .skip(first_allowed)
        .find(|(_, w)| w.n >= min_points && w.slope < 0.0 && w.r2_adj >= edge)
        .map_or(0, |(s, _)| s);
    let flags = if any_window && !any_candidate {
        // windows with enough points exist, but the curve rises over all of them
        NcaFlag::POSITIVE_SLOPE
    } else {
        NcaFlag::NONE
    };
    Chosen {
        start,
        has_fit: any_candidate,
        flags,
    }
}

/// The last window which still holds `n_points` regressable points.
///
/// The start is the largest index whose window has them, which is the shortest such window, and
/// a degenerate window has no count at all, so the rule reaches past it to a longer one.
/// `exclude_cmax` may push the start past that index and leave fewer points than asked for.
fn last_n(
    stats: &[WindowStats],
    first_allowed: usize,
    last: usize,
    phase: &TerminalPhase,
) -> Chosen {
    let Some(n_points) = phase.n_points else {
        return Chosen {
            start: 0,
            has_fit: false,
            flags: NcaFlag::NONE,
        };
    };
    let wanted = f64::from(n_points);
    let found = stats.iter().rposition(|w| w.n >= wanted);
    let start = found.unwrap_or(0).max(first_allowed).min(last);
    with_slope_test(stats, start, found.is_some(), phase)
}

/// The point count and the slope test `LastN` and `AllAfterTmax` share.
fn with_slope_test(
    stats: &[WindowStats],
    start: usize,
    has_fit: bool,
    phase: &TerminalPhase,
) -> Chosen {
    let window = stats.get(start).copied().unwrap_or(WindowStats::NONE);
    let enough = has_fit && window.n >= f64::from(phase.min_points);
    let declining = window.slope < 0.0;
    let positive = enough && !declining;
    Chosen {
        start,
        has_fit: enough && !positive,
        flags: if positive {
            NcaFlag::POSITIVE_SLOPE
        } else {
            NcaFlag::NONE
        },
    }
}

/// Read the statistics of the chosen window and set the flags of the row.
///
/// The start is snapped forward onto the first regressable point at or after it, so `t_first`
/// always names a point of the regression, and the window ends at the last regressable point of
/// the row, which is not the time of the last measured value when the caller kept that value
/// out of the regression.
fn collect(
    t: &[f64],
    stats: &[WindowStats],
    regressable: &[bool],
    chosen: Chosen,
    phase: &TerminalPhase,
) -> TerminalFit {
    let Chosen {
        start,
        has_fit,
        flags,
    } = chosen;
    let (start, end) = match (start..regressable.len()).find(|&j| regressable[j]) {
        Some(first) => (
            first,
            (first..regressable.len())
                .rev()
                .find(|&j| regressable[j])
                .unwrap_or(first),
        ),
        None => (start, start),
    };
    let window = stats.get(start).copied().unwrap_or(WindowStats::NONE);
    let passes = match phase.min_adj_r2 {
        Some(minimum) => window.r2_adj >= minimum,
        None => true,
    };
    let has_fit = has_fit && window.n >= f64::from(phase.min_points) && passes;
    if !has_fit {
        // a failure which raised no flag of its own is reported as too few points,
        // whatever really went wrong, `min_adj_r2` included
        let flags = if flags.is_empty() {
            NcaFlag::TOO_FEW_POINTS
        } else {
            flags
        };
        return TerminalFit::no_fit(flags);
    }
    TerminalFit {
        slope: window.slope,
        intercept: window.intercept,
        r2: window.r2,
        r2_adj: window.r2_adj,
        se_slope: window.se_slope,
        n_points: window.n,
        t_first: t.get(start).copied().unwrap_or(f64::NAN),
        t_last: t.get(end).copied().unwrap_or(f64::NAN),
        flags,
    }
}

/// `TERMINAL_LOW_CONCENTRATION` where a regressed point falls below a fraction of the peak.
///
/// The peak is the largest value of the whole row, not of the window, and the comparison is
/// strict. The diagnostic never changes the selection; a row without a window has a `NaN`
/// `t_first`, which no time compares against, so it is never flagged.
fn low_concentration(
    t: &[f64],
    c: &[f64],
    n_valid: usize,
    regressable: &[bool],
    fit: &TerminalFit,
    threshold: f64,
) -> NcaFlag {
    let mut peak = 0.0_f64;
    for &value in c.iter().take(n_valid) {
        if value.is_finite() && value > peak {
            peak = value;
        }
    }
    let limit = threshold * peak;
    let low = t
        .iter()
        .zip(c)
        .zip(regressable)
        .any(|((&time, &value), &point)| {
            point && time >= fit.t_first && time <= fit.t_last && value < limit
        });
    if low {
        NcaFlag::TERMINAL_LOW_CONCENTRATION
    } else {
        NcaFlag::NONE
    }
}

/// One window the selection may choose from.
///
/// The table of them reaches a result as the point variables `candidate_t_first`,
/// `candidate_n_points` and `candidate_r2_adj` of a batch of one row ([`crate::assemble`]),
/// which is every column python's `candidate_variables` reads. Python's table carries the slope
/// of a window as well, which no variable of a result reports, so it is not here.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Candidate {
    /// The time of the first point of the window.
    pub start_time: f64,
    /// The regressable points of the window.
    pub n: f64,
    /// The adjusted coefficient of determination.
    pub r2_adj: f64,
}

/// Every window the terminal selection may choose from, in the order of their start.
///
/// The set is looser than the selection and differently drawn: it applies no test on the sign of
/// the slope and no `min_adj_r2`, so a rising window is in it and a row the selection refuses
/// still has a table, while it does require the window to start at a regressable point, which
/// the selection does not. The two agree all the same, since a window starting at a point which
/// cannot be regressed has the statistics of the next point which can and the selection snaps
/// its start forward onto it.
///
/// The arguments are those of [`terminal_fit`] and mean the same.
///
/// Mirrors `candidate_table` of `src/pkpdutils/nca/terminal.py`, without the row index of the
/// batch, which the caller holds, and without the slope, which no result reports.
pub(crate) fn candidate_table(
    t: &[f64],
    c: &[f64],
    n_valid: usize,
    regressable: &[bool],
    tmax_idx: usize,
    phase: &TerminalPhase,
) -> Vec<Candidate> {
    let columns = t.len().min(c.len());
    let (t, c) = (&t[..columns], &c[..columns]);
    let (points, stats) = row_statistics(t, c, n_valid, regressable);
    let first_allowed = if phase.exclude_cmax {
        tmax_idx.saturating_add(1)
    } else {
        0
    };
    let min_points = f64::from(phase.min_points);
    t.iter()
        .zip(&stats)
        .zip(&points)
        .skip(first_allowed)
        .filter_map(|((&start_time, window), &point)| {
            let candidate = point && window.n >= min_points && window.r2_adj.is_finite();
            candidate.then_some(Candidate {
                start_time,
                n: window.n,
                r2_adj: window.r2_adj,
            })
        })
        .collect()
}

/// The half-lives the window of `fit` covers, `(t_last - t_first) / thalf`.
///
/// `NaN` for a row without a window, and negative for a rising manual window, whose half-life is
/// negative. Reported as `lambda_z_span`.
pub(crate) fn span(fit: &TerminalFit) -> f64 {
    let thalf = std::f64::consts::LN_2 / -fit.slope;
    (fit.t_last - fit.t_first) / thalf
}

/// `NcaFlag::SPAN_LOW` when the window covers fewer than [`SPAN_MINIMUM`] half-lives.
///
/// The comparison is strict, so a window of exactly two half-lives is not low, and a `NaN` span
/// compares false and sets nothing. The flag belongs to the parameters of the row rather than to
/// the regression, so it is not part of [`TerminalFit::flags`]; the rule lives here because the
/// window it measures does.
pub(crate) fn span_flag(fit: &TerminalFit) -> NcaFlag {
    if span(fit) < SPAN_MINIMUM {
        NcaFlag::SPAN_LOW
    } else {
        NcaFlag::NONE
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn an_exact_fit_has_no_residual_in_any_window() {
        // the residual `syy_c - slope * sxy_c` is zero for a perfect fit, so
        // what survived the subtraction was the rounding of the sums and its
        // sign decided between 0.0 and the square root of that noise, around
        // 1e-9. Half of a batch therefore carried a standard error made of
        // nothing, and which half differed between this kernel and the python
        // core: the engine gate of `pkpdutils` failed on linux while passing
        // on the developer's machine for the same commit. Every window of the
        // row is checked, because the floor is written against the uncentred
        // `syy` precisely for the short ones late in the curve, whose sums
        // carry the whole drop while their difference is small.
        let times = [0.25_f64, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0];
        let regressable = [true; 8];
        let mut out = [WindowStats::NONE; 8];
        let mut checked = 0_usize;
        for step in 0..512 {
            let k = 0.05 + 0.45 * (step as f64) / 511.0;
            let y: Vec<f64> = times.iter().map(|t| (10.0 * (-k * t).exp()).ln()).collect();
            window_statistics(&times, &y, &regressable, &mut out);
            for (start, stats) in out.iter().enumerate() {
                if stats.n >= 3.0 {
                    assert_eq!(
                        stats.se_slope, 0.0,
                        "k {k} start {start}: se_slope {}",
                        stats.se_slope
                    );
                    assert_eq!(stats.r2, 1.0, "k {k} start {start}: r2 {}", stats.r2);
                    let lambda_z = -stats.slope;
                    assert!(
                        (lambda_z - k).abs() < 1e-12,
                        "k {k} start {start}: {lambda_z}"
                    );
                    checked += 1;
                }
            }
        }
        // columns 0 to 5 hold 8 down to 3 points, the last two are below three
        assert_eq!(checked, 512 * 6, "every window of every curve was examined");
    }

    #[test]
    fn a_noisy_fit_keeps_its_residual() {
        // the floor must only reach an exact fit: four samples a few percent
        // off sit far above it, so the standard error has to survive untouched
        let times = [1.0_f64, 2.0, 4.0, 8.0];
        let offsets = [1.03_f64, 0.97, 1.04, 0.96];
        let y: Vec<f64> = times
            .iter()
            .zip(offsets)
            .map(|(t, off)| (10.0 * (-0.2 * t).exp() * off).ln())
            .collect();
        let regressable = [true; 4];
        let mut out = [WindowStats::NONE; 4];
        window_statistics(&times, &y, &regressable, &mut out);
        assert_eq!(out[0].n, 4.0);
        assert!(out[0].se_slope > 1e-6, "se_slope {}", out[0].se_slope);
        assert!(out[0].r2 < 1.0, "r2 {}", out[0].r2);
    }
    use super::*;

    use pkpdutils_abi::TerminalMethod;

    /// A tolerance of a last place or two against the doubles the python kernel prints,
    /// four orders tighter than the 1e-8 of the benchmark gate: the formulas and their
    /// order are reproduced exactly, only `ln` may round differently in the two libms.
    #[track_caller]
    fn close(actual: f64, expected: f64) {
        let tolerance = 1e-12 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} is not {expected}"
        );
    }

    /// The index of the first maximum over the valid points, as `nca.py` computes it for
    /// the row and hands it to the regression.
    fn tmax_index(c: &[f64], n_valid: usize) -> usize {
        let mut peak = f64::NEG_INFINITY;
        let mut index = 0;
        for (j, &value) in c.iter().take(n_valid).enumerate() {
            if value > peak {
                peak = value;
                index = j;
            }
        }
        index
    }

    /// The fit of one whole row, with every point permitted into the regression.
    fn fit_of(t: &[f64], c: &[f64], phase: &TerminalPhase) -> TerminalFit {
        let permitted = vec![true; t.len()];
        terminal_fit(t, c, t.len(), &permitted, tmax_index(c, t.len()), phase)
    }

    /// The statistics of every window of one whole row.
    fn stats_of(t: &[f64], c: &[f64]) -> Vec<WindowStats> {
        let y: Vec<f64> = c
            .iter()
            .map(|&v| if v > 0.0 { v.ln() } else { f64::NAN })
            .collect();
        let regressable: Vec<bool> = y.iter().map(|v| v.is_finite()).collect();
        let mut out = vec![WindowStats::NONE; t.len()];
        window_statistics(t, &y, &regressable, &mut out);
        out
    }

    /// `TerminalPhase::default()` with `method` replaced.
    fn method(method: TerminalMethod) -> TerminalPhase {
        TerminalPhase {
            method,
            ..TerminalPhase::default()
        }
    }

    #[test]
    fn a_noise_free_exponential_recovers_its_rate_exactly() {
        // c = 10 exp(-0.2 t) sampled to t = 48. Every window of the declining curve
        // lies on the line, so the whole band ties at an adjusted R squared of 1 and
        // the earliest start after the maximum wins: the eight points from t = 1 on.
        let t = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 36.0, 48.0];
        let c: Vec<f64> = t.iter().map(|&x| 10.0 * (-0.2_f64 * x).exp()).collect();
        let fit = fit_of(&t, &c, &TerminalPhase::default());

        assert_eq!(fit.n_points, 8.0);
        assert_eq!(fit.t_first, 1.0);
        assert_eq!(fit.t_last, 48.0);
        assert_eq!(fit.flags, NcaFlag::NONE);
        let lambda_z = -fit.slope;
        assert!((lambda_z - 0.2).abs() < 1e-9, "lambda_z {lambda_z}");
        let thalf = std::f64::consts::LN_2 / lambda_z;
        assert!(
            (thalf - std::f64::consts::LN_2 / 0.2).abs() < 1e-9,
            "thalf {thalf}"
        );
        // the doubles the python kernel prints for this row
        close(fit.slope, -0.199_999_999_999_999_98);
        close(fit.intercept, 2.302_585_092_994_045);
        close(fit.r2, 1.0);
        close(fit.r2_adj, 1.0);
        // an exact fit has no residual at all, not a small one: `RESIDUAL_FLOOR`
        // takes the rounding of the sums to zero, so this is exactly 0.0 on
        // every platform and agrees with the python core bit for bit
        assert_eq!(fit.se_slope, 0.0, "no residual: {}", fit.se_slope);

        // terminal.md section 1 step 1 and hazard 5: the sums are taken around the
        // first valid point of the row, so the same curve measured from t = 1e8 gives
        // the same slope. Raw moments of 1e8 would cancel to noise instead.
        let shifted: Vec<f64> = t.iter().map(|x| x + 1e8).collect();
        let far = fit_of(&shifted, &c, &TerminalPhase::default());
        assert_eq!(far.slope, fit.slope, "the origin shift makes the two equal");
        assert_eq!(far.t_first, 1e8 + 1.0);
        close(far.intercept, 20_000_002.302_585_09);
    }

    #[test]
    fn the_tie_band_prefers_the_window_with_more_points() {
        // ln c = 5, 4.51, 4, 3.5, 3 at t = 0..4. The three points from t = 2 lie on a
        // line, so their window holds the row's best adjusted R squared, 1.0. The four
        // points from t = 1 reach 0.999964429, which is 3.56e-5 below it and so inside
        // the absolute band of 1e-4. The earliest start in the band wins, which is the
        // longer window. A strict greater-than on the adjusted R squared would take the
        // window at t = 2 instead: three points and a slope of -0.5.
        let t = [0.0, 1.0, 2.0, 3.0, 4.0];
        let c: Vec<f64> = [5.0_f64, 4.51, 4.0, 3.5, 3.0]
            .iter()
            .map(|y| y.exp())
            .collect();
        let stats = stats_of(&t, &c);
        close(stats[1].r2_adj, 0.999_964_428_986_423);
        close(stats[2].r2_adj, 1.0); // the shorter window fits exactly

        let fit = fit_of(&t, &c, &TerminalPhase::default());
        assert_eq!(fit.n_points, 4.0);
        assert_eq!(fit.t_first, 1.0);
        assert_eq!(fit.t_last, 4.0);
        close(fit.slope, -0.502_999_999_999_999_8);
        close(fit.intercept, 5.01);
        assert_eq!(fit.flags, NcaFlag::NONE);
    }

    #[test]
    fn the_tie_band_is_absolute_and_not_relative() {
        // The band is `r2_adj >= best - tie_tolerance`, never `best * (1 - tolerance)`.
        // Near an adjusted R squared of 1 the two agree, so this row fits badly and the
        // tolerance is wide: ln c = 3, 1.7, 2, 1, 0.9 at t = 0..4 with tie_tolerance
        // 0.15. The best window is the three points from t = 2 at 0.63513514; the four
        // points from t = 1 reach 0.50813953. The absolute band reaches down to
        // 0.48513514 and holds the longer window, a relative band would stop at
        // 0.85 * 0.63513514 = 0.53986486 and take the shorter one.
        let t = [0.0, 1.0, 2.0, 3.0, 4.0];
        let c: Vec<f64> = [3.0_f64, 1.7, 2.0, 1.0, 0.9]
            .iter()
            .map(|y| y.exp())
            .collect();
        let stats = stats_of(&t, &c);
        close(stats[1].r2_adj, 0.508_139_534_883_720_8);
        close(stats[2].r2_adj, 0.635_135_135_135_133_8);

        let phase = TerminalPhase {
            tie_tolerance: 0.15,
            ..TerminalPhase::default()
        };
        let fit = fit_of(&t, &c, &phase);
        assert_eq!(fit.n_points, 4.0);
        assert_eq!(fit.t_first, 1.0);
        close(fit.slope, -0.340_000_000_000_000_2);
        assert_eq!(fit.flags, NcaFlag::NONE);
    }

    #[test]
    fn a_degenerate_window_is_not_a_window() {
        // terminal.md section 1 step 8: `enough = (n >= 3) & (sxx_c > 0) & (syy_c > 0)`
        // and every statistic, `n` among them, is NaN where it does not hold. So a
        // window of constant values or of one repeated time satisfies no `n >= k` test.
        let constant = stats_of(&[0.0, 1.0, 2.0, 3.0], &[5.0, 5.0, 5.0, 5.0]);
        let repeated = stats_of(&[2.0, 2.0, 2.0, 2.0], &[8.0, 4.0, 2.0, 1.0]);
        for (name, row) in [("constant values", constant), ("repeated times", repeated)] {
            for (s, window) in row.iter().enumerate() {
                assert!(window.n.is_nan(), "{name} n at {s}");
                assert!(window.slope.is_nan(), "{name} slope at {s}");
                assert!(window.intercept.is_nan(), "{name} intercept at {s}");
                assert!(window.r2.is_nan(), "{name} r2 at {s}");
                assert!(window.r2_adj.is_nan(), "{name} r2_adj at {s}");
                assert!(window.se_slope.is_nan(), "{name} se_slope at {s}");
                let counts_three_points = window.n >= 3.0;
                assert!(!counts_three_points, "{name} is a window at {s}");
            }
        }

        let fit = fit_of(
            &[0.0, 1.0, 2.0, 3.0],
            &[5.0, 5.0, 5.0, 5.0],
            &TerminalPhase::default(),
        );
        assert!(fit.t_first.is_nan());
        assert_eq!(fit.flags, NcaFlag::TOO_FEW_POINTS);
    }

    #[test]
    fn min_adj_r2_rejects_with_too_few_points_and_not_positive_slope() {
        // terminal.md hazard 2: `min_adj_r2` takes no part in the selection, it only
        // gates the window the selection chose, and a window it rejects leaves the row
        // with TOO_FEW_POINTS, never POSITIVE_SLOPE.
        let t = [0.0, 1.0, 2.0, 3.0, 4.0];
        let c: Vec<f64> = [5.0_f64, 4.0, 2.0, 3.0, 1.0]
            .iter()
            .map(|y| y.exp())
            .collect();

        let fit = fit_of(&t, &c, &TerminalPhase::default());
        assert_eq!(fit.t_first, 1.0, "without a threshold the window stands");
        close(fit.slope, -0.8);
        close(fit.r2_adj, 0.459_999_999_999_999_96);

        let phase = TerminalPhase {
            min_adj_r2: Some(0.9),
            ..TerminalPhase::default()
        };
        let rejected = fit_of(&t, &c, &phase);
        assert!(rejected.t_first.is_nan());
        assert!(rejected.slope.is_nan());
        assert_eq!(rejected.flags, NcaFlag::TOO_FEW_POINTS);
        assert!(!rejected.flags.contains(NcaFlag::POSITIVE_SLOPE));

        // and it does not fall back to a window which would pass: the row of the
        // absolute band has a window at 0.63513514, above the threshold of 0.6, but the
        // selection chose the longer one at 0.50813953 and the threshold ends there.
        let banded: Vec<f64> = [3.0_f64, 1.7, 2.0, 1.0, 0.9]
            .iter()
            .map(|y| y.exp())
            .collect();
        let phase = TerminalPhase {
            tie_tolerance: 0.15,
            min_adj_r2: Some(0.6),
            ..TerminalPhase::default()
        };
        let no_fallback = fit_of(&t, &banded, &phase);
        assert!(no_fallback.t_first.is_nan());
        assert_eq!(no_fallback.flags, NcaFlag::TOO_FEW_POINTS);
    }

    #[test]
    fn a_manual_window_with_a_rising_slope_reports_a_negative_rate_and_no_flag() {
        // terminal.md section 9 hazard 3, deliberately mirrored and raised with the
        // maintainer: a manual window applies no slope test, so a rising curve gives a
        // positive slope, a negative lambda_z and a negative half-life without a flag,
        // where BEST_FIT refuses the row.
        let t = [0.0, 1.0, 2.0, 3.0];
        let c: Vec<f64> = [1.0_f64, 2.0, 3.0, 4.0].iter().map(|y| y.exp()).collect();

        let fit = fit_of(&t, &c, &method(TerminalMethod::Manual));
        assert_eq!(fit.n_points, 4.0);
        assert_eq!(fit.t_first, 0.0);
        assert_eq!(fit.t_last, 3.0);
        close(fit.slope, 1.0);
        close(fit.intercept, 1.0);
        assert_eq!(fit.flags, NcaFlag::NONE, "no POSITIVE_SLOPE, no TOO_FEW");
        assert!(-fit.slope < 0.0, "lambda_z is negative");
        assert!(
            std::f64::consts::LN_2 / -fit.slope < 0.0,
            "thalf is negative"
        );
        assert_eq!(span_flag(&fit), NcaFlag::SPAN_LOW, "a negative span is low");

        // the same row under BEST_FIT, which does test the slope
        let refused = fit_of(
            &t,
            &c,
            &TerminalPhase {
                exclude_cmax: false,
                ..TerminalPhase::default()
            },
        );
        assert!(refused.t_first.is_nan());
        assert_eq!(refused.flags, NcaFlag::POSITIVE_SLOPE);
    }

    #[test]
    fn a_manual_mask_narrower_than_the_row_regresses_only_its_points() {
        // The manual window is the mask and nothing else: `TerminalPhase::points` is resolved
        // by the caller, which folds it into `regressable`, so the regression has to follow
        // the mask and not the row. The first two samples lie far above the terminal line
        // here, so a window which took them in would be a different fit altogether.
        let t = [0.0, 1.0, 2.0, 3.0, 4.0];
        let c = [100.0, 50.0, 8.0, 4.0, 2.0];
        let masked = [false, false, true, true, true];
        let phase = method(TerminalMethod::Manual);
        let fit = terminal_fit(&t, &c, 5, &masked, 0, &phase);

        assert_eq!(fit.n_points, 3.0);
        assert_eq!(fit.t_first, 2.0);
        assert_eq!(fit.t_last, 4.0);
        close(fit.slope, -0.693_147_180_559_945_4);
        close(fit.intercept, 3.465_735_902_799_727);
        close(fit.r2, 1.0);
        assert_eq!(fit.flags, NcaFlag::NONE);

        // the same row with every point permitted, which is what a caller that forgot to fold
        // the mask in would get: five points from t = 0 and a slope half again as steep
        let whole = fit_of(&t, &c, &phase);
        assert_eq!(whole.n_points, 5.0);
        assert_eq!(whole.t_first, 0.0);
        close(whole.slope, -1.034_977_465_516_455_2);
    }

    #[test]
    fn all_after_tmax_ignores_exclude_cmax() {
        // terminal.md section 2: ALL_AFTER_TMAX starts at tmax_idx + 1 whatever
        // `exclude_cmax` says. The maximum sits at index 1 here and index 0 is a
        // regressable point of a five point window, so a rule which read `exclude_cmax`
        // would start at 0 and report five points from t = 0.
        let t = [0.0, 1.0, 2.0, 3.0, 4.0];
        let c = [1.0, 10.0, 8.0, 6.0, 4.0];
        assert_eq!(tmax_index(&c, c.len()), 1);

        for exclude_cmax in [true, false] {
            let phase = TerminalPhase {
                exclude_cmax,
                ..method(TerminalMethod::AllAfterTmax)
            };
            let fit = fit_of(&t, &c, &phase);
            assert_eq!(fit.n_points, 3.0);
            assert_eq!(fit.t_first, 2.0);
            assert_eq!(fit.t_last, 4.0);
            close(fit.slope, -0.346_573_590_279_972_7);
            assert_eq!(fit.flags, NcaFlag::NONE);
        }
    }

    #[test]
    fn last_n_takes_the_longest_window_that_has_the_points() {
        // terminal.md section 2: the start is the LARGEST index whose window still
        // holds n_points, and `n` is NaN for a degenerate window. The last three
        // samples share a time here, so that window has no `n` at all and the rule
        // falls through to the longer one. A port which simply counts back three
        // regressable points lands on the degenerate window and reports nothing.
        let t = [0.0, 1.0, 2.0, 2.0, 2.0];
        let c = [20.0, 10.0, 5.0, 4.0, 3.0];
        assert!(stats_of(&t, &c)[2].n.is_nan(), "the three t = 2 points");

        let phase = TerminalPhase {
            n_points: Some(3),
            ..method(TerminalMethod::LastN)
        };
        let fit = fit_of(&t, &c, &phase);
        assert_eq!(fit.n_points, 4.0);
        assert_eq!(fit.t_first, 1.0);
        assert_eq!(fit.t_last, 2.0);
        close(fit.slope, -0.937_803_572_253_346_3);
        assert_eq!(fit.flags, NcaFlag::NONE);

        // on a row without a degenerate window it takes exactly n_points
        let clean = [20.0, 10.0, 5.0, 2.5, 1.25];
        let fit = fit_of(&[0.0, 1.0, 2.0, 3.0, 4.0], &clean, &phase);
        assert_eq!(fit.n_points, 3.0);
        assert_eq!(fit.t_first, 2.0);

        // and `exclude_cmax` may cut the window below n_points: the only window of
        // five points starts at 0, the maximum sits at 1, so three points are left
        let phase = TerminalPhase {
            n_points: Some(5),
            ..method(TerminalMethod::LastN)
        };
        let cut = fit_of(
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            &[1.0, 10.0, 8.0, 6.0, 4.0],
            &phase,
        );
        assert_eq!(cut.n_points, 3.0);
        assert_eq!(cut.t_first, 2.0);
    }

    #[test]
    fn the_window_runs_to_the_last_regressable_point_and_not_to_tlast() {
        // terminal.md hazard 7: a window ends at the last point of the row which may be
        // regressed, so a last sample the caller keeps out (a value below the limit a
        // BLQ rule kept) shortens it, although it is a positive measured value.
        let t = [0.0, 1.0, 2.0, 3.0, 4.0];
        let c = [20.0, 10.0, 5.0, 2.5, 1.25];
        let permitted = [true, true, true, true, false];
        let fit = terminal_fit(&t, &c, 5, &permitted, 0, &TerminalPhase::default());

        assert_eq!(fit.n_points, 3.0);
        assert_eq!(fit.t_first, 1.0);
        assert_eq!(fit.t_last, 3.0, "not 4.0, the time of the last valid point");
        close(fit.slope, -0.693_147_180_559_946_3);
        close(fit.r2, 1.0);
        assert_eq!(fit.flags, NcaFlag::NONE);
    }

    #[test]
    fn a_zero_inside_a_window_is_skipped_and_not_an_error() {
        // terminal.md section 0: `y = ln c` only where `c > 0`, so a zero or a negative
        // value in the middle of a window is skipped. It is not counted, it does not
        // end the window and it never becomes an infinity in the sums.
        let t = [0.0, 1.0, 2.0, 3.0, 4.0];
        for bad in [0.0, -1.0] {
            let c = [20.0, 10.0, bad, 2.5, 1.25];
            let fit = fit_of(&t, &c, &TerminalPhase::default());
            assert_eq!(fit.n_points, 3.0, "the bad point is not counted");
            assert_eq!(fit.t_first, 1.0);
            assert_eq!(fit.t_last, 4.0, "the window reaches over the gap");
            close(fit.slope, -0.693_147_180_559_944_7);
            assert!(fit.slope.is_finite(), "no infinity reached the sums");
            assert_eq!(fit.flags, NcaFlag::NONE);
        }
    }

    #[test]
    fn the_span_flag_is_strict_and_a_nan_span_sets_nothing() {
        // `span = (t_last - t_first) / thalf` with `thalf = ln 2 / -slope`, and
        // SPAN_LOW is set strictly below SPAN_MINIMUM, so a span of exactly two
        // half-lives is not low and a row without a window sets nothing.
        let two = TerminalFit {
            slope: -std::f64::consts::LN_2,
            t_first: 0.0,
            t_last: 2.0,
            ..TerminalFit::no_fit(NcaFlag::NONE)
        };
        assert_eq!(span(&two), 2.0);
        assert_eq!(span_flag(&two), NcaFlag::NONE, "two half-lives are enough");

        let just_below = TerminalFit {
            t_last: 1.999_999_999_999_999_8,
            ..two
        };
        assert!(span(&just_below) < SPAN_MINIMUM);
        assert_eq!(span_flag(&just_below), NcaFlag::SPAN_LOW);

        let none = TerminalFit::no_fit(NcaFlag::TOO_FEW_POINTS);
        assert!(span(&none).is_nan());
        assert_eq!(span_flag(&none), NcaFlag::NONE, "a NaN span sets nothing");

        // and a real row: the three points after the maximum cover one half-life
        let short = fit_of(
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            &[1.0, 10.0, 8.0, 6.0, 4.0],
            &method(TerminalMethod::AllAfterTmax),
        );
        close(span(&short), 1.000_000_000_000_000_2);
        assert_eq!(span_flag(&short), NcaFlag::SPAN_LOW);
    }

    #[test]
    fn the_candidate_table_is_looser_than_the_selection() {
        // terminal.md section 3: the table holds every window which starts at a
        // regressable point, lies after the maximum when `exclude_cmax` asks for it and
        // holds min_points points. It applies no slope test and no `min_adj_r2`, so a
        // rising window is in it and a row the selection refuses still has one.
        let t = [0.0, 1.0, 2.0, 3.0, 4.0];
        let c = [1.0, 10.0, 8.0, 6.0, 4.0];
        let phase = TerminalPhase {
            exclude_cmax: false,
            min_adj_r2: Some(0.99),
            ..TerminalPhase::default()
        };
        let permitted = vec![true; t.len()];
        let table = candidate_table(&t, &c, t.len(), &permitted, 1, &phase);

        assert_eq!(table.len(), 3);
        // the window of every point is in the table although the curve rises into its maximum
        // at t = 1, which no selection rule would regress
        assert_eq!(table[0].start_time, 0.0);
        assert_eq!(table[0].n, 5.0);
        close(table[0].r2_adj, -0.128_334_840_056_201_88);
        assert_eq!(table[1].start_time, 1.0);
        assert_eq!(table[1].n, 4.0);
        assert_eq!(table[2].start_time, 2.0);
        assert_eq!(table[2].n, 3.0);
        for candidate in &table {
            assert!(candidate.r2_adj < 0.99, "none of them passes min_adj_r2");
        }
        let fit = fit_of(&t, &c, &phase);
        assert!(fit.t_first.is_nan(), "while the selection refuses the row");

        // and a start which cannot be regressed is left out, although the selection
        // counts it: the zero at index 1 carries the statistics of index 2, and the
        // chosen start is snapped forward to the same point
        let c = [20.0, 0.0, 10.0, 5.0, 2.5];
        let table = candidate_table(&t, &c, t.len(), &permitted, 0, &TerminalPhase::default());
        assert_eq!(table.len(), 1);
        assert_eq!(table[0].start_time, 2.0);
        assert_eq!(table[0].n, 3.0);

        let fit = fit_of(&t, &c, &TerminalPhase::default());
        assert_eq!(fit.t_first, 2.0);
        assert_eq!(fit.n_points, 3.0);
    }

    #[test]
    fn a_row_of_two_valid_points_has_no_window() {
        // A window needs three points, so a row of two has none at all: every
        // statistic is NaN, the window has no first and no last time, and no flag
        // claims a window was found.
        let t = [0.0, 1.0, f64::NAN, f64::NAN];
        let c = [10.0, 5.0, f64::NAN, f64::NAN];
        let permitted = vec![true; t.len()];
        let fit = terminal_fit(&t, &c, 2, &permitted, 0, &TerminalPhase::default());

        assert!(fit.slope.is_nan());
        assert!(fit.intercept.is_nan());
        assert!(fit.r2.is_nan());
        assert!(fit.r2_adj.is_nan());
        assert!(fit.se_slope.is_nan());
        assert!(fit.n_points.is_nan());
        assert!(fit.t_first.is_nan());
        assert!(fit.t_last.is_nan());
        assert_eq!(fit.flags, NcaFlag::TOO_FEW_POINTS);
        assert!(!fit.flags.contains(NcaFlag::POSITIVE_SLOPE));
        assert_eq!(span_flag(&fit), NcaFlag::NONE);
    }

    #[test]
    fn the_low_concentration_warning_measures_against_the_peak_of_the_whole_row() {
        // terminal.md hazard 10: the warning compares a regressed point against
        // `relative_warning` times the peak of the WHOLE row, strictly below. The peak
        // is 10 here and lies outside the window, whose own largest value is 4, so a
        // rule which measured against the window would miss the point at 0.5.
        let t = [0.0, 1.0, 2.0, 3.0];
        let phase = TerminalPhase {
            relative_warning: Some(0.1),
            ..TerminalPhase::default()
        };
        let low = fit_of(&t, &[10.0, 4.0, 2.0, 0.5], &phase);
        assert_eq!(low.t_first, 1.0, "the window is found all the same");
        assert_eq!(low.flags, NcaFlag::TERMINAL_LOW_CONCENTRATION);

        // exactly at the threshold is not below it
        let at = fit_of(&t, &[10.0, 4.0, 2.0, 1.0], &phase);
        assert_eq!(at.t_first, 1.0);
        assert_eq!(at.flags, NcaFlag::NONE);
    }
}
