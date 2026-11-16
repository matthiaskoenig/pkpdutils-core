# Terminal phase kernel spec (extracted from Python, read-only)

Files: T = src/pkpdutils/nca/terminal.py, O = src/pkpdutils/nca/options.py, N = src/pkpdutils/nca/nca.py.
All of T works on "packed" rows: valid (finite t AND finite c) points moved to the front, stable order (auc.py:36-57). tp/cp are the packed arrays WITHOUT the inserted dose-time point (the c0 / zero point lives only in tp_area/cp_area, N:~735-760, and never enters the regression or tmax). Times are already relative to the dose (N:664-668).

## 0. Setup in terminal_fit (T:160-168) and the caller (N:~672-790)

- `in_row[j] = j < n_valid` (T:162). `y = ln(cp)` where `in_row & cp > 0`, else NaN (T:163-164).
- `regressable = in_row & isfinite(y) [& ~exclude]` (T:165-167). So zero, negative, NaN values are never regressable. A zero in the middle of a window is skipped, not an error.
- `exclude` (N:~675-690): (a) BLQ-kept/imputed points (`blq_packed`) unless `rules.terminal_regression` is true (N:~675); (b) IV_INFUSION with dose_duration not None: `tp <= nan_to_num(dose_duration, nan=0.0)` i.e. every sample at or before the end of infusion is excluded; a row with NaN duration excludes `tp <= 0`. Applies to ALL methods including MANUAL and per-row windows, because it is folded into `regressable`.
- `tmax_idx = argmax` over `where(in_row, cp, -inf)` along the packed row, FIRST occurrence on ties (N:~697-698). It is computed on all valid points (incl. zeros/BLQ-imputed), not on regressable ones, and on cp without the inserted dose-time point.
- Order of precedence in terminal_fit: per-row `windows` (given if both entries finite, T:170-174) -> `_fit_selected` on `regressable & tp>=t_first & tp<=t_last` (inclusive, T:179-181); rows without a window follow `phase.method` (merge T:182-193, T:266-288). exclude_cmax, min-slope sign, tie rule do NOT apply to given windows.

## 1. window_statistics (T:73-123)

Inputs x (times, N x n), y (ln c), valid (regressable). Output arrays N x n; column s = window "s..end of row" = all VALID points at index >= s (suffix of the packed row; windows always end at the last valid point of the row, never earlier).

Candidates/limits (this function): every start index s in 0..n-1 gets stats; a start s that is itself not valid gets exactly the stats of the next valid point (suffix sums unchanged). Minimum 3 points enforced by `enough` below (hard floor, independent of phase.min_points which must be >= 3 anyway, O:462).

Arithmetic, exactly (T:88-123):
1. Origin shift (T:90-97): `first = argmax(valid)` per row (index of first valid point of the ROW, not of the window; 0 if none), `x_origin = x[first]`, `y_origin = y[first]` (if there are zero columns: zeros). `x0 = where(valid, x - x_origin, 0)`, `y0 = where(valid, y - y_origin, 0)`.
2. Suffix sums `S(a)[s] = sum_{j>=s} a[j]`, computed as `cumsum(a[::-1])[::-1]` i.e. accumulated from the LAST column towards the first (T:68-70). Reproduce this summation order for bit-closeness.
   `n = S(valid as 0/1 f64)`, `sx = S(x0)`, `sy = S(y0)`, `sxx = S(x0*x0)`, `sxy = S(x0*y0)`, `syy = S(y0*y0)`.
3. Centered: `sxx_c = sxx - sx*sx/n`, `sxy_c = sxy - sx*sy/n`, `syy_c = syy - sy*sy/n` (T:105-107; division by n=0 gives inf/NaN, errors suppressed).
4. `slope = sxy_c / sxx_c`; `intercept = (sy - slope*sx)/n + y_origin - slope*x_origin` (intercept at original t=0, T:108-109); `ss_res = syy_c - slope*sxy_c` (T:110).
5. `r2 = clip(1 - ss_res/syy_c, 0, 1)` (T:111).
6. `r2_adj = 1 - (1 - r2) * (n - 1) / (n - 2)` (T:112): df = n-2, uses the CLIPPED r2, NOT clipped itself (can be negative).
7. `se_slope = sqrt(max(ss_res, 0) / (n - 2) / sxx_c)` (T:113).
8. `enough = (n >= 3) & (sxx_c > 0) & (syy_c > 0)` (T:114). Every output (`n, slope, intercept, r2, r2_adj, se_slope`) is NaN where `!enough` (T:116-123). Consequence: a window with identical ln values (syy_c = 0, e.g. constant concentrations) or identical times (sxx_c = 0) is NOT a window at all, `n` is NaN too, so it fails every `n >= k` test.

## 2. Selection rules (T:318-395, `_fit_by_method`); common pieces

- `first_allowed = tmax_idx + 1` if `exclude_cmax` else 0 (T:355). Index in PACKED space, tmax_idx = first max.
- Defaults: method BEST_FIT, min_points 3 (>=3), exclude_cmax True, n_points None (>=3), min_adj_r2 None, **tie_tolerance 1e-4 (>=0)**, relative_warning 1e-8 (0<x<1), windows None, keep_candidates False (O:459-467).
- Final gate in `_collect` (T:398-454) applies to every rule (see 4).

### BEST_FIT (T:357-372)
```
count_ok  = (s >= first_allowed) & (n[s] >= min_points)          # NaN n -> false
candidate = count_ok & (slope[s] < 0)                            # strict <
r2a       = candidate ? r2_adj[s] : -inf
best      = max_s r2a                                            # per row
within    = candidate & (r2a >= best - tie_tolerance)            # ABSOLUTE tolerance, inclusive >=
start     = first (smallest) s with within    ; 0 if none        # argmax of bool = first True
has_candidate = any(candidate)
flags     = POSITIVE_SLOPE if (!has_candidate && any(count_ok)) else 0
```
Tolerance: absolute (not relative) on adjusted R2, measured against the row's global best r2_adj among candidates (not chained, not against the neighbour). Winner = the SMALLEST start index inside the band = the window with the MOST points (n is non-increasing in s), NOT the one with the highest r2_adj (the best itself is only chosen when it is the earliest in the band). Strict `>` argmax is wrong. Note `count_ok`/`candidate` do NOT require `valid[s]`: a window starting at a non-valid index has the same stats as the next valid point, ties exactly (diff 0), and being earlier it wins the argmax; `_collect` then snaps start forward (section 4) so the reported window is the same. A port that only considers valid starts gets the same final result EXCEPT that the earliest-in-band logic must be evaluated on the band over valid starts (identical outcome since stats are equal and snapping goes forward to the same point). `min_adj_r2` does NOT take part in selection (T:360-361), only gates acceptance in `_collect`.

### LAST_N (T:374-384, 389-395)
```
enough_n = n[s] >= n_points                                      # NaN false
start    = enough_n.any ? (n_cols-1 - argmax(reverse(enough_n))) : 0   # LARGEST s with n[s] >= n_points
start    = clip(max(start, first_allowed), 0, n_cols-1)          # exclude_cmax can shrink the window below n_points
has_fit0 = enough_n.any
```
Then common tail: `n_at_start = n[start]`, `slope_at_start = slope[start]`; `enough = has_fit0 & (n_at_start >= min_points)`; `positive = enough & !(slope_at_start < 0)` (NaN slope also counts as positive); `flags = POSITIVE_SLOPE where positive`; fit exists iff `enough & !positive`. Trap: because `n` is NaN for degenerate windows (section 1.8), the "largest s with n >= n_points" can skip a degenerate window of exactly n_points and select a longer one. Without exclude_cmax, the window may reach before tmax.

### ALL_AFTER_TMAX (T:385-395)
`start = tmax_idx + 1` (ALWAYS, ignores exclude_cmax); `has_fit0 = start < n_valid` (n_valid counts all valid points incl. non-positive); `start = clip(start, 0, n_cols-1)`; same tail as LAST_N (min_points via n at start, positive-slope flag, no tie tolerance, no R2 comparison).

### MANUAL (T:348-351)
`_fit_selected(tp, y, regressable & manual_mask)`; points are original-order indices mapped to packed positions by N:~772-778 (`packed_mask`). Raises ValueError if manual_mask is None. exclude_cmax is NOT applied to the regression (but candidate_table does apply it, T:243-245). No slope check (see 8).

### Per-row windows (T:179-181, 291-315)
`_fit_selected`: stats over exactly the selected points (`y` zeroed outside, valid = selected), `start = first selected index` (0 if none), `has_fit0 = any(selected)`. No slope check, no tie rule.

## 3. keep_candidates / candidate_table (T:205-207, 210-263; N:931-969)

Only computed if `phase.keep_candidates`; returned in `TerminalFit.candidates` (pandas frame). Candidate set (T:247-253): `regressable[s] & s >= first_allowed (exclude_cmax-aware, even for MANUAL/ALL_AFTER_TMAX/LAST_N) & n[s] >= min_points & isfinite(r2_adj[s])`. NO slope<0 filter, NO min_adj_r2 filter, NO tie-band filter, and valid-start-only (unlike BEST_FIT's count_ok). Rows ordered by (row, start index). Columns: `row` int index of the batch row, `start_time` = tp[row, s], `n` int count of regressable points in the suffix window, `r2_adj`, `slope` (raw slope of ln c, i.e. -lambda_z, may be >= 0).
Exposed in the result only when the batch has exactly ONE row and the table is non-empty (N:954-969): `candidate_t_first = start_time`, `candidate_n_points = n` (as f64), `candidate_r2_adj = r2_adj`, each shape (1, K) over dim `candidate`; `row` and `slope` are dropped. Otherwise no candidate variables.

## 4. _collect: chosen window -> reported quantities (T:398-454)

Given `start` (index), `has_fit`, flags:
1. Snap forward (T:418-424): `at_or_after = regressable & (j >= start)`; if any: `start = first such j`, `end = last regressable index in the whole row` (last True of at_or_after); else start unchanged, `end = start`.
2. `n_points = n[start]`. `has_fit &= n_points >= min_points` (NaN -> false) (T:432). If `min_adj_r2` not None: `has_fit &= r2_adj[start] >= min_adj_r2` (T:433-434), inclusive.
3. Reported (NaN where !has_fit): `slope = slope[start]`, `intercept`, `r2`, `r2_adj`, `se_slope`, `n_points = n[start]`, `t_first = tp[start]`, `t_last = tp[end]` (time of the LAST REGRESSABLE point of the row, which can differ from tlast/clast when the last point is BLQ-excluded; windows always run to it), `start = start if has_fit else -1` (T:443-454).
4. Flags: `too_few = !has_fit & (flags == 0)`; `flags |= TOO_FEW_POINTS where too_few` (T:441-442). So a failure by `min_adj_r2` or `min_points` with no prior flag is reported as TOO_FEW_POINTS (flag text says "fewer than min_points"), while a positive slope leaves POSITIVE_SLOPE only.
Derived in N:~793-838 (all plain float64, errors suppressed so division gives inf/NaN):
- `lambda_z = -slope`; `thalf = ln(2)/lambda_z` (N:793-795).
- `span = (t_last - t_first) / thalf` (N:~833) (half-lives covered); reported as `lambda_z_span`; `lambda_z_t_first/t_last/r2/r2_adj/n_points/intercept` are fit.t_first, fit.t_last, fit.r2, fit.r2_adj, fit.n_points, fit.intercept; `lambda_z_stderr = se_slope` (N:868-877). Reported R2s are of the chosen window only (r2 clipped to [0,1]; r2_adj unclipped).
- `clast_pred = exp(intercept - lambda_z*tlast)`; `auc_inf_obs = auc_last + clast/lambda_z`; `auc_inf_pred = auc_last + clast_pred/lambda_z`; etc. (N:796-830). Note `tlast` here is the last positive non-BLQ-kept point (N:~705-710), which may differ from fit.t_last.
- relative_warning diagnostic (T:194-204): if not None: `selected = regressable & tp >= t_first & tp <= t_last` (plus `given | manual_mask` for MANUAL), `peak = max over in_row finite cp (0 if none)` over the WHOLE row (not just after tmax), `low = any(selected & cp < relative_warning * peak)` (strict) -> flag TERMINAL_LOW_CONCENTRATION on the fit flags. Never changes selection; NaN t_first -> no flag.

## 5. No window at all

Row has no fit (slope/intercept/r2/r2_adj/se/n_points/t_first/t_last = NaN, start = -1) when: BEST_FIT: no s with `s >= first_allowed & n>=min_points & slope<0` (-> POSITIVE_SLOPE if some window had enough points but none declining, else TOO_FEW_POINTS); LAST_N: no window with n >= n_points, or n at the (possibly shifted) start < min_points (TOO_FEW), or slope >= 0/NaN (POSITIVE_SLOPE); ALL_AFTER_TMAX: tmax is the last valid point (`tmax_idx+1 >= n_valid`) or < min_points regressable points (TOO_FEW), or slope >= 0 (POSITIVE_SLOPE); MANUAL/windows: no selected regressable point or fewer than min_points (or degenerate stats); any rule: chosen window fails min_adj_r2 (TOO_FEW_POINTS); fewer than 3 regressable points / zero variance of t or ln c (n NaN).
Also rows with `n_valid < 2` get NO_DATA and fit flags are not OR-ed (`flags |= where(has_data, fit.flags, 0)`, N:~791).
Dependents become NaN purely by arithmetic propagation (no special-casing): lambda_z, thalf, span, clast_pred, auc_inf_obs/pred, auc_extrap_fraction(_pred), aumc_inf(_pred), aumc_extrap(_pred), mrt, mrt_pred, thalf_eff (= ln2*mrt), cl/cl_f(_pred), vz/vz_f(_pred), vss, vss_pred, auc_inf_dn, auc_inf_pred_dn, aumc_back_extrap, auc_back_extrap_fraction(_pred) (N:793-930). NOT NaN: auc_last, aumc_last, auc_all, cmax, tmax, mrt_last, cmax_dn. NaN comparisons are false, so EXTRAPOLATION_HIGH and SPAN_LOW are NOT set for a no-fit row.

## 6. NCAFlag (O:133-170, IntFlag) and constants

| value | name | condition |
|---|---|---|
| 0 | NONE | |
| 1 | POSITIVE_SLOPE | T:369-371, T:394: windows with enough points exist but none with slope < 0 (BEST_FIT) / chosen window of LAST_N, ALL_AFTER_TMAX has `!(slope<0)` |
| 2 | TOO_FEW_POINTS | T:441-442: no fit and flags still 0 (includes min_adj_r2 failure) |
| 4 | EXTRAPOLATION_HIGH | N:829-831: `auc_extrap_fraction > options.extrapolation_warning` (strict) |
| 8 | NO_MAX | N:712: has_data & tmax_idx == n_valid-1 |
| 16 | NO_ABSORPTION | N:713-714: ORAL route & has_data & tmax_idx == 0 |
| 32 | BLQ_TRUNCATED | N:538-ish: `truncated` from apply_blq |
| 64 | NO_DATA | N:~699: n_valid < 2 (also sparse.py:428) |
| 128 | DELTA_WINDOW_CHANGE | uncertainty.py (delta method), out of scope |
| 256 | INCOMPLETE_INTERVAL | multiple dose, out of scope |
| 512 | EXTRAPOLATED_TROUGH | multiple dose, out of scope |
| 1024 | SPAN_LOW | N:~833-834: `span < SPAN_MINIMUM` (strict, span NaN -> not set) |
| 2048 | NOT_ACCEPTED | Acceptance thresholds, N:~1037-1060, out of scope |
| 4096 | PARTIAL_EXTRAPOLATED | partial_aucs beyond last measurable value |
| 8192 | TERMINAL_LOW_CONCENTRATION | T:194-204 as in section 4 |

Constants: `SPAN_MINIMUM = 2.0` (N:100); `CANDIDATE_DIM = "candidate"` (T:27); `CANDIDATE_PREFIX = "candidate_"` (T:30); `WINDOW_DEFAULT_KEY = "*"` (O:395); `TerminalPhase` defaults above (O:459-467). In terminal.py no other numeric constant; the min of 3 points is the literal `n >= 3` (T:114) and `Field(ge=3)` (O:460).

## 7. Points the regression may not see

- Dose-time inserted value (c0 after bolus, 0 for infusion/oral single dose): never, it exists only in tp_area/cp_area (N:~735-760).
- Zero, negative, NaN values: never (cp > 0 and finite y, T:163-165). NaN t or c are removed already by pack_valid.
- Points at or before tmax: forbidden only when `exclude_cmax` (BEST_FIT, LAST_N via start shift, candidate table) and always for ALL_AFTER_TMAX; index-based on first max (tmax_idx), so a later point equal to cmax is allowed. Allowed (if positive) with exclude_cmax=False for BEST_FIT/LAST_N, and for MANUAL and per-row windows regardless.
- Infusion: samples with t <= duration (relative times), all methods (N:~681-690).
- BLQ-kept/imputed points unless `blq.terminal_regression`; `blq_packed` (N:~672-676).
- `exclude` of `regressable` also applies to MANUAL and windows.
- Windows are contiguous in PACKED valid-point order: a non-regressable point inside a window is simply skipped (not counted in n).

## 8. Phoenix / PKNCA mirroring

- T:5-9 module docstring: BEST_FIT = largest adjusted R2, preferring more points within a tolerance, "the rule of Phoenix" (O:44, O:419-420: tie_tolerance doc). Phoenix's own rule is adjR2 >= best - 1e-4 (here default 1e-4, absolute).
- O:402-407 and N:~678-690: infusion windows start strictly after end of infusion (Phoenix WinNonlin).
- N:~735-748: value at dose inserted (0 for infusion/oral, c0 for bolus) as Phoenix does (areas only).
- N:~801-803, 812-818: clast_pred / auc_inf_pred / aumc_inf_pred follow Phoenix `Clast_pred`, `AUMCINF_pred` (PKNCA equivalents).
- N:821-824: thalf_eff = ln2*mrt verbatim PKNCA `pk.calc.thalf.eff`.
- O:435, 449: per-row windows mimic Phoenix `Lambda_z_lower/upper` and PKanalix "Check lambda_z".
- T:1-5 cites Gabrielsson & Weiner 2016 ch. 2.8 for lambda_z as -slope of ln c.

## 9. Silent-divergence hazards for a port (summary)

1. Tie rule: absolute tol, `>= best - tol`, band over candidates, winner = earliest start (most points), not best r2_adj.
2. `slope < 0` strict candidate filter in BEST_FIT; `min_adj_r2` is NOT in selection, only a post-gate that yields TOO_FEW_POINTS (not POSITIVE_SLOPE), and can reject the chosen window even if another window would pass.
3. MANUAL and per-row windows have NO slope check: a positive slope yields a fit with negative lambda_z, negative thalf, no POSITIVE_SLOPE flag, no NaN (SPAN_LOW then true because span negative < 2). LAST_N/ALL_AFTER_TMAX do check.
4. r2 is clipped to [0,1] before r2_adj; r2_adj is unclipped; df = n-2; windows with syy_c <= 0 or sxx_c <= 0 are NaN (not candidates, n NaN).
5. Origin-shifted suffix sums (origin = first valid point of the row) and backward accumulation order; intercept is reported at original t=0.
6. ALL_AFTER_TMAX ignores exclude_cmax; LAST_N "largest s with n >= n_points" uses the NaN-masked n.
7. t_last = time of the last REGRESSABLE point, not tlast; t_first snapped forward to a regressable point; span = (t_last - t_first)/thalf.
8. tmax_idx from all valid points (not regressable), first max, without the inserted dose point; infusion exclusion is `tp <= duration` incl. equality, NaN duration excludes `tp <= 0`.
9. candidate_table is looser (no slope/min_adj_r2 filter) and valid-start-only; exposed only for one-row batches; windows requiring `n >= min_points` where n is the regressable count.
10. relative_warning peak over whole row, strict `<`.
