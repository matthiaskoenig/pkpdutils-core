# Single-dose NCA kernel: areas and parameters (extracted from Python)

Sources: `A` = src/pkpdutils/nca/auc.py, `N` = src/pkpdutils/nca/nca.py, `T` = src/pkpdutils/nca/terminal.py, `O` = src/pkpdutils/nca/options.py.
All arithmetic is IEEE-754 float64 with numpy semantics: division by zero gives inf/NaN, there are almost no explicit guards (every `np.errstate` in the code only silences warnings). A reimplementation must not add guards that change inf/NaN results. Operation order below is the code's; keep it (left-associative).
Kernel entry: `compute_parameters` (N:618). Single-dose route into it: `_compute_chunk` (N:1262) calls `reference_dose(..., last=False)` (N:1070) then `compute_parameters(single_dose=True default)`. Rows are `(N, n)` arrays; every row independent.

## 0. Pipeline order inside compute_parameters (N:618-931)

1. Shift times: `shift = dose_time if finite else 0`; `t = t - shift` (N:660-664). A row with NaN dose time is NOT shifted (its dose-independent parameters still computed). Times may therefore be negative (pre-dose samples) - see trap T2.
2. BLQ (section 5): `resolve_lloq` then `apply_blq` (N:665-667).
3. `kind == EFFECT` returns `_effect_parameters` (N:668, defined N:514) - NOT concentration NCA, out of scope here (always linear AUC, parameters e0, emax_obs, temax, tlast, auec_last, auec_baseline, emax_baseline, time_above, flags). Mention only so the router is not forgotten.
4. `blq_packed`, `exclude` (N:673-680): `exclude = blq_packed` unless `rules.terminal_regression`.
5. `pack_valid` (section 1).
6. Infusion: `exclude |= (tp <= nan_to_num(dose_duration, nan=0))` when `route is IV_INFUSION and dose_duration is not None` (N:685-696). Inclusive `<=`. Times are dose-relative so end of infusion == duration.
7. `in_row = idx < n_valid`; `has_data = n_valid >= 2`; `flags = NO_DATA(64) where !has_data`; `|= BLQ_TRUNCATED(32)` where truncated (N:699-704).
8. cmax/tmax/cmin/tmin, NO_MAX, NO_ABSORPTION (3.1).
9. clast/tlast (3.1). tlag (3.7).
10. Route-specific insertion at the dose time (section 4), giving `tp_area, cp_area, n_area`.
11. Areas (3.2), terminal fit (3.3, uses `tp, cp, n_valid`, NOT the area arrays), extrapolations (3.4), cmax_half (3.7), route extras (3.8), dose parameters (3.9), flags (3.10).

## 1. pack_valid (A:36-57)

- valid point: `isfinite(t) & isfinite(c)` (both finite; inf and NaN are invalid). Inputs cast to float64.
- Layout: stable partition - valid points moved to the front keeping original relative order (`argsort(~valid, stable)`), the remaining columns are NaN in both arrays. NO sorting by time: it assumes input is already time ordered (see T1). Returns `(tp, cp, n_valid)`, `n_valid = valid.sum(axis=1)`.
- 0 valid points: all NaN row, n_valid 0. 1 valid point: n_valid 1. Both: `has_data = n_valid >= 2` False => flag NO_DATA, and (see N:704-931) every parameter is NaN with these exceptions to check against gating: `clast_pred` gated by `has_data` (N:~900); `cmax,tmax,cmin,tmin,clast,tlast,auc_last,auc_all,aumc_*,c0,c0_method,auc/aumc_back_extrap_fraction,cmax_half,tmax_half` gated by `has_data`. NOT gated: `tlag` (see T5), `lambda_*` (come out NaN from the fit because it needs >=3 points), `c0_method` is `C0_NONE=0` (gated). `candidate_*` untouched.
- segment_areas on such rows: `in_curve = arange(n-1) < n_valid-1`, so a row with 0 or 1 valid points has all segments = 0.0 (not NaN). Callers then overwrite with NaN through `has_data`.
- Edge: `n == 1` column arrays give `(N, 0)` segment arrays; sums are 0.

## 2. segment_areas (A:60-100) and auc_aumc (A:114-147)

Per segment between consecutive packed points `(t1,c1) -> (t2,c2)`, `dt = t2 - t1`:
- Linear area: `lin_area = mid(c1, c2) * dt`, `mid(a,b) = (a+b)*0.5`; if `a+b` is inf (overflow) use `0.5*a + 0.5*b` for that element (A:103-111). `lin_moment = mid(t1*c1, t2*c2) * dt` (AUMC segment).
- Log area: `L = ln(c1/c2)`; `log_area = (c1 - c2)/L * dt`; `log_moment = dt*(t1*c1 - t2*c2)/L + dt*dt*(c1 - c2)/(L*L)` (A:84-88). Note `ln(c1/c2)`, a ratio first, not a difference of logs.
- `positive = (c1 > 0) & (c2 > 0) & (c1 != c2)` (A:82). This is the guard: zero or negative value, or equal values => the segment is ALWAYS linear under every method (log rule never evaluated, no NaN).
- Method (A:90-97): `LINEAR`: never log. `LINEAR_LOG` ("linear-up/log-down"): `use_log = positive & (c2 < c1)`, i.e. "down" = strictly the later value lower than the earlier. Equal values -> linear (same area anyway). `LOG`: `use_log = positive` (rising segments also log).
- `area = where(use_log, log_area, lin_area)`, same for moment; then segments `>= n_valid-1` set to 0.0 (A:98-99).
- `auc_aumc`: sums segment areas/moments over axis 1 after optional masks: `keep &= tp[:, :-1] >= t_start` (if given), `keep &= tp[:, 1:] <= t_end` (if given) (A:140-147). Whole-segment semantics. NaN comparisons are False, so `t_end = NaN` (e.g. tlast NaN) drops every segment (sum 0.0).
- Other area.py functions (`time_above_threshold`, `interpolate_at`) are NOT on the single-dose concentration path except through partial_auc (N:2024-2160, not covered). `insert_point` (A:239-266): concatenates a new column, stable-argsorts by time with NaN as +inf, then repacks via `pack_valid`; a row whose new time or value is NaN is unchanged (the NaN column drops out in repack). An inserted point at t=0 goes before points with t>0; with ties (existing point at t=0 is never combined because insertion only happens when `tp[:,0] > 0`) stable sort puts existing points first. Negative t first points: insertion not done.

## 3. Parameters in dependency order (single dose, concentration)

Notation: `D` = `positive_dose(dose_amount)` (N:272-298): dose where `> 0` else NaN (placebo 0, NaN stay NaN). All `where(has_data, x, nan)` = NaN unless n_valid >= 2.

### 3.1 Observed extremes (N:706-731)
- `imax = argmax(where(in_row, cp, -inf))` (first max on ties); `cmax = cp[imax]`, `tmax = tp[imax]`; NaN unless has_data. (Includes imputed/kept BLQ values in cp. Excludes the inserted dose-time point.)
- `imin = argmin(where(in_row, cp, +inf))` (first min on ties); `cmin`, `tmin`, NaN unless has_data. Zeros count. Do not exclude BLQ-kept values.
- Flags: `NO_MAX(8)` where `has_data & imax == n_valid-1`; `NO_ABSORPTION(16)` where route is ORAL and `has_data & imax == 0`.
- `positive = in_row & (cp > 0) & ~blq_packed`; `ilast` = last True index; `clast = cp[ilast]`, `tlast = tp[ilast]` when `has_data & has_positive`, else NaN. (BLQ-kept/imputed values positive are not "measurable".)

### 3.2 Areas to tlast / all (N:741-775)
- `auc_last, aumc_last = auc_aumc(tp_area, cp_area, n_area, auc_method, t_end=tlast)`; NaN unless `has_data & has_positive`. Segments counted: those whose END time `<= tlast`, over the area arrays (so includes the inserted dose-time point and any imputed-BLQ points before tlast).
- `auc_all, aumc_all = auc_aumc(tp_area, cp_area, n_area, auc_method)` (no end mask); NaN unless `has_data`. Includes trailing zeros/imputed values (Phoenix AUCall, N:763-769).
- AUClast uses the chosen `AUCMethod` for both area and moment.

### 3.3 Terminal regression (needs tp, cp, n_valid, imax, options.terminal, exclude) (N:777-790; T:126-450)
`terminal_fit` returns slope, intercept, r2, r2_adj, se_slope, n_points, t_first, t_last, start, flags. NaN in all of them (start -1) for a row without a fit. Does NOT see the inserted dose point (it is fed `tp,cp`, not `tp_area,cp_area`) - state this in the brief.
- `y = ln(cp)` where `in_row & cp > 0` else NaN; `regressable = in_row & isfinite(y) & ~exclude` (T:171-176). Exclusions: BLQ kept/imputed (unless `terminal_regression`), infusion samples `t <= duration`.
- `window_statistics(x=tp, y, valid=regressable)` (T:73-123): for every start column s, the window is ALL regressable points from s to the end of the row (suffix sums). Values are centered at the row's first regressable point (`x_origin`, `y_origin`) for stability, then
  `n`, `sxx_c = sxx - sx*sx/n`, `sxy_c = sxy - sx*sy/n`, `syy_c = syy - sy*sy/n`; `slope = sxy_c/sxx_c`; `intercept = (sy - slope*sx)/n + y_origin - slope*x_origin` (ln c at t=0 of the dose-relative time axis); `ss_res = syy_c - slope*sxy_c`; `r2 = clip(1 - ss_res/syy_c, 0, 1)`; `r2_adj = 1 - (1-r2)*(n-1)/(n-2)`; `se_slope = sqrt(max(ss_res,0)/(n-2)/sxx_c)`. Window valid iff `n >= 3 & sxx_c > 0 & syy_c > 0`, else all six NaN (constant concentrations give NaN fit).
- Method (default `BEST_FIT`, `exclude_cmax=True`, `min_points=3`, `tie_tolerance=1e-4`, `min_adj_r2=None`, `relative_warning=1e-8`; O:458-465):
  - `BEST_FIT` (T:350-364): candidate s iff `s >= first_allowed` (`first_allowed = imax+1` if exclude_cmax else 0) `& n >= min_points & slope < 0`; `best = max r2_adj over candidates`; chosen start = FIRST (lowest s, i.e. most points) s in candidates with `r2_adj >= best - tie_tolerance`. Flag POSITIVE_SLOPE(1) when windows with enough points exist but none has slope<0.
  - `LAST_N` (T:366-376): start = largest s with `n >= n_points`; then `start = clip(max(start, first_allowed), 0, n-1)`; `has_fit = any(n >= n_points)`; then the common gate below.
  - `ALL_AFTER_TMAX`: `start = imax+1`; `has_fit = start < n_valid`.
  - LAST_N/ALL_AFTER_TMAX gate (T:377-387): `enough = has_fit & n_at_start >= min_points`; `positive = enough & ~(slope_at_start < 0)` -> flag POSITIVE_SLOPE and no fit.
  - `MANUAL`: regress exactly `regressable & manual_mask` (the packed positions of the given original indices; indices refer to original column positions before packing) (T:333-345, N:772-780).
  - Per-row `windows` (t_first,t_last) override: regress `regressable & tp>=t_first & tp<=t_last` (T:178-203).
- `_collect` (T:398-463): start snapped FORWARD to the first regressable point at/after start; `end` = last regressable point at/after start; `n_points = n[start]`; `has_fit &= n_points >= min_points` and, if `min_adj_r2` set, `r2_adj[start] >= min_adj_r2`; picks of slope, intercept, r2, r2_adj, se_slope, n, `t_first = tp[start]`, `t_last = tp[end]`, else NaN. `TOO_FEW_POINTS(2)` flag when no fit and flags == 0 at that point.
- Row-level: `flags |= fit.flags` only `where has_data` (N:791). `TERMINAL_LOW_CONCENTRATION(8192)` flag (T:~190-205): any selected regression point with `cp < relative_warning * peak` where `peak = max over in_row finite cp (floor 0)`; flag only, no change of numbers.
- Note t_last of the regression is the last REGRESSABLE point, which equals `tlast` only if no exclusion removed it.

### 3.4 Terminal-phase derived (N:791-835) - all inside errstate-ignore, no guards
- `lambda_z = -slope` (NaN w/o fit).
- `thalf = ln(2)/lambda_z`.
- `lambda_z_stderr = se_slope`, `lambda_z_intercept = intercept`, `lambda_z_r2 = r2`, `lambda_z_r2_adj = r2_adj`, `lambda_z_n_points = n_points`, `lambda_z_t_first = t_first`, `lambda_z_t_last = t_last`.
- `lambda_z_span = (t_last - t_first)/thalf`.
- `auc_inf_obs = auc_last + clast/lambda_z`.
- `clast_pred = exp(intercept - lambda_z*tlast)` (uses tlast, the last measurable time, not the window end). Output `clast_pred` is `where(has_data, clast_pred, nan)` (N:~905).
- `auc_inf_pred = auc_last + clast_pred/lambda_z`.
- `auc_extrap_fraction = (auc_inf_obs - auc_last)/auc_inf_obs`; `auc_extrap_fraction_pred = (auc_inf_pred - auc_last)/auc_inf_pred`. Fractions (0..1), not percent.
- `aumc_inf = aumc_last + clast*tlast/lambda_z + clast/(lambda_z*lambda_z)` (evaluated as `(aumc_last + ((clast*tlast)/lambda_z)) + (clast/(lambda_z*lambda_z))`).
- `aumc_inf_pred` = same with `clast_pred` in place of `clast`.
- `aumc_extrap_fraction = (aumc_inf - aumc_last)/aumc_inf`; `_pred` analog.
- `half_duration = nan_to_num(dose_duration, nan->0)/2` if `route is IV_INFUSION and dose_duration is not None` else 0.0 (N:817-822).
- `mrt = aumc_inf/auc_inf_obs - half_duration`; `mrt_last = aumc_last/auc_last - half_duration`; `mrt_pred = aumc_inf_pred/auc_inf_pred - half_duration`.
- `thalf_eff = ln(2) * mrt` (NOT ln2 / something; uses the infusion-corrected mrt).
- Flags: `EXTRAPOLATION_HIGH(4)` where `auc_extrap_fraction > options.extrapolation_warning` (default 0.2, strict; NaN -> no flag). `SPAN_LOW(1024)` where `lambda_z_span < 2.0` (SPAN_MINIMUM, N:100; strict; NaN -> no flag). Both `|=`ed without has_data gating, but NaN comparisons are False so equivalent.

NaN rules: all of this block is NaN when lambda_z is NaN (no fit) except `mrt_last`, `auc_last`, `aumc_last`, `clast`, `tlast`. `auc_last` NaN (no positive) makes everything else NaN. `lambda_z` itself is never 0 for a fitted row (slope < 0 enforced) but if `auc_inf_*`/`lambda_z^2` hit 0/inf the result is IEEE, not NaN-forced.

### 3.5 Extra output fields of the fit
All 10 terminal-fit outputs listed above (lambda_z, lambda_z_stderr, lambda_z_intercept, lambda_z_r2, lambda_z_r2_adj, lambda_z_n_points, lambda_z_t_first, lambda_z_t_last, lambda_z_span, thalf). `lambda_z_n_points` is a float array.

### 3.6 Optional candidate table
`TerminalPhase.keep_candidates` -> `candidate_variables` (N:933) -> point variables `candidate_t_first`, `candidate_n_points`, `candidate_r2_adj` (single-row analysis only; built by `candidate_table`, T:210). Not parameters of the kernel proper; include if the plan wants parity.

### 3.7 Route-independent-looking extras that are route-gated
- `tlag` (N:569-615; ORAL only, else all NaN): `after_dose = in_row & tp >= 0`; `measurable = after_dose & cp > 0 & ~blq_packed`; `first` = first True; if none -> NaN. `has_lag = has_measurable & first > 0 & after_dose[first-1]`; `tlag = tp[first-1]` if has_lag; else `0.0` (first sample after dose already measurable, or the sample before it is pre-dose) ; NaN if no measurable. Computed regardless of `has_data` (see T5). Only emitted into the output when route is ORAL (N:~895).
- `cmax_half` / `tmax_half` (N:838-846): ORAL only, `has_half = has_data & imax > 0 & route is ORAL`; among points `idx < imax` and in row, the index minimizing `|cp - 0.5*cmax|` (first on ties; argmin of where(before_max, |...|, +inf)); output `cp[ihalf]` / `tp[ihalf]`.
- IV bolus extras (N:861-881) see 4.

### 3.8 IV bolus only (route == IV_BOLUS)
- `c0`, `c0_method` (section 4), `auc_back_extrap_fraction`, `auc_back_extrap_fraction_pred`, `aumc_back_extrap_fraction`:
  `area, moment = segment_areas(tp_area, cp_area, n_area, auc_method)` (only computed if `insert.any()`; else zeros); `first = area[:,0]` (the segment from the inserted t=0 point to the first sample, same method rule); 
  `auc_back_extrap_fraction = where(has_data, where(insert, area[:,0], 0.0)/auc_inf_obs, nan)`; `_pred` divides by `auc_inf_pred`; `aumc_back_extrap_fraction = where(has_data, where(insert, moment[:,0], 0.0)/aumc_inf, nan)`. Note: when `insert` is False the numerator is 0.0 so the result is 0.0 (not NaN) as long as has_data and the denominator is finite non-zero (0/NaN = NaN when no terminal fit).

### 3.9 Dose-dependent (only if `dose_amount is not None and route is not None`) (N:901-929)
- `cl = D/auc_inf_obs`; `vz = cl/lambda_z`; `cl_pred = D/auc_inf_pred`; `vz_pred = cl_pred/lambda_z`.
- Names: suffix `""` for IV (bolus and infusion), `"_f"` otherwise: `cl`/`cl_f`, `vz`/`vz_f`, `cl_pred`/`cl_f_pred`, `vz_pred`/`vz_f_pred`.
- IV only: `vss = cl*mrt`; `vss_pred = cl_pred*mrt_pred` (note: `mrt` here already has the infusion half-duration subtracted).
- `auc_inf_dn = auc_inf_obs/D`; `auc_inf_pred_dn = auc_inf_pred/D`; `cmax_dn = cmax/D`.
- NaN when D is NaN (dose <= 0, or row with no dose), or the dependent quantity is NaN. No explicit guard beyond `positive_dose`.
- Route None or no dose array: none of these keys exist at all (key absent, not NaN).

### 3.10 flags (final `out["flags"]`, int64)
Bits set on this path: NO_DATA 64, BLQ_TRUNCATED 32, NO_MAX 8, NO_ABSORPTION 16 (oral), POSITIVE_SLOPE 1 / TOO_FEW_POINTS 2 / TERMINAL_LOW_CONCENTRATION 8192 (from the fit, only where has_data), EXTRAPOLATION_HIGH 4, SPAN_LOW 1024. NOT_ACCEPTED 2048, PARTIAL_EXTRAPOLATED, INCOMPLETE_INTERVAL, DELTA_WINDOW_CHANGE etc. belong to other paths (`nca()` N:1544 adds NOT_ACCEPTED after acceptance evaluation).

### 3.11 Parameter count (keys of `compute_parameters` output, excluding the optional `candidate_*` and including `flags`)
Common concentration block, always present: cmax, tmax, cmin, tmin, clast, clast_pred, tlast (7); auc_last, auc_all, auc_inf_obs, auc_inf_pred, auc_extrap_fraction, auc_extrap_fraction_pred (6); aumc_last, aumc_all, aumc_inf, aumc_inf_pred, aumc_extrap_fraction, aumc_extrap_fraction_pred (6); mrt, mrt_last, mrt_pred, thalf_eff (4); lambda_z, lambda_z_stderr, lambda_z_intercept, lambda_z_r2, lambda_z_r2_adj, lambda_z_n_points, lambda_z_t_first, lambda_z_t_last, lambda_z_span, thalf (10) = 33, plus `flags` = 34.
+ IV bolus: c0, c0_method, auc_back_extrap_fraction, auc_back_extrap_fraction_pred, aumc_back_extrap_fraction (5).
+ ORAL: tlag, cmax_half, tmax_half (3).
+ with dose and route: cl|cl_f, vz|vz_f, cl_pred|cl_f_pred, vz_pred|vz_f_pred (4), auc_inf_dn, auc_inf_pred_dn, cmax_dn (3), IV only vss, vss_pred (2).
Totals (with dose, excluding candidates): IV bolus 34+5+7+2 = 48; IV infusion 34+7+2 = 43; oral 34+3+7 = 44; no dose/route: 34 (+5 bolus needs a route anyway, +3 oral). Effect kind is a separate 7-8 variable set (not counted).
`PARAMETER_UNITS` (N:116) is the full unit-expression table (placeholders `{unit}`, `{time}`, `{dose}`; `c0_method` dimensionless, `lambda_z` `1 / ({time})`, `lambda_z_stderr` `1/({time})`, `lambda_z_intercept` dimensionless, `cl*` `({dose}) / (({unit}) * ({time}))`, `vz*`/`vss*` `({dose}) / ({unit})`, `aumc_*` `({unit}) * ({time}) ** 2`, `*_dn` per dose, fractions dimensionless). It also lists many multiple-dose/urine/sparse/effect names not in this section; the single-dose names above are the subset to port. `unit_expression` (N:242) handles derived suffixes; `x_dn` gets `(expr) / ({dose})`.

## 4. Value at the dose time (N:699-751; also `_insert_dose_value` N:1965, partial area only)

In `compute_parameters` the areas run on `tp_area, cp_area, n_area`, which equal `tp, cp, n_valid` unless a point is inserted. Only auc_last, aumc_last, auc_all, aumc_all and the three back-extrapolated fractions see the inserted point. NOT seen by: cmax/tmax/cmin/tmin, clast/tlast, tlag, cmax_half/tmax_half, imax (used for NO_MAX etc.), flags, and above all the terminal regression (`terminal_fit` receives `tp, cp`). `lambda_z`, `thalf`, `clast_pred` etc. therefore are independent of the insertion; `auc_inf_*` depend on it only through `auc_last`.
- IV bolus (N:723-735), executed regardless of `single_dose`: `(estimate, rule) = bolus_c0(tp, cp, n_valid, options)`; `c0 = where(has_data, estimate, nan)`; `c0_method = where(has_data, rule, 0)`; `insert = has_data & (tp[:,0] > 0) & isfinite(c0)`; insert `(0.0, c0)` for rows with `insert`.
  `bolus_c0` (N:301-356): `C0Method.NONE` -> c0 NaN, rule 0, no insertion. Else `first = cp[:,0]` (NaN if n_valid < 1), rule `C0_FIRST_VALUE=2` (0 if n_valid<1). `FIRST_VALUE` returns `first`. `LOG_BACK_EXTRAPOLATION` (default; C0Method O:84): with the first two packed points `(t1,c1)`, `(t2,c2)` (NaN if n_valid < 2):
  `back = exp( ln(c1) - (ln(c2) - ln(c1)) / (t2 - t1) * t1 )`
  used iff `n_valid >= 1 & c1 > 0 & c2 > 0 & c2 < c1 & t2 > t1`, rule `C0_BACK_EXTRAPOLATION=1`; otherwise `first` (rule 2). (`c1`, `c2` are the first two valid points AFTER BLQ dropping, relative times.) The extrapolation uses ln-difference form here (different from the `ln(c1/c2)` ratio in the log segment).
  `c0` is reported also when `tp[:,0] <= 0` (no insertion then; with t1 = 0 the formula returns c1; with t1 < 0 it is evaluated as written). With `C0Method.NONE`, c0 NaN, nothing inserted, `auc_back_extrap_fraction = 0.0/auc_inf_obs` (0.0 if auc_inf finite, see 3.8).
- IV infusion and ORAL (N:736-751): only when `single_dose` is True (steady-state passes False; single-dose path default True): `at_dose = has_data & (tp[:,0] > 0)`; insert `(0.0, 0.0)` for those rows. No `c0` variable for these routes.
- Route None (or any other value): nothing inserted anywhere in `compute_parameters` (the `if/elif` has no else); an area before the first sample is then simply not counted. (`_insert_dose_value` has the same `route is None` early return, N:1999.)
- `n_area` etc. come from `insert_point` -> `pack_valid` (A:239-266); result arrays are one column wider.

## 5. BLQ handling (single dose path) (N:359-494, O:54-60, 185-312)

- `resolve_lloq` (N:359-397): `limit = options.lloq` (scalar, wins) else per-row `lloq` else None; with `options.lloq_relative`: `peak = max over finite c>0 of the row (0 if none)`, `relative = lloq_relative*peak` (NaN if peak == 0), result `fmax(absolute, relative)` (relative alone if no absolute). Requires c (the shifted/pre-BLQ values). `lloq is None` -> `apply_blq` returns c unchanged, `truncated` all False, in_curve all False.
- `apply_blq`: `below = finite(c) & finite(limit) & (c < limit)` strict; `measurable = finite(c) & ~below`.
  - Default `options.blq = BLQHandling.NAN` (O:585) -> rules `first=middle=last=DROP` (O:268-272). `BLQHandling.ZERO_BEFORE_TMAX` -> `before_tmax=ZERO, after_tmax=DROP` (the "tmax axis"). Any `BLQRules` instead is used as-is (can be positional first/middle/last OR tmax-axis before/after, never both; `None` rule == DROP). Other presets: `ich_m13a` first=ZERO, middle=DROP, last=ZERO; `pkanalix` before_tmax=ZERO, after_tmax=HALF_LLOQ; `pumas` first=KEEP, middle=DROP, last=KEEP (O:274-312).
  - Positional axis: `first_idx` = first measurable index, `last_idx` = last measurable index (n beyond all columns when no measurable value). `first` group: `below & idx < first_idx`; `middle`: `below & first_idx < idx < last_idx`; `last`: `below & idx > last_idx`. A row with no measurable value: every finite value counts as `first`.
  - tmax axis: `tmax_idx = argmax(where(measurable, c, -inf))` (first maximum; 0 if no measurable value); `before_tmax`: `below & idx < tmax_idx`; `after_tmax`: `below & idx >= tmax_idx`. A row with no measurable value: all `after_tmax`.
  - Actions per group (N:480-494): `DROP`: value -> NaN (becomes invalid in pack_valid, so absent), `changed`; `KEEP`: value unchanged, `in_curve` True, `changed` NOT set (so no BLQ_TRUNCATED); `ZERO`: 0; `LLOQ`: limit; `HALF_LLOQ`: 0.5*limit; float: that number. Imputing sets `changed` and `in_curve`.
  - Outputs: `values`, `truncated = changed.any(axis=1)` (flag BLQ_TRUNCATED 32; set for drop and impute, not for KEEP), `blq_in_curve` (imputed or kept points; mask over original columns).
- Consequences (N:673-680, 713-715, 585-612): imputed/kept points are part of the curve (areas, cmax, cmin, imax) but: not `positive` for clast/tlast (so AUClast ends at last real measurable), excluded from the terminal regression unless `BLQRules.terminal_regression`, and not "measurable" for `tlag`. `packed_mask` (N:497) maps the original-column mask to packed positions: `valid = isfinite(t)&isfinite(c)` (after BLQ) , stable partition identical to `pack_valid`, `mask & valid`.
- Order: shifting by dose time happens before BLQ; BLQ before pack.

## 6. The `_pred` family (observed extrapolates from `clast`, predicted from the regression)
Predicted counterpart exists for: `clast_pred` (new quantity), `auc_inf_pred`, `auc_extrap_fraction_pred`, `aumc_inf_pred`, `aumc_extrap_fraction_pred`, `mrt_pred`, `cl_pred`/`cl_f_pred`, `vz_pred`/`vz_f_pred`, `vss_pred` (IV), `auc_inf_pred_dn`, `auc_back_extrap_fraction_pred` (bolus). No `_pred` for: thalf, thalf_eff, mrt_last, lambda_z, cmax/tmax, `auc_inf_dn` (observed `auc_inf_obs` -> `auc_inf_dn`), `aumc_back_extrap_fraction`, `c0`, `tlag`.
Difference: `clast_pred = exp(intercept - lambda_z*tlast)` replaces `clast` in `auc_inf` and `aumc_inf` extrapolation and in the derived quantities: `auc_inf_pred = auc_last + clast_pred/lambda_z`; `aumc_inf_pred = aumc_last + clast_pred*tlast/lambda_z + clast_pred/lambda_z^2`; `mrt_pred = aumc_inf_pred/auc_inf_pred - half_duration`; `cl_pred = D/auc_inf_pred`; `vz_pred = cl_pred/lambda_z`; `vss_pred = cl_pred*mrt_pred`. `auc_last`, `aumc_last`, `tlast` are shared. `thalf_eff` uses observed `mrt` only (no pred counterpart).

## 7. Dose normalization (companions)
Inside the kernel (N:925-927): only `auc_inf_dn = auc_inf_obs/D`, `auc_inf_pred_dn = auc_inf_pred/D`, `cmax_dn = cmax/D`, `D = positive_dose(reference dose amount)` (first dose of the protocol, N:1070/1788; the divisor is the dose AMOUNT in dose units, no body weight). `D` NaN if dose <= 0 or missing. Beyond the kernel: `NCAResult.dose_normalized` (src/pkpdutils/nca/result.py:201) adds `x_dn = x / dose_amount` (coordinate; `dose.where(dose > 0)`; name `auc_inf_dn` for `auc_inf_obs`, `x_dn` otherwise) for parameters whose unit expression is `{unit}` or `({unit}) * ({time})` (`dose_normalized_parameters` result.py:183) - e.g. auc_last_dn, auc_all_dn, c0_dn, cmax_dn. Those post-hoc variables are NOT part of `compute_parameters`. `PARAMETER_UNITS` also lists `auc_tau_dn`, `cavg_dn`, `cmax_ss_dn` (multiple-dose, other plan).

## 8. Phoenix WinNonlin / PKNCA mirrors (places a reimplementation silently diverges)
- N:9-12, A:1-16 Phoenix NCA definitions; `AUCMethod.LINEAR_LOG` is Phoenix "linear up/log down" (O:30-36, A:92-93).
- N:313-326 / N:345-355 `bolus_c0` fallback chain is the documented Phoenix rule: use first observed value if slope >= 0 or one of the first two values is zero; here also when `t2 <= t1` or fewer than two points.
- N:581-590 `tlag` only for extravascular; lag 0 when the first post-dose sample is already measurable (Phoenix); `tp >= 0` filter.
- N:685-696 terminal window of an infusion starts strictly after the infusion end (`t <= duration` excluded), Phoenix.
- N:743-751 and N:1976-1983 zero inserted at the dose time for extravascular and infusion single dose ("Phoenix inserts"); N:1980 quotes it. Bolus uses c0 instead.
- N:763-770 `AUCall` includes the area over trailing zero/imputed observations (Phoenix `AUCall`); `AUClast` ends at the last measurable value.
- N:795-800 `clast_pred = exp(intercept - lambda_z*tlast)` (Phoenix `Clast_pred`); N:802-810 `AUMCINF_pred` (Phoenix).
- N:808-822 MRT infusion correction subtracts half the infusion duration (Gabrielsson & Weiner). Applies to mrt, mrt_last, mrt_pred.
- N:824-828 and N:18-27 `thalf_eff = log(2)*mrt` copied verbatim from PKNCA `pk.calc.thalf.eff` (not a half-life of anything in the Phoenix sense: uses infusion-corrected MRT).
- N:883-896 `AUC_%Back_Ext_obs` Phoenix: back-extrapolated segment area / AUC(0-inf); 0 when there is no inserted segment.
- N:912-916 `Cl_pred`, `Vz_pred`, `Vss_pred` Phoenix (vss_pred = cl_pred * mrt_pred).
- T:1-12, T:350-364 (and options O:419-424) `BEST_FIT` = Phoenix best fit: largest adjusted R^2, ties (within `tie_tolerance` 1e-4) resolved to MORE points; windows are suffix windows; `min_points` 3; `exclude_cmax` windows start after tmax (Phoenix excludes Cmax from regression).
- N:1031 (acceptance, other path) PKanalix/Phoenix `Accepted` criterion: not single-dose kernel.
- O:306-312 BLQ presets (PKNCA conc.blq, PKanalix, Pumas, ICH M13A): the default is NOT any of them - default is drop everywhere (`BLQHandling.NAN`).
- A:1-16 formulas note "dt (c1 - c2) / L" for log area; this is the Phoenix log trapezoid.

## Traps and unsure items (also for the brief)
- T1 No time sorting anywhere in the kernel (`pack_valid` is a stable partition, not a sort). Segments assume monotone times. Unsure where ordering is enforced upstream (Timecourse/Timecourses); the kernel does not.
- T2 Samples with negative dose-relative time (pre-dose) are NOT dropped: `auc_aumc` sums every segment, so a pre-dose sample adds area (including negative-time segments). Insertion at the dose happens only if the FIRST sample time `> 0`. A pre-dose sample at t < 0 with later samples gives no dose-time insertion. Verify this is the intended behaviour against upstream filtering before porting.
- T3 c0-reported-without-insertion, and `auc_back_extrap_fraction == 0.0` (not NaN) when no point was inserted (incl. `C0Method.NONE`).
- T4 The terminal regression never sees the inserted point (it uses `tp, cp`, not `tp_area, cp_area`), and the BLQ-kept/imputed values and infusion-time samples are excluded from it; `tlast`/`clast` ignore BLQ-kept values but `cmax`/`cmin`/`imax` do not.
- T5 `tlag` is NOT gated by `has_data` (a row with one valid point can report a numeric tlag); `cmax_half` is.
- T6 `clast_pred` uses `tlast`, not the end of the regression window; `lambda_z_t_last` is the last regressable point and can differ from `tlast`.
- T7 Log area uses `ln(c1/c2)`; bolus back-extrapolation uses `ln(c1)` and `ln(c2)` separately; keep each form to match floating point.
- T8 `positive` guard in segments includes `c1 != c2`: equal positive values use the linear rule (also under LOG).
- T9 Linear-up/log-down "down" is strictly `c2 < c1`.
- T10 Bolus c0 insertion is NOT conditioned on `single_dose` (only infusion/oral are); steady-state path owns that.
- T11 Division is unguarded (IEEE inf/NaN flows into `*_extrap_fraction`, `mrt`, `cl`, `vz`); NaN comparisons suppress flags; `EXTRAPOLATION_HIGH` and `SPAN_LOW` are strict inequalities.
- T12 Dose-dependent keys are absent (not NaN) when `dose_amount is None` or `route is None`; `cl` vs `cl_f` is by route (`is_iv` = bolus or infusion).
- T13 `lambda_z_n_points` etc. come from float arrays (`n` is float64).
- T14 `window_statistics` centers at the first valid point of the row for numerical stability; results match naive formulas only to rounding, so reference tests must compare with a tolerance (the repo's benchmark tolerance is 1e-8, tests/nca/test_benchmark_datasets.py).
- T15 `Kind.EFFECT` takes a completely different branch right after BLQ (N:668).
- T16 Unsure/other-plan: `windows` per-row override, `MANUAL`, `partial_auc` / `_insert_dose_value`, `candidate_table`, `resolve_lloq` relative cutoff are included above but may belong to other plans.
