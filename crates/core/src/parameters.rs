//! The parameters of one single dose curve, in the order they depend on each other.
//!
//! This is `compute_parameters` of `src/pkpdutils/nca/nca.py`, row by row: the times are shifted
//! onto the dose, the rules for the values below the limit of quantification run, the row is
//! packed, the route puts its value at the dose time, the areas are integrated, the terminal
//! phase is regressed and everything else follows from those by arithmetic alone. The order is
//! the order of `areas.md` section 0 and it matters: the areas run on the row which carries the
//! inserted dose-time point and the regression runs on the row which does not, and a parameter
//! is `NaN` because the one it is computed from is `NaN`, never because a guard was added here.
//!
//! The division is the IEEE division of numpy: nothing guards against a zero denominator, so an
//! exposure of zero gives an infinite clearance and a row without a terminal phase gives a `NaN`
//! half-life. The comparisons which raise a flag are strict and a `NaN` answers none of them,
//! which is why a row without a window is flagged neither `EXTRAPOLATION_HIGH` nor `SPAN_LOW`.
//!
//! Which parameters a row carries at all depends on its route and on whether its batch has
//! doses: `c0` belongs to an intravenous bolus, `tlag` to an extravascular dose, and the
//! clearance is `cl` for an intravenous route and `cl_f` for every other one. A parameter which
//! does not belong to a row is **absent** from the result rather than `NaN` in it, which is what
//! [`PARAMETERS`] and [`Presence`] describe.

use pkpdutils_abi::{NcaFlag, NcaOptions, Route, TerminalMethod};

use crate::auc::{Areas, trapezoid, trapezoid_until};
use crate::blq::{apply_blq, packed_mask, resolve_lloq};
use crate::dose::{ReferenceDose, bolus_c0, dose_value, insert_dose_value};
use crate::pack::pack_valid;
use crate::terminal::{self, Candidate, TerminalFit};

/// Every parameter of one row, whether its route carries it or not.
///
/// A field which the route of the row does not define keeps the value it was born with, which is
/// `NaN` everywhere but in `c0_method`, the one integer variable of the set. The name a field is
/// reported under, and whether it is reported at all, is [`PARAMETERS`]: `cl` and `cl_f` are the
/// same field under the two names the route picks between.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowParameters {
    /// The largest observed value.
    pub cmax: f64,
    /// The time of the largest observed value.
    pub tmax: f64,
    /// The smallest observed value.
    pub cmin: f64,
    /// The time of the smallest observed value.
    pub tmin: f64,
    /// The last measurable value.
    pub clast: f64,
    /// The terminal regression at the time of the last measurable value.
    pub clast_pred: f64,
    /// The time of the last measurable value.
    pub tlast: f64,
    /// The area to the last measurable value.
    pub auc_last: f64,
    /// The area over every observation.
    pub auc_all: f64,
    /// The area extrapolated to infinity from the last measurable value.
    pub auc_inf_obs: f64,
    /// The area extrapolated to infinity from the predicted last value.
    pub auc_inf_pred: f64,
    /// The extrapolated share of the area, a fraction.
    pub auc_extrap_fraction: f64,
    /// The extrapolated share of the predicted area, a fraction.
    pub auc_extrap_fraction_pred: f64,
    /// The first moment to the last measurable value.
    pub aumc_last: f64,
    /// The first moment over every observation.
    pub aumc_all: f64,
    /// The first moment extrapolated to infinity.
    pub aumc_inf: f64,
    /// The first moment extrapolated from the predicted last value.
    pub aumc_inf_pred: f64,
    /// The extrapolated share of the first moment, a fraction.
    pub aumc_extrap_fraction: f64,
    /// The extrapolated share of the predicted first moment, a fraction.
    pub aumc_extrap_fraction_pred: f64,
    /// The mean residence time.
    pub mrt: f64,
    /// The mean residence time to the last measurable value.
    pub mrt_last: f64,
    /// The mean residence time from the predicted last value.
    pub mrt_pred: f64,
    /// The effective half-life, `ln(2) mrt`.
    pub thalf_eff: f64,
    /// The terminal elimination rate constant.
    pub lambda_z: f64,
    /// The standard error of the terminal elimination rate constant.
    pub lambda_z_stderr: f64,
    /// The intercept of the terminal regression at `t = 0`.
    pub lambda_z_intercept: f64,
    /// The coefficient of determination of the terminal regression.
    pub lambda_z_r2: f64,
    /// The adjusted coefficient of determination of the terminal regression.
    pub lambda_z_r2_adj: f64,
    /// The points of the terminal regression.
    pub lambda_z_n_points: f64,
    /// The first time of the terminal regression.
    pub lambda_z_t_first: f64,
    /// The last time of the terminal regression.
    pub lambda_z_t_last: f64,
    /// The half-lives the terminal window covers.
    pub lambda_z_span: f64,
    /// The terminal half-life.
    pub thalf: f64,
    /// The concentration at the dose time of an intravenous bolus.
    pub c0: f64,
    /// The rule which produced `c0`, one of the `C0Method::CODE_*` codes.
    pub c0_method: f64,
    /// The back-extrapolated share of the area, a fraction.
    pub auc_back_extrap_fraction: f64,
    /// The back-extrapolated share of the predicted area, a fraction.
    pub auc_back_extrap_fraction_pred: f64,
    /// The back-extrapolated share of the first moment, a fraction.
    pub aumc_back_extrap_fraction: f64,
    /// The lag time of the absorption.
    pub tlag: f64,
    /// The value closest to half the maximum during the absorption.
    pub cmax_half: f64,
    /// The time of that value.
    pub tmax_half: f64,
    /// The clearance, reported as `cl` or `cl_f`.
    pub cl: f64,
    /// The terminal volume of distribution, reported as `vz` or `vz_f`.
    pub vz: f64,
    /// The clearance from the predicted area, `cl_pred` or `cl_f_pred`.
    pub cl_pred: f64,
    /// The terminal volume from the predicted area, `vz_pred` or `vz_f_pred`.
    pub vz_pred: f64,
    /// The volume of distribution at steady state.
    pub vss: f64,
    /// The volume of distribution at steady state from the predicted area.
    pub vss_pred: f64,
    /// The dose normalized area to infinity.
    pub auc_inf_dn: f64,
    /// The dose normalized predicted area to infinity.
    pub auc_inf_pred_dn: f64,
    /// The dose normalized maximum.
    pub cmax_dn: f64,
    /// The flags of the row.
    pub flags: NcaFlag,
}

impl RowParameters {
    /// Every parameter undefined: the row before anything is computed.
    ///
    /// `c0_method` is `0` rather than `NaN`, the code of "no estimate": it is an integer variable
    /// of the python result (`INTEGER_VARIABLES` of `nca.py`) and carries no `NaN`.
    pub(crate) const EMPTY: Self = Self {
        cmax: f64::NAN,
        tmax: f64::NAN,
        cmin: f64::NAN,
        tmin: f64::NAN,
        clast: f64::NAN,
        clast_pred: f64::NAN,
        tlast: f64::NAN,
        auc_last: f64::NAN,
        auc_all: f64::NAN,
        auc_inf_obs: f64::NAN,
        auc_inf_pred: f64::NAN,
        auc_extrap_fraction: f64::NAN,
        auc_extrap_fraction_pred: f64::NAN,
        aumc_last: f64::NAN,
        aumc_all: f64::NAN,
        aumc_inf: f64::NAN,
        aumc_inf_pred: f64::NAN,
        aumc_extrap_fraction: f64::NAN,
        aumc_extrap_fraction_pred: f64::NAN,
        mrt: f64::NAN,
        mrt_last: f64::NAN,
        mrt_pred: f64::NAN,
        thalf_eff: f64::NAN,
        lambda_z: f64::NAN,
        lambda_z_stderr: f64::NAN,
        lambda_z_intercept: f64::NAN,
        lambda_z_r2: f64::NAN,
        lambda_z_r2_adj: f64::NAN,
        lambda_z_n_points: f64::NAN,
        lambda_z_t_first: f64::NAN,
        lambda_z_t_last: f64::NAN,
        lambda_z_span: f64::NAN,
        thalf: f64::NAN,
        c0: f64::NAN,
        c0_method: 0.0,
        auc_back_extrap_fraction: f64::NAN,
        auc_back_extrap_fraction_pred: f64::NAN,
        aumc_back_extrap_fraction: f64::NAN,
        tlag: f64::NAN,
        cmax_half: f64::NAN,
        tmax_half: f64::NAN,
        cl: f64::NAN,
        vz: f64::NAN,
        cl_pred: f64::NAN,
        vz_pred: f64::NAN,
        vss: f64::NAN,
        vss_pred: f64::NAN,
        auc_inf_dn: f64::NAN,
        auc_inf_pred_dn: f64::NAN,
        cmax_dn: f64::NAN,
        flags: NcaFlag::NONE,
    };
}

/// Which rows carry a parameter at all.
///
/// A parameter a row does not carry is absent from the result of a batch of such rows, and
/// `NaN` - `0` for the one integer variable - in the result of a batch which mixes routes, where
/// another row does carry it. `areas.md` trap T12: the dose dependent parameters need both a
/// dose and a route, and the clearance is named by the route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Presence {
    /// Every row.
    Always,
    /// An intravenous bolus.
    Bolus,
    /// An extravascular dose.
    Oral,
    /// A row whose batch has doses and whose route is known.
    Dose,
    /// Such a row with an intravenous route.
    DoseIv,
    /// Such a row with any other route.
    DoseExtravascular,
}

impl Presence {
    /// Whether a row of this route, in a batch which has or has not got doses, carries it.
    fn holds(self, route: Option<Route>, has_dose: bool) -> bool {
        match self {
            Self::Always => true,
            Self::Bolus => route == Some(Route::IvBolus),
            Self::Oral => route == Some(Route::Oral),
            Self::Dose => has_dose && route.is_some(),
            Self::DoseIv => has_dose && route.is_some_and(Route::is_iv),
            Self::DoseExtravascular => has_dose && route.is_some_and(|r| !r.is_iv()),
        }
    }
}

/// One reported parameter: its name, its unit, the field it reads and the rows which carry it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Parameter {
    /// The name of the variable, exactly the python one.
    pub name: &'static str,
    /// The unit expression of the variable, exactly the entry of `PARAMETER_UNITS` of
    /// `src/pkpdutils/nca/nca.py`, with the placeholders `{unit}`, `{time}` and `{dose}` which
    /// the units of a batch resolve ([`crate::assemble`]).
    pub unit: &'static str,
    /// The field of [`RowParameters`] it reports.
    pub read: fn(&RowParameters) -> f64,
    /// The rows which carry it.
    pub presence: Presence,
    /// The value a row which does not carry it contributes to the column of a mixed batch.
    pub fill: f64,
}

/// Every parameter of the single dose concentration analysis, in the order python writes it.
///
/// The order is the insertion order of the `out` dictionary of `compute_parameters`: the block
/// every row carries, then the extras of an intravenous bolus, then those of an extravascular
/// dose, then the dose dependent ones. A pair like `cl` and `cl_f` is one field under two names,
/// of which a row carries one, so a batch of one route reports them in exactly the python order
/// and a batch which mixes routes reports both names, as `merge_rows` of `nca.py` does.
///
/// `flags` is not here: it has its own section in `pkpdutils_abi::AnalysisResult` and is not a
/// parameter column.
pub(crate) const PARAMETERS: &[Parameter] = &[
    Parameter::new("cmax", "{unit}", |p| p.cmax, Presence::Always),
    Parameter::new("tmax", "{time}", |p| p.tmax, Presence::Always),
    Parameter::new("cmin", "{unit}", |p| p.cmin, Presence::Always),
    Parameter::new("tmin", "{time}", |p| p.tmin, Presence::Always),
    Parameter::new("clast", "{unit}", |p| p.clast, Presence::Always),
    Parameter::new("clast_pred", "{unit}", |p| p.clast_pred, Presence::Always),
    Parameter::new("tlast", "{time}", |p| p.tlast, Presence::Always),
    Parameter::new(
        "auc_last",
        "({unit}) * ({time})",
        |p| p.auc_last,
        Presence::Always,
    ),
    Parameter::new(
        "auc_all",
        "({unit}) * ({time})",
        |p| p.auc_all,
        Presence::Always,
    ),
    Parameter::new(
        "auc_inf_obs",
        "({unit}) * ({time})",
        |p| p.auc_inf_obs,
        Presence::Always,
    ),
    Parameter::new(
        "auc_inf_pred",
        "({unit}) * ({time})",
        |p| p.auc_inf_pred,
        Presence::Always,
    ),
    Parameter::new(
        "auc_extrap_fraction",
        "dimensionless",
        |p| p.auc_extrap_fraction,
        Presence::Always,
    ),
    Parameter::new(
        "auc_extrap_fraction_pred",
        "dimensionless",
        |p| p.auc_extrap_fraction_pred,
        Presence::Always,
    ),
    Parameter::new(
        "aumc_last",
        "({unit}) * ({time}) ** 2",
        |p| p.aumc_last,
        Presence::Always,
    ),
    Parameter::new(
        "aumc_all",
        "({unit}) * ({time}) ** 2",
        |p| p.aumc_all,
        Presence::Always,
    ),
    Parameter::new(
        "aumc_inf",
        "({unit}) * ({time}) ** 2",
        |p| p.aumc_inf,
        Presence::Always,
    ),
    Parameter::new(
        "aumc_inf_pred",
        "({unit}) * ({time}) ** 2",
        |p| p.aumc_inf_pred,
        Presence::Always,
    ),
    Parameter::new(
        "aumc_extrap_fraction",
        "dimensionless",
        |p| p.aumc_extrap_fraction,
        Presence::Always,
    ),
    Parameter::new(
        "aumc_extrap_fraction_pred",
        "dimensionless",
        |p| p.aumc_extrap_fraction_pred,
        Presence::Always,
    ),
    Parameter::new("mrt", "{time}", |p| p.mrt, Presence::Always),
    Parameter::new("mrt_last", "{time}", |p| p.mrt_last, Presence::Always),
    Parameter::new("mrt_pred", "{time}", |p| p.mrt_pred, Presence::Always),
    Parameter::new("thalf_eff", "{time}", |p| p.thalf_eff, Presence::Always),
    Parameter::new("lambda_z", "1 / ({time})", |p| p.lambda_z, Presence::Always),
    Parameter::new(
        "lambda_z_stderr",
        "1 / ({time})",
        |p| p.lambda_z_stderr,
        Presence::Always,
    ),
    Parameter::new(
        "lambda_z_intercept",
        "dimensionless",
        |p| p.lambda_z_intercept,
        Presence::Always,
    ),
    Parameter::new(
        "lambda_z_r2",
        "dimensionless",
        |p| p.lambda_z_r2,
        Presence::Always,
    ),
    Parameter::new(
        "lambda_z_r2_adj",
        "dimensionless",
        |p| p.lambda_z_r2_adj,
        Presence::Always,
    ),
    Parameter::new(
        "lambda_z_n_points",
        "dimensionless",
        |p| p.lambda_z_n_points,
        Presence::Always,
    ),
    Parameter::new(
        "lambda_z_t_first",
        "{time}",
        |p| p.lambda_z_t_first,
        Presence::Always,
    ),
    Parameter::new(
        "lambda_z_t_last",
        "{time}",
        |p| p.lambda_z_t_last,
        Presence::Always,
    ),
    Parameter::new(
        "lambda_z_span",
        "dimensionless",
        |p| p.lambda_z_span,
        Presence::Always,
    ),
    Parameter::new("thalf", "{time}", |p| p.thalf, Presence::Always),
    Parameter::new("c0", "{unit}", |p| p.c0, Presence::Bolus),
    Parameter::integer(
        "c0_method",
        "dimensionless",
        |p| p.c0_method,
        Presence::Bolus,
    ),
    Parameter::new(
        "auc_back_extrap_fraction",
        "dimensionless",
        |p| p.auc_back_extrap_fraction,
        Presence::Bolus,
    ),
    Parameter::new(
        "auc_back_extrap_fraction_pred",
        "dimensionless",
        |p| p.auc_back_extrap_fraction_pred,
        Presence::Bolus,
    ),
    Parameter::new(
        "aumc_back_extrap_fraction",
        "dimensionless",
        |p| p.aumc_back_extrap_fraction,
        Presence::Bolus,
    ),
    Parameter::new("tlag", "{time}", |p| p.tlag, Presence::Oral),
    Parameter::new("cmax_half", "{unit}", |p| p.cmax_half, Presence::Oral),
    Parameter::new("tmax_half", "{time}", |p| p.tmax_half, Presence::Oral),
    Parameter::new(
        "cl",
        "({dose}) / (({unit}) * ({time}))",
        |p| p.cl,
        Presence::DoseIv,
    ),
    Parameter::new(
        "cl_f",
        "({dose}) / (({unit}) * ({time}))",
        |p| p.cl,
        Presence::DoseExtravascular,
    ),
    Parameter::new("vz", "({dose}) / ({unit})", |p| p.vz, Presence::DoseIv),
    Parameter::new(
        "vz_f",
        "({dose}) / ({unit})",
        |p| p.vz,
        Presence::DoseExtravascular,
    ),
    Parameter::new(
        "cl_pred",
        "({dose}) / (({unit}) * ({time}))",
        |p| p.cl_pred,
        Presence::DoseIv,
    ),
    Parameter::new(
        "cl_f_pred",
        "({dose}) / (({unit}) * ({time}))",
        |p| p.cl_pred,
        Presence::DoseExtravascular,
    ),
    Parameter::new(
        "vz_pred",
        "({dose}) / ({unit})",
        |p| p.vz_pred,
        Presence::DoseIv,
    ),
    Parameter::new(
        "vz_f_pred",
        "({dose}) / ({unit})",
        |p| p.vz_pred,
        Presence::DoseExtravascular,
    ),
    Parameter::new("vss", "({dose}) / ({unit})", |p| p.vss, Presence::DoseIv),
    Parameter::new(
        "vss_pred",
        "({dose}) / ({unit})",
        |p| p.vss_pred,
        Presence::DoseIv,
    ),
    Parameter::new(
        "auc_inf_dn",
        "(({unit}) * ({time})) / ({dose})",
        |p| p.auc_inf_dn,
        Presence::Dose,
    ),
    Parameter::new(
        "auc_inf_pred_dn",
        "(({unit}) * ({time})) / ({dose})",
        |p| p.auc_inf_pred_dn,
        Presence::Dose,
    ),
    Parameter::new(
        "cmax_dn",
        "({unit}) / ({dose})",
        |p| p.cmax_dn,
        Presence::Dose,
    ),
];

impl Parameter {
    /// One entry of [`PARAMETERS`], a float of the result.
    const fn new(
        name: &'static str,
        unit: &'static str,
        read: fn(&RowParameters) -> f64,
        presence: Presence,
    ) -> Self {
        Self {
            name,
            unit,
            read,
            presence,
            fill: f64::NAN,
        }
    }

    /// One entry which is an integer variable of the python result.
    ///
    /// A row which does not carry it contributes `0`, the code of "no estimate", where a float
    /// contributes `NaN`: an integer variable of the python result carries no `NaN`
    /// (`INTEGER_VARIABLES` and `merge_rows` of `nca.py`).
    const fn integer(
        name: &'static str,
        unit: &'static str,
        read: fn(&RowParameters) -> f64,
        presence: Presence,
    ) -> Self {
        Self {
            name,
            unit,
            read,
            presence,
            fill: 0.0,
        }
    }

    /// Whether a row of this route, in a batch which has or has not got doses, reports it.
    pub(crate) fn reported_by(&self, route: Option<Route>, has_dose: bool) -> bool {
        self.presence.holds(route, has_dose)
    }

    /// The value a row which does not report it contributes to the column of a mixed batch.
    pub(crate) fn missing(&self) -> f64 {
        self.fill
    }
}

/// The row buffers of the analysis, long enough for the widest row of a batch.
///
/// One set per worker, reused by every row: the kernel allocates per batch and not per row.
/// Every buffer is as long as a row but the two the areas run on, which hold one point more, the
/// value the route puts at the dose time ([`insert_dose_value`], which cuts the row rather than
/// trapping when it does not fit).
#[derive(Debug)]
pub(crate) struct Scratch {
    /// The times of the row, shifted onto the dose.
    time: Vec<f64>,
    /// The values of the row, as the rules for the values below the limit left them.
    value: Vec<f64>,
    /// The values below the limit which are still part of the curve, over the columns.
    in_curve: Vec<bool>,
    /// The same, over the packed positions.
    blq: Vec<bool>,
    /// The packed times.
    tp: Vec<f64>,
    /// The packed values.
    cp: Vec<f64>,
    /// The packed points the terminal regression may use.
    permitted: Vec<bool>,
    /// The times of the row the areas run on.
    t_area: Vec<f64>,
    /// The values of the row the areas run on.
    c_area: Vec<f64>,
}

impl Scratch {
    /// Buffers for a row of `n_time` points.
    pub(crate) fn new(n_time: usize) -> Self {
        let width = n_time.saturating_add(1);
        Self {
            time: vec![f64::NAN; n_time],
            value: vec![f64::NAN; n_time],
            in_curve: vec![false; n_time],
            blq: vec![false; n_time],
            tp: vec![f64::NAN; n_time],
            cp: vec![f64::NAN; n_time],
            permitted: vec![false; n_time],
            t_area: vec![f64::NAN; width],
            c_area: vec![f64::NAN; width],
        }
    }
}

/// One row of a batch, with everything the analysis of that row needs besides the options.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowInput<'a> {
    /// The observation times, as the host sent them.
    pub times: &'a [f64],
    /// The observed values.
    pub values: &'a [f64],
    /// The dose the analysis runs against.
    pub dose: ReferenceDose,
    /// The route of the row.
    pub route: Option<Route>,
    /// Whether the batch carries dose arrays at all, which decides whether the dose dependent
    /// parameters exist (`areas.md` trap T12).
    pub has_dose: bool,
    /// The limit of quantification of the row, which `NcaOptions::lloq` overrides.
    pub lloq: Option<f64>,
    /// The terminal window of the row, which replaces `TerminalPhase::method` where it is given.
    pub window: Option<(f64, f64)>,
    /// The columns `TerminalMethod::Manual` selects, over the columns of the row as the host
    /// sent them; `None` under every other method.
    pub manual: Option<&'a [bool]>,
}

/// The parameters of one curve.
///
/// The pipeline is `areas.md` section 0 and the formulas are its section 3. The row is read
/// through `scratch`, whose buffers are overwritten, so nothing of the batch is changed.
///
/// No index here comes from the data: every loop runs to `n_valid`, which the packing bounds by
/// the length of the buffers it wrote, and every other read goes through a slice method which
/// answers a short row with `None`. A malformed row is `NaN` parameters and the flag `NO_DATA`,
/// never a panic.
///
/// `candidates`, where it is given, is filled with the windows the terminal selection could
/// have chosen from (`terminal::candidate_table`), which only an analysis of a single row
/// reports; the caller decides whether it wants them, as python's `candidate_variables` does.
///
/// Mirrors `compute_parameters` of `src/pkpdutils/nca/nca.py` for one row, with
/// `single_dose=True`.
pub(crate) fn compute_row(
    input: &RowInput<'_>,
    options: &NcaOptions,
    scratch: &mut Scratch,
    candidates: Option<&mut Vec<Candidate>>,
) -> RowParameters {
    let mut out = RowParameters::EMPTY;
    let Scratch {
        time,
        value,
        in_curve,
        blq,
        tp,
        cp,
        permitted,
        t_area,
        c_area,
    } = scratch;
    let columns = input.times.len().min(input.values.len()).min(time.len());

    // 1. the times of the row relative to its dose; a row whose dose time is not a number keeps
    // its own times and only its dose dependent parameters stay NaN
    let shift = if input.dose.time.is_finite() {
        input.dose.time
    } else {
        0.0
    };
    for j in 0..columns {
        time[j] = input.times[j] - shift;
        value[j] = input.values[j];
    }

    // 2. the values below the limit of quantification, before the row is packed
    let limit = resolve_lloq(options, input.lloq, &value[..columns]);
    let rules = options.blq_rules();
    let truncated = apply_blq(
        &mut value[..columns],
        limit,
        &rules,
        &mut in_curve[..columns],
    );

    // 4 and 5. the mask of the kept and imputed values onto the packed row, and the packing
    packed_mask(
        &time[..columns],
        &value[..columns],
        &in_curve[..columns],
        &mut blq[..],
    );
    let n_valid = pack_valid(&time[..columns], &value[..columns], tp, cp);
    // the packing never writes more points than the buffers hold, so every loop below which
    // runs to `n_valid` indexes both of them in range
    let n_valid = n_valid.min(tp.len()).min(cp.len());

    // 6. the points the terminal regression may use: the kept and imputed values are out unless
    // the rules let them in, and so is every sample of an infusion up to the end of the infusion
    let infusion_end = infusion_end(input.route, input.dose.duration);
    for (j, slot) in permitted.iter_mut().enumerate() {
        let kept = !rules.terminal_regression && blq.get(j).copied().unwrap_or(false);
        let during = infusion_end.is_some_and(|end| tp.get(j).is_some_and(|&t| t <= end));
        *slot = !kept && !during;
    }

    // 7. the row which is too short to be a curve
    let has_data = n_valid >= 2;
    if !has_data {
        out.flags |= NcaFlag::NO_DATA;
    }
    if truncated {
        out.flags |= NcaFlag::BLQ_TRUNCATED;
    }

    // 8. the observed extremes, over every valid point and over the row as it was measured
    let imax = first_maximum(cp, n_valid);
    let imin = first_minimum(cp, n_valid);
    if has_data {
        out.cmax = cp[imax];
        out.tmax = tp[imax];
        out.cmin = cp[imin];
        out.tmin = tp[imin];
        if imax == n_valid - 1 {
            out.flags |= NcaFlag::NO_MAX;
        }
        if input.route == Some(Route::Oral) && imax == 0 {
            out.flags |= NcaFlag::NO_ABSORPTION;
        }
    }

    // 9. the last measurable value: a value a rule kept or imputed is positive and not measured
    let last = (0..n_valid)
        .rev()
        .find(|&j| cp[j] > 0.0 && !blq.get(j).copied().unwrap_or(false));
    if let Some(ilast) = last
        && has_data
    {
        out.clast = cp[ilast];
        out.tlast = tp[ilast];
    }
    out.tlag = lag_time(tp, cp, n_valid, input.route, blq);

    // 10. the value the route puts at the dose time, which only the areas see
    if input.route == Some(Route::IvBolus) {
        let estimate = bolus_c0(tp, cp, n_valid, options.c0_method);
        if has_data {
            out.c0 = estimate.value;
            out.c0_method = f64::from(estimate.method);
        }
    }
    let inserted = insert_dose_value(
        tp,
        cp,
        n_valid,
        dose_value(input.route, true, out.c0),
        t_area,
        c_area,
    );
    let n_area = inserted.n.min(t_area.len()).min(c_area.len());
    let t_area = &t_area[..n_area];
    let c_area = &c_area[..n_area];

    // 11. the areas, over the row which carries the inserted point
    let to_last = trapezoid_until(t_area, c_area, out.tlast, options.auc_method);
    if has_data && last.is_some() {
        out.auc_last = to_last.auc;
        out.aumc_last = to_last.aumc;
    }
    let over_all = trapezoid(t_area, c_area, options.auc_method);
    if has_data {
        out.auc_all = over_all.auc;
        out.aumc_all = over_all.aumc;
    }

    // the terminal regression, over the row which does not
    let packed = Packed {
        t: tp,
        c: cp,
        n_valid,
        imax,
        permitted,
    };
    let fit = regress(input, options, &packed, &time[..columns], &value[..columns]);
    if let Some(table) = candidates {
        // the table is of the rule's own candidates and of the whole regressable row: neither
        // a window of the sample nor the manual mask narrows it (`terminal_fit` of terminal.py,
        // which builds it from `regressable` and `tmax_idx` whatever the method decided)
        *table = terminal::candidate_table(tp, cp, n_valid, permitted, imax, &options.terminal);
    }
    if has_data {
        out.flags |= fit.flags;
    }
    terminal_parameters(&mut out, &fit, input.route, input.dose.duration, options);
    if !has_data {
        // the regression at the last measurable time is reported only for a curve, although
        // everything which extrapolates with it is computed from it either way
        out.clast_pred = f64::NAN;
    }

    // the value closest to half the maximum, during the absorption of an extravascular dose
    if has_data && imax > 0 && input.route == Some(Route::Oral) {
        let ihalf = closest_to_half(cp, imax, out.cmax);
        out.cmax_half = cp[ihalf];
        out.tmax_half = tp[ihalf];
    }

    // the share of the exposure the value at the dose time contributes
    if input.route == Some(Route::IvBolus) && has_data {
        let first = first_segment(t_area, c_area, inserted.inserted, options);
        out.auc_back_extrap_fraction = first.auc / out.auc_inf_obs;
        out.auc_back_extrap_fraction_pred = first.auc / out.auc_inf_pred;
        out.aumc_back_extrap_fraction = first.aumc / out.aumc_inf;
    }

    if input.has_dose && input.route.is_some() {
        dose_parameters(&mut out, input.dose.amount);
    }
    out
}

/// The end of an infusion, the time up to which no sample may be regressed.
///
/// `Some` only for an infusion whose batch carries durations at all, as python's
/// `dose_duration is not None` has it; a duration which is not a number excludes the samples at
/// or before the dose, since `numpy.nan_to_num` reads it as a zero. An infinite duration excludes
/// every sample under both readings: python replaces it with the largest finite double, which no
/// finite time of a packed row is above either.
fn infusion_end(route: Option<Route>, duration: Option<f64>) -> Option<f64> {
    let duration = duration?;
    (route == Some(Route::IvInfusion)).then_some(if duration.is_nan() { 0.0 } else { duration })
}

/// The packed index of the first largest value of a row, `0` for a row without a point.
///
/// Over every valid point, the values a rule kept or imputed among them, and never over the
/// point the route put at the dose time, which lives in another row (`terminal.md` hazard 8).
fn first_maximum(cp: &[f64], n_valid: usize) -> usize {
    let mut index = 0;
    let mut peak = f64::NEG_INFINITY;
    for (j, &value) in cp.iter().take(n_valid).enumerate() {
        if value > peak {
            peak = value;
            index = j;
        }
    }
    index
}

/// The packed index of the first smallest value of a row, `0` for a row without a point.
fn first_minimum(cp: &[f64], n_valid: usize) -> usize {
    let mut index = 0;
    let mut trough = f64::INFINITY;
    for (j, &value) in cp.iter().take(n_valid).enumerate() {
        if value < trough {
            trough = value;
            index = j;
        }
    }
    index
}

/// The lag time of the absorption of an extravascular dose.
///
/// The time of the last sample at or after the dose before the first measurable value, `0` where
/// the first such sample is already measurable and `NaN` where nothing is measurable or the route
/// is not extravascular. It is **not** gated by `has_data`, so a row of one point reports it
/// (`areas.md` trap T5), which is mirrored rather than corrected.
///
/// Mirrors `_lag_time` of `src/pkpdutils/nca/nca.py`.
fn lag_time(tp: &[f64], cp: &[f64], n_valid: usize, route: Option<Route>, blq: &[bool]) -> f64 {
    if route != Some(Route::Oral) {
        return f64::NAN;
    }
    let time_at = |j: usize| (j < n_valid).then(|| tp.get(j).copied()).flatten();
    let measurable = |j: usize| {
        time_at(j).is_some_and(|t| t >= 0.0)
            && cp.get(j).is_some_and(|&c| c > 0.0)
            && !blq.get(j).copied().unwrap_or(false)
    };
    let Some(first) = (0..n_valid).find(|&j| measurable(j)) else {
        return f64::NAN;
    };
    // the sample before it, which must itself be a sample at or after the dose
    first
        .checked_sub(1)
        .and_then(time_at)
        .filter(|&t| t >= 0.0)
        .unwrap_or(0.0)
}

/// The index of the value closest to half the maximum, among the points before the maximum.
///
/// The first of them on a tie, as `numpy.argmin` picks it.
fn closest_to_half(cp: &[f64], imax: usize, cmax: f64) -> usize {
    let mut index = 0;
    let mut closest = f64::INFINITY;
    for (j, &value) in cp.iter().take(imax).enumerate() {
        let distance = (value - 0.5 * cmax).abs();
        if distance < closest {
            closest = distance;
            index = j;
        }
    }
    index
}

/// The packed row and what the terminal regression reads of it.
#[derive(Debug, Clone, Copy)]
struct Packed<'a> {
    /// The packed times.
    t: &'a [f64],
    /// The packed values.
    c: &'a [f64],
    /// How many of them are points.
    n_valid: usize,
    /// The packed index of the first largest value.
    imax: usize,
    /// The points the caller permits in a regression.
    permitted: &'a [bool],
}

/// The terminal regression of a row, under its own window or under the rule of the batch.
///
/// A row which was given a window regresses that window and nothing else; every other row
/// follows `TerminalPhase::method`, and under `TerminalMethod::Manual` the columns the options
/// name are mapped onto the packed row here, since only the caller of the regression knows how
/// the row was packed (`terminal.md` section 2). `time` and `value` are the row as the packing
/// read it, which is what carries that mask over.
fn regress(
    input: &RowInput<'_>,
    options: &NcaOptions,
    packed: &Packed<'_>,
    time: &[f64],
    value: &[f64],
) -> TerminalFit {
    let phase = &options.terminal;
    let Packed {
        t,
        c,
        n_valid,
        imax,
        permitted,
    } = *packed;
    if let Some(window) = input.window {
        // a window replaces the rule, so the manual mask does not narrow it
        return terminal::fit_in_window(t, c, n_valid, permitted, window, phase);
    }
    if phase.method == TerminalMethod::Manual {
        let mut manual = vec![false; t.len()];
        packed_mask(time, value, input.manual.unwrap_or(&[]), &mut manual);
        for (j, slot) in manual.iter_mut().enumerate() {
            *slot &= permitted.get(j).copied().unwrap_or(false);
        }
        return terminal::terminal_fit(t, c, n_valid, &manual, imax, phase);
    }
    terminal::terminal_fit(t, c, n_valid, permitted, imax, phase)
}

/// Everything which follows from the terminal regression, and the two flags it raises.
///
/// Every quantity here is `NaN` for a row without a window, by the arithmetic and not by a guard:
/// a `NaN` slope makes `lambda_z` and the half-life `NaN`, and they carry it into the
/// extrapolated areas, the residence times and the effective half-life. A `NaN` answers no
/// comparison, so neither `EXTRAPOLATION_HIGH` nor `SPAN_LOW` is raised for such a row.
fn terminal_parameters(
    out: &mut RowParameters,
    fit: &TerminalFit,
    route: Option<Route>,
    duration: Option<f64>,
    options: &NcaOptions,
) {
    let lambda_z = -fit.slope;
    // the same expression `terminal::span` evaluates, so that the two half-lives are one value
    let thalf = std::f64::consts::LN_2 / lambda_z;
    out.lambda_z = lambda_z;
    out.thalf = thalf;
    out.lambda_z_stderr = fit.se_slope;
    out.lambda_z_intercept = fit.intercept;
    out.lambda_z_r2 = fit.r2;
    out.lambda_z_r2_adj = fit.r2_adj;
    out.lambda_z_n_points = fit.n_points;
    out.lambda_z_t_first = fit.t_first;
    out.lambda_z_t_last = fit.t_last;
    out.lambda_z_span = terminal::span(fit);

    let (clast, tlast) = (out.clast, out.tlast);
    // Clast_pred of Phoenix: the regression at the last measurable time, which is not the end
    // of the window whenever a sample behind it is measurable (`areas.md` trap T6)
    let clast_pred = (fit.intercept - lambda_z * tlast).exp();
    out.clast_pred = clast_pred;
    out.auc_inf_obs = out.auc_last + clast / lambda_z;
    out.auc_inf_pred = out.auc_last + clast_pred / lambda_z;
    out.auc_extrap_fraction = (out.auc_inf_obs - out.auc_last) / out.auc_inf_obs;
    out.auc_extrap_fraction_pred = (out.auc_inf_pred - out.auc_last) / out.auc_inf_pred;
    out.aumc_inf = out.aumc_last + clast * tlast / lambda_z + clast / (lambda_z * lambda_z);
    out.aumc_inf_pred =
        out.aumc_last + clast_pred * tlast / lambda_z + clast_pred / (lambda_z * lambda_z);
    out.aumc_extrap_fraction = (out.aumc_inf - out.aumc_last) / out.aumc_inf;
    out.aumc_extrap_fraction_pred = (out.aumc_inf_pred - out.aumc_last) / out.aumc_inf_pred;

    // the drug of an infusion enters the body on average at the middle of the infusion
    let half_duration = match (route, duration) {
        (Some(Route::IvInfusion), Some(duration)) => {
            (if duration.is_nan() { 0.0 } else { duration }) / 2.0
        }
        _ => 0.0,
    };
    out.mrt = out.aumc_inf / out.auc_inf_obs - half_duration;
    out.mrt_last = out.aumc_last / out.auc_last - half_duration;
    out.mrt_pred = out.aumc_inf_pred / out.auc_inf_pred - half_duration;
    out.thalf_eff = std::f64::consts::LN_2 * out.mrt;

    if out.auc_extrap_fraction > options.extrapolation_warning {
        out.flags |= NcaFlag::EXTRAPOLATION_HIGH;
    }
    out.flags |= terminal::span_flag(fit);
}

/// The area and the moment of the first segment of the row the areas run on.
///
/// Zero where no point was inserted at the dose time, which is what python's
/// `where(insert, area[:, 0], 0.0)` writes: the fraction of a row with a sample at the dose is
/// `0`, not `NaN` (`areas.md` trap T3).
fn first_segment(t_area: &[f64], c_area: &[f64], inserted: bool, options: &NcaOptions) -> Areas {
    if !inserted {
        return Areas {
            auc: 0.0,
            aumc: 0.0,
        };
    }
    let (t, c) = (
        t_area.get(..2).unwrap_or(&[]),
        c_area.get(..2).unwrap_or(&[]),
    );
    trapezoid(t, c, options.auc_method)
}

/// The parameters which divide by the dose.
///
/// The amount is the dose where it is positive and `NaN` otherwise, so a placebo arm of dose `0`
/// and a row whose dose records are missing give `NaN` rather than an infinity
/// (`positive_dose` of `nca.py`).
fn dose_parameters(out: &mut RowParameters, amount: f64) {
    let dose = if amount > 0.0 { amount } else { f64::NAN };
    out.cl = dose / out.auc_inf_obs;
    out.vz = out.cl / out.lambda_z;
    out.cl_pred = dose / out.auc_inf_pred;
    out.vz_pred = out.cl_pred / out.lambda_z;
    out.vss = out.cl * out.mrt;
    out.vss_pred = out.cl_pred * out.mrt_pred;
    out.auc_inf_dn = out.auc_inf_obs / dose;
    out.auc_inf_pred_dn = out.auc_inf_pred / dose;
    out.cmax_dn = out.cmax / dose;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use pkpdutils_abi::{
        AnalysisResult, AucMethod, BATCH_CONTRACT, Batch, Coordinate, Kind, SampleCoordinate,
        TerminalPhase, WindowLabel,
    };

    use crate::{KernelError, analyze_batch};

    /// A tolerance of a last place or two against the doubles of the formulas of `areas.md`,
    /// evaluated by hand: the expressions and their order are reproduced exactly, only the
    /// origin the regression shifts its sums by and the libm of `ln` and `exp` may move the
    /// last bits.
    #[track_caller]
    pub(crate) fn close(actual: f64, expected: f64) {
        let tolerance = 1e-12 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} is not {expected}"
        );
    }

    /// A batch of several samples of the same width, without doses and without a route.
    ///
    /// The samples are labelled `A`, `B`, ... by the coordinate of their dimension, which is
    /// what a terminal window is keyed by.
    pub(crate) fn many(times: &[&[f64]], values: &[&[f64]]) -> Batch {
        let n_samples = times.len();
        let n_time = times.first().map_or(0, |row| row.len());
        let labels = (0..n_samples)
            .map(|i| {
                char::from_u32('A' as u32 + u32::try_from(i).unwrap())
                    .unwrap()
                    .to_string()
            })
            .collect();
        Batch {
            contract: BATCH_CONTRACT.to_string(),
            sample_dims: vec!["subject".to_string()],
            sample_shape: vec![n_samples],
            n_time,
            times: times.concat(),
            values: values.concat(),
            sd: None,
            se: None,
            n: None,
            nominal_time: None,
            lloq: None,
            max_doses: 0,
            dose_amount: Vec::new(),
            dose_time: Vec::new(),
            dose_duration: Vec::new(),
            time_unit: "h".to_string(),
            unit: "mg/l".to_string(),
            dose_unit: None,
            substance: Some("caffeine".to_string()),
            substances: None,
            route: None,
            routes: None,
            tissue: Some("plasma".to_string()),
            coords: vec![SampleCoordinate {
                name: "subject".to_string(),
                dims: vec!["subject".to_string()],
                values: Coordinate::Str(labels),
            }],
        }
    }

    /// A batch of one sample, without doses and without a route.
    pub(crate) fn one(times: &[f64], values: &[f64]) -> Batch {
        many(&[times], &[values])
    }

    /// `batch` with one dose of `amount` at `time` per sample, given by `route`.
    pub(crate) fn dosed(batch: Batch, route: Route, amount: f64, time: f64) -> Batch {
        let n_samples = batch.sample_shape.iter().product();
        Batch {
            max_doses: 1,
            dose_amount: vec![amount; n_samples],
            dose_time: vec![time; n_samples],
            dose_duration: vec![f64::NAN; n_samples],
            dose_unit: Some("mg".to_string()),
            route: Some(route),
            ..batch
        }
    }

    /// The options of a hand computed area: the linear rule, everything else as it comes.
    pub(crate) fn linear() -> NcaOptions {
        NcaOptions {
            auc_method: AucMethod::Linear,
            ..NcaOptions::default()
        }
    }

    /// The analysis of a batch which is expected to run.
    #[track_caller]
    pub(crate) fn run(batch: &Batch, options: &NcaOptions) -> AnalysisResult {
        analyze_batch(batch, options).expect("the batch is analysable")
    }

    /// The names of the parameter columns, in the order they are reported.
    pub(crate) fn names(result: &AnalysisResult) -> Vec<&str> {
        result.parameters.iter().map(|p| p.name.as_str()).collect()
    }

    /// One parameter of one sample.
    #[track_caller]
    pub(crate) fn at(result: &AnalysisResult, name: &str, row: usize) -> f64 {
        let column = result
            .parameters
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the result has no column {name}: {:?}", names(result)));
        column.values[row]
    }

    /// One parameter of the first sample.
    #[track_caller]
    pub(crate) fn value(result: &AnalysisResult, name: &str) -> f64 {
        at(result, name, 0)
    }

    /// The flags of the first sample.
    fn flags(result: &AnalysisResult) -> NcaFlag {
        NcaFlag::from_bits_truncate(result.flags[0])
    }

    /// The canonical oral curve: one dose of 100 at t = 0, a maximum of 8 at t = 2 and a
    /// terminal phase which halves every hour from t = 3 to t = 6.
    pub(crate) fn oral_curve() -> Batch {
        let batch = one(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[0.0, 4.0, 8.0, 4.0, 2.0, 1.0, 0.5],
        );
        dosed(batch, Route::Oral, 100.0, 0.0)
    }

    /// Every parameter of an extravascular curve of a dosed batch, in the reported order.
    const ORAL_PARAMETERS: &[&str] = &[
        "cmax",
        "tmax",
        "cmin",
        "tmin",
        "clast",
        "clast_pred",
        "tlast",
        "auc_last",
        "auc_all",
        "auc_inf_obs",
        "auc_inf_pred",
        "auc_extrap_fraction",
        "auc_extrap_fraction_pred",
        "aumc_last",
        "aumc_all",
        "aumc_inf",
        "aumc_inf_pred",
        "aumc_extrap_fraction",
        "aumc_extrap_fraction_pred",
        "mrt",
        "mrt_last",
        "mrt_pred",
        "thalf_eff",
        "lambda_z",
        "lambda_z_stderr",
        "lambda_z_intercept",
        "lambda_z_r2",
        "lambda_z_r2_adj",
        "lambda_z_n_points",
        "lambda_z_t_first",
        "lambda_z_t_last",
        "lambda_z_span",
        "thalf",
        "tlag",
        "cmax_half",
        "tmax_half",
        "cl_f",
        "vz_f",
        "cl_f_pred",
        "vz_f_pred",
        "auc_inf_dn",
        "auc_inf_pred_dn",
        "cmax_dn",
    ];

    #[test]
    fn an_oral_row_reports_the_oral_parameter_set_and_no_iv_parameter() {
        // areas.md section 3.11: the 33 parameters of every concentration curve, the three of
        // an extravascular dose and the seven a dose and a route add, `flags` travelling in
        // its own section of the result rather than as a column.
        let result = run(&oral_curve(), &linear());
        assert_eq!(names(&result), ORAL_PARAMETERS);
        assert_eq!(result.parameters.len(), 43);
        for iv_only in [
            "c0",
            "c0_method",
            "auc_back_extrap_fraction",
            "auc_back_extrap_fraction_pred",
            "aumc_back_extrap_fraction",
            "cl",
            "vz",
            "vss",
            "vss_pred",
        ] {
            assert!(
                !names(&result).contains(&iv_only),
                "{iv_only} belongs to an intravenous route"
            );
        }

        // the numbers, hand computed from the formulas of areas.md section 3
        assert_eq!(value(&result, "cmax"), 8.0);
        assert_eq!(value(&result, "tmax"), 2.0);
        assert_eq!(value(&result, "cmin"), 0.0);
        assert_eq!(value(&result, "tmin"), 0.0);
        assert_eq!(value(&result, "clast"), 0.5);
        assert_eq!(value(&result, "tlast"), 6.0);
        // the linear trapezoids 2 + 6 + 6 + 3 + 1.5 + 0.75 and their moments
        assert_eq!(value(&result, "auc_last"), 19.25);
        assert_eq!(value(&result, "aumc_last"), 46.5);
        assert_eq!(value(&result, "auc_all"), 19.25, "no trailing zero to add");
        assert_eq!(value(&result, "aumc_all"), 46.5);
        // the four points from t = 3 halve every hour: the slope of ln c is -ln 2
        assert_eq!(value(&result, "lambda_z_n_points"), 4.0);
        assert_eq!(value(&result, "lambda_z_t_first"), 3.0);
        assert_eq!(value(&result, "lambda_z_t_last"), 6.0);
        close(value(&result, "lambda_z"), 0.693_147_180_559_945_4);
        close(value(&result, "lambda_z_intercept"), 3.465_735_902_799_727);
        close(value(&result, "thalf"), 0.999_999_999_999_999_9);
        close(value(&result, "lambda_z_span"), 3.000_000_000_000_000_4);
        close(value(&result, "lambda_z_r2"), 1.0);
        close(value(&result, "auc_inf_obs"), 19.971_347_520_444_482);
        close(
            value(&result, "auc_extrap_fraction"),
            0.036_119_121_141_227_21,
        );
        close(value(&result, "aumc_inf"), 51.868_769_613_169_7);
        close(value(&result, "mrt"), 2.597_159_233_250_141);
        close(value(&result, "mrt_last"), 2.415_584_415_584_415_7);
        close(value(&result, "thalf_eff"), 1.800_213_599_992_564_6);
        close(value(&result, "cl_f"), 5.007_173_396_668_949);
        close(value(&result, "vz_f"), 7.223_824_228_245_438);
        close(value(&result, "auc_inf_dn"), 0.199_713_475_204_444_82);
        assert_eq!(value(&result, "cmax_dn"), 0.08);
        // the extravascular extras: the first sample is at the dose and measurable from the
        // second on, and 4 is exactly half the maximum
        assert_eq!(value(&result, "tlag"), 0.0);
        assert_eq!(value(&result, "cmax_half"), 4.0);
        assert_eq!(value(&result, "tmax_half"), 1.0);
        assert_eq!(flags(&result), NcaFlag::NONE);
    }

    #[test]
    fn a_bolus_row_reports_c0_and_the_back_extrapolated_fractions() {
        // areas.md sections 3.8 and 4: the value at the dose of an intravenous bolus is
        // back-extrapolated from the first two samples, inserted for the areas alone, and the
        // share of the exposure it contributes is reported.
        let batch = one(&[1.0, 2.0, 3.0, 4.0, 5.0], &[8.0, 4.0, 2.0, 1.0, 0.5]);
        let result = run(&dosed(batch, Route::IvBolus, 100.0, 0.0), &linear());

        for bolus_only in [
            "c0",
            "c0_method",
            "auc_back_extrap_fraction",
            "auc_back_extrap_fraction_pred",
            "aumc_back_extrap_fraction",
            "cl",
            "vz",
            "vss",
            "vss_pred",
        ] {
            assert!(names(&result).contains(&bolus_only), "{bolus_only}");
        }
        for extravascular in ["tlag", "cmax_half", "tmax_half", "cl_f", "vz_f"] {
            assert!(
                !names(&result).contains(&extravascular),
                "{extravascular} belongs to an extravascular dose"
            );
        }
        assert_eq!(result.parameters.len(), 47);

        // c0 = exp(ln 8 - (ln 4 - ln 8) / (2 - 1) * 1) = 8 * 8 / 4
        close(value(&result, "c0"), 16.0);
        assert_eq!(
            value(&result, "c0_method"),
            1.0,
            "the back-extrapolation, not the first value"
        );
        // the inserted point is no sample: the maximum stays the first measured value
        assert_eq!(value(&result, "cmax"), 8.0);
        assert_eq!(value(&result, "tmax"), 1.0);
        // (16 + 8)/2 + 6 + 3 + 1.5 + 0.75, with c0 one place below 16
        close(value(&result, "auc_last"), 23.249_999_999_999_996);
        // the moment of the first segment is (0 * c0 + 1 * 8) / 2, free of c0
        assert_eq!(value(&result, "aumc_last"), 27.25);
        close(value(&result, "auc_inf_obs"), 23.971_347_520_444_48);
        close(
            value(&result, "auc_back_extrap_fraction"),
            0.500_597_640_152_083_2,
        );
        close(
            value(&result, "auc_back_extrap_fraction_pred"),
            0.500_597_640_152_083_2,
        );
        close(
            value(&result, "aumc_back_extrap_fraction"),
            0.125_401_983_532_464_6,
        );
        close(value(&result, "cl"), 4.171_647_001_267_361);
        close(value(&result, "vss"), 5.550_993_122_426_232);
        assert_eq!(flags(&result), NcaFlag::NONE);
    }

    #[test]
    fn a_row_without_a_dose_omits_every_dose_dependent_parameter() {
        // areas.md trap T12: the dose dependent parameters exist only where the batch has
        // doses and the route is known; a row whose dose is missing in a batch which has them
        // reports them as NaN instead.
        let dose_dependent = [
            "cl",
            "cl_f",
            "vz",
            "vz_f",
            "vss",
            "auc_inf_dn",
            "auc_inf_pred_dn",
            "cmax_dn",
        ];

        // a batch with a route and no dose arrays at all: the route extras are there and
        // nothing which divides by the dose is
        let without_doses = Batch {
            route: Some(Route::Oral),
            ..one(
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                &[0.0, 4.0, 8.0, 4.0, 2.0, 1.0, 0.5],
            )
        };
        let result = run(&without_doses, &linear());
        assert!(names(&result).contains(&"tlag"), "the route is known");
        for name in dose_dependent {
            assert!(!names(&result).contains(&name), "{name} needs a dose");
        }
        assert_eq!(value(&result, "auc_last"), 19.25);
        assert!(
            names(&run(&oral_curve(), &linear())).contains(&"cl_f"),
            "the same curve with a dose does report it"
        );

        // a batch with doses and no route: no dose dependent parameter, no route extra, and
        // nothing inserted at the dose, so the area starts at the first sample
        let doseless_route = Batch {
            route: None,
            ..dosed(
                one(&[1.0, 2.0, 3.0, 4.0, 5.0], &[8.0, 4.0, 2.0, 1.0, 0.5]),
                Route::Oral,
                100.0,
                0.0,
            )
        };
        let without_route = run(&doseless_route, &linear());
        for name in dose_dependent.iter().chain(&["tlag", "c0"]) {
            assert!(
                !names(&without_route).contains(name),
                "{name} needs a route"
            );
        }
        assert_eq!(
            value(&without_route, "auc_last"),
            11.25,
            "6 + 3 + 1.5 + 0.75, nothing added before the first sample"
        );

        // a row whose dose records are missing in a batch which has them: the columns are
        // there and the values are NaN
        let missing = Batch {
            dose_amount: vec![f64::NAN],
            dose_time: vec![f64::NAN],
            ..oral_curve()
        };
        let result = run(&missing, &linear());
        assert!(names(&result).contains(&"cl_f"));
        assert!(value(&result, "cl_f").is_nan());
        assert!(value(&result, "cmax_dn").is_nan());
        assert_eq!(value(&result, "auc_last"), 19.25, "the curve is analysed");
        assert_eq!(value(&result, "tmax"), 2.0, "and its times are not shifted");
    }

    #[test]
    fn a_non_positive_dose_gives_nan_and_not_an_infinity() {
        // `positive_dose` of nca.py: a dose of 0 is the placebo arm and a negative dose is no
        // dose, so every parameter which divides by it is NaN rather than an infinity.
        for amount in [0.0, -5.0] {
            let batch = Batch {
                dose_amount: vec![amount],
                ..oral_curve()
            };
            let result = run(&batch, &linear());
            for name in [
                "cl_f",
                "vz_f",
                "cl_f_pred",
                "vz_f_pred",
                "auc_inf_dn",
                "auc_inf_pred_dn",
                "cmax_dn",
            ] {
                let value = value(&result, name);
                assert!(value.is_nan(), "{name} of a dose of {amount} is {value}");
            }
            assert_eq!(value(&result, "auc_last"), 19.25, "the curve is analysed");
            close(value(&result, "auc_inf_obs"), 19.971_347_520_444_482);
            assert_eq!(flags(&result), NcaFlag::NONE, "a dose raises no flag");
        }
    }

    #[test]
    fn clearance_follows_the_route_between_cl_and_cl_f() {
        // areas.md section 3.9: the suffix is `_f` for every route which is not intravenous,
        // and the volume at steady state belongs to the intravenous routes alone. The curve
        // starts at the dose, so no route inserts anything and the three agree on the number.
        let oral = run(&oral_curve(), &linear());
        let bolus = run(
            &dosed(
                one(
                    &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    &[0.0, 4.0, 8.0, 4.0, 2.0, 1.0, 0.5],
                ),
                Route::IvBolus,
                100.0,
                0.0,
            ),
            &linear(),
        );
        let infusion = run(
            &dosed(
                one(
                    &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    &[0.0, 4.0, 8.0, 4.0, 2.0, 1.0, 0.5],
                ),
                Route::IvInfusion,
                100.0,
                0.0,
            ),
            &linear(),
        );

        for iv in [&bolus, &infusion] {
            assert!(names(iv).contains(&"cl") && names(iv).contains(&"vz"));
            assert!(names(iv).contains(&"vss") && names(iv).contains(&"vss_pred"));
            assert!(!names(iv).contains(&"cl_f") && !names(iv).contains(&"vz_f_pred"));
            close(value(iv, "cl"), 5.007_173_396_668_949);
        }
        assert!(names(&oral).contains(&"cl_f") && names(&oral).contains(&"cl_f_pred"));
        assert!(!names(&oral).contains(&"cl") && !names(&oral).contains(&"vss"));
        close(value(&oral, "cl_f"), 5.007_173_396_668_949);
        assert_eq!(
            value(&oral, "cl_f"),
            value(&bolus, "cl"),
            "one quantity under two names"
        );
    }

    #[test]
    fn the_predicted_family_differs_from_the_observed_one_only_in_clast() {
        // areas.md section 6: the predicted family extrapolates from `clast_pred`, the
        // regression at the last measurable time, where the observed one extrapolates from the
        // measured `clast`; `auc_last`, `aumc_last`, `tlast` and the regression are shared.
        // ln c is 5, 4, 2, 3, 1 at t = 0 to 4, so the four points from t = 1 regress to a
        // slope of -0.8 and an intercept of 4.5, which the last point does not lie on.
        let values: Vec<f64> = [5.0_f64, 4.0, 2.0, 3.0, 1.0]
            .iter()
            .map(|y| y.exp())
            .collect();
        let batch = one(&[0.0, 1.0, 2.0, 3.0, 4.0], &values);
        let result = run(&dosed(batch, Route::Oral, 100.0, 0.0), &linear());

        close(value(&result, "lambda_z"), 0.8);
        close(value(&result, "lambda_z_intercept"), 4.5);
        assert_eq!(value(&result, "tlast"), 4.0);
        close(value(&result, "clast"), std::f64::consts::E);
        // exp(4.5 - 0.8 * 4) = exp(1.3), which is above the measured value
        close(value(&result, "clast_pred"), 3.669_296_667_619_243_6);

        close(value(&result, "auc_last"), 157.638_463_520_780_36);
        close(value(&result, "aumc_last"), 135.069_436_657_486_62);
        close(value(&result, "auc_inf_obs"), 161.036_315_806_354_17);
        close(value(&result, "auc_inf_pred"), 162.225_084_355_304_4);
        close(value(&result, "aumc_inf"), 152.908_161_156_749_1);
        close(value(&result, "aumc_inf_pred"), 159.149_196_038_737_9);
        close(
            value(&result, "auc_extrap_fraction"),
            0.021_099_913_200_074_17,
        );
        close(
            value(&result, "auc_extrap_fraction_pred"),
            0.028_273_191_243_830_403,
        );
        close(
            value(&result, "aumc_extrap_fraction"),
            0.116_662_998_000_320_28,
        );
        close(
            value(&result, "aumc_extrap_fraction_pred"),
            0.151_303_053_867_706_1,
        );
        close(value(&result, "mrt"), 0.949_525_952_522_540_5);
        close(value(&result, "mrt_pred"), 0.981_039_379_151_563_8);
        close(value(&result, "mrt_last"), 0.856_830_456_481_081_9);
        close(value(&result, "cl_f"), 0.620_977_942_144_738_2);
        close(value(&result, "cl_f_pred"), 0.616_427_480_357_973_5);
        close(value(&result, "vz_f"), 0.776_222_427_680_922_8);
        close(value(&result, "vz_f_pred"), 0.770_534_350_447_466_8);
        close(value(&result, "auc_inf_dn"), 1.610_363_158_063_541_6);
        close(value(&result, "auc_inf_pred_dn"), 1.622_250_843_553_044);
        // the effective half-life has no predicted counterpart: it follows the observed one
        close(value(&result, "thalf_eff"), 0.658_161_236_859_495_3);
        assert_eq!(
            flags(&result),
            NcaFlag::NO_ABSORPTION,
            "the maximum is the first sample of an extravascular curve"
        );
    }

    #[test]
    fn a_row_with_one_valid_point_carries_no_data_and_does_not_panic() {
        // areas.md section 1: a row of fewer than two points is no curve. Every parameter is
        // NaN and the row is flagged - every parameter but `tlag`, which trap T5 records as
        // ungated and which is mirrored rather than corrected.
        let batch = one(&[0.0, f64::NAN, 2.0], &[5.0, 3.0, f64::NAN]);
        let result = run(&dosed(batch, Route::Oral, 100.0, 0.0), &linear());

        assert_eq!(flags(&result), NcaFlag::NO_DATA);
        for name in names(&result) {
            if name == "tlag" {
                continue;
            }
            assert!(
                value(&result, name).is_nan(),
                "{name} of a row of one point is {}",
                value(&result, name)
            );
        }
        assert_eq!(
            value(&result, "tlag"),
            0.0,
            "the one point is measurable and sits at the dose"
        );

        // the same row as an intravenous bolus: no estimate of the value at the dose either
        let bolus = run(
            &dosed(
                one(&[1.0, f64::NAN, 2.0], &[5.0, 3.0, f64::NAN]),
                Route::IvBolus,
                100.0,
                0.0,
            ),
            &linear(),
        );
        assert_eq!(flags(&bolus), NcaFlag::NO_DATA);
        assert!(value(&bolus, "c0").is_nan());
        assert_eq!(value(&bolus, "c0_method"), 0.0, "no estimate");
    }

    #[test]
    fn a_row_of_all_nan_values_reports_every_parameter_nan_and_no_tmax_at_index_zero() {
        // The maximum of a row without a point is the index 0 of numpy's argmax over an empty
        // mask, and `has_data` is what keeps it out of the result: `tmax` is NaN and not the
        // first time of the row.
        let batch = one(&[0.0, 1.0, 2.0, 3.0], &[f64::NAN; 4]);
        let result = run(&dosed(batch, Route::Oral, 100.0, 0.0), &linear());

        assert_eq!(flags(&result), NcaFlag::NO_DATA);
        for name in names(&result) {
            assert!(
                value(&result, name).is_nan(),
                "{name} of an empty row is {}",
                value(&result, name)
            );
        }
        assert!(value(&result, "tmax").is_nan(), "not the time of index 0");
        assert!(value(&result, "cmax").is_nan());
        assert!(value(&result, "tlag").is_nan(), "nothing is measurable");
    }

    #[test]
    fn every_flag_of_the_single_dose_path_is_set_by_its_own_condition() {
        // terminal.md section 6, every flag this plan owns, each on a row which raises it and
        // none of them on the canonical curve.
        let baseline = run(&oral_curve(), &linear());
        assert_eq!(flags(&baseline), NcaFlag::NONE);

        // a rising curve has no declining window, although it has windows enough
        let rising: Vec<f64> = [1.0_f64, 2.0, 3.0, 4.0].iter().map(|y| y.exp()).collect();
        let positive = run(
            &one(&[0.0, 1.0, 2.0, 3.0], &rising),
            &NcaOptions {
                terminal: TerminalPhase {
                    exclude_cmax: false,
                    ..TerminalPhase::default()
                },
                ..linear()
            },
        );
        assert!(flags(&positive).contains(NcaFlag::POSITIVE_SLOPE));
        assert!(!flags(&positive).contains(NcaFlag::TOO_FEW_POINTS));

        // three points with the maximum in the middle leave one point after it
        let too_few = run(&one(&[0.0, 1.0, 2.0], &[1.0, 5.0, 2.0]), &linear());
        assert!(flags(&too_few).contains(NcaFlag::TOO_FEW_POINTS));
        assert!(!flags(&too_few).contains(NcaFlag::NO_DATA));

        // a slow decline sampled over three hours extrapolates 86 % of its exposure and
        // covers a seventh of a half-life
        let slow: Vec<f64> = [0.0, 1.0, 2.0, 3.0]
            .iter()
            .map(|t: &f64| 10.0 * (-0.05 * t).exp())
            .collect();
        let extrapolated = run(
            &dosed(
                one(&[0.0, 1.0, 2.0, 3.0], &slow),
                Route::IvBolus,
                100.0,
                0.0,
            ),
            &linear(),
        );
        assert!(flags(&extrapolated).contains(NcaFlag::EXTRAPOLATION_HIGH));
        close(
            value(&extrapolated, "auc_extrap_fraction"),
            0.860_683_001_157_990_3,
        );

        // the terminal window of this curve covers exactly one half-life
        let short_span = run(
            &one(&[0.0, 1.0, 2.0, 3.0, 4.0], &[1.0, 10.0, 8.0, 6.0, 4.0]),
            &linear(),
        );
        assert!(flags(&short_span).contains(NcaFlag::SPAN_LOW));
        close(value(&short_span, "lambda_z_span"), 1.0);

        // the maximum is the last point of the row
        let no_max = run(&one(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]), &linear());
        assert!(flags(&no_max).contains(NcaFlag::NO_MAX));

        // and the first one of an extravascular curve
        let no_absorption = run(
            &dosed(
                one(&[0.0, 1.0, 2.0], &[5.0, 2.0, 1.0]),
                Route::Oral,
                100.0,
                0.0,
            ),
            &linear(),
        );
        assert!(flags(&no_absorption).contains(NcaFlag::NO_ABSORPTION));
        assert!(!flags(&baseline).contains(NcaFlag::NO_ABSORPTION));

        // two values below the limit of quantification are dropped
        let truncated = run(
            &oral_curve(),
            &NcaOptions {
                lloq: Some(1.0),
                ..linear()
            },
        );
        assert!(flags(&truncated).contains(NcaFlag::BLQ_TRUNCATED));
        assert_eq!(value(&truncated, "clast"), 1.0, "the 0.5 is gone");

        // a row of one point is no curve
        let no_data = run(&one(&[0.0, 1.0], &[5.0, f64::NAN]), &linear());
        assert!(flags(&no_data).contains(NcaFlag::NO_DATA));

        // a regressed point four orders below the peak of the row
        let low = run(
            &one(&[0.0, 1.0, 2.0, 3.0], &[1e9, 1e5, 1e1, 1e-3]),
            &linear(),
        );
        assert!(flags(&low).contains(NcaFlag::TERMINAL_LOW_CONCENTRATION));
        assert!(
            !flags(&baseline).contains(NcaFlag::TERMINAL_LOW_CONCENTRATION),
            "the canonical curve stays within a factor of 1e8"
        );
    }

    #[test]
    fn auc_last_masks_whole_segments_by_time_and_does_not_cut_the_row() {
        // `auc_aumc` of auc.py masks the segments whose END time is at or before `tlast`,
        // which is not the same as cutting the packed row at the index of `tlast`: nothing in
        // this kernel sorts a row (trap T1), so a row whose times fall back on themselves
        // keeps the segments of the window wherever they sit.
        let batch = one(&[0.0, 1.0, 3.0, 2.0], &[10.0, 8.0, 0.0, 4.0]);
        let result = run(&batch, &linear());

        assert_eq!(value(&result, "tlast"), 2.0, "the last positive value");
        assert_eq!(value(&result, "clast"), 4.0);
        // the segments which end at or before t = 2: (0,10)-(1,8) and (3,0)-(2,4), the second
        // of them over a negative interval. Cutting the row at the index of tlast would add
        // the segment (1,8)-(3,0) and give 15.0.
        assert_eq!(value(&result, "auc_last"), 7.0);
        assert_eq!(value(&result, "aumc_last"), 0.0);
        assert_eq!(value(&result, "auc_all"), 15.0, "every segment of the row");
        assert_eq!(value(&result, "aumc_all"), 8.0);
    }

    #[test]
    fn a_protocol_with_a_hole_is_read_as_the_python_kernel_reads_it() {
        // An upstream oddity, mirrored rather than corrected: the reference dose is the first
        // COLUMN of the protocol where any column is a dose, not the first column which is
        // one. A protocol whose amount is missing and whose time is not therefore shifts the
        // curve onto a dose whose amount is NaN.
        let batch = Batch {
            max_doses: 2,
            dose_amount: vec![f64::NAN, 100.0],
            dose_time: vec![5.0, 0.0],
            dose_duration: vec![f64::NAN, f64::NAN],
            dose_unit: Some("mg".to_string()),
            route: Some(Route::Oral),
            ..one(
                &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                &[0.0, 4.0, 8.0, 4.0, 2.0, 1.0, 0.5],
            )
        };
        let result = run(&batch, &linear());

        assert_eq!(
            value(&result, "tmax"),
            2.0,
            "the times are shifted by the 5.0 of the first column"
        );
        assert_eq!(value(&result, "auc_last"), 19.25);
        assert!(
            value(&result, "cl_f").is_nan(),
            "and the amount of that column is no amount"
        );
    }

    #[test]
    fn a_terminal_window_replaces_the_rule_and_is_narrowed_before_the_statistics() {
        // terminal.md section 0: a row with a window of its own regresses the points of that
        // window and follows no rule. The statistics are of the selected points alone, so the
        // window ends where it was told to and not at the last point of the row.
        let times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let values = [1.0, 10.0, 8.0, 6.0, 4.0, 2.0];
        let windowed = NcaOptions {
            terminal: TerminalPhase {
                windows: Some([(WindowLabel::Str("A".to_string()), (1.0, 3.0))].into()),
                ..TerminalPhase::default()
            },
            ..linear()
        };
        let result = run(&one(&times, &values), &windowed);

        assert_eq!(value(&result, "lambda_z_n_points"), 3.0);
        assert_eq!(value(&result, "lambda_z_t_first"), 1.0);
        assert_eq!(
            value(&result, "lambda_z_t_last"),
            3.0,
            "the window ends where it was given, not at the last point"
        );
        close(value(&result, "lambda_z"), 0.255_412_811_882_995_47);

        // without the window the rule of the batch picks a different window altogether
        let by_rule = run(&one(&times, &values), &linear());
        assert_eq!(value(&by_rule, "lambda_z_t_first"), 3.0);
        assert_eq!(value(&by_rule, "lambda_z_t_last"), 5.0);
        close(value(&by_rule, "lambda_z"), 0.549_306_144_334_054_9);

        // the key `*` names every sample the mapping does not
        let default_key = NcaOptions {
            terminal: TerminalPhase {
                windows: Some([(WindowLabel::Str("*".to_string()), (1.0, 3.0))].into()),
                ..TerminalPhase::default()
            },
            ..linear()
        };
        let starred = run(&one(&times, &values), &default_key);
        assert_eq!(value(&starred, "lambda_z_t_last"), 3.0);

        // and a key which names no sample is refused rather than ignored
        let unknown = NcaOptions {
            terminal: TerminalPhase {
                windows: Some([(WindowLabel::Str("Z".to_string()), (1.0, 3.0))].into()),
                ..TerminalPhase::default()
            },
            ..linear()
        };
        let error = analyze_batch(&one(&times, &values), &unknown).unwrap_err();
        assert!(
            matches!(&error, KernelError::Options(message) if message.contains("Z")),
            "{error}"
        );
    }

    #[test]
    fn manual_points_are_columns_of_the_curve_and_not_packed_positions() {
        // terminal.md section 2: the indices of `TerminalPhase::points` are indices into the
        // curve as it was given, so a row with a gap in it regresses other points than a port
        // which read them as positions of the packed row would.
        let times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let values = [1.0, f64::NAN, 100.0, 50.0, 25.0, 12.5];
        let manual = |points: Vec<i64>| NcaOptions {
            terminal: TerminalPhase {
                method: TerminalMethod::Manual,
                points: Some(points),
                ..TerminalPhase::default()
            },
            ..linear()
        };

        let result = run(&one(&times, &values), &manual(vec![3, 4, 5]));
        assert_eq!(value(&result, "lambda_z_n_points"), 3.0);
        assert_eq!(value(&result, "lambda_z_t_first"), 3.0);
        assert_eq!(value(&result, "lambda_z_t_last"), 5.0);
        close(value(&result, "lambda_z"), 0.693_147_180_559_945_2);

        // a negative index counts from the end of the curve, as numpy counts it
        let from_the_end = run(&one(&times, &values), &manual(vec![-3, -2, -1]));
        assert_eq!(
            value(&from_the_end, "lambda_z_t_first"),
            value(&result, "lambda_z_t_first")
        );
        assert_eq!(value(&from_the_end, "lambda_z"), value(&result, "lambda_z"));

        // an index which is no column of the curve is refused by name
        let error = analyze_batch(&one(&times, &values), &manual(vec![4, 5, 6])).unwrap_err();
        assert!(
            matches!(&error, KernelError::Options(message) if message.contains('6')),
            "{error}"
        );
    }

    #[test]
    fn the_maximum_is_the_first_of_the_valid_points_and_never_the_inserted_one() {
        // terminal.md hazard 8: the index of the maximum is taken over every valid point, the
        // FIRST of them on a tie, and over the row as it was measured, which is not the row
        // the areas run on.
        let tied = run(
            &one(&[0.0, 1.0, 2.0, 3.0], &[5.0, 5.0, 3.0, 2.0]),
            &linear(),
        );
        assert_eq!(value(&tied, "cmax"), 5.0);
        assert_eq!(value(&tied, "tmax"), 0.0, "the first of the two maxima");

        // the value an intravenous bolus puts at the dose is twice the largest sample here,
        // and the maximum is still the sample
        let inserted = run(
            &dosed(
                one(&[1.0, 2.0, 3.0, 4.0, 5.0], &[8.0, 4.0, 2.0, 1.0, 0.5]),
                Route::IvBolus,
                100.0,
                0.0,
            ),
            &linear(),
        );
        close(value(&inserted, "c0"), 16.0);
        assert_eq!(value(&inserted, "cmax"), 8.0);
        assert_eq!(value(&inserted, "tmax"), 1.0);
    }

    #[test]
    fn an_infusion_excludes_its_own_samples_up_to_the_end_of_the_infusion() {
        // terminal.md hazard 8 and areas.md section 3.3: no sample at or before the end of an
        // infusion may be regressed, the comparison being inclusive, and the index of the
        // maximum is taken over every valid point, excluded ones included.
        let batch = Batch {
            dose_duration: vec![2.0],
            ..dosed(
                one(
                    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    &[10.0, 8.0, 6.0, 4.0, 3.0, 2.0],
                ),
                Route::IvInfusion,
                100.0,
                0.0,
            )
        };
        let result = run(&batch, &linear());

        assert_eq!(
            value(&result, "tmax"),
            1.0,
            "an excluded sample all the same"
        );
        assert_eq!(value(&result, "cmax"), 10.0);
        assert_eq!(
            value(&result, "lambda_z_t_first"),
            3.0,
            "the sample at t = 2 ends the infusion and is excluded with it"
        );
        assert_eq!(value(&result, "lambda_z_n_points"), 4.0);
        close(value(&result, "lambda_z"), 0.358_351_893_845_611);
        // the areas run on the row which carries the zero at the dose: 5 + 9 + 7 + 5 + 3.5
        // + 2.5, and the moments 5 + 13 + 17 + 17 + 15.5 + 13.5
        assert_eq!(value(&result, "auc_last"), 32.0);
        assert_eq!(value(&result, "aumc_last"), 81.0);
        close(value(&result, "auc_inf_obs"), 37.581_106_265_512_474);
        close(value(&result, "aumc_inf"), 130.061_011_166_546_1);
        // and the mean residence time of an infusion is corrected by half its duration
        close(value(&result, "mrt"), 2.460_808_477_740_338);
        assert_eq!(
            value(&result, "mrt_last"),
            1.531_25,
            "the correction reaches it too"
        );

        // a duration which is no number excludes the samples at and before the dose, which is
        // what `numpy.nan_to_num` makes of it
        let undated = Batch {
            dose_duration: vec![f64::NAN],
            ..dosed(
                one(&[0.0, 1.0, 2.0, 3.0], &[16.0, 8.0, 4.0, 2.0]),
                Route::IvInfusion,
                100.0,
                0.0,
            )
        };
        let whole_row = NcaOptions {
            terminal: TerminalPhase {
                exclude_cmax: false,
                ..TerminalPhase::default()
            },
            ..linear()
        };
        let result = run(&undated, &whole_row);
        assert_eq!(
            value(&result, "lambda_z_t_first"),
            1.0,
            "the sample at the dose is excluded"
        );
        assert_eq!(value(&result, "lambda_z_n_points"), 3.0);
    }

    #[test]
    fn an_effect_batch_is_refused_by_name() {
        // areas.md section 0 step 3: an effect curve is a different parameter set altogether.
        let options = NcaOptions {
            kind: Kind::Effect,
            ..NcaOptions::default()
        };
        let error = analyze_batch(&oral_curve(), &options).unwrap_err();
        assert!(
            matches!(&error, KernelError::Unsupported { what } if what.contains("effect")),
            "{error}"
        );
    }

    #[test]
    fn a_multiple_dose_batch_is_refused_by_name() {
        // `is_multiple_dose` of nca.py: a row whose protocol holds two doses, or any row at
        // all when `tau` is given, is analysed over its dosing intervals instead.
        let two_doses = Batch {
            max_doses: 2,
            dose_amount: vec![100.0, 100.0],
            dose_time: vec![0.0, 12.0],
            dose_duration: vec![f64::NAN, f64::NAN],
            ..oral_curve()
        };
        let error = analyze_batch(&two_doses, &linear()).unwrap_err();
        assert!(
            matches!(&error, KernelError::Unsupported { what } if what.contains("more than one dose")),
            "{error}"
        );

        let steady_state = NcaOptions {
            tau: Some(12.0),
            ..linear()
        };
        let error = analyze_batch(&oral_curve(), &steady_state).unwrap_err();
        assert!(
            matches!(&error, KernelError::Unsupported { what } if what.contains("tau")),
            "{error}"
        );
    }

    #[test]
    fn a_ragged_batch_is_refused_as_a_batch_error() {
        // The row loop reads a slice per sample, so a batch whose arrays do not fit its shape
        // is refused before any row runs rather than cut to what fits.
        let ragged = Batch {
            values: vec![1.0, 2.0],
            ..oral_curve()
        };
        let error = analyze_batch(&ragged, &linear()).unwrap_err();
        assert!(
            matches!(&error, KernelError::Batch(message) if message.contains("values")),
            "{error}"
        );
    }

    #[test]
    fn every_analysis_of_another_plan_is_refused_by_name() {
        // The scope of this build: an analysis it does not carry is a named error and never a
        // number which looks like an answer.
        let cases: [(NcaOptions, &str); 4] = [
            (
                NcaOptions {
                    acceptance: pkpdutils_abi::Acceptance {
                        r2_adj_min: Some(0.98),
                        ..pkpdutils_abi::Acceptance::default()
                    },
                    ..linear()
                },
                "acceptance",
            ),
            (
                NcaOptions {
                    partial_aucs: [("auc_0_2".to_string(), (0.0, 2.0))].into(),
                    ..linear()
                },
                "partial",
            ),
            (
                NcaOptions {
                    uncertainty: Some(pkpdutils_abi::UncertaintyMethod::Bootstrap),
                    ..linear()
                },
                "uncertainty",
            ),
            (
                NcaOptions {
                    units: [("auc_last".to_string(), "mg*h/l".to_string())].into(),
                    ..linear()
                },
                "unit",
            ),
        ];
        for (options, named) in cases {
            let error = analyze_batch(&oral_curve(), &options).unwrap_err();
            assert!(
                matches!(&error, KernelError::Unsupported { what } if what.contains(named)),
                "{named}: {error}"
            );
        }

        // a batch which carries a spread resolves to the bootstrap on its own
        let group = Batch {
            se: Some(vec![0.1; 7]),
            ..oral_curve()
        };
        let error = analyze_batch(&group, &linear()).unwrap_err();
        assert!(
            matches!(&error, KernelError::Unsupported { what } if what.contains("uncertainty")),
            "{error}"
        );
    }

    #[test]
    fn a_batch_which_mixes_routes_reports_the_parameters_of_both() {
        // `run_rows` of nca.py splits a batch by route and merges the groups again, so the
        // result holds the union of the two parameter sets and a row carries NaN in the
        // parameters of the other route.
        let batch = Batch {
            routes: Some(vec![Route::Oral, Route::IvBolus]),
            route: None,
            ..dosed(
                many(
                    &[&[0.0, 1.0, 2.0, 3.0, 4.0], &[0.0, 1.0, 2.0, 3.0, 4.0]],
                    &[&[0.0, 8.0, 4.0, 2.0, 1.0], &[8.0, 4.0, 2.0, 1.0, 0.5]],
                ),
                Route::Oral,
                100.0,
                0.0,
            )
        };
        let result = run(&batch, &linear());

        for name in ["tlag", "cl_f", "c0", "c0_method", "cl", "vss"] {
            assert!(names(&result).contains(&name), "{name}");
        }
        assert!(at(&result, "tlag", 0).is_finite(), "the extravascular row");
        assert!(at(&result, "tlag", 1).is_nan(), "the intravenous row");
        assert!(at(&result, "c0", 0).is_nan());
        close(at(&result, "c0", 1), 8.0);
        assert_eq!(
            at(&result, "c0_method", 0),
            0.0,
            "an integer variable is filled with the code of no estimate, not with NaN"
        );
        assert!(at(&result, "cl", 0).is_nan() && at(&result, "cl_f", 0).is_finite());
        assert!(at(&result, "cl", 1).is_finite() && at(&result, "cl_f", 1).is_nan());
    }

    #[test]
    fn an_extravascular_curve_which_starts_after_the_dose_gets_a_zero_at_the_dose() {
        // areas.md section 4: an extravascular dose and an infusion put a zero at the dose
        // time of a row whose first sample comes after it, which only the areas see.
        let curve = || one(&[1.0, 2.0, 3.0, 4.0, 5.0], &[8.0, 4.0, 2.0, 1.0, 0.5]);
        for route in [Route::Oral, Route::IvInfusion] {
            let result = run(&dosed(curve(), route, 100.0, 0.0), &linear());
            // the segment (0,0) to (1,8) adds 4.0 to the 11.25 of the measured row
            assert_eq!(value(&result, "auc_last"), 15.25, "{route:?}");
            // and (0*0 + 1*8)/2 to its moment
            assert_eq!(value(&result, "aumc_last"), 27.25, "{route:?}");
            assert_eq!(value(&result, "auc_all"), 15.25);
            assert_eq!(value(&result, "cmax"), 8.0, "the zero is no sample");
            assert_eq!(value(&result, "tmax"), 1.0);
            assert_eq!(value(&result, "clast"), 0.5);
        }

        // the lag time of that row is 0 all the same: there is no sample before the first
        // measurable one to carry it (`_lag_time` of nca.py, mirrored)
        let oral = run(&dosed(curve(), Route::Oral, 100.0, 0.0), &linear());
        assert_eq!(value(&oral, "tlag"), 0.0);
    }

    #[test]
    fn a_window_is_keyed_by_the_label_of_the_sample() {
        // `sample_keys` of nca.py: the label of a batch with one sample dimension is the value
        // of its coordinate, and the position of the sample where the dimension has none.
        let times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let values = [1.0, 10.0, 8.0, 6.0, 4.0, 2.0];
        let windowed = |key: WindowLabel| NcaOptions {
            terminal: TerminalPhase {
                windows: Some([(key, (1.0, 3.0))].into()),
                ..TerminalPhase::default()
            },
            ..linear()
        };

        let numbered = Batch {
            coords: vec![SampleCoordinate {
                name: "subject".to_string(),
                dims: vec!["subject".to_string()],
                values: Coordinate::Int(vec![7]),
            }],
            ..one(&times, &values)
        };
        let result = run(&numbered, &windowed(WindowLabel::Int(7)));
        assert_eq!(value(&result, "lambda_z_t_last"), 3.0);
        let error =
            analyze_batch(&numbered, &windowed(WindowLabel::Str("7".to_string()))).unwrap_err();
        assert!(
            matches!(&error, KernelError::Options(_)),
            "a number is no string: {error}"
        );

        let anonymous = Batch {
            coords: Vec::new(),
            ..one(&times, &values)
        };
        let result = run(&anonymous, &windowed(WindowLabel::Int(0)));
        assert_eq!(
            value(&result, "lambda_z_t_last"),
            3.0,
            "the position of the sample is its label"
        );
    }
}
