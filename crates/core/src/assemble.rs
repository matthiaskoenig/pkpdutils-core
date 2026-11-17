//! The result of an analysis: its columns, their units and the attributes of the batch.
//!
//! The rows of the analysis arrive as a value per parameter per row and leave as the sections of
//! a [`pkpdutils_abi::AnalysisResult`]: one column per parameter any row reports, the candidate
//! windows of a single row over the dimension `candidate`, the acceptance variables and the
//! dataset attributes. Mirrors `_to_result` and the tail of `nca` of
//! `src/pkpdutils/nca/nca.py`.
//!
//! The unit of a column is not a label: [`pkpdutils_units::parameter_unit`] returns the
//! canonical unit **and the factor** from the unit the kernel computed in to the reported one,
//! since a volume is reported in liter and a clearance in liter per hour whatever the batch was
//! measured in. The factor multiplies the values here, exactly as `array * factor` of
//! `_to_result` does; recording the unit and dropping the factor would report every normalized
//! parameter in the wrong magnitude.

use pkpdutils_abi::{
    AnalysisResult, Attributes, Batch, BooleanColumn, Coordinate, NcaOptions, ParameterColumn,
    PointVariable, Route, SampleCoordinate, UncertaintyMethod,
};
use pkpdutils_units::{UnitContext, parameter_unit};

use crate::KernelError;
use crate::intervals::{INTERVAL_VARIABLES, Interval};
use crate::parameters::{PARAMETERS, Parameter, RowParameters};
use crate::terminal::Candidate;

/// The unit of a dimensionless variable, as python spells it.
///
/// Both the expression of `PARAMETER_UNITS` and what it resolves to, which are the same string
/// for a quantity without a dimension.
const DIMENSIONLESS: &str = "dimensionless";

/// The name of the dimension the candidate windows live on.
const CANDIDATE_DIM: &str = "candidate";

/// The extra dimension of the per-interval variables.
const INTERVAL_DIM: &str = "interval";

/// The name of the coordinate which names the dose every sample was analysed against.
///
/// `DOSE_COORDINATE` of `src/pkpdutils/nca/result.py`.
const DOSE_COORDINATE: &str = "dose_amount";

/// The number of subjects behind a sample, which every result carries.
///
/// `n` is no entry of `PARAMETER_UNITS`; `unit_expression` of `nca.py` answers it, and every
/// name ending in `_n`, with `dimensionless`.
const SUBJECT_COUNT: &str = "n";

/// What one row of a batch contributes to the result.
pub(crate) struct RowResult {
    /// The route of the row, which decides which parameters it reports.
    pub route: Option<Route>,
    /// The amount of the dose the row was analysed against, as the protocol spells it: the
    /// amount of no dose at all is `NaN` and a dose of `0` is a `0`, since this is the
    /// coordinate of the protocol and not the divisor of `positive_dose`.
    pub dose_amount: f64,
    /// The parameters of the row.
    pub parameters: RowParameters,
    /// The dosing intervals of the row, empty unless the batch is a multiple dose one and the
    /// options ask for them (`NcaOptions::intervals`).
    pub intervals: Vec<Interval>,
}

/// One variable of the candidate table: its name, its unit and the column it reads.
struct CandidateVariable {
    /// The name of the variable, exactly the python one.
    name: &'static str,
    /// The unit expression, as `PARAMETER_UNITS` of `nca.py` has it.
    unit: &'static str,
    /// The column of the table it reports.
    read: fn(&Candidate) -> f64,
}

/// The candidate windows as variables of the result, with their unit expressions.
///
/// `candidate_variables` of `nca.py` writes exactly these three, from the columns `start_time`,
/// `n` and `r2_adj` of the table.
const CANDIDATE_VARIABLES: [CandidateVariable; 3] = [
    CandidateVariable {
        name: "candidate_t_first",
        unit: "{time}",
        read: |c| c.start_time,
    },
    CandidateVariable {
        name: "candidate_n_points",
        unit: DIMENSIONLESS,
        read: |c| c.n,
    },
    CandidateVariable {
        name: "candidate_r2_adj",
        unit: DIMENSIONLESS,
        read: |c| c.r2_adj,
    },
];

/// The result of an analysis of `rows`, with every column in its reported unit.
///
/// `candidates` are the windows the terminal selection could have chosen from, which only an
/// analysis of a single row reports and which the caller therefore collects only for one.
///
/// # Errors
///
/// Whatever [`unit_of`] refuses: the units of a batch are no part of `Batch::validate`, so a
/// unit this build cannot read and a batch with doses but without a dose unit both reach this
/// point. The first is a [`KernelError::Unsupported`] a host routes its own path on, the second
/// a [`KernelError::Batch`] naming the variable.
pub(crate) fn result_of(
    batch: &Batch,
    options: &NcaOptions,
    rows: &[RowResult],
    candidates: &[Candidate],
) -> Result<AnalysisResult, KernelError> {
    let context = UnitContext {
        unit: &batch.unit,
        time: &batch.time_unit,
        dose: batch.dose_unit.as_deref(),
        // the urinary excretion analysis is another plan and another front end
        amount: None,
        volume: None,
    };
    let has_dose = batch.max_doses > 0;
    let n_samples = rows.len();
    let mut parameters = parameter_columns(rows, has_dose, &context)?;
    // `n` goes after every parameter and before the boolean variables, where `nca` writes it
    parameters.push(subject_counts(batch, n_samples, &context)?);
    Ok(AnalysisResult {
        parameters,
        point_variables: {
            let mut variables = candidate_variables(candidates, n_samples, &context)?;
            variables.extend(interval_variables(batch, rows, &context)?);
            variables
        },
        boolean_variables: acceptance_variables(n_samples),
        text_variables: Vec::new(),
        flags: rows.iter().map(|r| r.parameters.flags.bits()).collect(),
        coords: coordinates(batch, rows, has_dose),
        ..AnalysisResult::new(
            batch.sample_dims.clone(),
            batch.sample_shape.clone(),
            Attributes::of(batch, resolved_options(batch, options)),
        )
    })
}

/// The uncertainty the analysis really propagates, which is what the result records.
///
/// `None` is the automatic choice: the parametric bootstrap for a batch which carries a spread
/// per point, nothing for a batch of individual curves. Mirrors `NCAOptions.resolve_uncertainty`
/// of `src/pkpdutils/nca/options.py`, which `nca` reads both to decide what to run and to write
/// the resolved options onto the result.
pub(crate) fn resolved_uncertainty(batch: &Batch, options: &NcaOptions) -> UncertaintyMethod {
    options.uncertainty.unwrap_or({
        if batch.sd.is_some() || batch.se.is_some() {
            UncertaintyMethod::Bootstrap
        } else {
            UncertaintyMethod::None
        }
    })
}

/// The options the analysis really ran with, which travel in the attributes of the result.
///
/// Python writes `options.model_copy(update={"uncertainty": method})` into
/// `attrs["nca_options"]`, so `uncertainty` is the one field whose automatic value is replaced
/// by what it resolved to; every other field is already what it says. The remaining choices an
/// analysis makes are not fields and are not rewritten: the BLQ rules of `NCAOptions.blq`, the
/// limit of quantification of `resolve_lloq` and the dose of a row are resolved per row and per
/// batch and have no place in the options.
fn resolved_options(batch: &Batch, options: &NcaOptions) -> NcaOptions {
    NcaOptions {
        uncertainty: Some(resolved_uncertainty(batch, options)),
        ..options.clone()
    }
}

/// The parameter columns of a batch, one per parameter any of its rows reports.
///
/// A batch of one route reports exactly the parameters of that route, in the order
/// [`PARAMETERS`] lists them. A batch which mixes routes reports the union of theirs, and a row
/// which does not carry a parameter contributes `NaN` to its column - `0` where the variable is
/// an integer code - as `merge_rows` of `nca.py` fills it.
///
/// Python orders the union of a mixed batch after the group which reports the most variables,
/// so its order depends on which route appears first; the order here is [`PARAMETERS`] either
/// way. A result is keyed by name on both sides, so this is a divergence in the order of a
/// dataset and in no value of it.
fn parameter_columns(
    rows: &[RowResult],
    has_dose: bool,
    context: &UnitContext,
) -> Result<Vec<ParameterColumn>, KernelError> {
    let mut routes: Vec<Option<Route>> = Vec::new();
    for row in rows {
        if !routes.contains(&row.route) {
            routes.push(row.route);
        }
    }
    let reported = |p: &Parameter| routes.iter().any(|&r| p.reported_by(r, has_dose));
    PARAMETERS
        .iter()
        .filter(|p| reported(p))
        .map(|p| {
            let (unit, factor) = unit_of(p.name, p.unit, context)?;
            Ok(ParameterColumn {
                name: p.name.to_string(),
                unit,
                values: rows
                    .iter()
                    .map(|row| {
                        if p.reported_by(row.route, has_dose) {
                            // python converts every variable but the integer ones, whose
                            // expression is dimensionless and whose factor is therefore 1: a
                            // multiplication by 1 leaves every double as it was, the codes of
                            // `c0_method` included
                            (p.read)(&row.parameters) * factor
                        } else {
                            // the fill of a parameter the row does not report, which no unit
                            // converts: a NaN stays a NaN and the 0 of an integer code stays 0
                            p.missing()
                        }
                    })
                    .collect(),
            })
        })
        .collect()
}

/// The number of subjects behind every sample, the variable `n` of every result.
///
/// `Timecourses.n_subjects` is the stored count when the batch keeps one per sample and the
/// largest over the time points when it keeps one per point, "the number of subjects of a group
/// is the number behind its best covered point". The wire of this build has only the second
/// form (`Batch::n` is one value per point), so the count of a sample is the `numpy.nanmax` of
/// its row: the largest count which is a number, and `NaN` for a row which holds none, as numpy
/// answers an all-`NaN` slice. A batch which does not carry `n` at all gives a column of `NaN`,
/// which is what `nca` writes for a batch of individual curves.
///
/// # Errors
///
/// [`KernelError::Batch`] if `dimensionless` is no unit, which cannot happen.
fn subject_counts(
    batch: &Batch,
    n_samples: usize,
    context: &UnitContext,
) -> Result<ParameterColumn, KernelError> {
    let (unit, _) = unit_of(SUBJECT_COUNT, DIMENSIONLESS, context)?;
    let values = (0..n_samples)
        .map(|row| match &batch.n {
            // `f64::max` answers a NaN with the other value and two NaNs with a NaN, which is
            // the rule of `numpy.nanmax` down to the empty row
            Some(n) => crate::row_slice(n, row, batch.n_time)
                .iter()
                .fold(f64::NAN, |most, &count| most.max(count)),
            None => f64::NAN,
        })
        .collect();
    Ok(ParameterColumn {
        name: SUBJECT_COUNT.to_string(),
        unit,
        values,
    })
}

/// The coordinates of the result: those of the batch, and the dose every sample was analysed
/// against.
///
/// `_to_result` of `nca.py` adds `dose_amount` over the sample dimensions for a batch which has
/// doses and does not already name the coordinate itself, so that a parameter of the result can
/// be normalized by the dose afterwards (`NCAResult.dose_normalized`). The amount is the one the
/// row really ran against (`reference_dose_amount`), which is why it is read back from the rows
/// rather than from the first column of the protocol.
///
/// `SampleCoordinate` carries no unit, where python writes `attrs={"units": dose_unit}` onto the
/// coordinate. The unit is `Batch::dose_unit`, which the host sent and still holds; nothing in
/// this contract can carry it across, so nothing is invented for it here.
fn coordinates(batch: &Batch, rows: &[RowResult], has_dose: bool) -> Vec<SampleCoordinate> {
    let mut coords = batch.coords.clone();
    if has_dose && !coords.iter().any(|c| c.name == DOSE_COORDINATE) {
        coords.push(SampleCoordinate {
            name: DOSE_COORDINATE.to_string(),
            dims: batch.sample_dims.clone(),
            values: Coordinate::Float(rows.iter().map(|r| r.dose_amount).collect()),
        });
    }
    coords
}

/// The candidate windows of the terminal selection, over the dimension `candidate`.
///
/// Empty unless the analysis is of a single row which has windows at all, which is what
/// `candidate_variables` of `nca.py` writes: the rows of a batch need not have equally many
/// windows, so a batch would need a padded dimension every later step would carry along.
fn candidate_variables(
    candidates: &[Candidate],
    n_samples: usize,
    context: &UnitContext,
) -> Result<Vec<PointVariable>, KernelError> {
    if candidates.is_empty() || n_samples != 1 {
        return Ok(Vec::new());
    }
    CANDIDATE_VARIABLES
        .iter()
        .map(|variable| {
            let (unit, factor) = unit_of(variable.name, variable.unit, context)?;
            Ok(PointVariable {
                name: variable.name.to_string(),
                unit,
                dim: CANDIDATE_DIM.to_string(),
                size: candidates.len(),
                values: candidates
                    .iter()
                    .map(|c| (variable.read)(c) * factor)
                    .collect(),
            })
        })
        .collect()
}

/// The per-interval variables of a multiple dose batch, over the dimension `interval`.
///
/// Empty unless some row carries intervals, which only a multiple dose batch analysed with
/// `NcaOptions::intervals` does. The dimension is as wide as the widest protocol of the batch
/// (`max_doses`, at least one for a `tau` without a protocol), a row with fewer doses is `NaN`
/// padded, and the values are sample major: sample `i`, interval `k` at `i * size + k`.
///
/// Mirrors the `interval_*` variables `compute_intervals` returns, over `(N, K)`.
fn interval_variables(
    batch: &Batch,
    rows: &[RowResult],
    context: &UnitContext,
) -> Result<Vec<PointVariable>, KernelError> {
    if rows.iter().all(|row| row.intervals.is_empty()) {
        return Ok(Vec::new());
    }
    let size = batch
        .max_doses
        .max(rows.iter().map(|r| r.intervals.len()).max().unwrap_or(0));
    INTERVAL_VARIABLES
        .iter()
        .filter(|variable| (variable.present)(batch))
        .map(|variable| {
            let (unit, factor) = unit_of(variable.name, variable.unit, context)?;
            let values = rows
                .iter()
                .flat_map(|row| {
                    (0..size).map(|k| {
                        row.intervals
                            .get(k)
                            .map_or(f64::NAN, |interval| (variable.read)(interval) * factor)
                    })
                })
                .collect();
            Ok(PointVariable {
                name: variable.name.to_string(),
                unit,
                dim: INTERVAL_DIM.to_string(),
                size,
                values,
            })
        })
        .collect()
}

/// `accepted` and `excluded` of every sample, which every result carries.
///
/// `nca` of `nca.py` writes both onto every result whatever the acceptance criteria say, and
/// without a single threshold `evaluate_acceptance` accepts every row, so nothing is excluded
/// either. The text variable `excluded_reason` is written only where a row is excluded, so a
/// result of this build has none; this build refuses the acceptance criteria outright, which
/// leaves exactly this case.
fn acceptance_variables(n_samples: usize) -> Vec<BooleanColumn> {
    vec![
        BooleanColumn {
            name: "accepted".to_string(),
            unit: DIMENSIONLESS.to_string(),
            values: vec![true; n_samples],
        },
        BooleanColumn {
            name: "excluded".to_string(),
            unit: DIMENSIONLESS.to_string(),
            values: vec![false; n_samples],
        },
    ]
}

/// The unit of one variable and the factor from the unit it was computed in to it.
///
/// # Errors
///
/// [`KernelError::Unsupported`] for a unit of the batch the registry of this build could not
/// read, whatever the reason (`UnitError::is_unsupported`): a time in `week`, a concentration in
/// `U/l`, a prefixed `kilohour`, pint's `mg l` or `hr ** 0.5`, a spelling pint over-forgives
/// (`mg//l`, which it reads as `mg / l`) and a string which is no unit anywhere. What this build
/// reads is a property of this build, so a host whose own engine reads it analyses the batch
/// there and the refusal is one a host routes on rather than an error at its user. Every one of
/// these quotes the offending unit string itself, which is why the refusal does not carry the
/// variable name: the variable is wherever the derivation happened to reach it first and says
/// nothing about the batch.
///
/// [`KernelError::Batch`] naming the variable for the failures which are not about reading a
/// string: a placeholder the batch does not provide, which is a batch with doses and no dose
/// unit and whose variable is the one piece of information in that message. Python raises in the
/// same place.
fn unit_of(
    name: &str,
    expression: &str,
    context: &UnitContext,
) -> Result<(String, f64), KernelError> {
    parameter_unit(expression, context).map_err(|e| {
        if e.is_unsupported() {
            KernelError::Unsupported {
                what: format!("a unit which the registry of this build does not read ({e})"),
            }
        } else {
            KernelError::Batch(format!("the unit of {name} cannot be derived: {e}"))
        }
    })
}

#[cfg(test)]
mod tests {
    use pkpdutils_abi::{
        ANALYSIS_CONTRACT, AnalysisResult, BATCH_CONTRACT, Batch, Coordinate, NcaOptions, Route,
        SampleCoordinate, TerminalPhase, UncertaintyMethod,
    };

    use crate::parameters::tests::{at, close, dosed, linear, many, names, one, oral_curve, run};

    /// The unit of one parameter column.
    #[track_caller]
    fn unit_of<'a>(result: &'a AnalysisResult, name: &str) -> &'a str {
        result
            .parameters
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the result has no column {name}"))
            .unit
            .as_str()
    }

    /// One point variable of a result.
    #[track_caller]
    fn point_variable<'a>(
        result: &'a AnalysisResult,
        name: &str,
    ) -> &'a pkpdutils_abi::PointVariable {
        result
            .point_variables
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the result has no point variable {name}"))
    }

    /// The canonical curve of one dose of 100 mg given as an intravenous bolus.
    fn bolus_curve() -> Batch {
        dosed(
            one(&[1.0, 2.0, 3.0, 4.0, 5.0], &[8.0, 4.0, 2.0, 1.0, 0.5]),
            Route::IvBolus,
            100.0,
            0.0,
        )
    }

    #[test]
    fn every_parameter_column_carries_the_unit_python_derives() {
        // `PARAMETER_UNITS` of nca.py resolved by `parameter_unit` of nca/result.py: the
        // expression of every parameter against the units of the batch, in the canonical long
        // form of pint. The canonical batch measures milligram per liter over hours and doses
        // in milligram, under which the clearance is already a liter per hour and the volume a
        // liter, so no magnitude moves and the strings stand alone.
        let oral = run(&oral_curve(), &linear());
        let bolus = run(&bolus_curve(), &linear());

        for (name, unit) in [
            // the values and the times, one representative of every family
            ("cmax", "milligram / liter"),
            ("cmin", "milligram / liter"),
            ("clast", "milligram / liter"),
            ("clast_pred", "milligram / liter"),
            ("tmax", "hour"),
            ("tmin", "hour"),
            ("tlast", "hour"),
            ("tlag", "hour"),
            ("cmax_half", "milligram / liter"),
            ("tmax_half", "hour"),
            // the areas and their moments
            ("auc_last", "hour * milligram / liter"),
            ("auc_all", "hour * milligram / liter"),
            ("auc_inf_obs", "hour * milligram / liter"),
            ("auc_inf_pred", "hour * milligram / liter"),
            ("aumc_last", "hour ** 2 * milligram / liter"),
            ("aumc_all", "hour ** 2 * milligram / liter"),
            ("aumc_inf", "hour ** 2 * milligram / liter"),
            ("aumc_inf_pred", "hour ** 2 * milligram / liter"),
            // the fractions of an area, which are fractions and not percentages
            ("auc_extrap_fraction", "dimensionless"),
            ("auc_extrap_fraction_pred", "dimensionless"),
            ("aumc_extrap_fraction", "dimensionless"),
            ("aumc_extrap_fraction_pred", "dimensionless"),
            // the residence times and the terminal phase
            ("mrt", "hour"),
            ("mrt_last", "hour"),
            ("mrt_pred", "hour"),
            ("thalf_eff", "hour"),
            ("thalf", "hour"),
            ("lambda_z", "1 / hour"),
            ("lambda_z_stderr", "1 / hour"),
            ("lambda_z_intercept", "dimensionless"),
            ("lambda_z_r2", "dimensionless"),
            ("lambda_z_r2_adj", "dimensionless"),
            ("lambda_z_n_points", "dimensionless"),
            ("lambda_z_t_first", "hour"),
            ("lambda_z_t_last", "hour"),
            ("lambda_z_span", "dimensionless"),
            // the dose dependent ones of an extravascular dose: milligram over milligram per
            // liter is a liter, and the normalized ones cancel the dose out of the unit
            ("cl_f", "liter / hour"),
            ("cl_f_pred", "liter / hour"),
            ("vz_f", "liter"),
            ("vz_f_pred", "liter"),
            ("auc_inf_dn", "hour / liter"),
            ("auc_inf_pred_dn", "hour / liter"),
            ("cmax_dn", "1 / liter"),
        ] {
            assert_eq!(unit_of(&oral, name), unit, "{name}");
        }

        for (name, unit) in [
            ("c0", "milligram / liter"),
            ("c0_method", "dimensionless"),
            ("auc_back_extrap_fraction", "dimensionless"),
            ("auc_back_extrap_fraction_pred", "dimensionless"),
            ("aumc_back_extrap_fraction", "dimensionless"),
            ("cl", "liter / hour"),
            ("cl_pred", "liter / hour"),
            ("vz", "liter"),
            ("vz_pred", "liter"),
            ("vss", "liter"),
            ("vss_pred", "liter"),
        ] {
            assert_eq!(unit_of(&bolus, name), unit, "{name}");
        }

        // and not one column is left without a unit
        for result in [&oral, &bolus] {
            for column in &result.parameters {
                assert!(!column.unit.is_empty(), "{} has no unit", column.name);
            }
        }
    }

    #[test]
    fn a_normalized_unit_converts_the_values_and_not_only_the_string() {
        // `parameter_unit` reports a volume in liter and a clearance in liter per hour whatever
        // the batch was measured in, and the factor from the raw unit to the reported one
        // belongs to the values (`array * factor` of `_to_result`). The same curve as the
        // canonical one, measured in nanogram per milliliter over minutes and dosed in
        // milligram: a milligram over a nanogram per milliliter is 1e3 liter, and the
        // clearance of that volume per minute is 6e4 liter per hour.
        let batch = Batch {
            time_unit: "min".to_string(),
            unit: "ng/ml".to_string(),
            ..oral_curve()
        };
        let result = run(&batch, &linear());

        // the arithmetic of the kernel is the arithmetic of the canonical curve: the raw
        // clearance is 100 / 19.971347520444482 and the raw volume is that over ln 2
        assert_eq!(unit_of(&result, "vz_f"), "liter");
        close(at(&result, "vz_f", 0), 7_223.824_228_245_438);
        assert_eq!(unit_of(&result, "cl_f"), "liter / hour");
        close(at(&result, "cl_f", 0), 300_430.403_800_136_94);
        assert_eq!(unit_of(&result, "vz_f_pred"), "liter");
        close(at(&result, "vz_f_pred", 0), 7_223.824_228_245_438);
        assert_eq!(unit_of(&result, "cl_f_pred"), "liter / hour");
        close(at(&result, "cl_f_pred", 0), 300_430.403_800_136_94);

        // every other family is reported as it was measured, so its values stand untouched
        assert_eq!(
            unit_of(&result, "auc_last"),
            "minute * nanogram / milliliter"
        );
        assert_eq!(at(&result, "auc_last", 0), 19.25);
        assert_eq!(
            unit_of(&result, "aumc_last"),
            "minute ** 2 * nanogram / milliliter"
        );
        assert_eq!(at(&result, "aumc_last", 0), 46.5);
        assert_eq!(unit_of(&result, "cmax"), "nanogram / milliliter");
        assert_eq!(at(&result, "cmax", 0), 8.0);
        assert_eq!(unit_of(&result, "lambda_z"), "1 / minute");
        close(at(&result, "lambda_z", 0), 0.693_147_180_559_945_4);
        // a nanogram does not cancel a milligram, so the dose stays in the unit and the
        // dimension is no volume: the value is the raw one
        assert_eq!(
            unit_of(&result, "auc_inf_dn"),
            "minute * nanogram / milligram / milliliter"
        );
        close(at(&result, "auc_inf_dn", 0), 0.199_713_475_204_444_82);
    }

    #[test]
    fn the_candidate_table_travels_as_point_variables_for_a_one_row_batch_only() {
        // `candidate_variables` of nca.py: the table of `TerminalPhase.keep_candidates` reaches
        // a result as the three variables over the dimension `candidate`, and only for an
        // analysis of a single row, since the rows of a batch need not have equally many
        // windows.
        let keep = NcaOptions {
            terminal: TerminalPhase {
                keep_candidates: true,
                ..TerminalPhase::default()
            },
            ..linear()
        };
        let result = run(&oral_curve(), &keep);

        // the canonical curve has its maximum at t = 2 and halves from t = 3 to t = 6, so the
        // windows of at least three points after the maximum start at t = 3 and at t = 4
        assert_eq!(
            result
                .point_variables
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "candidate_t_first",
                "candidate_n_points",
                "candidate_r2_adj"
            ]
        );
        for name in [
            "candidate_t_first",
            "candidate_n_points",
            "candidate_r2_adj",
        ] {
            let variable = point_variable(&result, name);
            assert_eq!(variable.dim, "candidate", "{name}");
            assert_eq!(variable.size, 2, "{name}");
        }
        assert_eq!(
            point_variable(&result, "candidate_t_first").values,
            [3.0, 4.0]
        );
        assert_eq!(
            point_variable(&result, "candidate_n_points").values,
            [4.0, 3.0]
        );
        // both windows lie on the exact halving, so the regression of each is exact
        for &r2_adj in &point_variable(&result, "candidate_r2_adj").values {
            close(r2_adj, 1.0);
        }
        assert_eq!(unit_of_point(&result, "candidate_t_first"), "hour");
        assert_eq!(
            unit_of_point(&result, "candidate_n_points"),
            "dimensionless"
        );
        assert_eq!(unit_of_point(&result, "candidate_r2_adj"), "dimensionless");

        // a batch of two rows carries none of them, and neither does one which did not ask
        let two_rows = many(
            &[&[0.0, 1.0, 2.0, 3.0, 4.0], &[0.0, 1.0, 2.0, 3.0, 4.0]],
            &[&[0.0, 8.0, 4.0, 2.0, 1.0], &[0.0, 8.0, 4.0, 2.0, 1.0]],
        );
        assert!(run(&two_rows, &keep).point_variables.is_empty());
        assert!(run(&oral_curve(), &linear()).point_variables.is_empty());
    }

    #[test]
    fn every_result_names_the_number_of_subjects_behind_its_samples() {
        // `nca` of nca.py writes `values["n"] = timecourses.n_subjects` onto every result, NaN
        // for a batch which does not say, and `unit_expression` of nca.py gives the name `n`
        // the unit `dimensionless`. It is written after every parameter and before the boolean
        // variables, so it is the last column here.
        let individual = run(&oral_curve(), &linear());
        assert_eq!(
            names(&individual).last(),
            Some(&"n"),
            "after every parameter of the analysis"
        );
        assert_eq!(unit_of(&individual, "n"), "dimensionless");
        assert!(
            at(&individual, "n", 0).is_nan(),
            "a batch of individual curves counts no subjects"
        );

        // `Timecourses.n_subjects` is the largest count over the time points of a sample, the
        // number of subjects behind its best covered point, and `numpy.nanmax` of a sample
        // without a single count is NaN rather than an error
        let times: &[f64] = &[0.0, 1.0, 2.0];
        let values: &[f64] = &[8.0, 4.0, 2.0];
        let group = Batch {
            n: Some(vec![4.0, 6.0, 5.0, f64::NAN, f64::NAN, f64::NAN]),
            ..many(&[times, times], &[values, values])
        };
        let result = run(&group, &linear());
        assert_eq!(at(&result, "n", 0), 6.0, "the largest of 4, 6 and 5");
        assert!(at(&result, "n", 1).is_nan(), "a sample without a count");
    }

    #[test]
    fn a_dosed_batch_carries_the_amount_it_was_analysed_against_as_a_coordinate() {
        // `_to_result` of nca.py adds the coordinate `dose_amount` (`DOSE_COORDINATE` of
        // nca/result.py) over the sample dimensions, the amount of the reference dose of every
        // row, so that a parameter can be normalized by it afterwards
        // (`NCAResult.dose_normalized`). It is added only where the batch has doses and does
        // not already carry the coordinate itself.
        let dosed_result = run(&oral_curve(), &linear());
        let dose = coordinate(&dosed_result, "dose_amount");
        assert_eq!(dose.dims, vec!["subject"]);
        assert_eq!(dose.values, Coordinate::Float(vec![100.0]));
        assert!(
            dosed_result.coords.iter().any(|c| c.name == "subject"),
            "beside the coordinates of the batch"
        );

        // a batch without doses has no amount to name
        let undosed = run(&one(&[0.0, 1.0, 2.0], &[8.0, 4.0, 2.0]), &linear());
        assert!(!undosed.coords.iter().any(|c| c.name == "dose_amount"));

        // the amount of the dose the row was really analysed against, which may be no number
        let missing = Batch {
            dose_amount: vec![f64::NAN],
            ..oral_curve()
        };
        let result = run(&missing, &linear());
        let Coordinate::Float(values) = &coordinate(&result, "dose_amount").values else {
            panic!("the amount of a dose is a number");
        };
        assert!(values[0].is_nan(), "{values:?}");

        // and a batch which names the coordinate itself keeps what it named
        let own = Batch {
            coords: vec![SampleCoordinate {
                name: "dose_amount".to_string(),
                dims: vec!["subject".to_string()],
                values: Coordinate::Float(vec![42.0]),
            }],
            ..oral_curve()
        };
        let result = run(&own, &linear());
        assert_eq!(
            coordinate(&result, "dose_amount").values,
            Coordinate::Float(vec![42.0])
        );
        assert_eq!(result.coords.len(), 1, "and gains no second one");
    }

    /// One coordinate of a result.
    #[track_caller]
    fn coordinate<'a>(result: &'a AnalysisResult, name: &str) -> &'a SampleCoordinate {
        result
            .coords
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| {
                panic!(
                    "the result has no coordinate {name}: {:?}",
                    result.coords.iter().map(|c| &c.name).collect::<Vec<_>>()
                )
            })
    }

    /// The unit of one point variable.
    #[track_caller]
    fn unit_of_point<'a>(result: &'a AnalysisResult, name: &str) -> &'a str {
        point_variable(result, name).unit.as_str()
    }

    #[test]
    fn the_result_carries_the_batch_attributes_and_its_contract() {
        // `_to_result` of nca.py writes the substance, the route and the tissue of a batch
        // which has one of each onto the result, and `nca` writes the options the analysis
        // really ran with: the automatic `uncertainty` replaced by what it resolved to
        // (`resolve_uncertainty` of options.py), which is `NONE` for a batch without a spread.
        let result = run(&oral_curve(), &linear());

        assert_eq!(result.contract, ANALYSIS_CONTRACT);
        assert_eq!(result.attributes.substance.as_deref(), Some("caffeine"));
        assert_eq!(result.attributes.route, Some(Route::Oral));
        assert_eq!(result.attributes.tissue.as_deref(), Some("plasma"));
        assert_eq!(
            result.attributes.options.uncertainty,
            Some(UncertaintyMethod::None),
            "the resolved options and not the options as they were given"
        );
        assert_eq!(
            result.attributes.options,
            NcaOptions {
                uncertainty: Some(UncertaintyMethod::None),
                ..linear()
            },
            "and nothing else of them is rewritten"
        );
        assert_eq!(linear().uncertainty, None, "which the caller left open");
    }

    #[test]
    fn the_acceptance_variables_are_written_onto_every_result() {
        // `nca` of nca.py writes `accepted` and `excluded` onto every result, whatever the
        // acceptance criteria say; without a single threshold every row is accepted and none is
        // excluded (`evaluate_acceptance`). `excluded_reason` is written only where a row is
        // excluded, so a result of this build carries no text variable at all.
        let times: &[f64] = &[0.0, 1.0, 2.0];
        let values: &[f64] = &[8.0, 4.0, 2.0];
        let result = run(&many(&[times, times], &[values, values]), &linear());

        assert_eq!(
            result
                .boolean_variables
                .iter()
                .map(|b| b.name.as_str())
                .collect::<Vec<_>>(),
            vec!["accepted", "excluded"]
        );
        assert_eq!(result.boolean_variables[0].values, vec![true, true]);
        assert_eq!(result.boolean_variables[1].values, vec![false, false]);
        for variable in &result.boolean_variables {
            assert_eq!(variable.unit, "dimensionless", "{}", variable.name);
        }
        assert!(result.text_variables.is_empty(), "nothing is excluded");
    }

    #[test]
    fn the_column_order_is_stable_across_runs() {
        // The columns follow `PARAMETERS`, the order `compute_parameters` writes its output in,
        // and a batch which mixes routes reports the union of the two sets in that same order.
        // Python's `merge_rows` orders the union after the group which reports the most
        // variables and so depends on which route appears first; the order here does not, which
        // is a divergence in the order of a dataset and in nothing else.
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
        assert_eq!(names(&result), MIXED_ROUTE_PARAMETERS);
        assert_eq!(
            result.parameters.len(),
            55,
            "the union and the count of subjects"
        );
        assert_eq!(
            names(&run(&batch, &linear())),
            names(&result),
            "the same batch gives the same columns in the same order"
        );
        // the reversed batch reports the same union in the same order, where python would
        // report the intravenous set first
        let reversed = Batch {
            routes: Some(vec![Route::IvBolus, Route::Oral]),
            ..batch.clone()
        };
        assert_eq!(names(&run(&reversed, &linear())), MIXED_ROUTE_PARAMETERS);

        // `merge_rows` of nca.py fills the column of a variable a group does not report with
        // NaN, and an integer variable of `INTEGER_VARIABLES` with its 0
        assert!(at(&result, "c0", 0).is_nan(), "the extravascular row");
        assert_eq!(at(&result, "c0_method", 0), 0.0);
        assert!(at(&result, "tlag", 1).is_nan(), "the intravenous row");
        assert!(at(&result, "cl", 0).is_nan() && at(&result, "cl_f", 1).is_nan());
        // and the unit of such a column is the unit of the variable all the same
        assert_eq!(unit_of(&result, "c0"), "milligram / liter");
        assert_eq!(unit_of(&result, "c0_method"), "dimensionless");
    }

    /// Every parameter of a batch which mixes an extravascular with an intravenous dose, and
    /// the count of subjects every result carries.
    const MIXED_ROUTE_PARAMETERS: &[&str] = &[
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
        "c0",
        "c0_method",
        "auc_back_extrap_fraction",
        "auc_back_extrap_fraction_pred",
        "aumc_back_extrap_fraction",
        "tlag",
        "cmax_half",
        "tmax_half",
        "cl",
        "cl_f",
        "vz",
        "vz_f",
        "cl_pred",
        "cl_f_pred",
        "vz_pred",
        "vz_f_pred",
        "vss",
        "vss_pred",
        "auc_inf_dn",
        "auc_inf_pred_dn",
        "cmax_dn",
        "n",
    ];

    #[test]
    fn a_result_of_a_two_dimensional_batch_keeps_its_sample_shape_and_coordinates() {
        // The result is over the sample dimensions of the batch, row major as the batch is, and
        // the non-dimension coordinates travel with it (`sample_coordinates` of result.py),
        // which `stats/bioequivalence.py` reads back.
        let batch = grid();
        let result = run(&batch, &linear());

        assert_eq!(result.sample_dims, vec!["subject", "period"]);
        assert_eq!(result.sample_shape, vec![2, 2]);
        assert_eq!(result.coords, batch.coords);
        assert_eq!(result.flags.len(), 4);
        // the rows are the samples in the row major order of the batch: (A,1), (A,2), (B,1),
        // (B,2), whose curves are the canonical one scaled by 1, 2, 3 and 4
        let cmax = result
            .parameters
            .iter()
            .find(|p| p.name == "cmax")
            .expect("the result carries the maximum");
        assert_eq!(cmax.values, vec![8.0, 16.0, 24.0, 32.0]);
        // the linear trapezoids 4 + 6 + 3 + 1.5 of the first curve, scaled with it
        let auc_last = result
            .parameters
            .iter()
            .find(|p| p.name == "auc_last")
            .expect("the result carries the area");
        assert_eq!(auc_last.values, vec![14.5, 29.0, 43.5, 58.0]);
        for column in &result.parameters {
            assert_eq!(column.values.len(), 4, "{}", column.name);
        }
    }

    /// A batch of two subjects in two periods, the canonical curve scaled per sample.
    fn grid() -> Batch {
        let times: Vec<f64> = [0.0_f64, 1.0, 2.0, 3.0, 4.0].repeat(4);
        let values: Vec<f64> = (1..=4)
            .flat_map(|k| {
                [0.0_f64, 8.0, 4.0, 2.0, 1.0]
                    .into_iter()
                    .map(move |c| c * f64::from(k))
            })
            .collect();
        Batch {
            sample_dims: vec!["subject".to_string(), "period".to_string()],
            sample_shape: vec![2, 2],
            n_time: 5,
            times,
            values,
            coords: vec![
                SampleCoordinate {
                    name: "subject".to_string(),
                    dims: vec!["subject".to_string()],
                    values: Coordinate::Str(vec!["A".to_string(), "B".to_string()]),
                },
                SampleCoordinate {
                    name: "period".to_string(),
                    dims: vec!["period".to_string()],
                    values: Coordinate::Int(vec![1, 2]),
                },
                SampleCoordinate {
                    name: "weight".to_string(),
                    dims: vec!["subject".to_string()],
                    values: Coordinate::Float(vec![70.0, 80.0]),
                },
            ],
            ..one(&[0.0, 1.0, 2.0, 3.0, 4.0], &[0.0, 8.0, 4.0, 2.0, 1.0])
        }
    }

    /// The batch of `oral_curve` with one of its three unit strings replaced.
    fn with_unit(which: &str, unit: &str) -> Batch {
        let mut batch = oral_curve();
        match which {
            "unit" => batch.unit = unit.to_string(),
            "time_unit" => batch.time_unit = unit.to_string(),
            "dose_unit" => batch.dose_unit = Some(unit.to_string()),
            other => panic!("'{other}' is no unit of a batch"),
        }
        batch
    }

    #[test]
    fn a_unit_the_registry_cannot_read_is_a_refusal_and_not_a_batch_error() {
        // What this build reads is a property of this build, so a unit it cannot read is not a
        // defect of the batch: the batch is analysable, by a host whose own engine reads the
        // string, and `KernelError::Unsupported` is what says so. Reported as
        // `KernelError::Batch` it reaches the user of that host as an exception, since the
        // fallback catches the refusal and deliberately nothing broader, and none of these is a
        // corner case: `week` is the time base of a monoclonal antibody study, `U/l` a clinical
        // concentration, and `mg//l` is read by pint as `mg / l` and therefore computed by
        // `pkpdutils` 1.4.0.
        //
        // Every way the parse can fail, as the unit of the values, of the times and of the
        // doses, since each reaches the derivation through another placeholder and a dose unit
        // reaches it only for some parameters.
        let deep = format!("{}hr{}", "(".repeat(80), ")".repeat(80));
        for spelling in [
            // a unit pint has and this registry does not
            "week",
            "month",
            "year",
            "U/l",
            // a prefix on a unit which is not prefixable
            "kilohour",
            "hectoliter",
            // pint's implicit product and its fractional exponent
            "mg l",
            "hr**0.5",
            // the two limits this build puts on an expression it would otherwise read
            "hr**100",
            deep.as_str(),
            // a spelling pint over-forgives, which 1.4.0 computes: `mg/l` and `hr**2`
            "mg//l",
            "hr**--2",
            // and a string which is no unit anywhere. The host refuses these before a batch is
            // built at all (`Timecourses.from_arrays` parses every unit with pint), so the
            // classification is invisible from there; it is uniform because the rule is one
            // line, and a direct caller of the binding gets the same answer as everyone else
            "mg/",
            "/mg",
            "()",
            "(mg",
            "mg)",
            "2*mg",
            "mg**",
            "mg.l",
            "",
        ] {
            for which in ["unit", "time_unit", "dose_unit"] {
                let batch = with_unit(which, spelling);
                let error = crate::analyze_batch(&batch, &linear()).unwrap_err();
                let crate::KernelError::Unsupported { what } = &error else {
                    panic!("{which} = '{spelling}' is {error}, not a refusal");
                };
                // the refusal quotes the unit itself, which is why it names no variable: the
                // variable is wherever the derivation reached it first. Clipped, as every error
                // of the registry clips the host string it echoes
                let quoted: String = spelling
                    .chars()
                    .take(pkpdutils_units::MAX_ERROR_TEXT)
                    .collect();
                assert!(
                    what.contains(&quoted),
                    "the refusal of {which} = '{spelling}' does not quote it: {what}"
                );
            }
        }
    }

    #[test]
    fn a_unit_which_reads_is_analysed_and_one_the_batch_lacks_is_a_batch_error() {
        // The other side of the rule above, and the mutation which would catch a classification
        // that refuses wholesale: the ordinary units of a batch are read and analysed here and
        // not handed back to the host.
        for (unit, time_unit, dose_unit) in [
            ("mg/l", "hr", "mg"),
            ("ug/ml", "min", "mg/kg"),
            ("mmol/l", "day", "mmol"),
            ("IU/l", "s", "IU"),
            ("mg/l", "hour", "milligram"),
        ] {
            let batch = Batch {
                unit: unit.to_string(),
                time_unit: time_unit.to_string(),
                dose_unit: Some(dose_unit.to_string()),
                ..oral_curve()
            };
            let result = crate::analyze_batch(&batch, &linear());
            assert!(
                result.is_ok(),
                "'{unit}' over '{time_unit}' with a dose in '{dose_unit}' is {:?}",
                result.err()
            );
        }

        // and the one unit failure which is no failure to read a string: a dose dependent
        // parameter needs a dose unit, which a batch with doses may lack. The batch is
        // incomplete rather than unreadable, python raises in the same place, and the variable
        // the message names is the one piece of information in it
        let doseless = Batch {
            dose_unit: None,
            ..oral_curve()
        };
        let error = crate::analyze_batch(&doseless, &linear()).unwrap_err();
        assert!(
            matches!(&error, crate::KernelError::Batch(message) if message.contains("cl_f")),
            "{error}"
        );
    }

    #[test]
    fn a_batch_of_another_contract_is_refused_by_both_entry_points() {
        // `analyze_batch` is a public entry of its own, so it checks the contract of the batch
        // it is handed rather than trusting a caller which went through `analyze`.
        let foreign = Batch {
            contract: "batch/0".to_string(),
            ..oral_curve()
        };
        let error = crate::analyze_batch(&foreign, &linear()).unwrap_err();
        assert!(matches!(&error, crate::KernelError::Contract(_)), "{error}");
        assert!(error.to_string().contains(BATCH_CONTRACT), "{error}");
    }
}
