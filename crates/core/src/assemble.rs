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
    AnalysisResult, Attributes, Batch, BooleanColumn, NcaOptions, ParameterColumn, PointVariable,
    Route, UncertaintyMethod,
};
use pkpdutils_units::{UnitContext, parameter_unit};

use crate::KernelError;
use crate::parameters::{PARAMETERS, Parameter, RowParameters};
use crate::terminal::Candidate;

/// The unit of a dimensionless variable, as python spells it.
///
/// Both the expression of `PARAMETER_UNITS` and what it resolves to, which are the same string
/// for a quantity without a dimension.
const DIMENSIONLESS: &str = "dimensionless";

/// The name of the dimension the candidate windows live on.
const CANDIDATE_DIM: &str = "candidate";

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
/// [`KernelError::Batch`] naming the variable whose unit could not be derived: the units of a
/// batch are no part of `Batch::validate`, so a string which is no unit, and a batch with doses
/// but without a dose unit, reach this point. Python raises in the same place.
pub(crate) fn result_of(
    batch: &Batch,
    options: &NcaOptions,
    rows: &[(Option<Route>, RowParameters)],
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
    let n_samples = rows.len();
    Ok(AnalysisResult {
        parameters: parameter_columns(rows, batch.max_doses > 0, &context)?,
        point_variables: candidate_variables(candidates, n_samples, &context)?,
        boolean_variables: acceptance_variables(n_samples),
        text_variables: Vec::new(),
        flags: rows.iter().map(|(_, p)| p.flags.bits()).collect(),
        coords: batch.coords.clone(),
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
    rows: &[(Option<Route>, RowParameters)],
    has_dose: bool,
    context: &UnitContext,
) -> Result<Vec<ParameterColumn>, KernelError> {
    let mut routes: Vec<Option<Route>> = Vec::new();
    for (route, _) in rows {
        if !routes.contains(route) {
            routes.push(*route);
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
                    .map(|(route, values)| {
                        if p.reported_by(*route, has_dose) {
                            // python converts every variable but the integer ones, whose
                            // expression is dimensionless and whose factor is therefore 1: a
                            // multiplication by 1 leaves every double as it was, the codes of
                            // `c0_method` included
                            (p.read)(values) * factor
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
/// [`KernelError::Batch`] naming the variable and the reason, which is a unit of the batch the
/// registry does not read or a placeholder the batch does not provide.
fn unit_of(
    name: &str,
    expression: &str,
    context: &UnitContext,
) -> Result<(String, f64), KernelError> {
    parameter_unit(expression, context)
        .map_err(|e| KernelError::Batch(format!("the unit of {name} cannot be derived: {e}")))
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
        assert_eq!(result.parameters.len(), 54);
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

    /// Every parameter of a batch which mixes an extravascular with an intravenous dose.
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

    #[test]
    fn a_unit_the_batch_cannot_spell_is_refused_by_name() {
        // The units of a batch are no part of `Batch::validate`, so a string which is no unit
        // reaches the assembly; python raises there too (`ureg.parse_units`).
        let nonsense = Batch {
            unit: "banana".to_string(),
            ..oral_curve()
        };
        let error = crate::analyze_batch(&nonsense, &linear()).unwrap_err();
        assert!(
            matches!(&error, crate::KernelError::Batch(message) if message.contains("banana")),
            "{error}"
        );

        // a dose dependent parameter needs a dose unit, which a batch with doses may lack
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
