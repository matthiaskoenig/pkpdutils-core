//! The urinary excretion analysis, the port of `nca_urine` of `src/pkpdutils/nca/urine.py`.
//!
//! The excretion rate `A_k / (e_k - s_k)` of every collection is analysed against the midpoint
//! `(s_k + e_k) / 2` of its interval with the rules of a concentration curve (Phoenix WinNonlin
//! urine models 210 to 212), so the area under the rate curve is an amount. The parameters are
//! then renamed (`RATE_PARAMETERS`), and what was collected is reported as measured:
//! `A_e = sum A_k`, `100 A_e / D` and `sum V_k`. With a plasma area the renal clearance is
//! `CL_R = A_e / AUC` over the collection span.

use crate::KernelError;
use crate::parameters::{PARAMETERS, RowInput, RowParameters, Scratch, compute_row};
use crate::sparse::numpy_sum;
use pkpdutils_abi::{
    AnalysisResult, Attributes, CheckedUrine, NcaOptions, ParameterColumn, PointVariable,
};
use pkpdutils_units::{UnitContext, UnitError, parameter_unit, parse_unit};

/// The dimension of the collection intervals of the rate curve.
const COLLECTION_DIM: &str = "collection";

/// The parameters of the rate curve: the name in the result, the name the core computes it
/// under and the unit expression, in the order of `RATE_PARAMETERS` of the host. The unit
/// expressions are the urine entries of `PARAMETER_UNITS` of `nca.py`; the others are those of
/// the concentration parameter, `None` here.
const RATE_PARAMETERS: [(&str, &str, Option<&str>); 18] = [
    ("max_rate", "cmax", Some("({amount}) / ({time})")),
    ("tmax_rate", "tmax", Some("{time}")),
    ("rate_last", "clast", Some("({amount}) / ({time})")),
    ("mid_pt_last", "tlast", Some("{time}")),
    ("aurc_last", "auc_last", Some("{amount}")),
    ("aurc_all", "auc_all", Some("{amount}")),
    ("aurc_inf_obs", "auc_inf_obs", Some("{amount}")),
    ("aurc_inf_pred", "auc_inf_pred", Some("{amount}")),
    ("lambda_z", "lambda_z", None),
    ("lambda_z_stderr", "lambda_z_stderr", None),
    ("lambda_z_intercept", "lambda_z_intercept", None),
    ("lambda_z_r2", "lambda_z_r2", None),
    ("lambda_z_r2_adj", "lambda_z_r2_adj", None),
    ("lambda_z_n_points", "lambda_z_n_points", None),
    ("lambda_z_t_first", "lambda_z_t_first", None),
    ("lambda_z_t_last", "lambda_z_t_last", None),
    ("lambda_z_span", "lambda_z_span", None),
    ("thalf", "thalf", None),
];

/// `numpy.nansum`: the sum with every `NaN` replaced by 0, pairwise as numpy sums.
fn nansum(values: &[f64]) -> f64 {
    let cleaned: Vec<f64> = values
        .iter()
        .map(|&v| if v.is_nan() { 0.0 } else { v })
        .collect();
    numpy_sum(&cleaned)
}

/// The error of a unit which cannot be derived: a unit the registry of this build does not read
/// is [`KernelError::Unsupported`], which a host answers with its own path.
fn unit_error(what: &str, e: &UnitError) -> KernelError {
    if e.is_unsupported() {
        KernelError::Unsupported {
            what: format!("a unit which the registry of this build does not read ({e})"),
        }
    } else {
        KernelError::Urine(format!("{what}: {e}"))
    }
}

/// The recovered amount in multiples of one unit of the dose, `A_e` in the unit of the dose.
///
/// Python's `(Q_(total, unit) / Q_(1.0, dose.unit)).to("dimensionless")`: the dimension check,
/// run before the dose amount is looked at, and the factor between the two units.
fn recovery_factor(unit: &str, dose_unit: &str) -> Result<f64, KernelError> {
    let amount = parse_unit(unit).map_err(|e| unit_error("the amount unit", &e))?;
    let dose = parse_unit(dose_unit).map_err(|e| unit_error("the dose unit", &e))?;
    match amount.over(&dose) {
        Some(ratio) if ratio.dimension().is_dimensionless() => Ok(ratio.factor()),
        _ => Err(KernelError::Urine(format!(
            "the recovered amount in '{unit}' cannot be compared with the dose in '{dose_unit}'"
        ))),
    }
}

/// The non-compartmental analysis of the urinary excretion of one subject.
///
/// The rate `A_k / (e_k - s_k)` at the midpoint `(s_k + e_k) / 2` of every collection runs
/// through the analysis of a curve without a dose (`compute_row`), of which only `auc_method`,
/// `c0_method`, `terminal` and `extrapolation_warning` are read: the rules which read a
/// concentration do not apply to a rate. The result carries the eighteen renamed parameters,
/// `amount_recovered` (`A_e = sum A_k`, `NaN` amounts ignored), `vol_ur` (`sum V_k`, `NaN`
/// without volumes), `percent_recovered` (`100 A_e / D`, `NaN` for a dose of 0), `clr`
/// (`A_e / AUC`, an infinity for an area of 0) when the request carries an area, the point
/// variables `midpoint` and `rate` over `collection` and the flags. It has no sample dimension.
///
/// # Errors
///
/// [`KernelError::Urine`] when the recovered amount cannot be compared with the dose, whatever
/// the dose amount, or when the dose amount and its unit do not come together;
/// [`KernelError::Unsupported`] for a unit the registry of this build does not read.
pub fn nca_urine(urine: &CheckedUrine) -> Result<AnalysisResult, KernelError> {
    let request = urine.request();
    let midpoint: Vec<f64> = request
        .start
        .iter()
        .zip(&request.end)
        .map(|(s, e)| 0.5 * (s + e))
        .collect();
    let rate: Vec<f64> = request
        .amount
        .iter()
        .zip(request.start.iter().zip(&request.end))
        .map(|(a, (s, e))| a / (e - s))
        .collect();

    let options = NcaOptions {
        auc_method: request.options.auc_method,
        c0_method: request.options.c0_method,
        terminal: request.options.terminal.clone(),
        extrapolation_warning: request.options.extrapolation_warning,
        ..NcaOptions::default()
    };
    let dose = match (request.dose_amount, request.dose_unit.as_deref()) {
        (Some(amount), Some(unit)) => Some((amount, unit)),
        (None, None) => None,
        _ => {
            return Err(KernelError::Urine(
                "'dose_amount' and 'dose_unit' come together".to_string(),
            ));
        }
    };
    // python passes no dose to the analysis of the rate curve, and the route of the dose only
    // when there is one
    let input = RowInput {
        times: &midpoint,
        values: &rate,
        dose: crate::dose::ReferenceDose {
            amount: f64::NAN,
            time: f64::NAN,
            duration: None,
        },
        route: dose.map(|_| request.route),
        has_dose: false,
        lloq: None,
        window: None,
        manual: None,
        single_dose: true,
        dose_times: &[],
        dose_amounts: &[],
    };
    let mut scratch = Scratch::new(midpoint.len());
    let values: RowParameters = compute_row(&input, &options, &mut scratch, None);

    let context = UnitContext {
        unit: "dimensionless",
        time: &request.time_unit,
        dose: dose.map(|(_, unit)| unit),
        amount: Some(&request.unit),
        volume: Some(&request.volume_unit),
    };
    let resolve = |name: &str, expression: &str| {
        parameter_unit(expression, &context)
            .map_err(|e| unit_error(&format!("the unit of {name} cannot be derived"), &e))
    };
    let mut result = AnalysisResult::new(
        Vec::new(),
        Vec::new(),
        Attributes {
            substance: None,
            route: None,
            tissue: None,
            options: NcaOptions::default(),
        },
    );
    let mut write = |name: &str, value: f64, expression: &str| -> Result<(), KernelError> {
        let (unit, factor) = resolve(name, expression)?;
        result.parameters.push(ParameterColumn {
            name: name.to_string(),
            unit,
            values: vec![value * factor],
        });
        Ok(())
    };
    for (name, source, expression) in RATE_PARAMETERS {
        let parameter = PARAMETERS
            .iter()
            .find(|p| p.name == source)
            .ok_or_else(|| KernelError::Internal(format!("no parameter named {source}")))?;
        write(
            name,
            (parameter.read)(&values),
            expression.unwrap_or(parameter.unit),
        )?;
    }
    let total = nansum(&request.amount);
    write("amount_recovered", total, "{amount}")?;
    let total_volume = if request.volume.iter().all(|v| v.is_nan()) {
        f64::NAN
    } else {
        nansum(&request.volume)
    };
    write("vol_ur", total_volume, "{volume}")?;

    let mut percent = f64::NAN;
    if let Some((amount, unit)) = dose {
        // the unit check and the factor come first, and the dose amount is divided out
        // afterwards, so a placebo in an incompatible unit is an error and one in a compatible
        // unit is NaN
        let recovered = recovery_factor(&request.unit, unit)? * total;
        if amount > 0.0 {
            percent = 100.0 * recovered / amount;
        }
    }
    write("percent_recovered", percent, "percent")?;

    if let (Some(area), Some(area_unit)) = (request.area, request.area_unit.as_deref()) {
        let expression = format!("({}) / ({area_unit})", request.unit);
        let clearance_context = UnitContext {
            unit: "dimensionless",
            time: &request.time_unit,
            dose: None,
            amount: None,
            volume: None,
        };
        let (unit, factor) = parameter_unit(&expression, &clearance_context)
            .map_err(|e| unit_error("the unit of clr cannot be derived", &e))?;
        result.parameters.push(ParameterColumn {
            name: "clr".to_string(),
            unit,
            values: vec![total / area * factor],
        });
    }

    for (name, array, expression) in [
        ("midpoint", &midpoint, "{time}"),
        ("rate", &rate, "({amount}) / ({time})"),
    ] {
        let (unit, factor) = resolve(name, expression)?;
        result.point_variables.push(PointVariable {
            name: name.to_string(),
            unit,
            dim: COLLECTION_DIM.to_string(),
            size: array.len(),
            values: array.iter().map(|v| v * factor).collect(),
        });
    }
    result.flags = vec![values.flags.bits()];
    result.validate().map_err(KernelError::Internal)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pkpdutils_abi::{Route, UrineRequest};

    fn request() -> UrineRequest {
        UrineRequest {
            start: vec![0.0, 2.0, 4.0],
            end: vec![2.0, 4.0, 8.0],
            amount: vec![300.0, 200.0, 100.0],
            volume: vec![120.0, 140.0, 260.0],
            unit: "mg".to_string(),
            time_unit: "hr".to_string(),
            volume_unit: "ml".to_string(),
            route: Route::Oral,
            dose_amount: Some(1.0),
            dose_unit: Some("g".to_string()),
            area: None,
            area_unit: None,
            options: NcaOptions::default(),
        }
    }

    fn run(r: &UrineRequest) -> Result<AnalysisResult, KernelError> {
        nca_urine(&r.checked().map_err(KernelError::Urine)?)
    }

    fn param<'a>(r: &'a AnalysisResult, name: &str) -> &'a ParameterColumn {
        r.parameters.iter().find(|p| p.name == name).unwrap()
    }

    fn value(r: &AnalysisResult, name: &str) -> f64 {
        param(r, name).values[0]
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-9 * b.abs().max(1.0), "{a} != {b}");
    }

    /// Two collections of 300 and 200 mg, the recovered total of 500 of the python oracle.
    fn recovered(unit: &str, dose_amount: f64, dose_unit: &str) -> UrineRequest {
        UrineRequest {
            start: vec![0.0, 2.0],
            end: vec![2.0, 4.0],
            amount: vec![300.0, 200.0],
            volume: vec![f64::NAN; 2],
            unit: unit.to_string(),
            dose_amount: Some(dose_amount),
            dose_unit: Some(dose_unit.to_string()),
            ..request()
        }
    }

    #[test]
    fn the_rate_curve_is_the_amount_over_the_duration_at_the_midpoint() {
        let r = run(&request()).unwrap();
        let point = |name: &str| r.point_variables.iter().find(|p| p.name == name).unwrap();
        assert_eq!(point("midpoint").values, vec![1.0, 3.0, 6.0]);
        assert_eq!(point("rate").values, vec![150.0, 100.0, 25.0]);
        assert_eq!(point("rate").unit, "milligram / hour");
        assert_eq!(point("midpoint").unit, "hour");
        assert_eq!(point("rate").dim, "collection");
        assert_eq!(point("rate").size, 3);
        // python oracle
        // the exact value of python, 0x1.e3eefc5e55c69p+8, bit for bit
        assert_eq!(value(&r, "aurc_last").to_bits(), 4647221065031179369);
        assert_eq!(value(&r, "aurc_all").to_bits(), 4647221065031179369);
        assert_eq!(value(&r, "max_rate"), 150.0);
        assert_eq!(value(&r, "rate_last"), 25.0);
        assert_eq!(value(&r, "tmax_rate"), 1.0);
        assert_eq!(value(&r, "mid_pt_last"), 6.0);
        assert!(value(&r, "aurc_inf_obs").is_nan());
        assert!(value(&r, "aurc_inf_pred").is_nan());
        assert_eq!(value(&r, "amount_recovered"), 600.0);
        assert_eq!(param(&r, "amount_recovered").unit, "milligram");
        close(value(&r, "percent_recovered"), 60.0);
        assert_eq!(param(&r, "percent_recovered").unit, "percent");
        assert_eq!(r.flags.len(), 1);
        assert!(r.sample_dims.is_empty());
    }

    #[test]
    fn the_parameters_carry_the_urine_names_not_the_plasma_names() {
        let r = run(&request()).unwrap();
        let names: Vec<&str> = r.parameters.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "max_rate",
                "tmax_rate",
                "rate_last",
                "mid_pt_last",
                "aurc_last",
                "aurc_all",
                "aurc_inf_obs",
                "aurc_inf_pred",
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
                "amount_recovered",
                "vol_ur",
                "percent_recovered",
            ]
        );
        for plasma in ["cmax", "tmax", "clast", "tlast", "auc_last", "auc_all"] {
            assert!(r.parameters.iter().all(|p| p.name != plasma));
        }
    }

    #[test]
    fn an_excretion_without_volumes_has_a_nan_volume_and_resolves_its_unit() {
        let mut q = request();
        q.volume = vec![f64::NAN; 3];
        let r = run(&q).unwrap();
        assert!(value(&r, "vol_ur").is_nan());
        assert_eq!(param(&r, "vol_ur").unit, "liter");
        // with volumes in ml the volume is normalized to liter
        let r = run(&request()).unwrap();
        close(value(&r, "vol_ur"), 0.52);
        assert_eq!(param(&r, "vol_ur").unit, "liter");
    }

    #[test]
    fn the_percent_recovered_carries_the_factor_between_the_amount_unit_and_the_dose_unit() {
        // the oracle of python, the recovered total being 500 in every case
        for (unit, dose, dose_unit, expected) in [
            ("mg", 1.0, "g", 50.0),
            ("mg", 1000.0, "mg", 50.0),
            ("ug", 1.0, "mg", 50.0),
            ("mmol", 1.0, "mmol", 50000.0),
        ] {
            let r = run(&recovered(unit, dose, dose_unit)).unwrap();
            close(value(&r, "percent_recovered"), expected);
        }
    }

    #[test]
    fn a_dose_amount_of_zero_leaves_the_percent_recovered_nan_and_is_no_error() {
        let r = run(&recovered("mg", 0.0, "mg")).unwrap();
        assert!(value(&r, "percent_recovered").is_nan());
        assert_eq!(value(&r, "amount_recovered"), 500.0);
    }

    #[test]
    fn a_dose_unit_of_another_dimension_is_an_error_even_for_a_placebo() {
        for dose in [1.0, 0.0] {
            let e = run(&recovered("mg", dose, "mmol")).unwrap_err();
            assert!(
                matches!(&e, KernelError::Urine(m) if m.contains(
                    "the recovered amount in 'mg' cannot be compared with the dose")),
                "{e}"
            );
        }
    }

    #[test]
    fn the_renal_clearance_is_the_total_amount_over_the_plasma_area() {
        let mut q = request();
        q.area = Some(30.0);
        q.area_unit = Some("mg*h/l".to_string());
        let r = run(&q).unwrap();
        // 600 mg / 30 mg h / l = 20 l / h
        close(value(&r, "clr"), 20.0);
        assert_eq!(param(&r, "clr").unit, "liter / hour");
        let names: Vec<&str> = r.parameters.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names.last(), Some(&"clr"));
        assert!(
            run(&request())
                .unwrap()
                .parameters
                .iter()
                .all(|p| p.name != "clr")
        );
    }

    #[test]
    fn a_plasma_area_of_zero_gives_an_infinite_clearance_rather_than_an_error() {
        let mut q = request();
        q.area = Some(0.0);
        q.area_unit = Some("mg*h/l".to_string());
        assert_eq!(value(&run(&q).unwrap(), "clr"), f64::INFINITY);
    }

    #[test]
    fn the_same_collections_in_any_order_give_the_same_result() {
        let ordered = run(&request()).unwrap();
        let mut q = request();
        q.start = vec![4.0, 0.0, 2.0];
        q.end = vec![8.0, 2.0, 4.0];
        q.amount = vec![100.0, 300.0, 200.0];
        q.volume = vec![260.0, 120.0, 140.0];
        let shuffled = run(&q).unwrap();
        assert_eq!(ordered.parameters.len(), shuffled.parameters.len());
        for (a, b) in ordered.parameters.iter().zip(&shuffled.parameters) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.unit, b.unit);
            assert_eq!(a.values[0].to_bits(), b.values[0].to_bits(), "{}", a.name);
        }
        for (a, b) in ordered
            .point_variables
            .iter()
            .zip(&shuffled.point_variables)
        {
            let bits = |v: &PointVariable| v.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(a), bits(b));
        }
        assert_eq!(ordered.flags, shuffled.flags);
    }
}
