//! Units of the parameters: the expression of a parameter resolved against the
//! units of an analysis and normalized to liters and hours.

use crate::registry::{Dimension, Unit, UnitError, parse_unit};

/// The units a parameter expression is resolved against.
#[derive(Clone, Debug)]
pub struct UnitContext<'a> {
    /// Unit of the values, for `{unit}`.
    pub unit: &'a str,
    /// Unit of the times, for `{time}`.
    pub time: &'a str,
    /// Unit of the doses, for `{dose}`; `None` without doses.
    pub dose: Option<&'a str>,
    /// Unit of an excreted amount, for `{amount}`; `None` outside the urine analysis.
    pub amount: Option<&'a str>,
    /// Unit of a collected volume, for `{volume}`; `None` outside the urine analysis.
    pub volume: Option<&'a str>,
}

/// Convert `unit` to `target` (parsed from a string) and return it with the
/// factor a magnitude in `unit` is multiplied with.
fn convert(unit: &Unit, target: &str) -> Result<(Unit, f64), UnitError> {
    let target = parse_unit(target)?;
    let factor = unit.factor() / target.factor();
    Ok((target, factor))
}

/// Convert a volume to `liter` and a volume per body weight to `liter / kilogram`.
///
/// The unit is classified by its dimension, so a prefixed or compound spelling
/// is recognized. Anything else, a clearance included, is returned unchanged.
///
/// # Errors
///
/// [`UnitError`] never in practice; the targets are units of the registry.
///
/// Returns the new unit and the factor applied (1 when unchanged).
pub fn normalize_volume(unit: &Unit) -> Result<(Unit, f64), UnitError> {
    let d = unit.dimension();
    if d == Dimension::VOLUME {
        convert(unit, "liter")
    } else if d == Dimension::VOLUME.div(Dimension::MASS) {
        convert(unit, "liter / kilogram")
    } else {
        Ok((unit.clone(), 1.0))
    }
}

/// Convert a clearance to `liter / hour` and one per body weight to
/// `liter / hour / kilogram`.
///
/// Anything else is returned unchanged.
///
/// # Errors
///
/// [`UnitError`] never in practice; the targets are units of the registry.
///
/// Returns the new unit and the factor applied (1 when unchanged).
pub fn normalize_clearance(unit: &Unit) -> Result<(Unit, f64), UnitError> {
    let d = unit.dimension();
    let per_time = Dimension::VOLUME.div(Dimension::TIME);
    if d == per_time {
        convert(unit, "liter / hour")
    } else if d == per_time.div(Dimension::MASS) {
        convert(unit, "liter / hour / kilogram")
    } else {
        Ok((unit.clone(), 1.0))
    }
}

/// The unit of a parameter and the factor from its raw unit to the reported one.
///
/// `expression` uses the placeholders `{unit}`, `{time}`, `{dose}`, `{amount}`
/// and `{volume}`, e.g. `"({unit}) * ({time})"`. The resolved unit is
/// normalized by [`normalize_volume`], then by [`normalize_clearance`]; the
/// factor is the product of both. Returns the canonical unit string.
///
/// # Errors
///
/// [`UnitError::MissingPlaceholder`] if the expression uses a placeholder the
/// context does not provide, other [`UnitError`] if the result is no unit.
pub fn parameter_unit(expression: &str, ctx: &UnitContext) -> Result<(String, f64), UnitError> {
    let mut raw = expression.to_string();
    for (name, value) in [
        ("unit", Some(ctx.unit)),
        ("time", Some(ctx.time)),
        ("dose", ctx.dose),
        ("amount", ctx.amount),
        ("volume", ctx.volume),
    ] {
        let placeholder = format!("{{{name}}}");
        if !raw.contains(&placeholder) {
            continue;
        }
        let value = value.ok_or_else(|| UnitError::MissingPlaceholder {
            expression: expression.to_string(),
            placeholder: name.to_string(),
        })?;
        raw = raw.replace(&placeholder, value);
    }
    let (unit, f1) = normalize_volume(&parse_unit(&raw)?)?;
    let (unit, f2) = normalize_clearance(&unit)?;
    let text = if unit.terms().is_empty() {
        "dimensionless".to_string()
    } else {
        crate::format::long_form(&unit)
    };
    Ok((text, f1 * f2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholders_resolve_and_volumes_and_clearances_normalize() {
        let ctx = UnitContext {
            unit: "mg/l",
            time: "hr",
            dose: Some("mg"),
            amount: None,
            volume: None,
        };
        assert_eq!(
            parameter_unit("{unit}", &ctx).unwrap(),
            ("milligram / liter".into(), 1.0)
        );
        assert_eq!(
            parameter_unit("({unit}) * ({time})", &ctx).unwrap(),
            ("hour * milligram / liter".into(), 1.0)
        );
        assert_eq!(
            parameter_unit("1 / ({time})", &ctx).unwrap(),
            ("1 / hour".into(), 1.0)
        );
        assert_eq!(
            parameter_unit("dimensionless", &ctx).unwrap(),
            ("dimensionless".into(), 1.0)
        );
        // (mg)/(mg/l) reduces to liter, already normalized
        assert_eq!(
            parameter_unit("({dose}) / ({unit})", &ctx).unwrap(),
            ("liter".into(), 1.0)
        );
        // (mg)/((mg/l)*(hr)) reduces to liter / hour, already normalized
        assert_eq!(
            parameter_unit("({dose}) / (({unit}) * ({time}))", &ctx).unwrap(),
            ("liter / hour".into(), 1.0)
        );
    }

    #[test]
    fn normalization_converts_and_reports_the_factor() {
        let ctx = UnitContext {
            unit: "ng/ml",
            time: "min",
            dose: Some("mg"),
            amount: None,
            volume: None,
        };
        // (mg)/(ng/ml) is 1e6 ml = 1e3 liter
        let (unit, factor) = parameter_unit("({dose}) / ({unit})", &ctx).unwrap();
        assert_eq!(unit, "liter");
        assert!((factor - 1e3).abs() < 1e-9 * 1e3);
        // (mg)/((ng/ml)*(min)) is 1e3 liter / minute = 6e4 liter / hour
        let (unit, factor) = parameter_unit("({dose}) / (({unit}) * ({time}))", &ctx).unwrap();
        assert_eq!(unit, "liter / hour");
        assert!((factor - 6e4).abs() < 1e-9 * 6e4);
    }

    #[test]
    fn per_bodyweight_doses_normalize_to_the_per_kilogram_forms() {
        let ctx = UnitContext {
            unit: "mg/l",
            time: "hr",
            dose: Some("mg/kg"),
            amount: None,
            volume: None,
        };
        assert_eq!(
            parameter_unit("({dose}) / ({unit})", &ctx).unwrap().0,
            "liter / kilogram"
        );
        assert_eq!(
            parameter_unit("({dose}) / (({unit}) * ({time}))", &ctx)
                .unwrap()
                .0,
            "liter / hour / kilogram"
        );
    }

    #[test]
    fn a_missing_placeholder_is_an_error_naming_it() {
        let ctx = UnitContext {
            unit: "mg/l",
            time: "hr",
            dose: None,
            amount: None,
            volume: None,
        };
        let err = parameter_unit("({dose}) / ({unit})", &ctx)
            .unwrap_err()
            .to_string();
        assert!(err.contains("dose"));
    }

    #[test]
    fn volume_normalization_leaves_a_clearance_alone_and_only_volumes_normalize() {
        let ml_min = parse_unit("ml/min").unwrap();
        let (unit, factor) = normalize_volume(&ml_min).unwrap();
        assert_eq!((unit, factor), (ml_min.clone(), 1.0));
        let (unit, factor) = normalize_clearance(&ml_min).unwrap();
        assert_eq!(crate::format::long_form(&unit), "liter / hour");
        assert!((factor - 0.06).abs() < 1e-12);
        let ctx = UnitContext {
            unit: "mg/l",
            time: "min",
            dose: None,
            amount: None,
            volume: None,
        };
        assert_eq!(
            parameter_unit("1 / ({time})", &ctx).unwrap(),
            ("1 / minute".into(), 1.0)
        );
    }
}
