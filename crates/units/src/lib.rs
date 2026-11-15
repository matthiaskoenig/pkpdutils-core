//! Units of the package: a curated registry, parsing and the canonical spelling.
//!
//! The registry covers exactly the dimensions the package uses (mass, volume,
//! time, amount of substance, activity, pressure and the dimensionless units)
//! and is not a general unit system. It reproduces the helpers of
//! `pkpdutils.units`: [`parse_unit`], [`unit_str`], [`short_unit`],
//! [`check_dose_unit`] and [`is_per_bodyweight`].

mod format;
mod registry;

pub use format::{short_unit, unit_str};
pub use registry::{Dimension, Term, Unit, UnitError, parse_unit};

/// Dimensionalities a dose may have: an amount as mass, as substance or as
/// activity, or such an amount per body weight.
///
/// As in pint, a mass per mass is the dimensionless dimension, so a ratio
/// such as `mg/kg` (and the dimensionless unit itself) passes this check.
const DOSE_DIMENSIONS: [Dimension; 6] = [
    Dimension::MASS,
    Dimension::SUBSTANCE,
    Dimension::ACTIVITY,
    Dimension::NONE,
    Dimension::SUBSTANCE.div(Dimension::MASS),
    Dimension::ACTIVITY.div(Dimension::MASS),
];

/// The dimensionalities of [`DOSE_DIMENSIONS`] which are an amount per body weight.
const PER_BODYWEIGHT_DIMENSIONS: [Dimension; 3] = [
    Dimension::NONE,
    Dimension::SUBSTANCE.div(Dimension::MASS),
    Dimension::ACTIVITY.div(Dimension::MASS),
];

/// Check that a unit is a dose unit: a mass, an amount of substance or an
/// activity, or one of them per body weight.
///
/// # Errors
///
/// [`UnitError`] if the string is not a unit or has another dimensionality.
pub fn check_dose_unit(s: &str) -> Result<(), UnitError> {
    let unit = parse_unit(s)?;
    let dimension = unit.dimension();
    if DOSE_DIMENSIONS.contains(&dimension) {
        Ok(())
    } else {
        Err(UnitError::NotADose {
            unit: s.to_string(),
            dimension: dimension.to_string(),
        })
    }
}

/// Check whether a dose unit is an amount per body weight, such as `mg/kg`.
///
/// # Errors
///
/// [`UnitError`] if the string is not a unit.
pub fn is_per_bodyweight(s: &str) -> Result<bool, UnitError> {
    Ok(PER_BODYWEIGHT_DIMENSIONS.contains(&parse_unit(s)?.dimension()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_spelling_matches_pint() {
        for (input, expected) in [
            ("mg/l", "milligram / liter"),
            ("hr", "hour"),
            ("mg", "milligram"),
            ("mg*hr/l", "hour * milligram / liter"),
            ("1/hr", "1 / hour"),
            ("(mg)/((mg/l)*(hr))", "liter / hour"),
            ("(mg)/(mg/l)", "liter"),
            ("(mg/l)*(hr)**2", "hour ** 2 * milligram / liter"),
            ("ng/ml", "nanogram / milliliter"),
            ("mg/l/hr", "milligram / hour / liter"),
            ("mg/(l*hr)", "milligram / hour / liter"),
            (
                "mg*ug/(l*hr*kg)",
                "microgram * milligram / hour / kilogram / liter",
            ),
            ("hr**-2", "1 / hour ** 2"),
            ("l/hr/kg", "liter / hour / kilogram"),
            ("dimensionless", "dimensionless"),
            ("IU", "IU"),
            ("none", "none"),
            ("mmHg", "millimeter_Hg"),
            ("mg/kg", "milligram / kilogram"),
        ] {
            assert_eq!(unit_str(input).unwrap(), expected, "input {input}");
        }
    }

    #[test]
    fn short_unit_keeps_what_the_user_wrote_and_shortens_the_canonical_form() {
        // pint's `~P` format: the middle dot U+22C5, Unicode superscripts, and
        // the micro sign U+03BC. These are the exact bytes, read from a live run.
        assert_eq!(short_unit("mg/l"), "mg/l");
        assert_eq!(short_unit("milligram / liter"), "mg/l");
        assert_eq!(short_unit("liter / hour / kilogram"), "l/h/kg");
        assert_eq!(short_unit("hour * milligram / liter"), "h\u{22c5}mg/l");
        assert_eq!(
            short_unit("hour ** 2 * milligram / liter"),
            "h\u{b2}\u{22c5}mg/l"
        );
        assert_eq!(short_unit("1 / hour"), "1/h");
        assert_eq!(short_unit("1 / hour ** 2"), "1/h\u{b2}");
        assert_eq!(short_unit("nanogram / milliliter"), "ng/ml");
        assert_eq!(
            short_unit("microgram * milligram / hour / kilogram / liter"),
            "\u{3bc}g\u{22c5}mg/h/kg/l"
        );
        assert_eq!(short_unit("millimeter_Hg"), "mmHg");
        assert_eq!(short_unit("dimensionless"), "");
        assert_eq!(short_unit(""), "");
        assert_eq!(short_unit("mg/foo"), "mg/foo");
    }

    #[test]
    fn an_unknown_unit_names_itself_and_the_supported_dimensions() {
        let err = parse_unit("mg/foo").unwrap_err().to_string();
        assert!(
            err.contains("foo"),
            "the error must name the unknown unit: {err}"
        );
        assert!(
            err.contains("mass"),
            "the error must name what is supported: {err}"
        );
    }

    #[test]
    fn the_empty_unit_is_an_error_and_points_at_dimensionless() {
        let err = parse_unit("  ").unwrap_err().to_string();
        assert!(err.contains("dimensionless"));
    }

    #[test]
    fn dimensionless_and_none_are_different_units() {
        assert_eq!(unit_str("dimensionless").unwrap(), "dimensionless");
        assert_eq!(unit_str("none").unwrap(), "none");
        assert_ne!(
            parse_unit("dimensionless").unwrap(),
            parse_unit("none").unwrap()
        );
    }

    #[test]
    fn dose_units_accept_the_six_dimensions_and_reject_the_rest() {
        for ok in ["mg", "mmol", "IU", "mg/kg", "mmol/kg", "IU/kg"] {
            assert!(check_dose_unit(ok).is_ok(), "{ok}");
        }
        for bad in ["mg/l", "hr", "mmHg"] {
            assert!(check_dose_unit(bad).is_err(), "{bad}");
        }
        assert!(is_per_bodyweight("mg/kg").unwrap());
        assert!(!is_per_bodyweight("mg").unwrap());
    }
}
