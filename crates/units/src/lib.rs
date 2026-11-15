//! Units of the package: a curated registry, parsing and the canonical spelling.
//!
//! The registry covers exactly the dimensions the package uses (mass, volume,
//! time, amount of substance, activity, pressure and the dimensionless units)
//! and is not a general unit system. It reproduces the helpers of
//! `pkpdutils.units`: [`parse_unit`], [`unit_str`], [`short_unit`],
//! [`check_dose_unit`] and [`is_per_bodyweight`].

mod format;
mod parameter;
mod registry;

pub use format::{short_unit, unit_str};
pub use parameter::{UnitContext, normalize_clearance, normalize_volume, parameter_unit};
pub use registry::{Dimension, MAX_DEPTH, MAX_EXPONENT, Term, Unit, UnitError, parse_unit};

/// Dimensionalities a dose may have: an amount as mass, as substance or as
/// activity, or such an amount per body weight.
///
/// The per-bodyweight dimensionalities admit any dimensionless unit
/// (`dimensionless`, `none`, `%`, `mg/g`, ...) because pint reduces
/// `[mass] / [mass]` to dimensionless. This mirrors `check_dose_unit` in
/// `src/pkpdutils/units.py` deliberately: it must not be tightened without
/// changing Python first, or a batch Python accepts would fail here.
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

    #[test]
    fn a_dimensionless_dose_is_accepted_as_in_python() {
        // pint reduces [mass] / [mass] to dimensionless, so Python accepts it
        assert!(check_dose_unit("dimensionless").is_ok());
        assert!(is_per_bodyweight("dimensionless").unwrap());
    }

    #[test]
    fn percent_is_a_dimensionless_unit() {
        assert_eq!(unit_str("percent").unwrap(), "percent");
        assert_eq!(unit_str("%").unwrap(), "percent");
        assert_eq!(short_unit("percent"), "%");
        assert!(check_dose_unit("%").is_ok());
    }

    #[test]
    fn malformed_and_oversized_input_is_an_error_and_never_a_panic() {
        use UnitError::*;
        let syntax = |e: UnitError| matches!(e, Syntax { .. });
        for input in [
            "mg/", "/mg", "mg*", "()", "mg//l", "mg**", "mg**1.5", "(mg", "mg)",
        ] {
            assert!(syntax(parse_unit(input).unwrap_err()), "{input}");
        }
        assert!(matches!(parse_unit("milli").unwrap_err(), Unknown { .. }));
        for input in [
            "mg**99999999999",
            "l**100",
            "mg**200",
            "(mg**8)**9",
            "1/mg**65",
        ] {
            assert!(
                matches!(parse_unit(input).unwrap_err(), Exponent { .. }),
                "{input}"
            );
        }
        assert!(parse_unit("mg**64").is_ok());
        assert!(parse_unit("mg**-64").is_ok());
        // a signed exponent takes one sign only
        for input in ["mg**--2", "mg**+-2", "mg**-+2"] {
            assert!(syntax(parse_unit(input).unwrap_err()), "{input}");
        }
        assert_eq!(unit_str("mg**+2").unwrap(), "milligram ** 2");
        // a dimensionless dose must not come from an overflowing exponent
        assert!(check_dose_unit("mg**200").is_err());
        assert!(is_per_bodyweight("mg**200").is_err());
    }

    #[test]
    fn deep_nesting_is_an_error_and_not_a_stack_overflow() {
        let deep = format!("{}mg{}", "(".repeat(100_000), ")".repeat(100_000));
        assert!(matches!(
            parse_unit(&deep).unwrap_err(),
            UnitError::Nesting { .. }
        ));
        let ok = format!("{}mg{}", "(".repeat(MAX_DEPTH), ")".repeat(MAX_DEPTH));
        assert_eq!(unit_str(&ok).unwrap(), "milligram");
    }
}
