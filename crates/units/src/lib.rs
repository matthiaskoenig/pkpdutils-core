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
pub use registry::{
    Dimension, MAX_DEPTH, MAX_ERROR_TEXT, MAX_EXPONENT, Term, Unit, UnitError, parse_unit,
};

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
            unit: registry::clipped(s),
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
    fn every_pint_spelling_of_a_registry_unit_resolves_to_the_same_unit() {
        // pint pairs any spelling of a prefix with any spelling of a unit and
        // takes one trailing plural `s`, so a string written against pint has
        // four spellings per prefixed unit and a plural of each. The expected
        // values are a live run of `ureg.parse_units`.
        for (input, expected) in [
            // the four prefix/base combinations, name and symbol on both sides
            ("millimole", "millimole"),
            ("mmol", "millimole"),
            ("millimol", "millimole"),
            // `tests/nca/test_urine.py` spells a dose `mmole`
            ("mmole", "millimole"),
            ("mliter", "milliliter"),
            ("mgram", "milligram"),
            ("ksecond", "kilosecond"),
            ("ugram", "microgram"),
            ("micromol", "micromole"),
            ("msec", "millisecond"),
            ("mIU", "milliIU"),
            ("kIU", "kiloIU"),
            // the micro prefix: `mc` is the CDISC PKUNIT synonym, `mu` the
            // ASCII transliteration of the Greek letter
            ("mcg", "microgram"),
            ("mcl", "microliter"),
            ("mcmol", "micromole"),
            ("mug", "microgram"),
            // meter, whose symbol is also the symbol of the milli prefix: a
            // bare spelling wins, so `m` is a meter and `mm`, `mg` and `ml`
            // decompose, as they do in pint
            ("m", "meter"),
            ("metre", "meter"),
            ("meters", "meter"),
            ("metres", "meter"),
            ("mm", "millimeter"),
            ("mmeter", "millimeter"),
            ("mcm", "micrometer"),
            ("mum", "micrometer"),
            ("um", "micrometer"),
            ("nm", "nanometer"),
            ("km", "kilometer"),
            ("cm", "centimeter"),
            ("dm", "decimeter"),
            ("am", "attometer"),
            ("Mm", "megameter"),
            ("mg", "milligram"),
            ("ml", "milliliter"),
            ("mmol", "millimole"),
            ("ms", "millisecond"),
            ("mol", "mole"),
            ("mmHg", "millimeter_Hg"),
            // the aliases of liter
            ("litre", "liter"),
            ("\u{2113}", "liter"),
            ("m\u{2113}", "milliliter"),
            ("millilitre", "milliliter"),
            // one trailing `s`
            ("litres", "liter"),
            ("mliters", "milliliter"),
            ("hours", "hour"),
            ("days", "day"),
            ("mins", "minute"),
            ("secs", "second"),
            ("mols", "mole"),
            ("mmoles", "millimole"),
            ("IUs", "IU"),
            ("counts", "count"),
            ("nones", "none"),
            ("percents", "percent"),
            ("mmHgs", "millimeter_Hg"),
        ] {
            assert_eq!(unit_str(input).unwrap(), expected, "input {input}");
        }
        // the canonical output is the same whichever spelling came in
        assert_eq!(short_unit(&unit_str("mmole").unwrap()), "mmol");
        assert_eq!(short_unit(&unit_str("mcg").unwrap()), "\u{3bc}g");
        assert_eq!(short_unit(&unit_str("litre").unwrap()), "l");
        assert!(check_dose_unit("mmole").is_ok());
        assert!(!is_per_bodyweight("mmole").unwrap());
    }

    #[test]
    fn a_volume_per_body_weight_in_cubic_meters_normalizes_as_in_python() {
        // `tests/test_units.py` asserts
        // `normalize_volume(Q_(0.5, "m**3/kg")) == 500 liter / kilogram`, which
        // is why the registry carries `meter`: it is the SI base of a dimension
        // `liter` already covers, not a dimension of its own
        let unit = parse_unit("m**3/kg").unwrap();
        assert_eq!(unit_str("m**3/kg").unwrap(), "meter ** 3 / kilogram");
        assert_eq!(short_unit("meter ** 3 / kilogram"), "m\u{b3}/kg");
        let (normalized, factor) = normalize_volume(&unit);
        assert_eq!(normalized, parse_unit("liter / kilogram").unwrap());
        assert_eq!(factor, 1000.0);
        assert_eq!(0.5 * factor, 500.0);
        // and through a parameter expression, which is how it reaches a result
        let ctx = UnitContext {
            unit: "m**3/kg",
            time: "hr",
            dose: None,
            amount: None,
            volume: None,
        };
        assert_eq!(
            parameter_unit("{unit}", &ctx).unwrap(),
            ("liter / kilogram".to_string(), 1000.0)
        );
        // a length which is no volume is left alone, as in pint
        assert_eq!(normalize_volume(&parse_unit("cm").unwrap()).1, 1.0);
    }

    #[test]
    fn fm_is_a_femtometer_here_and_a_fermi_in_pint() {
        // the one spelling of a unit the registry holds which pint names
        // differently, found by probing the whole spelling space against pint.
        // `fermi` is a unit of its own there, its symbol is `fm` and its exact
        // match beats the prefix decomposition. Same dimension, same factor and
        // the same short form, and pint prints a femtometer as `fm` as well, so
        // its own name does not survive its own round trip. Nothing in
        // pharmacokinetics spells either; this is a record, not a problem.
        assert_eq!(unit_str("fm").unwrap(), "femtometer");
        assert_eq!(short_unit("femtometer"), "fm");
        assert_eq!(parse_unit("fm").unwrap().factor(), 1e-15);
    }

    #[test]
    fn a_plural_needs_more_than_one_character_of_base_as_in_pint() {
        // pint skips a plural candidate whose base part is a single character,
        // so these are unknown there as well
        for input in ["ls", "Ls", "mgs", "kgs", "ugs", "mcgs", "mLs", "mls", "ss"] {
            assert!(
                matches!(parse_unit(input).unwrap_err(), UnitError::Unknown { .. }),
                "{input} must stay unknown"
            );
        }
        // `dimensionless` is the parser's own token and takes no plural
        assert!(parse_unit("dimensionlesss").is_err());
        // the singular wins over the plural: `ds` is a decisecond, not days
        assert_eq!(unit_str("ds").unwrap(), "decisecond");
        assert_eq!(unit_str("us").unwrap(), "microsecond");
        // a bare spelling wins over a prefixed one
        assert_eq!(unit_str("min").unwrap(), "minute");
        assert_eq!(unit_str("d").unwrap(), "day");
        assert_eq!(unit_str("mmHg").unwrap(), "millimeter_Hg");
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
    fn every_error_clips_the_host_string_it_echoes() {
        // a unit string crosses a host boundary and is unbounded, so no error
        // message may carry it whole into a console or a traceback
        let long = "a".repeat(10_000);
        let digits = "9".repeat(10_000);
        let errors = [
            parse_unit(&long).unwrap_err(),
            parse_unit(&format!("mg/{long}")).unwrap_err(),
            parse_unit(&digits).unwrap_err(),
            parse_unit(&format!("mg {long}")).unwrap_err(),
            parse_unit(&format!("mg**{digits}")).unwrap_err(),
            parse_unit(&format!("{}mg{}", "(".repeat(5_000), ")".repeat(5_000))).unwrap_err(),
            check_dose_unit(&format!("mmHg{}", "*1".repeat(5_000))).unwrap_err(),
            parameter_unit(
                &format!("{{bogus}}{long}"),
                &UnitContext {
                    unit: "mg/l",
                    time: "hr",
                    dose: None,
                    amount: None,
                    volume: None,
                },
            )
            .unwrap_err(),
            parameter_unit(
                &format!("({{dose}}){long}"),
                &UnitContext {
                    unit: "mg/l",
                    time: "hr",
                    dose: None,
                    amount: None,
                    volume: None,
                },
            )
            .unwrap_err(),
        ];
        // one input per variant which echoes a host string, so every variant of
        // `UnitError` but `Empty`, which echoes none
        for err in &errors {
            let message = err.to_string();
            assert!(
                !message.contains(&long[..MAX_ERROR_TEXT + 1])
                    && !message.contains(&digits[..MAX_ERROR_TEXT + 1]),
                "the message carries the host string: {message}"
            );
            assert!(
                message.chars().count() <= 400,
                "the message is unbounded ({} characters): {message}",
                message.chars().count()
            );
        }
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
