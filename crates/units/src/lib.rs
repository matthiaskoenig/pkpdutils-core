//! Units of the package: a curated registry, parsing and the canonical spelling.
//!
//! The registry covers exactly the dimensions the package uses (mass, length,
//! volume, time, amount of substance, activity, pressure and the dimensionless
//! units) and is not a general unit system. It reproduces the helpers of
//! `pkpdutils.units`: [`parse_unit`], [`unit_str`], [`short_unit`],
//! [`check_dose_unit`] and [`is_per_bodyweight`].
//!
//! Of the spellings it accepts, one names its unit differently than pint does:
//! `fm` is a femtometer here, where pint reads the same spelling as its `fermi`,
//! the same dimension, the same factor of `1e-15 meter` and the same short form
//! `fm`.

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
            "mg/", "/mg", "mg*", "()", "mg//l", "mg**", "(mg", "mg)", "mg.l", "mg**1.", ".5*mg",
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
    fn the_two_refusals_of_the_curation_have_an_error_of_their_own() {
        // pint's implicit product and its fractional exponent are refused here deliberately
        // (see the module documentation of `registry`), and each says so instead of landing in
        // the catch-all of the parser: "a product needs a '*'" is what a person reading a
        // fallback warning can act on, where "unexpected Name(\"l\")" is not.
        for input in ["mg l", "mg (l)", "(mg)(l)", "(mg l)/hr", "mg l hr", "1 mg"] {
            assert!(
                matches!(
                    parse_unit(input).unwrap_err(),
                    UnitError::ImplicitProduct { .. }
                ),
                "{input}"
            );
        }
        for input in [
            "hr**0.5",
            "mg**1.5",
            "hr ** 0.5",
            "mg**-1.5",
            "mg**1.0",
            "hr**.5",
        ] {
            assert!(
                matches!(
                    parse_unit(input).unwrap_err(),
                    UnitError::FractionalExponent { .. }
                ),
                "{input}"
            );
        }
    }

    #[test]
    fn a_string_this_registry_cannot_read_is_told_apart_from_a_unit_the_context_cannot_use() {
        // The classification a host routes on, and the rule is one line: a string this build
        // cannot read is not supported by this build, whatever the reason the parser gives. A
        // new variant of `UnitError` is classified by hand in `is_unsupported`, where the
        // exhaustive match refuses to compile without it.
        let deep = format!(
            "{}mg{}",
            "(".repeat(MAX_DEPTH + 1),
            ")".repeat(MAX_DEPTH + 1)
        );
        for input in [
            // the time base of a monoclonal antibody, a depot or an oncology study, and the
            // clinical spelling of an enzyme activity: the defect this classification was
            // written for
            "week",
            "weeks",
            "month",
            "year",
            "yr",
            "U",
            "U/l",
            "mU/l",
            // a prefix on a unit the registry does not let a prefix on
            "kilohour",
            "hectoliter",
            // pint's syntax beyond this parser
            "mg l",
            "hr**0.5",
            // the two limits of this build on an expression it would otherwise read
            "mg**65",
            deep.as_str(),
            // a spelling pint over-forgives and `pkpdutils` 1.4.0 therefore computes (`mg/l`
            // and `mg**2`): refusing the batch for it would narrow what a released library
            // accepts, which is the defect and not the spelling
            "mg//l",
            "mg**--2",
            // and a string which is no unit anywhere. Uniform with the rest because the rule is
            // one line and because the host parses every unit with pint before a batch exists,
            // so nothing of this class reaches the kernel through it anyway
            "",
            "mg/",
            "/mg",
            "()",
            "(mg",
            "mg)",
            "2*mg",
            "mg.l",
        ] {
            let error = parse_unit(input).unwrap_err();
            assert!(error.is_unsupported(), "{input}: {error}");
        }
        // every spelling of the package reads, so none of them is ever classified at all
        for input in [
            "mg/l", "ug/ml", "hr", "min", "day", "s", "IU", "mIU/l", "mmol/l", "%",
        ] {
            assert!(parse_unit(input).is_ok(), "{input}");
        }
        // a unit which reads and is no dose is a defect of the batch and not of this build
        let not_a_dose = check_dose_unit("mg/l").unwrap_err();
        assert!(!not_a_dose.is_unsupported(), "{not_a_dose}");
        // and so is an expression which needs a unit the context does not carry
        let context = UnitContext {
            unit: "mg/l",
            time: "hr",
            dose: None,
            amount: None,
            volume: None,
        };
        let missing = parameter_unit("{dose}/({unit})", &context).unwrap_err();
        assert!(!missing.is_unsupported(), "{missing}");
        // while the same expression over a unit this registry does not carry is a refusal
        let week = UnitContext {
            time: "week",
            ..context.clone()
        };
        let refused = parameter_unit("{unit} * {time}", &week).unwrap_err();
        assert!(refused.is_unsupported(), "{refused}");
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
            parse_unit(&format!("mg**1.{digits}")).unwrap_err(),
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
