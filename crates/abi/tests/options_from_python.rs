use std::collections::BTreeMap;

use pkpdutils_abi::{
    Acceptance, AucMethod, Blq, BlqAction, BlqHandling, BlqRules, BlqSetting,
    BootstrapDistribution, BootstrapSpread, C0Method, Kind, NcaFlag, NcaOptions, Route,
    TerminalMethod, UncertaintyMethod, WindowLabel,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Deserialize)]
struct EnumFixture {
    values: BTreeMap<String, Vec<String>>,
    flags: BTreeMap<String, u32>,
    route_is_iv: BTreeMap<String, bool>,
    c0_codes: BTreeMap<String, i32>,
}

fn value_of<T: Serialize>(v: &T) -> String {
    match serde_json::to_value(v).unwrap() {
        serde_json::Value::String(s) => s,
        other => panic!("an enum must serialize as a string, found {other}"),
    }
}

fn assert_values<T: Serialize>(fx: &EnumFixture, name: &str, all: &[T]) {
    let mut rust: Vec<String> = all.iter().map(value_of).collect();
    rust.sort();
    assert_eq!(rust, fx.values[name], "the serde values of {name}");
}

fn assert_round_trip<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(all: &[T]) {
    for v in all {
        let json = serde_json::to_string(v).unwrap();
        assert_eq!(&serde_json::from_str::<T>(&json).unwrap(), v, "{json}");
    }
}

#[test]
fn the_python_default_options_deserialize() {
    let json = include_str!("data/options_default.json");
    let opts: NcaOptions = serde_json::from_str(json).unwrap();
    assert_eq!(opts, NcaOptions::default());
}

#[test]
fn every_python_option_round_trips_without_loss() {
    for json in [
        include_str!("data/options_default.json"),
        include_str!("data/options_full.json"),
    ] {
        let opts: NcaOptions = serde_json::from_str(json).unwrap();
        let back: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&opts).unwrap()).unwrap();
        let original: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(back, original);
    }
}

#[test]
fn the_full_python_options_carry_their_values() {
    let opts: NcaOptions = serde_json::from_str(include_str!("data/options_full.json")).unwrap();
    assert_eq!(opts.terminal.method, TerminalMethod::LastN);
    assert_eq!(opts.terminal.n_points, Some(4));
    assert!(opts.terminal.keep_candidates);
    assert_eq!(opts.acceptance.r2_adj_min, Some(0.98));
    assert!(opts.acceptance.exclude);
    assert_eq!(
        opts.acceptance,
        Acceptance::pkanalix(true),
        "the constructor builds the fixture's object"
    );
    assert_eq!(opts.blq_rules(), pkpdutils_abi::BlqRules::ich_m13a());
    assert_eq!(opts.partial_aucs["early"], (0.0, 4.0));
    assert_eq!(opts.tau, Some(12.0));
    assert_eq!(opts.units["auc_last"], "mg*hr/l");
    assert_eq!(opts.seed, Some(7));
}

#[test]
fn a_foreign_options_version_is_rejected_by_name() {
    let json = r#"{"_nca_options_version": 2, "kind": "concentration"}"#;
    let err = serde_json::from_str::<NcaOptions>(json)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains('2'),
        "the error must name the version found: {err}"
    );
}

#[test]
fn an_unknown_option_field_is_rejected_by_name() {
    let json = r#"{"_nca_options_version": 1, "auc_methodd": "linear"}"#;
    let err = serde_json::from_str::<NcaOptions>(json)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("auc_methodd"),
        "the error must name the field: {err}"
    );
}

fn windows_options() -> NcaOptions {
    serde_json::from_str(include_str!("data/options_windows.json")).unwrap()
}

#[test]
fn windows_of_all_eight_label_types_round_trip_without_loss() {
    let opts = windows_options();
    let windows = opts.terminal.windows.as_ref().unwrap();
    assert_eq!(windows.len(), 9);
    let nested = WindowLabel::Tuple(vec![
        "grp".into(),
        WindowLabel::Tuple(vec![1.into(), WindowLabel::None]),
    ]);
    for (label, window) in [
        (WindowLabel::from("s1"), (1.0, 8.0)),
        (WindowLabel::Int(3), (2.0, 9.0)),
        (WindowLabel::Float(2.5), (0.5, 6.0)),
        (WindowLabel::Bool(false), (1.5, 7.0)),
        (WindowLabel::None, (0.0, 5.0)),
        (
            WindowLabel::Datetime("2024-01-02T03:04:05".into()),
            (1.0, 4.0),
        ),
        (WindowLabel::Date("2024-01-02".into()), (1.0, 3.0)),
        (
            WindowLabel::Tuple(vec!["subject".into(), 3.into()]),
            (2.0, 10.0),
        ),
        (nested, (0.25, 12.0)),
    ] {
        assert_eq!(windows[&label], window, "{label}");
    }
    // record order is not significant: the windows are a lookup table keyed by label, and the
    // ordered map re-sorts Python's insertion order, so the records are compared as a decoded map
    let python: serde_json::Value =
        serde_json::from_str(include_str!("data/options_windows.json")).unwrap();
    let back: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&opts).unwrap()).unwrap();
    let (mut python, mut back) = (python, back);
    let sorted = |v: &mut serde_json::Value| {
        let mut records = v["_terminal_windows"].take().as_array().unwrap().clone();
        records.sort_by_key(|r| r.to_string());
        records
    };
    let (rp, rb) = (sorted(&mut python), sorted(&mut back));
    assert_eq!(rp, rb, "the records, order aside");
    assert_eq!(python, back, "everything else");
    assert_eq!(
        serde_json::from_str::<NcaOptions>(&serde_json::to_string(&opts).unwrap()).unwrap(),
        opts
    );
}

#[test]
fn an_unknown_window_label_type_is_rejected_naming_it_and_the_known_ones() {
    let mut v: serde_json::Value =
        serde_json::from_str(include_str!("data/options_windows.json")).unwrap();
    v["_terminal_windows"][0]["key"]["type"] = "uuid".into();
    let err = serde_json::from_value::<NcaOptions>(v)
        .unwrap_err()
        .to_string();
    assert!(err.contains("uuid"), "{err}");
    for known in [
        "tuple", "none", "datetime", "date", "bool", "str", "int", "float",
    ] {
        assert!(err.contains(known), "{known} missing: {err}");
    }
}

#[test]
fn a_window_which_does_not_end_after_it_starts_is_rejected_with_a_readable_label() {
    let mut v: serde_json::Value =
        serde_json::from_str(include_str!("data/options_windows.json")).unwrap();
    v["_terminal_windows"][7]["window"] = serde_json::json!([5.0, 1.0]);
    let err = serde_json::from_value::<NcaOptions>(v)
        .unwrap_err()
        .to_string();
    assert!(err.contains("(\"subject\", 3)"), "{err}");
}

#[test]
fn non_finite_option_values_round_trip() {
    let opts = NcaOptions {
        effect_threshold: Some(f64::INFINITY),
        ..NcaOptions::default()
    };
    let json = serde_json::to_string(&opts).unwrap();
    assert!(json.contains("\"inf\""));
    assert_eq!(serde_json::from_str::<NcaOptions>(&json).unwrap(), opts);
}

#[test]
fn invalid_options_are_rejected() {
    for json in [
        r#"{"blq": {"first": "zero", "before_tmax": "zero"}}"#,
        r#"{"terminal": {"method": "last_n"}}"#,
        r#"{"ci_level": 1.5}"#,
        r#"{"terminal": {"min_adj_r2": 1.5}}"#,
        r#"{"terminal": {"min_adj_r2": -0.1}}"#,
        r#"{"partial_aucs": {"a": [4.0, 0.0]}}"#,
    ] {
        assert!(serde_json::from_str::<NcaOptions>(json).is_err(), "{json}");
    }
}

#[test]
fn every_enum_value_set_matches_python() {
    // keyed by serde value, not by variant name, so the two naming styles need no mapping;
    // catches a missing variant, an invented one and a misspelled value alike
    let fx: EnumFixture = serde_json::from_str(include_str!("data/enum_values.json")).unwrap();
    assert_values(&fx, "Kind", Kind::ALL);
    assert_values(&fx, "AUCMethod", AucMethod::ALL);
    assert_values(&fx, "TerminalMethod", TerminalMethod::ALL);
    assert_values(&fx, "BLQHandling", BlqHandling::ALL);
    assert_values(&fx, "BLQAction", BlqAction::ALL);
    assert_values(&fx, "C0Method", C0Method::ALL);
    assert_values(&fx, "UncertaintyMethod", UncertaintyMethod::ALL);
    assert_values(&fx, "BootstrapSpread", BootstrapSpread::ALL);
    assert_values(&fx, "BootstrapDistribution", BootstrapDistribution::ALL);
    assert_values(&fx, "Route", Route::ALL);
    assert_eq!(
        fx.values.len(),
        10,
        "every enumeration of the fixture is asserted"
    );
}

#[test]
fn every_enum_value_deserializes_back_to_its_variant() {
    assert_round_trip(Kind::ALL);
    assert_round_trip(AucMethod::ALL);
    assert_round_trip(TerminalMethod::ALL);
    assert_round_trip(BlqHandling::ALL);
    assert_round_trip(BlqAction::ALL);
    assert_round_trip(C0Method::ALL);
    assert_round_trip(UncertaintyMethod::ALL);
    assert_round_trip(BootstrapSpread::ALL);
    assert_round_trip(BootstrapDistribution::ALL);
    assert_round_trip(Route::ALL);
}

#[test]
fn flag_bits_route_is_iv_and_c0_codes_match_python() {
    let fx: EnumFixture = serde_json::from_str(include_str!("data/enum_values.json")).unwrap();
    for (name, bits) in &fx.flags {
        assert_eq!(
            NcaFlag::from_name(name).unwrap().bits(),
            *bits,
            "flag {name}"
        );
    }
    assert_eq!(
        fx.flags.len(),
        14,
        "NONE is excluded, the 14 named bits remain"
    );
    for (value, is_iv) in &fx.route_is_iv {
        assert_eq!(
            serde_json::from_str::<Route>(&format!("\"{value}\""))
                .unwrap()
                .is_iv(),
            *is_iv
        );
    }
    assert_eq!(C0Method::CODE_NONE, fx.c0_codes["C0_NONE"]);
    assert_eq!(
        C0Method::CODE_BACK_EXTRAPOLATION,
        fx.c0_codes["C0_BACK_EXTRAPOLATION"]
    );
    assert_eq!(C0Method::CODE_FIRST_VALUE, fx.c0_codes["C0_FIRST_VALUE"]);
}

#[test]
fn decode_handles_the_zero_value_and_an_unknown_bit() {
    assert_eq!(NcaFlag::NONE.decode(), Vec::<&str>::new());
    assert_eq!(
        NcaFlag::from_bits_retain(1 << 30).decode(),
        Vec::<&str>::new()
    );
}

#[test]
fn every_field_of_the_all_fields_fixture_parses_to_its_own_value() {
    let o: NcaOptions = serde_json::from_str(include_str!("data/options_all_fields.json")).unwrap();
    assert_eq!(o.kind, Kind::Concentration);
    assert_eq!(o.auc_method, AucMethod::Log);
    let t = &o.terminal;
    assert_eq!(t.method, TerminalMethod::Manual);
    assert_eq!(t.min_points, 4);
    assert!(!t.exclude_cmax);
    assert_eq!(t.n_points, Some(5));
    assert_eq!(t.points, Some(vec![1_i64, 2, 3, 4, 5]));
    assert_eq!(t.min_adj_r2, Some(0.8));
    assert_eq!(t.tie_tolerance, 1e-3);
    assert_eq!(t.relative_warning, Some(1e-6));
    assert_eq!(t.windows.as_ref().unwrap()[&"a".into()], (1.0, 5.0));
    assert!(t.keep_candidates);
    assert_eq!(o.lloq, Some(0.5));
    assert_eq!(o.lloq_relative, Some(0.1));
    assert_eq!(
        o.blq,
        Blq::Rules(BlqRules {
            first: Some(BlqSetting::Value(0.5)),
            middle: Some(BlqSetting::Action(BlqAction::Keep)),
            last: Some(BlqSetting::Action(BlqAction::HalfLloq)),
            terminal_regression: true,
            ..BlqRules::default()
        })
    );
    assert_eq!(o.c0_method, C0Method::FirstValue);
    assert_eq!(o.extrapolation_warning, 0.3);
    assert_eq!(
        o.acceptance,
        Acceptance {
            r2_adj_min: Some(0.9),
            extrapolation_max: Some(0.3),
            span_min: Some(2.5),
            n_points_min: Some(4),
            exclude: true,
        }
    );
    assert_eq!(o.partial_aucs["p"], (1.0, 6.0));
    assert_eq!(o.tau, Some(8.0));
    assert_eq!(o.tau_tolerance, 0.2);
    assert!(!o.intervals);
    assert_eq!(o.units["cmax"], "ng/ml");
    assert_eq!(o.effect_threshold, Some(1.25));
    assert_eq!(o.n_workers, Some(3));
    assert_eq!(o.chunk_rows, 777);
    assert_eq!(o.uncertainty, Some(UncertaintyMethod::Bootstrap));
    assert_eq!(o.n_boot, 250);
    assert_eq!(o.seed, Some(42));
    assert_eq!(o.ci_level, 0.9);
    assert_eq!(o.bootstrap_spread, BootstrapSpread::Sd);
    assert_eq!(o.bootstrap_distribution, BootstrapDistribution::Lognormal);
    assert_eq!(o.delta_step, 0.05);
}

#[test]
fn the_effect_fixture_parses_and_both_new_fixtures_round_trip() {
    let o: NcaOptions = serde_json::from_str(include_str!("data/options_effect.json")).unwrap();
    assert_eq!(o.kind, Kind::Effect);
    assert_eq!(o.lloq, Some(0.25));
    assert_eq!(o.effect_threshold, Some(0.5));
    assert_eq!(o.blq, Blq::Handling(BlqHandling::ZeroBeforeTmax));
    assert_eq!(o.uncertainty, Some(UncertaintyMethod::Delta));
    assert_eq!(o.c0_method, C0Method::None);
    for json in [
        include_str!("data/options_effect.json"),
        include_str!("data/options_all_fields.json"),
    ] {
        let o: NcaOptions = serde_json::from_str(json).unwrap();
        let back: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&o).unwrap()).unwrap();
        assert_eq!(
            back,
            serde_json::from_str::<serde_json::Value>(json).unwrap()
        );
    }
}

#[test]
fn manual_points_keep_negative_indices_in_order() {
    let json = include_str!("data/options_manual_points.json");
    let o: NcaOptions = serde_json::from_str(json).unwrap();
    assert_eq!(o.terminal.method, TerminalMethod::Manual);
    assert_eq!(o.terminal.points, Some(vec![4, -3, 6, -1]));
    let back: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&o).unwrap()).unwrap();
    assert_eq!(
        back,
        serde_json::from_str::<serde_json::Value>(json).unwrap()
    );
}

#[test]
fn a_nan_effect_threshold_equals_itself_and_survives_a_round_trip() {
    // Python accepts any float here, so NaN is a legal options object
    let options = NcaOptions {
        effect_threshold: Some(f64::NAN),
        ..NcaOptions::default()
    };
    assert_eq!(options, options.clone());
    let back: NcaOptions = serde_json::from_str(&serde_json::to_string(&options).unwrap()).unwrap();
    assert!(back.effect_threshold.unwrap().is_nan());
    assert_eq!(back, options);
    let other = NcaOptions {
        effect_threshold: Some(1.0),
        ..NcaOptions::default()
    };
    assert_ne!(other, options);
}
