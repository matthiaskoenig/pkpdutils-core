use std::collections::BTreeMap;

use pkpdutils_abi::{
    AucMethod, BlqAction, BlqHandling, BootstrapDistribution, BootstrapSpread, C0Method, Kind,
    NcaFlag, NcaOptions, Route, TerminalMethod, UncertaintyMethod,
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

#[test]
fn terminal_windows_round_trip_through_the_records() {
    let mut opts = NcaOptions::default();
    opts.terminal.windows = Some([("s1".to_string(), (1.0, 8.0))].into_iter().collect());
    let json = serde_json::to_string(&opts).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(v["terminal"].get("windows").is_none());
    assert_eq!(v["_terminal_windows"][0]["key"]["type"], "str");
    assert_eq!(serde_json::from_str::<NcaOptions>(&json).unwrap(), opts);
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
