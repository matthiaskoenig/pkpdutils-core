//! The pint gate: every derived parameter unit agrees with pint.
//!
//! The fixture `data/pint_gate.json` is pint's answer, written by
//! `scripts/pint_gate.py`. It is never edited by hand.

use pkpdutils_units::{UnitContext, parameter_unit, short_unit};
use serde::Deserialize;

/// One record of the fixture.
#[derive(Deserialize)]
struct Record {
    parameter: String,
    expression: String,
    unit: String,
    time: String,
    dose: Option<String>,
    amount: Option<String>,
    volume: Option<String>,
    expected_unit: String,
    expected_factor: f64,
    expected_short: String,
}

/// The provenance header of the fixture: which `pkpdutils` produced it.
#[derive(Deserialize)]
struct Provenance {
    pkpdutils_version: String,
    pkpdutils_source: String,
    n_records: usize,
    n_parameters: usize,
    parameter_units_digest: String,
}

/// The fixture: the provenance header and the records.
#[derive(Deserialize)]
struct Fixture {
    provenance: Provenance,
    records: Vec<Record>,
}

fn records() -> Vec<Record> {
    let fixture: Fixture =
        serde_json::from_str(include_str!("data/pint_gate.json")).expect("fixture parses");
    let p = &fixture.provenance;
    assert!(
        !p.pkpdutils_version.is_empty()
            && matches!(p.pkpdutils_source.as_str(), "checkout" | "release"),
        "the fixture does not record the pkpdutils it was generated with"
    );
    assert!(
        p.parameter_units_digest.len() == 16
            && p.parameter_units_digest
                .chars()
                .all(|c| c.is_ascii_hexdigit()),
        "the fixture carries no well formed PARAMETER_UNITS digest"
    );
    assert_eq!(
        p.n_records,
        fixture.records.len(),
        "the fixture header disagrees with its records"
    );
    let parameters: std::collections::HashSet<&str> = fixture
        .records
        .iter()
        .map(|r| r.parameter.as_str())
        .collect();
    assert_eq!(
        p.n_parameters,
        parameters.len(),
        "the fixture header disagrees with the parameters of its records"
    );
    fixture.records
}

fn describe(r: &Record) -> String {
    format!(
        "{} [{}] with {} / {} / {:?} / {:?} / {:?}",
        r.parameter, r.expression, r.unit, r.time, r.dose, r.amount, r.volume
    )
}

#[test]
fn every_parameter_unit_matches_pint_character_for_character() {
    let records = records();
    let mut failures = Vec::new();
    for r in &records {
        let ctx = UnitContext {
            unit: &r.unit,
            time: &r.time,
            dose: r.dose.as_deref(),
            amount: r.amount.as_deref(),
            volume: r.volume.as_deref(),
        };
        // Measured against pint: 110 of 2720 records differ from pint's double by at
        // most 2.4e-16 relative, through multiplication order. 1e-14 is the bound chosen
        // with headroom over that, so a failure near it is a real change, not noise.
        match parameter_unit(&r.expression, &ctx) {
            Ok((unit, factor)) => {
                if unit != r.expected_unit
                    || (factor - r.expected_factor).abs() > 1e-14 * r.expected_factor.abs().max(1.0)
                {
                    failures.push(format!(
                        "{}: got {unit} x {factor}, pint says {} x {}",
                        describe(r),
                        r.expected_unit,
                        r.expected_factor
                    ));
                }
            }
            Err(e) => failures.push(format!("{}: {e}", describe(r))),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} disagree with pint:\n{}",
        failures.len(),
        records.len(),
        failures.join("\n")
    );
}

#[test]
fn every_short_unit_matches_pint() {
    let records = records();
    let failures: Vec<String> = records
        .iter()
        .filter_map(|r| {
            let got = short_unit(&r.expected_unit);
            (got != r.expected_short).then(|| {
                format!(
                    "{}: short_unit({:?}) = {got:?}, pint says {:?}",
                    describe(r),
                    r.expected_unit,
                    r.expected_short
                )
            })
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} short units disagree with pint:\n{}",
        failures.len(),
        records.len(),
        failures.join("\n")
    );
}
