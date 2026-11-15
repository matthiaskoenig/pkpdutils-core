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
    pkpdutils_path: String,
    n_records: usize,
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
        !p.pkpdutils_version.is_empty() && !p.pkpdutils_path.is_empty(),
        "the fixture does not record the pkpdutils it was generated with"
    );
    assert_eq!(
        p.n_records,
        fixture.records.len(),
        "the fixture header disagrees with its records"
    );
    let records = fixture.records;
    assert!(
        records.len() > 1000,
        "the gate fixture looks truncated: {}",
        records.len()
    );
    records
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
        match parameter_unit(&r.expression, &ctx) {
            Ok((unit, factor)) => {
                if unit != r.expected_unit
                    || (factor - r.expected_factor).abs() > 1e-12 * r.expected_factor.abs().max(1.0)
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
