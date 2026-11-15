//! The pint gate: every unit spelling of the package and every derived
//! parameter unit agrees with pint.
//!
//! The fixture `data/pint_gate.json` is pint's answer, written by
//! `scripts/pint_gate.py` from the `pkpdutils` checkout this crate mirrors. It
//! is never edited by hand.

use std::collections::{BTreeSet, HashMap};

use pkpdutils_units::{UnitContext, parameter_unit, parse_unit, short_unit, unit_str};
use serde::Deserialize;

/// The sizes of the fixture, independent of its own header.
///
/// The header describes the records next to it, so a fixture regenerated against
/// a degraded `pkpdutils` would agree with itself and `--check` would agree with
/// the same degraded library. These are the backstop that fires in continuous
/// integration, where there is no `pkpdutils` to compare against at all, in the
/// same way as `MANIFEST_ENTRIES` in `crates/abi/tests/oracles_present.rs`.
/// Update them deliberately when `PARAMETER_UNITS` or the unit literals of
/// `pkpdutils` change, and never to make a failing gate pass.
const N_RECORDS: usize = 5078;
const N_PARAMETERS: usize = 135;
const N_EXPRESSIONS: usize = 16;
const N_SPELLINGS: usize = 39;

/// One record of the parameter unit section.
#[derive(Deserialize)]
struct Record {
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

/// One unit literal of `pkpdutils` with pint's unit and short form.
#[derive(Deserialize)]
struct Spelling {
    literal: String,
    unit: String,
    short: String,
}

/// One spelling pint accepts which the curated registry deliberately does not.
#[derive(Deserialize)]
struct Rejected {
    literal: String,
    unit: String,
    reason: String,
}

/// The provenance header of the fixture: which `pkpdutils` produced it and how.
#[derive(Deserialize)]
struct Provenance {
    pkpdutils_version: String,
    pkpdutils_source: String,
    n_records: usize,
    n_parameters: usize,
    n_expressions: usize,
    n_spellings: usize,
    n_rejected: usize,
    parameter_units_digest: String,
}

/// The fixture: the provenance header and the four sections.
#[derive(Deserialize)]
struct Fixture {
    provenance: Provenance,
    /// Every parameter of `PARAMETER_UNITS` with its expression.
    parameters: HashMap<String, String>,
    spellings: Vec<Spelling>,
    rejected: Vec<Rejected>,
    records: Vec<Record>,
}

fn fixture() -> Fixture {
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
    // the header against the constants of this file, then the sections against
    // the header, so neither a trimmed fixture nor a regenerated one can shrink
    assert_eq!(
        (p.n_records, p.n_parameters, p.n_expressions, p.n_spellings),
        (N_RECORDS, N_PARAMETERS, N_EXPRESSIONS, N_SPELLINGS),
        "the fixture is not the size this gate was written for"
    );
    assert_eq!(
        (
            fixture.records.len(),
            fixture.parameters.len(),
            fixture.spellings.len(),
            fixture.rejected.len()
        ),
        (p.n_records, p.n_parameters, p.n_spellings, p.n_rejected),
        "the fixture header disagrees with its sections"
    );
    let gated: BTreeSet<&str> = fixture
        .records
        .iter()
        .map(|r| r.expression.as_str())
        .collect();
    assert_eq!(
        gated.len(),
        p.n_expressions,
        "the records miss an expression"
    );
    let ungated: Vec<&String> = fixture
        .parameters
        .iter()
        .filter(|(_, e)| !gated.contains(e.as_str()))
        .map(|(name, _)| name)
        .collect();
    assert!(
        ungated.is_empty(),
        "the expression of these parameters is not gated: {ungated:?}"
    );
    fixture
}

fn describe(r: &Record) -> String {
    format!(
        "[{}] with {} / {} / {:?} / {:?} / {:?}",
        r.expression, r.unit, r.time, r.dose, r.amount, r.volume
    )
}

#[test]
fn every_parameter_unit_matches_pint_character_for_character() {
    let records = fixture().records;
    let mut failures = Vec::new();
    for r in &records {
        let ctx = UnitContext {
            unit: &r.unit,
            time: &r.time,
            dose: r.dose.as_deref(),
            amount: r.amount.as_deref(),
            volume: r.volume.as_deref(),
        };
        // Measured against pint: a minority of the records differ from pint's
        // double by at most 2.4e-16 relative, through multiplication order
        // (found by switching the comparison to `==`). 1e-14 is the bound chosen
        // with headroom over that, so a failure near it is a real change, not
        // noise.
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
    let records = fixture().records;
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

/// The gate on the input side: every unit literal `pkpdutils` spells anywhere in
/// its source, tests, documentation and examples resolves to the unit pint
/// resolves it to, in the long and in the short form.
///
/// This is the axis the parameter records do not pin: they cross the expressions
/// exhaustively, and a spelling the registry does not know is simply absent from
/// them.
#[test]
fn every_unit_spelling_of_the_package_matches_pint() {
    let spellings = fixture().spellings;
    let failures: Vec<String> = spellings
        .iter()
        .filter_map(|s| match unit_str(&s.literal) {
            Err(e) => Some(format!("{:?}: {e}, pint says {:?}", s.literal, s.unit)),
            Ok(unit) if unit != s.unit => Some(format!(
                "{:?}: got {unit:?}, pint says {:?}",
                s.literal, s.unit
            )),
            Ok(unit) => {
                let short = short_unit(&unit);
                (short != s.short).then(|| {
                    format!(
                        "{:?}: short_unit({unit:?}) = {short:?}, pint says {:?}",
                        s.literal, s.short
                    )
                })
            }
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} spellings disagree with pint:\n{}",
        failures.len(),
        spellings.len(),
        failures.join("\n")
    );
}

/// The exclusions are deliberate, so they are a tested fact: widening the
/// registry to one of them fails here until `REJECTED` in `scripts/pint_gate.py`
/// drops the entry and the fixture is regenerated.
#[test]
fn the_deliberate_exclusions_are_rejected() {
    let rejected = fixture().rejected;
    assert!(!rejected.is_empty(), "the fixture records no exclusion");
    for r in &rejected {
        assert!(
            parse_unit(&r.literal).is_err(),
            "{:?} is pint's {:?} and is supposed to be out of the registry ({}), \
             but it parses",
            r.literal,
            r.unit,
            r.reason
        );
    }
}
