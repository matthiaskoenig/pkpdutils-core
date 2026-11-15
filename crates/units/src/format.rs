//! The canonical long spelling of a unit and its short symbols.

use crate::UnitError;
use crate::registry::{Term, Unit, parse_unit};

/// The canonical long form of a unit.
///
/// Positive exponents come first, alphabetically by long name and joined by
/// ` * `, `1` when there are none, then every negative one as ` / name`,
/// alphabetically. An exponent beyond 1 is ` ** n`.
pub(crate) fn long_form(unit: &Unit) -> String {
    render(
        unit,
        |t| t.name.clone(),
        " * ",
        " / ",
        |n| format!(" ** {n}"),
    )
}

/// The short form, pint's `~P` format: middle dot products, superscript
/// exponents, one `/` per denominator unit.
fn short_form(unit: &Unit) -> String {
    render(unit, |t| t.symbol.clone(), "\u{22c5}", "/", superscript)
}

fn render(
    unit: &Unit,
    name: impl Fn(&Term) -> String,
    product: &str,
    divide: &str,
    power: impl Fn(i32) -> String,
) -> String {
    let spell = |t: &Term, n: i32| {
        let exp = if n > 1 { power(n) } else { String::new() };
        format!("{}{}", name(t), exp)
    };
    let numerator: Vec<String> = unit
        .terms()
        .iter()
        .filter(|t| t.exponent > 0)
        .map(|t| spell(t, t.exponent))
        .collect();
    let mut out = if numerator.is_empty() {
        "1".to_string()
    } else {
        numerator.join(product)
    };
    for t in unit.terms().iter().filter(|t| t.exponent < 0) {
        out.push_str(divide);
        out.push_str(&spell(t, -t.exponent));
    }
    out
}

fn superscript(n: i32) -> String {
    const DIGITS: [char; 10] = [
        '\u{2070}', '\u{b9}', '\u{b2}', '\u{b3}', '\u{2074}', '\u{2075}', '\u{2076}', '\u{2077}',
        '\u{2078}', '\u{2079}',
    ];
    n.to_string()
        .chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| DIGITS[d as usize])
        .collect()
}

/// The canonical long form of a unit string, `nanogram / milliliter` for `ng/ml`.
///
/// # Errors
///
/// [`UnitError`] if the string is not a unit.
pub fn unit_str(s: &str) -> Result<String, UnitError> {
    let unit = parse_unit(s)?;
    if unit.terms().is_empty() {
        return Ok("dimensionless".to_string());
    }
    Ok(long_form(&unit))
}

/// A unit in short symbols, `mg/l` for `milligram / liter`.
///
/// Only a string which already is the canonical long form is shortened; what
/// the user spelled themselves is kept as it is. An empty or dimensionless
/// unit gives the empty string and a string which is no unit is returned
/// unchanged.
pub fn short_unit(s: &str) -> String {
    if s.trim().is_empty() || s == "dimensionless" {
        return String::new();
    }
    match parse_unit(s) {
        Ok(unit) if !unit.terms().is_empty() && s == long_form(&unit) => short_form(&unit),
        _ => s.to_string(),
    }
}
