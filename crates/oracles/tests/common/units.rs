//! Reading a unit of an oracle and expressing a magnitude of a result in it.
//!
//! Shared by the numeric gates and by them only. `tests/common/mod.rs` is compiled into every
//! test binary of the crate, `tests/oracles_present.rs` among them, and that one converts
//! nothing, so a helper put there would be dead code under `-D warnings`. This file is pulled
//! in with `#[path = "common/units.rs"] mod units;` by the gates which convert and left out of
//! the one which does not. It is a module of each of them and no test binary of its own, since
//! cargo takes only the files directly under `tests/` for that.

use pkpdutils_units::{Unit, parse_unit};

/// A unit of an oracle or of a result, naming what could not be read.
#[track_caller]
pub fn unit_of(spelling: &str, what: &str) -> Unit {
    parse_unit(spelling).unwrap_or_else(|e| panic!("{what}: {spelling} is no unit: {e}"))
}

/// A magnitude in `from`, expressed in `to`.
///
/// This is `Quantity.to(unit)` of pint, which the python gates convert with
/// (`test_reference.py:108`). An oracle writes a magnitude in a unit of its own, which is not
/// always the canonical one the kernel reports, so an oracle gate cannot avoid converting.
///
/// Converting with the registry the kernel itself uses looks circular, and the dimension
/// assertion below does not undo that: a dimension is a vector of exponents and a factor is a
/// separate accumulation, so equal dimensions say nothing about a wrong factor. What makes it
/// sound is that the expected magnitudes are fixed absolute numbers and that every factor which
/// could hide an error cancels out of the comparison. Take the hardest case, the Divoll
/// clearance of `tests/reference.rs`: the kernel divides the raw
/// `milligram * milliliter / (microgram * hour)` by the factor of `liter / hour` to report in
/// `liter / hour`, and this function multiplies it back by that same factor before dividing by
/// the factor of the oracle's `milliliter / hour`. The factor of `liter / hour` therefore
/// appears once in each direction, `milliliter` cancels against the raw unit, and the compared
/// number reduces to `raw * f_milligram / f_microgram`. The factors that cancel are exactly the
/// ones the correct answer does not depend on either. What does not cancel is a ratio of two
/// spellings of one dimension, and each of those is a plain prefix ratio whose error would be a
/// power of ten, orders of magnitude outside the tightest tolerance of either gate.
///
/// The dimension assertion is here for a different error: a unit the kernel reports for a
/// parameter of another dimension than the reference then fails by name instead of as a number
/// which is merely out by a factor.
///
/// How much of this is live differs between the two gates:
///
/// - `tests/reference.rs` really converts. `nca_reference.json` writes every magnitude in a
///   unit of its own, and the ratios which do not cancel there are `f_milligram / f_microgram`
///   for the Divoll and Kim clearances, `f_mole / f_nanomole` for the synthetic curve,
///   `f_milliliter / f_liter` for the volumes and `f_minute / f_hour` for the midazolam slope.
/// - `tests/validation.rs` converts nothing. The validation oracle writes every number in the
///   unit the kernel reports for it: the indomethacin numbers in `microgram / milliliter`, as
///   the data is measured, and `cl` in `liter / hour` and `vz`/`vss` in `liter`, as the kernel
///   normalizes them. `from` equals `to` for all 1043 numbers of that gate, so only the
///   dimension assertion is live there. The conversion stays because the identity is a property
///   of that one file and not of the gate: an oracle which published, say, a WinNonlin table in
///   `mg/L` would need it, and would then be carried by the argument above.
#[track_caller]
pub fn in_unit(value: f64, from: &str, to: &str, what: &str) -> f64 {
    let from_unit = unit_of(from, what);
    let to_unit = unit_of(to, what);
    assert_eq!(
        from_unit.dimension(),
        to_unit.dimension(),
        "{what}: the kernel reports {from}, whose dimension is not that of the reference unit {to}"
    );
    value * from_unit.factor() / to_unit.factor()
}
