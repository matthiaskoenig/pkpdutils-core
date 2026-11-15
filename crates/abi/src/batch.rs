//! The batch of timecourses going into an analysis, the wire form of `Timecourses`.
//!
//! Every array is row major over the flattened sample dimensions. A batch is full of NaN by
//! design (the dose padding, a subject without a protocol, a BLQ value), so equality here is
//! bit equality: two NaNs in the same position are equal and a batch equals its round trip.

use crate::nonfinite;
use crate::{BATCH_CONTRACT, Route, check_contract};
use serde::{Deserialize, Serialize};

/// The values of a coordinate, which keep their dtype through an analysis.
///
/// `nca` carries `period`, `sequence` and covariates into the result and
/// `stats/bioequivalence.py` reads them back, so a string must not turn into a number.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum Coordinate {
    /// Text labels.
    Str(Vec<String>),
    /// Integer labels.
    Int(Vec<i64>),
    /// Floating point values, NaN and infinities allowed.
    Float(#[serde(with = "nonfinite::vec")] Vec<f64>),
    /// Booleans.
    Bool(Vec<bool>),
}

impl PartialEq for Coordinate {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Str(a), Self::Str(b)) => a == b,
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => bits_eq(a, b),
            (Self::Bool(a), Self::Bool(b)) => a == b,
            _ => false,
        }
    }
}

/// Bit equality of two slices of numbers, so that NaN equals NaN in the same position.
pub(crate) fn bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Bit equality of two optional slices of numbers.
fn opt_bits_eq(a: &Option<Vec<f64>>, b: &Option<Vec<f64>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => bits_eq(a, b),
        _ => false,
    }
}

/// The product of a sample shape, `None` on overflow.
pub(crate) fn checked_product(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |a, &b| a.checked_mul(b))
}

/// Accept `found` only when it is exactly `expected`, naming the field otherwise.
pub(crate) fn check_len(field: &str, expected: usize, found: usize) -> Result<(), String> {
    if expected == found {
        Ok(())
    } else {
        Err(format!("{field} has length {found}, expected {expected}"))
    }
}

/// The length a field of `per_sample` entries per sample must have, `n_samples * per_sample`.
pub(crate) fn expected_len(
    field: &str,
    n_samples: usize,
    per_sample: usize,
) -> Result<usize, String> {
    n_samples
        .checked_mul(per_sample)
        .ok_or_else(|| format!("{field}: {n_samples} samples times {per_sample} overflows"))
}

/// A coordinate over some of the sample dimensions.
///
/// A sample coordinate of `pkpdutils` lives on a *subset* of the sample
/// dimensions: `sample_coordinates` (`src/pkpdutils/result.py`) selects every
/// coordinate whose dims are a subset of them. So the dimensions are part of the
/// wire, and a coordinate is not one value per sample: `read_adnca(analytes=[..])`
/// puts `substance` along `analyte` only, a crossover batch has `period` of
/// length 2, and neither can be rebuilt from a column flattened over all samples.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SampleCoordinate {
    /// The name.
    pub name: String,
    /// The sample dimensions it lives on, a subset of `sample_dims` in their
    /// order; empty for a coordinate of a single value.
    pub dims: Vec<String>,
    /// The values, row major over `dims`.
    pub values: Coordinate,
}

/// Check every coordinate against the dimensions it names.
pub(crate) fn check_coords(
    coords: &[SampleCoordinate],
    sample_dims: &[String],
    sample_shape: &[usize],
) -> Result<(), String> {
    let mut seen: Vec<&str> = Vec::with_capacity(coords.len());
    for c in coords {
        let field = format!("coords[{}]", c.name);
        if seen.contains(&c.name.as_str()) {
            return Err(format!("{field} is given twice"));
        }
        seen.push(&c.name);
        let mut expected = 1usize;
        for (i, dim) in c.dims.iter().enumerate() {
            if c.dims[..i].contains(dim) {
                return Err(format!("{field} names the dimension {dim} twice"));
            }
            let axis = sample_dims
                .iter()
                .position(|d| d == dim)
                .ok_or_else(|| format!("{field} is on {dim}, which is no sample dimension"))?;
            expected = expected
                .checked_mul(sample_shape[axis])
                .ok_or_else(|| format!("{field}: the size of its dimensions overflows"))?;
        }
        check_len(&field, expected, c.values.len())?;
    }
    Ok(())
}

impl Coordinate {
    /// The number of entries.
    pub fn len(&self) -> usize {
        match self {
            Self::Str(v) => v.len(),
            Self::Int(v) => v.len(),
            Self::Float(v) => v.len(),
            Self::Bool(v) => v.len(),
        }
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The dosing protocol of one row, the doses without the `NaN` padding.
///
/// Mirrors `pkpdutils.timecourse.Dosing`: one unit and one route for every dose. The unit and
/// the route are `None` only when the batch does not state them, never a default.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct Dosing {
    /// The dose amounts.
    #[serde(with = "nonfinite::vec")]
    pub amounts: Vec<f64>,
    /// The dose times.
    #[serde(with = "nonfinite::vec")]
    pub times: Vec<f64>,
    /// The infusion durations, `Some` only for the route `iv_infusion`, as in Python.
    #[serde(with = "nonfinite::option_vec", default)]
    pub durations: Option<Vec<f64>>,
    /// The unit of the doses, `None` when the batch has no `dose_unit`.
    #[serde(default)]
    pub unit: Option<String>,
    /// The route of the row, `None` when the batch states neither `routes` nor `route`.
    #[serde(default)]
    pub route: Option<Route>,
}

impl PartialEq for Dosing {
    fn eq(&self, other: &Self) -> bool {
        // exhaustive, so that a new field breaks the build instead of leaving equality
        let Self {
            amounts,
            times,
            durations,
            unit,
            route,
        } = self;
        bits_eq(amounts, &other.amounts)
            && bits_eq(times, &other.times)
            && opt_bits_eq(durations, &other.durations)
            && *unit == other.unit
            && *route == other.route
    }
}

/// A batch of timecourses over the flattened sample dimensions.
#[derive(Clone, Debug)]
pub struct Batch {
    /// The contract, [`BATCH_CONTRACT`].
    pub contract: String,
    /// The names of the sample dimensions, outermost first.
    pub sample_dims: Vec<String>,
    /// The size of every sample dimension.
    pub sample_shape: Vec<usize>,
    /// The number of time points of every sample.
    pub n_time: usize,
    /// The observation times, `n_samples * n_time`, row major.
    pub times: Vec<f64>,
    /// The observed values, same layout as `times`.
    pub values: Vec<f64>,
    /// The standard deviation of every point, if the batch has one.
    pub sd: Option<Vec<f64>>,
    /// The standard error of every point, if the batch has one.
    pub se: Option<Vec<f64>>,
    /// The number of subjects behind every point, if the batch has it.
    pub n: Option<Vec<f64>>,
    /// The scheduled time of every point, if the batch has it.
    pub nominal_time: Option<Vec<f64>>,
    /// The lower limit of quantification, one per sample.
    pub lloq: Option<Vec<f64>>,
    /// The largest number of doses of any row, the row length of the dose arrays.
    pub max_doses: usize,
    /// The dose amounts, `n_samples * max_doses`, `NaN` padded at the back of every row.
    pub dose_amount: Vec<f64>,
    /// The dose times, same layout as `dose_amount`.
    pub dose_time: Vec<f64>,
    /// The infusion durations, same layout as `dose_amount`.
    pub dose_duration: Vec<f64>,
    /// The unit of the times.
    pub time_unit: String,
    /// The unit of the values.
    pub unit: String,
    /// The unit of the doses.
    pub dose_unit: Option<String>,
    /// The substance when every sample shares one, `None` when the batch is mixed.
    pub substance: Option<String>,
    /// The substance of every sample when they differ.
    pub substances: Option<Vec<String>>,
    /// The route when every sample shares one.
    pub route: Option<Route>,
    /// The route of every sample when they differ.
    pub routes: Option<Vec<Route>>,
    /// The tissue the values were measured in, `"plasma"`, `"urine"`, ...
    ///
    /// Batch level only, with no per-sample counterpart: `Timecourses` carries it
    /// in `attrs` and `from_timecourses` refuses curves which disagree on it, so
    /// a mixed batch does not exist in Python. The analysis hands it back in
    /// [`crate::result::Attributes`], where `cdisc.to_pp` reads it for `PPSPEC`
    /// and otherwise falls back to `plasma`.
    pub tissue: Option<String>,
    /// The non-dimension coordinates, each over the dimensions it names.
    pub coords: Vec<SampleCoordinate>,
}

impl PartialEq for Batch {
    fn eq(&self, o: &Self) -> bool {
        // exhaustive, so that a new field breaks the build instead of leaving equality
        let Self {
            contract,
            sample_dims,
            sample_shape,
            n_time,
            times,
            values,
            sd,
            se,
            n,
            nominal_time,
            lloq,
            max_doses,
            dose_amount,
            dose_time,
            dose_duration,
            time_unit,
            unit,
            dose_unit,
            substance,
            substances,
            route,
            routes,
            tissue,
            coords,
        } = self;
        *contract == o.contract
            && *sample_dims == o.sample_dims
            && *sample_shape == o.sample_shape
            && *n_time == o.n_time
            && bits_eq(times, &o.times)
            && bits_eq(values, &o.values)
            && opt_bits_eq(sd, &o.sd)
            && opt_bits_eq(se, &o.se)
            && opt_bits_eq(n, &o.n)
            && opt_bits_eq(nominal_time, &o.nominal_time)
            && opt_bits_eq(lloq, &o.lloq)
            && *max_doses == o.max_doses
            && bits_eq(dose_amount, &o.dose_amount)
            && bits_eq(dose_time, &o.dose_time)
            && bits_eq(dose_duration, &o.dose_duration)
            && *time_unit == o.time_unit
            && *unit == o.unit
            && *dose_unit == o.dose_unit
            && *substance == o.substance
            && *substances == o.substances
            && *route == o.route
            && *routes == o.routes
            && *tissue == o.tissue
            && *coords == o.coords
    }
}

impl Batch {
    /// Check every array against the sample shape, so that no row of the batch can index out
    /// of range.
    ///
    /// A batch read from the wire is validated on the way in. A host which builds the struct
    /// directly - the path the design mandates where the layouts already agree, with the numpy
    /// buffers or a `SimulationResult` passed through and no JSON - must call this before
    /// reading a row, since [`Batch::n_doses_of_row`] and [`Batch::dosing_of_row`] panic on a
    /// batch whose arrays do not fit its shape, and a panic is a trap which kills a wasm
    /// instance.
    ///
    /// # Errors
    ///
    /// The field, the expected length and the length found, or the overflow of a shape.
    pub fn validate(&self) -> Result<(), String> {
        check_len(
            "sample_shape",
            self.sample_dims.len(),
            self.sample_shape.len(),
        )?;
        let n = checked_product(&self.sample_shape)
            .ok_or_else(|| "sample_shape: the number of samples overflows".to_string())?;
        let points = expected_len("times", n, self.n_time)?;
        check_len("times", points, self.times.len())?;
        check_len("values", points, self.values.len())?;
        for (name, v) in [
            ("sd", &self.sd),
            ("se", &self.se),
            ("n", &self.n),
            ("nominal_time", &self.nominal_time),
        ] {
            if let Some(v) = v {
                check_len(name, points, v.len())?;
            }
        }
        if let Some(v) = &self.lloq {
            check_len("lloq", n, v.len())?;
        }
        if let Some(v) = &self.substances {
            check_len("substances", n, v.len())?;
        }
        if let Some(v) = &self.routes {
            check_len("routes", n, v.len())?;
        }
        let doses = expected_len("dose_amount", n, self.max_doses)?;
        check_len("dose_amount", doses, self.dose_amount.len())?;
        check_len("dose_time", doses, self.dose_time.len())?;
        check_len("dose_duration", doses, self.dose_duration.len())?;
        check_coords(&self.coords, &self.sample_dims, &self.sample_shape)
    }

    /// The number of samples, the product of the sample shape.
    ///
    /// `0` when that product overflows, which [`Batch::validate`] rejects, so that a row of
    /// such a batch is out of range rather than wrapped into it.
    pub fn n_samples(&self) -> usize {
        checked_product(&self.sample_shape).unwrap_or(0)
    }

    /// The number of doses of a row: the leading entries of its dose slice which are not `NaN`.
    ///
    /// A subject without a protocol (`io.py` `_build_batch`) has none. The padding is `NaN` at
    /// the back, so a row with a `NaN` in the middle is malformed and is treated as ending at
    /// its first `NaN`.
    ///
    /// # Panics
    ///
    /// When `row` is not a sample of the batch, or when the dose arrays do not fit the shape,
    /// a caller bug: a batch read from the wire is validated, and one built directly must pass
    /// [`Batch::validate`] first.
    pub fn n_doses_of_row(&self, row: usize) -> usize {
        assert!(
            row < self.n_samples(),
            "row {row} out of range for a batch of {} samples",
            self.n_samples()
        );
        let slice = &self.dose_amount[row * self.max_doses..(row + 1) * self.max_doses];
        slice.iter().take_while(|x| !x.is_nan()).count()
    }

    /// The route of a row: `routes[row]` when the batch carries per-sample routes, otherwise
    /// the batch-level `route`, `None` when the batch states neither.
    pub fn route_of_row(&self, row: usize) -> Option<Route> {
        match &self.routes {
            Some(routes) => routes.get(row).copied(),
            None => self.route,
        }
    }

    /// The dosing protocol of a row, without the padding, the analogue of
    /// `Timecourses.dosing_of()`.
    ///
    /// # Panics
    ///
    /// When `row` is not a sample of the batch, see [`Batch::n_doses_of_row`].
    pub fn dosing_of_row(&self, row: usize) -> Dosing {
        let n = self.n_doses_of_row(row);
        let start = row * self.max_doses;
        let route = self.route_of_row(row);
        Dosing {
            amounts: self.dose_amount[start..start + n].to_vec(),
            times: self.dose_time[start..start + n].to_vec(),
            durations: (route == Some(Route::IvInfusion))
                .then(|| self.dose_duration[start..start + n].to_vec()),
            unit: self.dose_unit.clone(),
            route,
        }
    }

    /// Two samples of four points, the first without any dose.
    #[cfg(test)]
    pub(crate) fn example_with_undosed_row() -> Self {
        let nan = f64::NAN;
        Self {
            contract: BATCH_CONTRACT.to_string(),
            sample_dims: vec!["subject".to_string()],
            sample_shape: vec![2],
            n_time: 4,
            times: vec![0.0, 1.0, 2.0, 4.0, 0.0, 1.0, 2.0, 4.0],
            values: vec![0.0, 5.0, 3.0, 1.0, 0.0, 6.0, 4.0, f64::NAN],
            sd: None,
            se: Some(vec![0.1; 8]),
            n: None,
            nominal_time: Some(vec![0.0, 1.0, 2.0, 4.0, 0.0, 1.0, 2.0, 4.0]),
            lloq: Some(vec![0.1, f64::NAN]),
            max_doses: 2,
            dose_amount: vec![nan, nan, 100.0, nan],
            dose_time: vec![nan, nan, 0.0, nan],
            dose_duration: vec![nan, nan, nan, nan],
            time_unit: "h".to_string(),
            unit: "mg/l".to_string(),
            dose_unit: Some("mg".to_string()),
            substance: Some("caffeine".to_string()),
            substances: None,
            route: Some(Route::Oral),
            routes: None,
            tissue: Some("plasma".to_string()),
            coords: vec![SampleCoordinate {
                name: "subject".to_string(),
                dims: vec!["subject".to_string()],
                values: Coordinate::Str(vec!["A".to_string(), "B".to_string()]),
            }],
        }
    }
}

/// The wire form of [`Batch`], which checks the contract when it is read.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchWire {
    contract: String,
    sample_dims: Vec<String>,
    sample_shape: Vec<usize>,
    n_time: usize,
    #[serde(with = "nonfinite::vec")]
    times: Vec<f64>,
    #[serde(with = "nonfinite::vec")]
    values: Vec<f64>,
    #[serde(with = "nonfinite::option_vec", default)]
    sd: Option<Vec<f64>>,
    #[serde(with = "nonfinite::option_vec", default)]
    se: Option<Vec<f64>>,
    #[serde(with = "nonfinite::option_vec", default)]
    n: Option<Vec<f64>>,
    #[serde(with = "nonfinite::option_vec", default)]
    nominal_time: Option<Vec<f64>>,
    #[serde(with = "nonfinite::option_vec", default)]
    lloq: Option<Vec<f64>>,
    max_doses: usize,
    #[serde(with = "nonfinite::vec")]
    dose_amount: Vec<f64>,
    #[serde(with = "nonfinite::vec")]
    dose_time: Vec<f64>,
    #[serde(with = "nonfinite::vec")]
    dose_duration: Vec<f64>,
    time_unit: String,
    unit: String,
    #[serde(default)]
    dose_unit: Option<String>,
    #[serde(default)]
    substance: Option<String>,
    #[serde(default)]
    substances: Option<Vec<String>>,
    #[serde(default)]
    route: Option<Route>,
    #[serde(default)]
    routes: Option<Vec<Route>>,
    #[serde(default)]
    tissue: Option<String>,
    #[serde(default)]
    coords: Vec<SampleCoordinate>,
}

impl From<Batch> for BatchWire {
    fn from(b: Batch) -> Self {
        Self {
            contract: b.contract,
            sample_dims: b.sample_dims,
            sample_shape: b.sample_shape,
            n_time: b.n_time,
            times: b.times,
            values: b.values,
            sd: b.sd,
            se: b.se,
            n: b.n,
            nominal_time: b.nominal_time,
            lloq: b.lloq,
            max_doses: b.max_doses,
            dose_amount: b.dose_amount,
            dose_time: b.dose_time,
            dose_duration: b.dose_duration,
            time_unit: b.time_unit,
            unit: b.unit,
            dose_unit: b.dose_unit,
            substance: b.substance,
            substances: b.substances,
            route: b.route,
            routes: b.routes,
            tissue: b.tissue,
            coords: b.coords,
        }
    }
}

impl TryFrom<BatchWire> for Batch {
    type Error = String;

    fn try_from(w: BatchWire) -> Result<Self, String> {
        check_contract(&w.contract, BATCH_CONTRACT).map_err(|e| e.to_string())?;
        let batch = Self {
            contract: w.contract,
            sample_dims: w.sample_dims,
            sample_shape: w.sample_shape,
            n_time: w.n_time,
            times: w.times,
            values: w.values,
            sd: w.sd,
            se: w.se,
            n: w.n,
            nominal_time: w.nominal_time,
            lloq: w.lloq,
            max_doses: w.max_doses,
            dose_amount: w.dose_amount,
            dose_time: w.dose_time,
            dose_duration: w.dose_duration,
            time_unit: w.time_unit,
            unit: w.unit,
            dose_unit: w.dose_unit,
            substance: w.substance,
            substances: w.substances,
            route: w.route,
            routes: w.routes,
            tissue: w.tissue,
            coords: w.coords,
        };
        batch.validate()?;
        Ok(batch)
    }
}

impl Serialize for Batch {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        BatchWire::from(self.clone()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Batch {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = BatchWire::deserialize(deserializer)?;
        Batch::try_from(wire).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_without_a_dosing_protocol_round_trips() {
        // io.py `_build_batch` gives a subject with no doses a row of NaN dose entries
        let batch = Batch::example_with_undosed_row();
        let back: Batch = serde_json::from_str(&serde_json::to_string(&batch).unwrap()).unwrap();
        assert!(back.dose_amount[0].is_nan());
        assert!(back.dose_time[0].is_nan());
        assert_eq!(back.n_doses_of_row(0), 0);
        assert_eq!(back, batch);
    }

    #[test]
    fn the_equality_still_tells_batches_apart() {
        type Mutation = (&'static str, fn(&mut Batch));
        let mutations: [Mutation; 23] = [
            ("contract", |b| b.contract = "x".into()),
            ("sample_dims", |b| b.sample_dims = vec!["y".into()]),
            ("sample_shape", |b| b.sample_shape = vec![1, 2]),
            ("n_time", |b| b.n_time = 5),
            ("times", |b| b.times[3] = 9.0),
            ("values", |b| b.values[0] = 1.0),
            ("sd", |b| b.sd = Some(vec![])),
            ("se", |b| b.se = None),
            ("n", |b| b.n = Some(vec![1.0])),
            ("nominal_time", |b| b.nominal_time = None),
            ("lloq", |b| b.lloq = Some(vec![0.1, 0.2])),
            ("max_doses", |b| b.max_doses = 3),
            ("dose_amount", |b| b.dose_amount[2] = 50.0),
            ("dose_time", |b| b.dose_time[2] = 1.0),
            ("dose_duration", |b| b.dose_duration[0] = 1.0),
            ("time_unit", |b| b.time_unit = "min".into()),
            ("unit", |b| b.unit = "ng/ml".into()),
            ("dose_unit", |b| b.dose_unit = None),
            ("substance", |b| b.substance = None),
            ("substances", |b| b.substances = Some(vec![])),
            ("route", |b| b.route = Some(Route::IvBolus)),
            ("routes", |b| b.routes = Some(vec![Route::Oral; 2])),
            ("tissue", |b| b.tissue = None),
        ];
        let batch = Batch::example_with_undosed_row();
        for (name, mutate) in mutations {
            let mut other = batch.clone();
            mutate(&mut other);
            assert_ne!(other, batch, "{name}");
        }
        let mut other = batch.clone();
        other.coords[0].values = Coordinate::Str(vec!["A".into(), "C".into()]);
        assert_ne!(other, batch, "coords");
        let mut other = batch.clone();
        other.coords[0].dims = vec![];
        assert_ne!(other, batch, "coords dims");
        let mut other = batch.clone();
        other.lloq = Some(vec![f64::NAN, 0.1]);
        assert_ne!(other, batch, "NaN position");
    }

    #[test]
    fn a_nan_in_the_middle_of_a_dose_row_ends_the_row() {
        let mut batch = Batch::example_with_undosed_row();
        batch.max_doses = 3;
        let nan = f64::NAN;
        batch.dose_amount = vec![nan, nan, nan, 100.0, nan, 50.0];
        batch.dose_time = vec![nan, nan, nan, 0.0, nan, 12.0];
        batch.dose_duration = vec![nan; 6];
        assert_eq!(batch.n_doses_of_row(1), 1);
        assert_eq!(batch.dosing_of_row(1).amounts, vec![100.0]);
    }

    #[test]
    #[should_panic(expected = "row 2 out of range")]
    fn a_row_outside_the_batch_is_a_caller_bug() {
        Batch::example_with_undosed_row().n_doses_of_row(2);
    }

    #[test]
    fn a_batch_built_without_the_wire_is_validated_by_hand() {
        // the design's bypass of JSON: a host builds the struct from the numpy
        // buffers or from a `SimulationResult`, and `n_doses_of_row` panics on a
        // batch whose dose arrays do not fit its shape, which in wasm is a trap
        let mut batch = Batch::example_with_undosed_row();
        assert!(batch.validate().is_ok());
        batch.max_doses = 3;
        let err = batch.validate().unwrap_err();
        assert!(
            err.contains("dose_amount has length 4, expected 6"),
            "{err}"
        );
        // an overflowing shape is no number of samples, so every row of such a
        // batch is out of range instead of wrapping into it
        let mut batch = Batch::example_with_undosed_row();
        batch.sample_dims = vec!["a".to_string(), "b".to_string()];
        batch.sample_shape = vec![usize::MAX, 2];
        assert!(batch.validate().unwrap_err().contains("overflows"));
        assert_eq!(batch.n_samples(), 0);
    }

    #[test]
    fn short_arrays_are_rejected_naming_field_expected_and_found() {
        type Break = (&'static str, fn(&mut serde_json::Value));
        let breaks: [Break; 9] = [
            ("dose_amount", |j| {
                j["dose_amount"] = serde_json::json!([1.0])
            }),
            ("dose_duration", |j| {
                j["dose_duration"] = serde_json::json!([])
            }),
            ("dose_time", |j| j["dose_time"] = serde_json::json!(["NaN"])),
            ("times", |j| j["times"] = serde_json::json!([1.0])),
            ("values", |j| j["values"] = serde_json::json!([])),
            ("sd", |j| j["sd"] = serde_json::json!([1.0])),
            ("lloq", |j| j["lloq"] = serde_json::json!([1.0])),
            ("routes", |j| j["routes"] = serde_json::json!(["oral"])),
            ("coords[subject]", |j| {
                j["coords"] = serde_json::json!([{"name": "subject", "dims": ["subject"],
                                        "values": {"str": ["A"]}}])
            }),
        ];
        for (field, brk) in breaks {
            let mut json = serde_json::to_value(Batch::example_with_undosed_row()).unwrap();
            brk(&mut json);
            let err = serde_json::from_value::<Batch>(json)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(&format!("{field} has length")) && err.contains("expected"),
                "{field}: {err}"
            );
        }
    }

    #[test]
    fn a_shape_which_overflows_is_rejected_not_wrapped() {
        let mut json = serde_json::to_value(Batch::example_with_undosed_row()).unwrap();
        json["sample_dims"] = serde_json::json!(["a", "b"]);
        json["sample_shape"] = serde_json::json!([usize::MAX, 2]);
        let err = serde_json::from_value::<Batch>(json)
            .unwrap_err()
            .to_string();
        assert!(err.contains("overflows"), "{err}");
        let mut json = serde_json::to_value(Batch::example_with_undosed_row()).unwrap();
        json["sample_dims"] = serde_json::json!(["a", "b"]);
        let err = serde_json::from_value::<Batch>(json)
            .unwrap_err()
            .to_string();
        assert!(err.contains("sample_shape has length"), "{err}");
    }

    #[test]
    fn an_empty_optional_array_is_not_an_absent_one() {
        // no time points, so an empty `sd` is valid and differs from an absent `se`
        let mut batch = Batch::example_with_undosed_row();
        batch.n_time = 0;
        batch.times.clear();
        batch.values.clear();
        batch.nominal_time = None;
        batch.sd = Some(vec![]);
        batch.se = None;
        let back: Batch = serde_json::from_str(&serde_json::to_string(&batch).unwrap()).unwrap();
        assert_eq!(back.sd, Some(vec![]));
        assert_eq!(back.se, None);
    }

    #[test]
    fn the_dosing_of_a_row_drops_the_padding() {
        let batch = Batch::example_with_undosed_row();
        assert_eq!(batch.n_doses_of_row(1), 1);
        let dosing = batch.dosing_of_row(1);
        assert_eq!(dosing.amounts, vec![100.0]);
        assert_eq!(dosing.times, vec![0.0]);
        assert_eq!(dosing.unit.as_deref(), Some("mg"));
        assert_eq!(dosing.route, Some(Route::Oral));
        assert_eq!(dosing.durations, None);
        let undosed = batch.dosing_of_row(0);
        assert!(undosed.amounts.is_empty() && undosed.times.is_empty());
        assert_eq!(batch.n_samples(), 2);
    }

    #[test]
    fn the_dosing_takes_the_route_of_the_row_and_durations_only_for_an_infusion() {
        let mut batch = Batch::example_with_undosed_row();
        batch.dose_duration = vec![f64::NAN, f64::NAN, 0.5, f64::NAN];
        batch.routes = Some(vec![Route::Oral, Route::IvInfusion]);
        assert_eq!(batch.dosing_of_row(0).route, Some(Route::Oral));
        let infusion = batch.dosing_of_row(1);
        assert_eq!(infusion.route, Some(Route::IvInfusion));
        assert_eq!(infusion.durations, Some(vec![0.5]));
        assert_eq!(batch.dosing_of_row(0).durations, None);
        // the batch-level route is not used when a row has its own
        assert_ne!(batch.route, infusion.route);
    }

    #[test]
    fn the_dosing_invents_neither_unit_nor_route() {
        let mut batch = Batch::example_with_undosed_row();
        batch.route = None;
        batch.dose_unit = None;
        let dosing = batch.dosing_of_row(1);
        assert_eq!((dosing.unit, dosing.route), (None, None));
    }

    #[test]
    fn coordinates_keep_their_dtype() {
        let along = |name: &str, values| SampleCoordinate {
            name: name.to_string(),
            dims: vec!["subject".to_string()],
            values,
        };
        let c = vec![
            along("subject", Coordinate::Str(vec!["A".into(), "B".into()])),
            along("period", Coordinate::Int(vec![1, 2])),
            along("weight", Coordinate::Float(vec![70.0, 82.5])),
        ];
        let back: Vec<SampleCoordinate> =
            serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
    }

    /// A batch of two analytes and two subjects, the substance along `analyte`.
    #[cfg(test)]
    fn two_analytes() -> Batch {
        let mut batch = Batch::example_with_undosed_row();
        batch.sample_dims = vec!["analyte".to_string(), "subject".to_string()];
        batch.sample_shape = vec![2, 1];
        batch.coords = vec![
            SampleCoordinate {
                name: "substance".to_string(),
                dims: vec!["analyte".to_string()],
                values: Coordinate::Str(vec!["parent".into(), "metabolite".into()]),
            },
            SampleCoordinate {
                name: "subject".to_string(),
                dims: vec!["subject".to_string()],
                values: Coordinate::Str(vec!["A".into()]),
            },
        ];
        batch
    }

    #[test]
    fn a_coordinate_on_a_subset_of_the_sample_dimensions_round_trips() {
        // `read_adnca(analytes=[..])` puts `substance` along `analyte` only, so a
        // batch of 2 analytes and 1 subject has 2 substances and not 2 * 1 of them
        // broadcast; `metabolite_ratio` and `cdisc.py` read the coordinate back
        let batch = two_analytes();
        let back: Batch = serde_json::from_str(&serde_json::to_string(&batch).unwrap()).unwrap();
        assert_eq!(back.coords[0].dims, vec!["analyte".to_string()]);
        assert_eq!(back.coords[0].values.len(), 2);
        assert_eq!(back, batch);
        // a coordinate of a single value names no dimension
        let mut batch = batch;
        batch.coords.push(SampleCoordinate {
            name: "study".to_string(),
            dims: vec![],
            values: Coordinate::Str(vec!["one".into()]),
        });
        let back: Batch = serde_json::from_str(&serde_json::to_string(&batch).unwrap()).unwrap();
        assert_eq!(back, batch);
    }

    #[test]
    fn a_coordinate_is_checked_against_the_dimensions_it_names() {
        type Break = (&'static str, fn(&mut Batch));
        let breaks: [Break; 4] = [
            ("coords[substance] has length 2, expected 1", |b| {
                b.coords[0].dims = vec!["subject".to_string()]
            }),
            (
                "coords[substance] is on period, which is no sample dimension",
                |b| b.coords[0].dims = vec!["period".to_string()],
            ),
            ("coords[substance] names the dimension analyte twice", |b| {
                b.coords[0].dims = vec!["analyte".to_string(), "analyte".to_string()]
            }),
            ("coords[substance] is given twice", |b| {
                let first = b.coords[0].clone();
                b.coords.push(first)
            }),
        ];
        for (message, brk) in breaks {
            let mut batch = two_analytes();
            brk(&mut batch);
            let err = serde_json::from_str::<Batch>(&serde_json::to_string(&batch).unwrap())
                .unwrap_err()
                .to_string();
            assert!(err.contains(message), "expected {message:?}, got {err}");
        }
    }

    #[test]
    fn a_float_coordinate_carries_nan_and_stays_a_float() {
        let c = Coordinate::Float(vec![1.0, f64::NAN]);
        let back: Coordinate = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
        assert_ne!(Coordinate::Int(vec![1]), Coordinate::Float(vec![1.0]));
    }

    #[test]
    fn a_foreign_batch_contract_is_rejected_by_name() {
        // a complete, valid batch with only the contract changed, so the error is
        // about the contract and not about a missing field
        let mut batch = Batch::example_with_undosed_row();
        batch.contract = "batch/2".to_string();
        let err = serde_json::from_str::<Batch>(&serde_json::to_string(&batch).unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("batch/2") && err.contains("batch/1"), "{err}");
    }

    #[test]
    fn an_unknown_field_is_rejected() {
        let mut json: serde_json::Value =
            serde_json::to_value(Batch::example_with_undosed_row()).unwrap();
        json["bogus"] = serde_json::json!(1);
        assert!(serde_json::from_value::<Batch>(json).is_err());
    }
}
