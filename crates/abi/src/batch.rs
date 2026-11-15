//! The batch of timecourses going into an analysis, the wire form of `Timecourses`.
//!
//! Every array is row major over the flattened sample dimensions. A batch is full of NaN by
//! design (the dose padding, a subject without a protocol, a BLQ value), so equality here is
//! bit equality: two NaNs in the same position are equal and a batch equals its round trip.

use crate::nonfinite;
use crate::{BATCH_CONTRACT, Route, check_contract};
use serde::{Deserialize, Serialize};

/// A coordinate along a sample dimension, which keeps its dtype through an analysis.
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
        bits_eq(&self.amounts, &other.amounts)
            && bits_eq(&self.times, &other.times)
            && opt_bits_eq(&self.durations, &other.durations)
            && self.unit == other.unit
            && self.route == other.route
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
    /// The non-dimension coordinates, each along the flattened samples.
    pub coords: Vec<(String, Coordinate)>,
}

impl PartialEq for Batch {
    fn eq(&self, o: &Self) -> bool {
        self.contract == o.contract
            && self.sample_dims == o.sample_dims
            && self.sample_shape == o.sample_shape
            && self.n_time == o.n_time
            && bits_eq(&self.times, &o.times)
            && bits_eq(&self.values, &o.values)
            && opt_bits_eq(&self.sd, &o.sd)
            && opt_bits_eq(&self.se, &o.se)
            && opt_bits_eq(&self.n, &o.n)
            && opt_bits_eq(&self.nominal_time, &o.nominal_time)
            && opt_bits_eq(&self.lloq, &o.lloq)
            && self.max_doses == o.max_doses
            && bits_eq(&self.dose_amount, &o.dose_amount)
            && bits_eq(&self.dose_time, &o.dose_time)
            && bits_eq(&self.dose_duration, &o.dose_duration)
            && self.time_unit == o.time_unit
            && self.unit == o.unit
            && self.dose_unit == o.dose_unit
            && self.substance == o.substance
            && self.substances == o.substances
            && self.route == o.route
            && self.routes == o.routes
            && self.coords == o.coords
    }
}

impl Batch {
    /// The number of samples, the product of the sample shape.
    pub fn n_samples(&self) -> usize {
        self.sample_shape.iter().product()
    }

    /// The number of doses of a row: the leading entries of its dose slice which are not `NaN`.
    ///
    /// A subject without a protocol (`io.py` `_build_batch`) has none.
    pub fn n_doses_of_row(&self, row: usize) -> usize {
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
            se: Some(vec![]),
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
            coords: vec![(
                "subject".to_string(),
                Coordinate::Str(vec!["A".to_string(), "B".to_string()]),
            )],
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
    coords: Vec<(String, Coordinate)>,
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
            coords: b.coords,
        }
    }
}

impl TryFrom<BatchWire> for Batch {
    type Error = String;

    fn try_from(w: BatchWire) -> Result<Self, String> {
        check_contract(&w.contract, BATCH_CONTRACT).map_err(|e| e.to_string())?;
        Ok(Self {
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
            coords: w.coords,
        })
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
        let batch = Batch::example_with_undosed_row();
        let mut other = batch.clone();
        other.values[0] = 1.0;
        assert_ne!(other, batch);
        let mut other = batch.clone();
        other.sd = Some(vec![]);
        assert_ne!(other, batch);
        let mut other = batch.clone();
        other.se = None;
        assert_ne!(other, batch);
    }

    #[test]
    fn an_empty_optional_array_is_not_an_absent_one() {
        let batch = Batch::example_with_undosed_row();
        let back: Batch = serde_json::from_str(&serde_json::to_string(&batch).unwrap()).unwrap();
        assert_eq!(back.se, Some(vec![]));
        assert_eq!(back.sd, None);
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
        let c = vec![
            (
                "subject".to_string(),
                Coordinate::Str(vec!["A".into(), "B".into()]),
            ),
            ("period".to_string(), Coordinate::Int(vec![1, 2])),
            ("weight".to_string(), Coordinate::Float(vec![70.0, 82.5])),
        ];
        let back: Vec<(String, Coordinate)> =
            serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
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
