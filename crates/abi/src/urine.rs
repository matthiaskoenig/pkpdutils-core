//! The request of the urine analysis.

use crate::nonfinite;
use crate::{NcaOptions, Route};
use serde::{Deserialize, Serialize};

/// The collection intervals of one subject and what the analysis needs of them, `nca_urine` of
/// the host.
///
/// The amount of every collection is already derived from the concentration and the volume by
/// the python `Excretion`, which stays the only place that derivation lives.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UrineRequest {
    /// The start of every collection interval, sorted.
    #[serde(with = "nonfinite::vec")]
    pub start: Vec<f64>,
    /// The end of every collection interval.
    #[serde(with = "nonfinite::vec")]
    pub end: Vec<f64>,
    /// The amount excreted in every collection.
    #[serde(with = "nonfinite::vec")]
    pub amount: Vec<f64>,
    /// The volume of every collection, every entry `NaN` when the excretion carries no volumes.
    #[serde(with = "nonfinite::vec")]
    pub volume: Vec<f64>,
    /// The unit of the amounts.
    pub unit: String,
    /// The unit of the times.
    pub time_unit: String,
    /// The unit of the volumes.
    pub volume_unit: String,
    /// The route of the dose.
    pub route: Route,
    /// The amount of the dose, if there is one.
    #[serde(with = "nonfinite::option")]
    pub dose_amount: Option<f64>,
    /// The unit of the dose, if there is one.
    pub dose_unit: Option<String>,
    /// The plasma area the renal clearance divides by, if there is one (`NaN` is legal).
    #[serde(with = "nonfinite::option")]
    pub area: Option<f64>,
    /// The unit of the plasma area, present exactly when the area is.
    pub area_unit: Option<String>,
    /// The options of the analysis.
    pub options: NcaOptions,
}

impl UrineRequest {
    /// The request with its collections in the order of `start`, the way python sorts an
    /// `Excretion` (`np.argsort(start, kind="stable")`, one permutation for `start`, `end`,
    /// `amount` and `volume`; a tie keeps the input order, `-0.0` ties with `0.0`, a `NaN` start goes last).
    ///
    /// The numerics consume the request through this method, since the collections of a request
    /// are not necessarily in order. An array whose length differs from `start` is left alone,
    /// which [`validate`](Self::validate) rejects.
    pub(crate) fn sorted(&self) -> Self {
        let n = self.start.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            // numpy: -0.0 equals 0.0 (the stable sort keeps the input order), NaN goes last
            let (a, b) = (self.start[i], self.start[j]);
            a.partial_cmp(&b)
                .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
        });
        let permute = |v: &Vec<f64>| -> Vec<f64> {
            if v.len() == n {
                order.iter().map(|&i| v[i]).collect()
            } else {
                v.clone()
            }
        };
        Self {
            start: permute(&self.start),
            end: permute(&self.end),
            amount: permute(&self.amount),
            volume: permute(&self.volume),
            ..self.clone()
        }
    }

    /// Validate the request and put its collections in order, the only way to a [`CheckedUrine`].
    pub fn checked(&self) -> Result<CheckedUrine, String> {
        self.validate()?;
        Ok(CheckedUrine(self.sorted()))
    }

    /// Check the lengths, that every interval ends after it starts, that the intervals do not
    /// overlap (once sorted by `sorted`, like python) and that the area and its unit come
    /// together.
    ///
    /// These are the checks of the python `Excretion`, whose `_prepare` sorts the collections
    /// before they are validated, so unsorted collections are valid. A `NaN` bound fails the
    /// comparisons and is rejected like python's `np.all`.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.start.len();
        for (name, len) in [
            ("end", self.end.len()),
            ("amount", self.amount.len()),
            ("volume", self.volume.len()),
        ] {
            if len != n {
                return Err(format!(
                    "'start' has {n} entries, '{name}' has {len}; every collection needs one each"
                ));
            }
        }
        if n == 0 {
            return Err("an excretion needs at least one collection interval".to_string());
        }
        let r = self.sorted();
        if !r.start.iter().zip(&r.end).all(|(s, e)| e > s) {
            return Err("every collection interval needs 'end' above 'start'".to_string());
        }
        if !r
            .end
            .windows(2)
            .zip(r.start.windows(2))
            .all(|(e, s)| e[0] <= s[1])
        {
            return Err("the collection intervals may not overlap".to_string());
        }
        if self.area.is_some() != self.area_unit.is_some() {
            return Err("'area' and 'area_unit' come together".to_string());
        }
        Ok(())
    }
}

/// A [`UrineRequest`] which was validated and put in the order of `start`.
///
/// Holding one is the proof of both: it is built only by [`UrineRequest::checked`], so the
/// numerics cannot be handed an unsorted or malformed request.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedUrine(UrineRequest);

impl CheckedUrine {
    /// The validated request, sorted by `start`.
    pub fn request(&self) -> &UrineRequest {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> UrineRequest {
        UrineRequest {
            start: vec![0.0, 2.0, 6.0],
            end: vec![2.0, 6.0, 12.0],
            amount: vec![1.0, 2.0, 1.5],
            volume: vec![f64::NAN; 3],
            unit: "mg".to_string(),
            time_unit: "h".to_string(),
            volume_unit: "ml".to_string(),
            route: Route::Oral,
            dose_amount: Some(10.0),
            dose_unit: Some("mg".to_string()),
            area: None,
            area_unit: None,
            options: NcaOptions::default(),
        }
    }

    #[test]
    fn a_well_formed_request_validates() {
        let r = request();
        assert_eq!(r.validate(), Ok(()));
        let back: UrineRequest = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert!(back.volume[0].is_nan());
        assert_eq!(back.amount, r.amount);
    }

    #[test]
    fn arrays_of_different_lengths_are_rejected() {
        for name in ["end", "amount", "volume"] {
            let mut r = request();
            match name {
                "end" => r.end.pop(),
                "amount" => r.amount.pop(),
                _ => r.volume.pop(),
            };
            assert!(
                r.validate()
                    .unwrap_err()
                    .contains(&format!("'{name}' has 2"))
            );
        }
    }

    #[test]
    fn an_empty_collection_list_is_rejected() {
        let mut r = request();
        r.start.clear();
        r.end.clear();
        r.amount.clear();
        r.volume.clear();
        assert!(r.validate().unwrap_err().contains("at least one"));
    }

    #[test]
    fn an_end_which_is_not_after_its_start_is_rejected() {
        for end in [0.0, -1.0, f64::NAN] {
            let mut r = request();
            r.end[0] = end;
            assert!(r.validate().unwrap_err().contains("'end' above 'start'"));
        }
    }

    #[test]
    fn overlapping_collections_are_rejected() {
        let mut r = request();
        r.end[0] = 3.0;
        assert!(r.validate().unwrap_err().contains("overlap"));
        r.end[0] = 2.0;
        r.start[1] = 2.5;
        r.end[1] = 6.0;
        assert!(r.validate().is_ok(), "a gap between collections is valid");
    }

    #[test]
    fn a_request_whose_starts_are_unsorted_is_sorted_the_way_python_sorts_it() {
        let mut r = request();
        r.start = vec![6.0, 0.0, 2.0];
        r.end = vec![12.0, 2.0, 6.0];
        r.amount = vec![1.5, 1.0, 2.0];
        r.volume = vec![3.0, 1.0, 2.0];
        assert_eq!(r.validate(), Ok(()));
        let s = r.sorted();
        assert_eq!(s.start, vec![0.0, 2.0, 6.0]);
        assert_eq!(s.end, vec![2.0, 6.0, 12.0]);
        assert_eq!(s.amount, vec![1.0, 2.0, 1.5]);
        assert_eq!(s.volume, vec![1.0, 2.0, 3.0]);
        // a tie on start (a zero-length-free case needs an invalid request, so check the
        // permutation alone): the stable sort keeps the input order
        let mut t = request();
        t.start = vec![1.0, 0.0, 1.0, 0.0];
        t.end = vec![2.0; 4];
        t.amount = vec![10.0, 20.0, 30.0, 40.0];
        t.volume = vec![1.0, 2.0, 3.0, 4.0];
        let s = t.sorted();
        assert_eq!(s.amount, vec![20.0, 40.0, 10.0, 30.0]);
        assert_eq!(s.volume, vec![2.0, 4.0, 1.0, 3.0]);
    }

    #[test]
    fn checked_validates_then_sorts() {
        let mut r = request();
        r.start = vec![6.0, 0.0, 2.0];
        r.end = vec![12.0, 2.0, 6.0];
        r.amount = vec![1.5, 1.0, 2.0];
        let c = r.checked().unwrap();
        assert_eq!(c.request().start, vec![0.0, 2.0, 6.0]);
        assert_eq!(c.request().amount, vec![1.0, 2.0, 1.5]);
        r.end[0] = 6.0;
        assert!(r.checked().is_err());
    }

    #[test]
    fn negative_zero_ties_with_zero_in_input_order() {
        let mut r = request();
        r.start = vec![0.0, -0.0];
        r.end = vec![1.0; 2];
        r.amount = vec![1.0, 2.0];
        r.volume = vec![f64::NAN; 2];
        assert_eq!(r.sorted().amount, vec![1.0, 2.0]);
    }

    #[test]
    fn a_nan_area_and_a_nan_dose_amount_round_trip_and_validate() {
        let mut r = request();
        r.area = Some(f64::NAN);
        r.area_unit = Some("h*mg/l".to_string());
        r.dose_amount = Some(f64::NAN);
        let back: UrineRequest = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert!(back.area.unwrap().is_nan());
        assert!(back.dose_amount.unwrap().is_nan());
        assert_eq!(back.validate(), Ok(()));
    }

    #[test]
    fn a_single_collection_without_volumes_is_accepted() {
        let mut r = request();
        r.start.truncate(1);
        r.end.truncate(1);
        r.amount.truncate(1);
        r.volume.truncate(1);
        assert_eq!(r.validate(), Ok(()));
    }

    #[test]
    fn an_area_without_its_unit_is_rejected() {
        let mut r = request();
        r.area = Some(5.0);
        assert!(r.validate().unwrap_err().contains("come together"));
        r.area_unit = Some("mg*h/l".to_string());
        assert_eq!(r.validate(), Ok(()));
        r.area = None;
        assert!(r.validate().is_err());
    }

    #[test]
    fn an_unknown_field_is_rejected() {
        let mut v = serde_json::to_value(request()).unwrap();
        v["extra"] = serde_json::json!(1);
        assert!(serde_json::from_value::<UrineRequest>(v).is_err());
    }
}
