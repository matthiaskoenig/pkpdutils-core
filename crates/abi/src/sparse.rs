//! The request of the sparse (destructive and batch design) analysis.

use crate::nonfinite;
use serde::{Deserialize, Serialize};

/// The sampling design, `design` of `nca_sparse` of the host.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SparseDesign {
    /// One sample per animal.
    Serial,
    /// Several samples per animal.
    Batch,
}

/// The concentrations of `n_rows` animals at `n_cols` common times, `nca_sparse` of the host.
///
/// `values` is row-major `(n_rows, n_cols)`; a `NaN` entry is a sample not taken.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseRequest {
    /// The sampling times, strictly increasing.
    #[serde(with = "nonfinite::vec")]
    pub times: Vec<f64>,
    /// The concentrations, row-major `(n_rows, n_cols)`.
    #[serde(with = "nonfinite::vec")]
    pub values: Vec<f64>,
    /// The number of animals.
    pub n_rows: usize,
    /// The number of sampling times.
    pub n_cols: usize,
    /// The sampling design.
    pub design: SparseDesign,
    /// The unit of the concentrations.
    pub unit: String,
    /// The unit of the times.
    pub time_unit: String,
    /// The unit of the dose, if there is one.
    pub dose_unit: Option<String>,
}

impl SparseRequest {
    /// Check the shape (without overflowing), that the times strictly increase and that a serial
    /// design has at most one finite sample per animal.
    pub fn validate(&self) -> Result<(), String> {
        if self.n_rows == 0 || self.n_cols == 0 {
            return Err(format!(
                "{} rows x {} columns: both must be positive",
                self.n_rows, self.n_cols
            ));
        }
        let size = self
            .n_rows
            .checked_mul(self.n_cols)
            .ok_or_else(|| format!("{} rows x {} columns overflow", self.n_rows, self.n_cols))?;
        if self.values.len() != size {
            return Err(format!(
                "{} values for {} rows x {} columns",
                self.values.len(),
                self.n_rows,
                self.n_cols
            ));
        }
        if self.times.len() != self.n_cols {
            return Err(format!(
                "{} times for {} columns",
                self.times.len(),
                self.n_cols
            ));
        }
        if self
            .times
            .windows(2)
            .any(|w| w[0].partial_cmp(&w[1]) != Some(std::cmp::Ordering::Less))
        {
            return Err("the times must be strictly increasing".to_string());
        }
        if self.design == SparseDesign::Serial {
            let k = self
                .values
                .chunks(self.n_cols)
                .filter(|row| row.iter().filter(|v| v.is_finite()).count() > 1)
                .count();
            if k > 0 {
                return Err(format!(
                    "{k} of {} animals carry more than one sample, which is not a serial design; use design='batch'",
                    self.n_rows
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> SparseRequest {
        SparseRequest {
            times: vec![1.0, 2.0, 3.0],
            values: vec![1.0, f64::NAN, f64::NAN, f64::NAN, 2.0, f64::NAN],
            n_rows: 2,
            n_cols: 3,
            design: SparseDesign::Serial,
            unit: "mg/l".to_string(),
            time_unit: "h".to_string(),
            dose_unit: None,
        }
    }

    #[test]
    fn a_well_formed_request_validates() {
        assert_eq!(request().validate(), Ok(()));
    }

    #[test]
    fn a_values_length_which_is_not_n_rows_times_n_cols_is_rejected() {
        let mut r = request();
        r.values.push(1.0);
        assert!(r.validate().unwrap_err().contains("7 values"));
    }

    #[test]
    fn a_times_length_which_is_not_n_cols_is_rejected() {
        let mut r = request();
        r.times.pop();
        assert!(r.validate().unwrap_err().contains("2 times"));
    }

    #[test]
    fn times_which_do_not_increase_are_rejected() {
        for times in [
            vec![1.0, 1.0, 3.0],
            vec![1.0, 3.0, 2.0],
            vec![1.0, f64::NAN, 3.0],
        ] {
            let mut r = request();
            r.times = times;
            assert!(r.validate().unwrap_err().contains("strictly increasing"));
        }
    }

    #[test]
    fn zero_rows_or_zero_columns_is_rejected() {
        let mut r = request();
        r.n_rows = 0;
        r.values.clear();
        assert!(r.validate().is_err());
        let mut r = request();
        r.n_cols = 0;
        r.values.clear();
        r.times.clear();
        assert!(r.validate().is_err());
    }

    #[test]
    fn a_serial_design_with_an_animal_of_two_samples_is_rejected() {
        let mut r = request();
        r.values[1] = 5.0;
        assert_eq!(
            r.validate().unwrap_err(),
            "1 of 2 animals carry more than one sample, which is not a serial design; use design='batch'"
        );
        r.design = SparseDesign::Batch;
        assert_eq!(r.validate(), Ok(()));
    }

    #[test]
    fn nan_values_survive_a_round_trip_through_serde() {
        let r = request();
        let back: SparseRequest =
            serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back.values.len(), 6);
        assert!(back.values[1].is_nan());
        assert_eq!(back.values[0], 1.0);
        assert_eq!(back.design, SparseDesign::Serial);
    }

    #[test]
    fn an_unknown_field_is_rejected() {
        let mut v = serde_json::to_value(request()).unwrap();
        v["extra"] = serde_json::json!(1);
        assert!(serde_json::from_value::<SparseRequest>(v).is_err());
    }

    #[test]
    fn a_row_of_infinities_counts_as_no_sample_in_a_serial_design() {
        let mut r = request();
        r.values[1] = f64::INFINITY;
        r.values[2] = f64::NEG_INFINITY;
        assert_eq!(r.validate(), Ok(()));
    }

    #[test]
    fn a_design_which_is_not_serial_or_batch_is_rejected_by_serde() {
        for design in ["Serial", "stratified"] {
            let mut v = serde_json::to_value(request()).unwrap();
            v["design"] = serde_json::json!(design);
            assert!(serde_json::from_value::<SparseRequest>(v).is_err());
        }
    }
}
