//! The request and the answer of the time to steady state.

use crate::nonfinite;
use serde::{Deserialize, Serialize};

/// The estimator of the time to steady state, `TSSMethod` of the host.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TssMethod {
    /// The monoexponential approach of the troughs to a plateau.
    Monoexponential,
    /// The first interval from which the troughs no longer trend.
    Stepwise,
}

/// The troughs of the dosing intervals of `n_rows` samples, `time_to_steady_state` of the host.
///
/// `time`, `value`, `start` and `valid` are row-major `(n_rows, n_cols)`: the time every trough
/// was taken at (the end of its interval), the trough, the start of its interval and which
/// entries are troughs. The troughs of a row are packed to the front of it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TssRequest {
    /// The times of the troughs.
    #[serde(with = "nonfinite::vec")]
    pub time: Vec<f64>,
    /// The troughs.
    #[serde(with = "nonfinite::vec")]
    pub value: Vec<f64>,
    /// The start of the interval of every trough.
    #[serde(with = "nonfinite::vec")]
    pub start: Vec<f64>,
    /// Which entries are troughs.
    pub valid: Vec<bool>,
    /// The number of samples.
    pub n_rows: usize,
    /// The number of dosing intervals per sample.
    pub n_cols: usize,
    /// The estimator.
    pub method: TssMethod,
    /// The fraction of the plateau of the monoexponential estimate, strictly between 0 and 1.
    #[serde(with = "nonfinite")]
    pub fraction: f64,
    /// The level of the trend test of the stepwise estimate, strictly between 0 and 1.
    #[serde(with = "nonfinite")]
    pub alpha: f64,
}

impl TssRequest {
    /// Check that the four arrays have `n_rows * n_cols` entries (without overflowing), that the
    /// troughs of every row are packed to the front of it and that `fraction` and `alpha` are
    /// strictly between 0 and 1.
    pub fn validate(&self) -> Result<(), String> {
        let size = self
            .n_rows
            .checked_mul(self.n_cols)
            .ok_or_else(|| format!("{} rows x {} columns overflow", self.n_rows, self.n_cols))?;
        let lengths = [
            ("time", self.time.len()),
            ("value", self.value.len()),
            ("start", self.start.len()),
            ("valid", self.valid.len()),
        ];
        for (name, length) in lengths {
            if length != size {
                return Err(format!(
                    "{length} {name} entries for {} rows x {} columns",
                    self.n_rows, self.n_cols
                ));
            }
        }
        if self.n_cols > 0 {
            for (r, row) in self.valid.chunks(self.n_cols).enumerate() {
                let count = row.iter().take_while(|&&v| v).count();
                if row[count..].iter().any(|&v| v) {
                    return Err(format!(
                        "the troughs of row {r} are not packed to the front of the row"
                    ));
                }
            }
        }
        if !(self.fraction > 0.0 && self.fraction < 1.0) {
            return Err(format!(
                "fraction {} is not strictly between 0 and 1",
                self.fraction
            ));
        }
        if !(self.alpha > 0.0 && self.alpha < 1.0) {
            return Err(format!(
                "alpha {} is not strictly between 0 and 1",
                self.alpha
            ));
        }
        Ok(())
    }
}

/// The time to steady state of every sample.
///
/// `c_ss` is the plateau of the monoexponential estimate and empty for the stepwise one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TssAnswer {
    /// The time to steady state per sample, `NaN` where the troughs carry no estimate.
    #[serde(with = "nonfinite::vec")]
    pub tss: Vec<f64>,
    /// The plateau per sample, `NaN` where there is no estimate.
    #[serde(with = "nonfinite::vec")]
    pub c_ss: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> TssRequest {
        TssRequest {
            time: vec![1.0, 2.0, 3.0, 1.0, 2.0, 0.0],
            value: vec![1.0; 6],
            start: vec![0.0; 6],
            valid: vec![true, true, true, true, true, false],
            n_rows: 2,
            n_cols: 3,
            method: TssMethod::Stepwise,
            fraction: 0.9,
            alpha: 0.05,
        }
    }

    #[test]
    fn a_well_formed_request_validates() {
        assert_eq!(request().validate(), Ok(()));
    }

    #[test]
    fn each_malformation_is_named() {
        let mut r = request();
        r.value.pop();
        assert!(r.validate().unwrap_err().contains("5 value entries"));
        let mut r = request();
        r.valid[4] = false;
        r.valid[5] = true;
        assert!(r.validate().unwrap_err().contains("row 1"));
        let mut r = request();
        r.n_rows = usize::MAX;
        assert!(r.validate().unwrap_err().contains("overflow"));
        for bad in [0.0, 1.0, f64::NAN, -0.5] {
            let mut r = request();
            r.fraction = bad;
            assert!(r.validate().unwrap_err().contains("fraction"));
            let mut r = request();
            r.alpha = bad;
            assert!(r.validate().unwrap_err().contains("alpha"));
        }
    }
}
