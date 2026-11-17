//! The request and the answer of the summary statistics of a parameter table.

use crate::nonfinite;
use serde::{Deserialize, Serialize};

/// Raw parameter values to reduce over their samples, `ParameterResult.summarize` of python.
///
/// `values` is row-major `(n_vars, n_groups, n_samples)`. `lognormal` and `discrete` hold one
/// flag per variable and say which statistics are meaningful; the kernel computes every
/// statistic of every variable and the host drops the ones it does not want.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SummaryRequest {
    /// The values, `NaN` or infinite for a missing one.
    #[serde(with = "nonfinite::vec")]
    pub values: Vec<f64>,
    /// The number of variables.
    pub n_vars: usize,
    /// The number of groups of every variable.
    pub n_groups: usize,
    /// The number of samples of every group.
    pub n_samples: usize,
    /// The level of the confidence interval of the mean, strictly between 0 and 1.
    #[serde(with = "nonfinite")]
    pub ci_level: f64,
    /// Per variable: the parameter is log-normal.
    pub lognormal: Vec<bool>,
    /// Per variable: the parameter is discrete.
    pub discrete: Vec<bool>,
}

impl SummaryRequest {
    /// Check that the lengths of `values`, `lognormal` and `discrete` agree with the dimensions
    /// (without overflowing) and that `ci_level` is strictly between 0 and 1.
    pub fn validate(&self) -> Result<(), String> {
        let size = self
            .n_vars
            .checked_mul(self.n_groups)
            .and_then(|n| n.checked_mul(self.n_samples))
            .ok_or_else(|| {
                format!(
                    "{} variables x {} groups x {} samples overflow",
                    self.n_vars, self.n_groups, self.n_samples
                )
            })?;
        if size != self.values.len() {
            return Err(format!(
                "{} values for {} variables x {} groups x {} samples",
                self.values.len(),
                self.n_vars,
                self.n_groups,
                self.n_samples
            ));
        }
        if self.lognormal.len() != self.n_vars || self.discrete.len() != self.n_vars {
            return Err(format!(
                "{} lognormal and {} discrete flags for {} variables",
                self.lognormal.len(),
                self.discrete.len(),
                self.n_vars
            ));
        }
        if !(self.ci_level > 0.0 && self.ci_level < 1.0) {
            return Err(format!(
                "ci_level {} is not strictly between 0 and 1",
                self.ci_level
            ));
        }
        Ok(())
    }
}

/// The statistics of every group of every variable, each row-major `(n_vars, n_groups)`.
///
/// `sd`, `se`, `cv`, `ci_low` and `ci_high` are `NaN` below two finite values, the geometric
/// statistics below one (`geomean`) and two (`geocv`) positive values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SummaryAnswer {
    /// The arithmetic mean.
    #[serde(with = "nonfinite::vec")]
    pub mean: Vec<f64>,
    /// The sample standard deviation.
    #[serde(with = "nonfinite::vec")]
    pub sd: Vec<f64>,
    /// The standard error of the mean.
    #[serde(with = "nonfinite::vec")]
    pub se: Vec<f64>,
    /// The coefficient of variation as a fraction.
    #[serde(with = "nonfinite::vec")]
    pub cv: Vec<f64>,
    /// The lower bound of the t interval of the mean.
    #[serde(with = "nonfinite::vec")]
    pub ci_low: Vec<f64>,
    /// The upper bound of the t interval of the mean.
    #[serde(with = "nonfinite::vec")]
    pub ci_high: Vec<f64>,
    /// The median.
    #[serde(with = "nonfinite::vec")]
    pub median: Vec<f64>,
    /// The first quartile.
    #[serde(with = "nonfinite::vec")]
    pub q25: Vec<f64>,
    /// The third quartile.
    #[serde(with = "nonfinite::vec")]
    pub q75: Vec<f64>,
    /// The minimum.
    #[serde(with = "nonfinite::vec")]
    pub min: Vec<f64>,
    /// The maximum.
    #[serde(with = "nonfinite::vec")]
    pub max: Vec<f64>,
    /// The number of finite values.
    #[serde(with = "nonfinite::vec")]
    pub count: Vec<f64>,
    /// The geometric mean of the positive values.
    #[serde(with = "nonfinite::vec")]
    pub geomean: Vec<f64>,
    /// The geometric coefficient of variation of the positive values.
    #[serde(with = "nonfinite::vec")]
    pub geocv: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_round_trips_with_missing_values_and_checks_its_lengths() {
        let request = SummaryRequest {
            values: vec![1.0, f64::NAN, f64::INFINITY, 4.0],
            n_vars: 1,
            n_groups: 2,
            n_samples: 2,
            ci_level: 0.95,
            lognormal: vec![true],
            discrete: vec![false],
        };
        let text = serde_json::to_string(&request).unwrap();
        assert!(text.contains(r#""NaN""#) && text.contains(r#""inf""#));
        let back: SummaryRequest = serde_json::from_str(&text).unwrap();
        assert_eq!(back.values[0], 1.0);
        assert!(back.values[1].is_nan());
        assert_eq!(back.values[2], f64::INFINITY);
        assert!(request.validate().is_ok());
        let short = SummaryRequest {
            n_samples: 3,
            ..request.clone()
        };
        assert!(short.validate().unwrap_err().contains("values for"));
        let flags = SummaryRequest {
            discrete: vec![],
            ..request
        };
        assert!(flags.validate().unwrap_err().contains("flags"));
    }
}
