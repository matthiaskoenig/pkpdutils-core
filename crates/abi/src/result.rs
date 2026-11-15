//! The analysis request going into the core and the result coming out of it.

use crate::batch::{Batch, Coordinate, bits_eq};
use crate::{ANALYSIS_CONTRACT, NcaOptions, check_contract, nonfinite};
use serde::{Deserialize, Serialize};

/// A batch with the options to analyse it with.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalysisRequest {
    /// The contract, [`ANALYSIS_CONTRACT`].
    pub contract: String,
    /// The timecourses.
    pub batch: Batch,
    /// The options of the analysis.
    pub options: NcaOptions,
}

/// The wire form of [`AnalysisRequest`], which checks the contract when it is read.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnalysisRequestWire {
    contract: String,
    batch: Batch,
    options: NcaOptions,
}

impl TryFrom<AnalysisRequestWire> for AnalysisRequest {
    type Error = String;

    fn try_from(w: AnalysisRequestWire) -> Result<Self, String> {
        check_contract(&w.contract, ANALYSIS_CONTRACT).map_err(|e| e.to_string())?;
        Ok(Self {
            contract: w.contract,
            batch: w.batch,
            options: w.options,
        })
    }
}

impl Serialize for AnalysisRequest {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        AnalysisRequestWire {
            contract: self.contract.clone(),
            batch: self.batch.clone(),
            options: self.options.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AnalysisRequest {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = AnalysisRequestWire::deserialize(deserializer)?;
        AnalysisRequest::try_from(wire).map_err(serde::de::Error::custom)
    }
}

/// One parameter of a result, a value per sample.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ParameterColumn {
    /// The name of the variable.
    pub name: String,
    /// The unit, as `attrs["units"]` of the variable.
    pub unit: String,
    /// One value per sample, row major, NaN where it is not defined.
    #[serde(with = "nonfinite::vec")]
    pub values: Vec<f64>,
}

impl PartialEq for ParameterColumn {
    fn eq(&self, o: &Self) -> bool {
        self.name == o.name && self.unit == o.unit && bits_eq(&self.values, &o.values)
    }
}

/// A variable with an extra dimension: the `interval`, `candidate` and `collection`
/// dimensions of `nca/intervals.py`, `nca/terminal.py` and `nca/urine.py`.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct PointVariable {
    /// The name of the variable.
    pub name: String,
    /// The unit.
    pub unit: String,
    /// The name of the extra dimension.
    pub dim: String,
    /// The size of the extra dimension.
    pub size: usize,
    /// `n_samples * size` values, row major, NaN padded.
    #[serde(with = "nonfinite::vec")]
    pub values: Vec<f64>,
}

impl PartialEq for PointVariable {
    fn eq(&self, o: &Self) -> bool {
        self.name == o.name
            && self.unit == o.unit
            && self.dim == o.dim
            && self.size == o.size
            && bits_eq(&self.values, &o.values)
    }
}

/// The result of a non-compartmental analysis, one value per sample and parameter.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AnalysisResult {
    /// The names of the sample dimensions, outermost first.
    pub sample_dims: Vec<String>,
    /// The size of every sample dimension.
    pub sample_shape: Vec<usize>,
    /// The parameters.
    pub parameters: Vec<ParameterColumn>,
    /// The variables over an extra dimension.
    pub point_variables: Vec<PointVariable>,
    /// The flag bits of every sample, see `NcaFlag`.
    pub flags: Vec<u32>,
    /// The non-dimension coordinates of the batch, carried into the result.
    pub coords: Vec<(String, Coordinate)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result() -> AnalysisResult {
        AnalysisResult {
            sample_dims: vec!["subject".into()],
            sample_shape: vec![2],
            parameters: vec![ParameterColumn {
                name: "cmax".into(),
                unit: "mg/l".into(),
                values: vec![6.0, f64::NAN],
            }],
            point_variables: vec![PointVariable {
                name: "interval_auc".into(),
                unit: "h*mg/l".into(),
                dim: "interval".into(),
                size: 2,
                values: vec![1.0, 2.0, 3.0, f64::NAN],
            }],
            flags: vec![0, 4],
            coords: vec![("period".into(), Coordinate::Int(vec![1, 2]))],
        }
    }

    #[test]
    fn a_result_with_nan_round_trips() {
        let r = result();
        let back: AnalysisResult =
            serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert!(back.parameters[0].values[1].is_nan());
        assert_eq!(back, r);
    }

    #[test]
    fn a_request_round_trips_and_checks_its_contract() {
        let req = AnalysisRequest {
            contract: ANALYSIS_CONTRACT.into(),
            batch: Batch::example_with_undosed_row(),
            options: NcaOptions::default(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(serde_json::from_str::<AnalysisRequest>(&json).unwrap(), req);
        let bad = json.replacen("analysis/1", "analysis/2", 1);
        let err = serde_json::from_str::<AnalysisRequest>(&bad)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("analysis/2") && err.contains("analysis/1"),
            "{err}"
        );
    }
}
