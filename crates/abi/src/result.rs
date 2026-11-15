//! The analysis request going into the core and the result coming out of it.

use crate::batch::{
    Batch, Coordinate, bits_eq, check_coords, check_len, checked_product, expected_len,
};
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
        // exhaustive, so that a new field breaks the build instead of leaving equality
        let Self { name, unit, values } = self;
        *name == o.name && *unit == o.unit && bits_eq(values, &o.values)
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
        // exhaustive, so that a new field breaks the build instead of leaving equality
        let Self {
            name,
            unit,
            dim,
            size,
            values,
        } = self;
        *name == o.name
            && *unit == o.unit
            && *dim == o.dim
            && *size == o.size
            && bits_eq(values, &o.values)
    }
}

/// The result of a non-compartmental analysis, one value per sample and parameter.
///
/// Read through a wire struct which checks the contract and the length of every array.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalysisResult {
    /// The contract, [`ANALYSIS_CONTRACT`].
    pub contract: String,
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

impl AnalysisResult {
    /// A result of this build's contract.
    pub fn new(
        sample_dims: Vec<String>,
        sample_shape: Vec<usize>,
        parameters: Vec<ParameterColumn>,
        point_variables: Vec<PointVariable>,
        flags: Vec<u32>,
        coords: Vec<(String, Coordinate)>,
    ) -> Self {
        Self {
            contract: ANALYSIS_CONTRACT.to_string(),
            sample_dims,
            sample_shape,
            parameters,
            point_variables,
            flags,
            coords,
        }
    }
}

/// The wire form of [`AnalysisResult`], which checks the contract and the lengths when read.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnalysisResultWire {
    contract: String,
    sample_dims: Vec<String>,
    sample_shape: Vec<usize>,
    parameters: Vec<ParameterColumn>,
    point_variables: Vec<PointVariable>,
    flags: Vec<u32>,
    coords: Vec<(String, Coordinate)>,
}

impl TryFrom<AnalysisResultWire> for AnalysisResult {
    type Error = String;

    fn try_from(w: AnalysisResultWire) -> Result<Self, String> {
        check_contract(&w.contract, ANALYSIS_CONTRACT).map_err(|e| e.to_string())?;
        check_len("sample_shape", w.sample_dims.len(), w.sample_shape.len())?;
        let n = checked_product(&w.sample_shape)
            .ok_or_else(|| "sample_shape: the number of samples overflows".to_string())?;
        for p in &w.parameters {
            check_len(&format!("parameters[{}].values", p.name), n, p.values.len())?;
        }
        for p in &w.point_variables {
            let field = format!("point_variables[{}].values", p.name);
            check_len(&field, expected_len(&field, n, p.size)?, p.values.len())?;
        }
        check_len("flags", n, w.flags.len())?;
        check_coords(&w.coords, n)?;
        Ok(Self {
            contract: w.contract,
            sample_dims: w.sample_dims,
            sample_shape: w.sample_shape,
            parameters: w.parameters,
            point_variables: w.point_variables,
            flags: w.flags,
            coords: w.coords,
        })
    }
}

impl Serialize for AnalysisResult {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        AnalysisResultWire {
            contract: self.contract.clone(),
            sample_dims: self.sample_dims.clone(),
            sample_shape: self.sample_shape.clone(),
            parameters: self.parameters.clone(),
            point_variables: self.point_variables.clone(),
            flags: self.flags.clone(),
            coords: self.coords.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AnalysisResult {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = AnalysisResultWire::deserialize(deserializer)?;
        AnalysisResult::try_from(wire).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result() -> AnalysisResult {
        AnalysisResult::new(
            vec!["subject".into()],
            vec![2],
            vec![ParameterColumn {
                name: "cmax".into(),
                unit: "mg/l".into(),
                values: vec![6.0, f64::NAN],
            }],
            vec![PointVariable {
                name: "interval_auc".into(),
                unit: "h*mg/l".into(),
                dim: "interval".into(),
                size: 2,
                values: vec![1.0, 2.0, 3.0, f64::NAN],
            }],
            vec![0, 4],
            vec![("period".into(), Coordinate::Int(vec![1, 2]))],
        )
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

    #[test]
    fn a_foreign_result_contract_is_rejected_by_name() {
        let mut r = result();
        r.contract = "analysis/2".into();
        let err = serde_json::from_str::<AnalysisResult>(&serde_json::to_string(&r).unwrap())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("analysis/2") && err.contains("analysis/1"),
            "{err}"
        );
    }

    #[test]
    fn a_result_with_a_short_array_is_rejected() {
        let mut json = serde_json::to_value(result()).unwrap();
        json["flags"] = serde_json::json!([0]);
        let err = serde_json::from_value::<AnalysisResult>(json)
            .unwrap_err()
            .to_string();
        assert!(err.contains("flags has length 1, expected 2"), "{err}");
        let mut json = serde_json::to_value(result()).unwrap();
        json["point_variables"][0]["values"] = serde_json::json!([1.0]);
        let err = serde_json::from_value::<AnalysisResult>(json)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("point_variables[interval_auc].values"),
            "{err}"
        );
    }
}
