//! The analysis request going into the core and the result coming out of it.

use crate::batch::{
    Batch, SampleCoordinate, bits_eq, check_coords, check_len, checked_product, expected_len,
};
use crate::{ANALYSIS_CONTRACT, NcaOptions, Route, check_contract, nonfinite};
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
///
/// An integer variable of `nca.py`'s `INTEGER_VARIABLES` which is not `flags`
/// (`c0_method`) travels here as well: the codes are small, so a double holds
/// them exactly, and Python casts them back by that set. A boolean and a text
/// variable do not, and have their own sections,
/// [`AnalysisResult::boolean_variables`] and
/// [`AnalysisResult::text_variables`].
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

/// A boolean variable of a result, a value per sample.
///
/// `nca.py` declares `BOOLEAN_VARIABLES = {"accepted", "excluded"}` and writes
/// both onto every result, driven by `NCAOptions.acceptance`, which is part of
/// this contract, so the kernel computes them. A column of 0.0 and 1.0 would
/// carry them by a convention written nowhere, and a host without
/// `BOOLEAN_VARIABLES` to consult could not tell them from a measurement.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BooleanColumn {
    /// The name of the variable.
    pub name: String,
    /// The unit, as `attrs["units"]` of the variable; Python writes
    /// `dimensionless` for these.
    pub unit: String,
    /// One value per sample, row major.
    pub values: Vec<bool>,
}

/// A text variable of a result, a value per sample.
///
/// `nca.py` declares `TEXT_VARIABLES = {"excluded_reason"}` and writes it
/// whenever a sample is excluded; `to_dataframe` reports it and `cdisc.py` reads
/// `ds.data_vars`. Nothing numeric can carry it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TextColumn {
    /// The name of the variable.
    pub name: String,
    /// The unit, as `attrs["units"]` of the variable; Python writes
    /// `dimensionless` for these.
    pub unit: String,
    /// One value per sample, row major, the empty string where there is none.
    pub values: Vec<String>,
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

/// The dataset level attributes of a result, the `attrs` of `NCAResult.ds`.
///
/// `nca` writes `substance`, `route` and `tissue` onto the result when the batch
/// has one of each, so that a writer of the result finds what it describes:
/// `cdisc.to_pp` reads `tissue` for `PPSPEC` and falls back to `plasma`, which is
/// the wrong answer for a urine analysis and a silent one. It also writes the
/// options the analysis really ran with, which `report.py` reads back for its
/// methods line.
///
/// Python's fourth attribute, `partial_aucs`, is `options.partial_aucs` of those
/// resolved options and is not carried a second time.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Attributes {
    /// The substance when every sample of the batch shares one, `None` when the
    /// batch is mixed and names it in a coordinate instead.
    #[serde(default)]
    pub substance: Option<String>,
    /// The route when every sample of the batch shares one.
    #[serde(default)]
    pub route: Option<Route>,
    /// The tissue the values were measured in.
    #[serde(default)]
    pub tissue: Option<String>,
    /// The options the analysis ran with, resolved: the automatic choices
    /// (`uncertainty`, ...) replaced by what they resolved to.
    pub options: NcaOptions,
}

impl Attributes {
    /// The attributes of a batch analysed with `options`, the resolved ones.
    pub fn of(batch: &Batch, options: NcaOptions) -> Self {
        Self {
            substance: batch.substance.clone(),
            route: batch.route,
            tissue: batch.tissue.clone(),
            options,
        }
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
    /// The boolean variables, `accepted` and `excluded`.
    pub boolean_variables: Vec<BooleanColumn>,
    /// The text variables, `excluded_reason`.
    pub text_variables: Vec<TextColumn>,
    /// The flag bits of every sample, see `NcaFlag`.
    pub flags: Vec<u32>,
    /// The non-dimension coordinates of the batch, carried into the result.
    pub coords: Vec<SampleCoordinate>,
    /// The dataset level attributes, see [`Attributes`].
    pub attributes: Attributes,
}

impl AnalysisResult {
    /// Check every section against the sample shape, so that no column of the
    /// result is read out of range.
    ///
    /// A result read from the wire is validated on the way in. The kernel, which
    /// builds the struct, calls this before handing it to a host.
    ///
    /// # Errors
    ///
    /// The field, the expected length and the length found, or the overflow of a
    /// shape.
    pub fn validate(&self) -> Result<(), String> {
        check_len(
            "sample_shape",
            self.sample_dims.len(),
            self.sample_shape.len(),
        )?;
        let n = checked_product(&self.sample_shape)
            .ok_or_else(|| "sample_shape: the number of samples overflows".to_string())?;
        for p in &self.parameters {
            check_len(&format!("parameters[{}].values", p.name), n, p.values.len())?;
        }
        for p in &self.point_variables {
            let field = format!("point_variables[{}].values", p.name);
            check_len(&field, expected_len(&field, n, p.size)?, p.values.len())?;
        }
        for b in &self.boolean_variables {
            let field = format!("boolean_variables[{}].values", b.name);
            check_len(&field, n, b.values.len())?;
        }
        for x in &self.text_variables {
            let field = format!("text_variables[{}].values", x.name);
            check_len(&field, n, x.values.len())?;
        }
        check_len("flags", n, self.flags.len())?;
        check_coords(&self.coords, &self.sample_dims, &self.sample_shape)
    }

    /// A result of this build's contract for a sample shape, with its flags
    /// cleared and no variables yet.
    ///
    /// The variable sections are filled in by the caller, which keeps this
    /// constructor from growing an argument per section. `sample_shape` is the
    /// shape of a batch which was validated, so its product is a real number of
    /// samples; one which overflows leaves the flags empty and
    /// [`AnalysisResult::validate`] then reports it.
    ///
    /// ```
    /// use pkpdutils_abi::{AnalysisResult, Attributes, NcaOptions, ParameterColumn};
    ///
    /// let attributes = Attributes {
    ///     substance: Some("caffeine".to_string()),
    ///     route: None,
    ///     tissue: Some("plasma".to_string()),
    ///     options: NcaOptions::default(),
    /// };
    /// let result = AnalysisResult {
    ///     parameters: vec![ParameterColumn {
    ///         name: "cmax".to_string(),
    ///         unit: "mg/l".to_string(),
    ///         values: vec![6.0, 4.0],
    ///     }],
    ///     ..AnalysisResult::new(vec!["subject".to_string()], vec![2], attributes)
    /// };
    /// assert_eq!(result.flags, vec![0, 0]);
    /// assert!(result.validate().is_ok());
    /// ```
    pub fn new(sample_dims: Vec<String>, sample_shape: Vec<usize>, attributes: Attributes) -> Self {
        let n_samples = checked_product(&sample_shape).unwrap_or(0);
        Self {
            contract: ANALYSIS_CONTRACT.to_string(),
            sample_dims,
            sample_shape,
            parameters: Vec::new(),
            point_variables: Vec::new(),
            boolean_variables: Vec::new(),
            text_variables: Vec::new(),
            flags: vec![0; n_samples],
            coords: Vec::new(),
            attributes,
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
    #[serde(default)]
    boolean_variables: Vec<BooleanColumn>,
    #[serde(default)]
    text_variables: Vec<TextColumn>,
    flags: Vec<u32>,
    coords: Vec<SampleCoordinate>,
    attributes: Attributes,
}

impl TryFrom<AnalysisResultWire> for AnalysisResult {
    type Error = String;

    fn try_from(w: AnalysisResultWire) -> Result<Self, String> {
        check_contract(&w.contract, ANALYSIS_CONTRACT).map_err(|e| e.to_string())?;
        let result = Self {
            contract: w.contract,
            sample_dims: w.sample_dims,
            sample_shape: w.sample_shape,
            parameters: w.parameters,
            point_variables: w.point_variables,
            boolean_variables: w.boolean_variables,
            text_variables: w.text_variables,
            flags: w.flags,
            coords: w.coords,
            attributes: w.attributes,
        };
        result.validate()?;
        Ok(result)
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
            boolean_variables: self.boolean_variables.clone(),
            text_variables: self.text_variables.clone(),
            flags: self.flags.clone(),
            coords: self.coords.clone(),
            attributes: self.attributes.clone(),
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
    use crate::batch::Coordinate;

    fn attributes() -> Attributes {
        Attributes::of(&Batch::example_with_undosed_row(), NcaOptions::default())
    }

    fn result() -> AnalysisResult {
        AnalysisResult {
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
            boolean_variables: vec![
                BooleanColumn {
                    name: "accepted".into(),
                    unit: "dimensionless".into(),
                    values: vec![true, false],
                },
                BooleanColumn {
                    name: "excluded".into(),
                    unit: "dimensionless".into(),
                    values: vec![false, true],
                },
            ],
            text_variables: vec![TextColumn {
                name: "excluded_reason".into(),
                unit: "dimensionless".into(),
                values: vec![String::new(), "acceptance criteria".into()],
            }],
            flags: vec![0, 4],
            coords: vec![SampleCoordinate {
                name: "period".into(),
                dims: vec!["subject".into()],
                values: Coordinate::Int(vec![1, 2]),
            }],
            ..AnalysisResult::new(vec!["subject".into()], vec![2], attributes())
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

    #[test]
    fn the_attributes_of_a_urine_batch_reach_the_result() {
        // `cdisc.to_pp` reads `tissue` for `PPSPEC` and falls back to `plasma`,
        // so a urine analysis whose tissue does not cross is a wrong submission
        // dataset with no error anywhere
        let mut batch = Batch::example_with_undosed_row();
        batch.tissue = Some("urine".to_string());
        batch.substance = Some("caffeine".to_string());
        let options = NcaOptions {
            tau: Some(12.0),
            ..NcaOptions::default()
        };
        let r = AnalysisResult {
            ..AnalysisResult::new(
                vec!["subject".into()],
                vec![2],
                Attributes::of(&batch, options.clone()),
            )
        };
        let back: AnalysisResult =
            serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back.attributes.tissue.as_deref(), Some("urine"));
        assert_eq!(back.attributes.substance.as_deref(), Some("caffeine"));
        assert_eq!(back.attributes.route, Some(crate::Route::Oral));
        // the resolved options, which `report.py` reads back
        assert_eq!(back.attributes.options, options);
        assert_eq!(back, r);
        assert_eq!(back.flags, vec![0, 0]);
    }

    #[test]
    fn the_acceptance_variables_keep_their_type() {
        // `accepted`, `excluded` and `excluded_reason` are on every result
        // (`nca.py` BOOLEAN_VARIABLES and TEXT_VARIABLES) and are driven by
        // `NCAOptions.acceptance`, which is part of this contract
        let r = result();
        let back: AnalysisResult =
            serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back.boolean_variables[0].values, vec![true, false]);
        assert_eq!(back.boolean_variables[1].name, "excluded");
        assert_eq!(
            back.text_variables[0].values,
            vec![String::new(), "acceptance criteria".to_string()]
        );
        assert_eq!(back, r);
        // a boolean is not a number and a number is not a boolean
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["boolean_variables"][0]["values"][0], true);
        let mut broken = json.clone();
        broken["boolean_variables"][0]["values"] = serde_json::json!([1.0, 0.0]);
        assert!(serde_json::from_value::<AnalysisResult>(broken).is_err());
        // and both are checked against the sample count by name
        for (field, section) in [
            ("boolean_variables[accepted]", "boolean_variables"),
            ("text_variables[excluded_reason]", "text_variables"),
        ] {
            let mut broken = json.clone();
            broken[section][0]["values"] = serde_json::json!([]);
            let err = serde_json::from_value::<AnalysisResult>(broken)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(&format!("{field}.values has length 0")),
                "{err}"
            );
        }
    }

    #[test]
    fn a_result_without_attributes_is_rejected() {
        // every result comes out of an analysis, so the resolved options and the
        // batch level attributes are not optional
        let mut json = serde_json::to_value(result()).unwrap();
        json.as_object_mut().unwrap().remove("attributes");
        let err = serde_json::from_value::<AnalysisResult>(json)
            .unwrap_err()
            .to_string();
        assert!(err.contains("attributes"), "{err}");
    }

    #[test]
    fn the_attributes_tell_results_apart() {
        let r = result();
        for (name, mutate) in [
            (
                "substance",
                (|a: &mut Attributes| a.substance = None) as fn(&mut Attributes),
            ),
            ("route", |a: &mut Attributes| a.route = None),
            ("tissue", |a: &mut Attributes| {
                a.tissue = Some("urine".into())
            }),
            ("options", |a: &mut Attributes| a.options.intervals = false),
        ] {
            let mut other = r.clone();
            mutate(&mut other.attributes);
            assert_ne!(other, r, "{name}");
        }
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
    fn a_result_built_without_the_wire_is_validated_by_hand() {
        let mut r = result();
        assert!(r.validate().is_ok());
        r.flags.pop();
        let err = r.validate().unwrap_err();
        assert!(err.contains("flags has length 1, expected 2"), "{err}");
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
