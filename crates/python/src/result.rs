//! The result of the kernel, from the wire shape of the abi into numpy arrays.
//!
//! The dict which goes back carries six keys. `parameters` is the mapping `pkpdutils` consumes,
//! one `float64` array per variable of length `prod(sample_shape)` in the row order of the
//! batch; `flags` is a word of counted bits per sample and not a measurement, so it travels as
//! the unsigned integers it is; `point_variables`, `booleans` and `texts` are the three
//! sections a `float64` array per sample cannot carry, each with the name and the size of its
//! extra dimension or its own dtype; and `units` is one mapping over the variables of all four
//! of them, as `attrs["units"]` of python's result is one mapping over every variable of it.
//!
//! The sections stay apart here rather than being flattened into the one mapping `nca()`
//! consumes, because the host compares the ones it does not consume against its own result.
//! What does not travel at all is what the host already holds: the contract, which the two
//! sides agree on by version, and the sample dimensions, their shape, the coordinates and the
//! attributes, every one of which is what the host handed over in the batch.
//!
//! The arrays are moved and not copied: a column of the result is a `Vec<f64>` and
//! `PyArray1::from_vec` takes it by value, so the numbers of a large batch cross once.

use numpy::PyArray1;
use pkpdutils_abi::AnalysisResult;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// The result of the kernel as the dict `pkpdutils` reads back.
///
/// The module documentation describes the dict. The result is consumed, so that every column of
/// it becomes a numpy array without a second allocation.
///
/// # Errors
///
/// Whatever building the dicts of the sections raises, which is what python raises when it runs
/// out of memory and nothing else: every key is a string the kernel wrote and every value is an
/// array of it.
pub(crate) fn result_into_python(
    py: Python<'_>,
    result: AnalysisResult,
) -> PyResult<Bound<'_, PyDict>> {
    let parameters = PyDict::new(py);
    let units = PyDict::new(py);
    let point_variables = PyDict::new(py);
    let booleans = PyDict::new(py);
    let texts = PyDict::new(py);

    for column in result.parameters {
        units.set_item(&column.name, column.unit)?;
        parameters.set_item(column.name, PyArray1::from_vec(py, column.values))?;
    }
    for variable in result.point_variables {
        let entry = PyDict::new(py);
        entry.set_item("dim", variable.dim)?;
        entry.set_item("size", variable.size)?;
        entry.set_item("values", PyArray1::from_vec(py, variable.values))?;
        units.set_item(&variable.name, variable.unit)?;
        point_variables.set_item(variable.name, entry)?;
    }
    for column in result.boolean_variables {
        units.set_item(&column.name, column.unit)?;
        booleans.set_item(column.name, PyArray1::from_vec(py, column.values))?;
    }
    for column in result.text_variables {
        units.set_item(&column.name, column.unit)?;
        texts.set_item(column.name, PyList::new(py, column.values)?)?;
    }

    let dict = PyDict::new(py);
    dict.set_item("parameters", parameters)?;
    dict.set_item("units", units)?;
    dict.set_item("flags", PyArray1::from_vec(py, result.flags))?;
    dict.set_item("point_variables", point_variables)?;
    dict.set_item("booleans", booleans)?;
    dict.set_item("texts", texts)?;
    Ok(dict)
}

#[cfg(test)]
mod tests {
    use super::*;

    use numpy::PyReadonlyArray1;
    use pkpdutils_abi::{
        AnalysisResult, Attributes, BooleanColumn, NcaOptions, ParameterColumn, PointVariable,
        TextColumn,
    };

    /// The attributes of a result, which the dict does not carry: the host wrote them.
    fn attributes() -> Attributes {
        Attributes {
            substance: Some("caffeine".to_string()),
            route: None,
            tissue: Some("plasma".to_string()),
            options: NcaOptions::default(),
        }
    }

    /// A result of two samples with one variable of every section.
    fn result() -> AnalysisResult {
        AnalysisResult {
            parameters: vec![
                ParameterColumn {
                    name: "cmax".to_string(),
                    unit: "mg/l".to_string(),
                    values: vec![6.0, f64::NAN],
                },
                ParameterColumn {
                    name: "auc_last".to_string(),
                    unit: "hour*mg/l".to_string(),
                    values: vec![24.0, 12.0],
                },
            ],
            point_variables: vec![PointVariable {
                name: "candidate_r2_adj".to_string(),
                unit: "dimensionless".to_string(),
                dim: "candidate".to_string(),
                size: 2,
                values: vec![0.99, 0.98, 0.97, f64::NAN],
            }],
            boolean_variables: vec![BooleanColumn {
                name: "accepted".to_string(),
                unit: "dimensionless".to_string(),
                values: vec![true, false],
            }],
            text_variables: vec![TextColumn {
                name: "excluded_reason".to_string(),
                unit: "dimensionless".to_string(),
                values: vec![String::new(), "acceptance criteria".to_string()],
            }],
            flags: vec![0, 4],
            ..AnalysisResult::new(vec!["subject".to_string()], vec![2], attributes())
        }
    }

    /// One section of the dict, as a dict of its own.
    fn section<'py>(dict: &Bound<'py, PyDict>, key: &str) -> Bound<'py, PyDict> {
        dict.get_item(key)
            .unwrap()
            .unwrap_or_else(|| panic!("the result has no section `{key}`"))
            .cast_into::<PyDict>()
            .unwrap()
    }

    /// The numbers of a `float64` array of the dict `dict`, by its key.
    fn numbers(dict: &Bound<'_, PyDict>, key: &str) -> Vec<f64> {
        dict.get_item(key)
            .unwrap()
            .unwrap_or_else(|| panic!("`{key}` is not in the mapping"))
            .extract::<PyReadonlyArray1<'_, f64>>()
            .unwrap_or_else(|e| panic!("`{key}` is no `float64` array: {e}"))
            .as_slice()
            .unwrap()
            .to_vec()
    }

    #[test]
    fn every_section_of_a_result_travels() {
        Python::attach(|py| {
            let dict = result_into_python(py, result()).unwrap();

            let parameters = section(&dict, "parameters");
            assert_eq!(parameters.len(), 2);
            let cmax = numbers(&parameters, "cmax");
            assert_eq!(cmax[0], 6.0);
            assert!(cmax[1].is_nan(), "{cmax:?}");
            assert_eq!(numbers(&parameters, "auc_last"), [24.0, 12.0]);

            // one units mapping over every section, as `attrs["units"]` of python's result is
            // one mapping over every variable of it
            let units = section(&dict, "units");
            for (name, unit) in [
                ("cmax", "mg/l"),
                ("auc_last", "hour*mg/l"),
                ("candidate_r2_adj", "dimensionless"),
                ("accepted", "dimensionless"),
                ("excluded_reason", "dimensionless"),
            ] {
                let found: String = units.get_item(name).unwrap().unwrap().extract().unwrap();
                assert_eq!(found, unit, "the unit of {name}");
            }

            // the flags are counted bits, so they travel as the unsigned integers they are and
            // not as the doubles every parameter is
            let flags = dict
                .get_item("flags")
                .unwrap()
                .unwrap()
                .extract::<PyReadonlyArray1<'_, u32>>()
                .unwrap()
                .as_slice()
                .unwrap()
                .to_vec();
            assert_eq!(flags, [0, 4]);

            // a point variable keeps the name and the size of its extra dimension: its values
            // are `n_samples * size` long and cannot be reshaped without them
            let point = section(&dict, "point_variables");
            let candidate = point
                .get_item("candidate_r2_adj")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let dim: String = candidate
                .get_item("dim")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            let size: usize = candidate
                .get_item("size")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(dim, "candidate");
            assert_eq!(size, 2);
            let values = numbers(&candidate, "values");
            assert_eq!(values[..3], [0.99, 0.98, 0.97]);
            assert!(values[3].is_nan(), "{values:?}");

            let booleans = section(&dict, "booleans");
            let accepted = booleans
                .get_item("accepted")
                .unwrap()
                .unwrap()
                .extract::<PyReadonlyArray1<'_, bool>>()
                .unwrap_or_else(|e| panic!("`accepted` is no `bool` array: {e}"))
                .as_slice()
                .unwrap()
                .to_vec();
            assert_eq!(accepted, [true, false]);

            let texts = section(&dict, "texts");
            let reason: Vec<String> = texts
                .get_item("excluded_reason")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(reason, ["", "acceptance criteria"]);
        });
    }

    #[test]
    fn a_result_of_parameters_alone_carries_the_other_sections_empty() {
        // the single dose analysis of a batch of more than one row reports no point variable,
        // no exclusion and so no text, and the host indexes the sections rather than asking
        // whether they are there
        let bare = AnalysisResult {
            parameters: vec![ParameterColumn {
                name: "cmax".to_string(),
                unit: "mg/l".to_string(),
                values: vec![6.0, 4.0],
            }],
            ..AnalysisResult::new(vec!["subject".to_string()], vec![2], attributes())
        };

        Python::attach(|py| {
            let dict = result_into_python(py, bare).unwrap();
            assert_eq!(section(&dict, "parameters").len(), 1);
            assert_eq!(section(&dict, "units").len(), 1);
            assert_eq!(section(&dict, "point_variables").len(), 0);
            assert_eq!(section(&dict, "booleans").len(), 0);
            assert_eq!(section(&dict, "texts").len(), 0);
            assert_eq!(dict.len(), 6);
        });
    }
}
