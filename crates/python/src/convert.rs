//! The batch of timecourses, from python into the wire shape of the abi.
//!
//! The dict `pkpdutils` hands over carries one key per field of [`Batch`], spelled as the abi
//! spells it, with two deliberate exceptions: `contract` is not a key, since the binding speaks
//! one contract and writes it itself, and an enumeration arrives as its serde name (`"oral"`,
//! not `Route::Oral`), so that the python side and a JSON host agree on one spelling. A key
//! which the abi gives a serde default - every `Option`, and `coords` - may be absent or
//! `None`; the rest must be there.
//!
//! The arrays arrive as numpy buffers and are copied, never borrowed: a [`Batch`] owns its
//! numbers and outlives the call. Nothing here validates. [`Batch::validate`] is the most
//! expensive check of the kernel, it runs on every entry point of `pkpdutils_core` already, and
//! a second run would buy the same answer at twice the price.

use numpy::ndarray::Dimension;
use numpy::{PyReadonlyArray, PyReadonlyArray1, PyReadonlyArray2};
use pkpdutils_abi::{BATCH_CONTRACT, Batch, Coordinate, Route, SampleCoordinate};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A batch of timecourses, built from the dict `pkpdutils` hands over.
///
/// The module documentation describes the dict. The result is not validated here, so a caller
/// hands it to `pkpdutils_core`, which validates every entry point, and never reads a row of it
/// itself.
///
/// # Errors
///
/// A `KeyError` naming a required key the dict does not carry, a `TypeError` naming the key
/// whose value has the wrong type - a `float32` array where the batch is `float64`, a 1-D array
/// where it is 2-D - or a `ValueError` naming a route the contract does not spell.
#[allow(
    dead_code,
    reason = "the analysis entry point which calls it is the next slice of the binding; the \
              tests of this module are its only caller today"
)]
pub(crate) fn batch_from_python(py: Python<'_>, obj: &Bound<'_, PyDict>) -> PyResult<Batch> {
    Ok(Batch {
        contract: BATCH_CONTRACT.to_string(),
        sample_dims: field(py, obj, "sample_dims")?,
        sample_shape: field(py, obj, "sample_shape")?,
        n_time: field(py, obj, "n_time")?,
        times: points(py, obj, "times")?,
        values: points(py, obj, "values")?,
        sd: optional_points(py, obj, "sd")?,
        se: optional_points(py, obj, "se")?,
        n: optional_points(py, obj, "n")?,
        nominal_time: optional_points(py, obj, "nominal_time")?,
        lloq: optional_samples(py, obj, "lloq")?,
        max_doses: field(py, obj, "max_doses")?,
        dose_amount: points(py, obj, "dose_amount")?,
        dose_time: points(py, obj, "dose_time")?,
        dose_duration: points(py, obj, "dose_duration")?,
        time_unit: field(py, obj, "time_unit")?,
        unit: field(py, obj, "unit")?,
        dose_unit: optional_field(py, obj, "dose_unit")?,
        substance: optional_field(py, obj, "substance")?,
        substances: optional_field(py, obj, "substances")?,
        route: optional_route(py, obj, "route")?,
        routes: optional_routes(py, obj, "routes")?,
        tissue: optional_field(py, obj, "tissue")?,
        coords: coordinates(py, obj, "coords")?,
    })
}

/// The value of a key the dict must carry.
fn required<'py>(obj: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    obj.get_item(key)?.ok_or_else(|| {
        PyKeyError::new_err(format!(
            "{BATCH_CONTRACT} needs the key `{key}`, which is absent"
        ))
    })
}

/// The value of a key the dict may leave out, `None` for an absent key and for `None` itself.
fn optional<'py>(obj: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(obj.get_item(key)?.filter(|value| !value.is_none()))
}

/// Name the key in an error, keeping the exception type python would otherwise have raised.
///
/// A dict has no stack to point at: without the key, a wrong dtype reads as `type mismatch:
/// from=float32, to=float64` and leaves the caller to guess which of the ten arrays it was.
fn at_key(py: Python<'_>, key: &str, err: PyErr) -> PyErr {
    PyErr::from_type(
        err.get_type(py).clone(),
        format!("{key}: {}", err.value(py)),
    )
}

/// The value of a required key, extracted into `T` and named on a type mismatch.
fn field<'py, T>(py: Python<'_>, obj: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
{
    required(obj, key)?
        .extract::<T>()
        .map_err(|err| at_key(py, key, err))
}

/// The value of an optional key, extracted into `T` and named on a type mismatch.
fn optional_field<'py, T>(
    py: Python<'_>,
    obj: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<T>>
where
    T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
{
    optional(obj, key)?
        .map(|value| value.extract::<T>().map_err(|err| at_key(py, key, err)))
        .transpose()
}

/// The numbers of an array in C order, copied out of numpy.
///
/// A contiguous array in C order is copied from its buffer; anything else is read through its
/// strides, which is what an iterator of `ndarray` does, rather than being refused: `pkpdutils`
/// hands out a view as readily as a buffer (`values[:, ::2]` of a partial area, the transpose
/// of a frame) and a copy costs less than asking python to materialize one. Reading the buffer
/// is left to `ndarray`, whose `as_slice` is `Some` only for the standard layout: numpy calls
/// an array contiguous when it is contiguous in *either* order, so the buffer of a Fortran
/// ordered array would be the transpose of what the batch means.
fn c_order<D: Dimension>(array: &PyReadonlyArray<'_, f64, D>) -> Vec<f64> {
    let view = array.as_array();
    match view.as_slice() {
        Some(buffer) => buffer.to_vec(),
        None => view.iter().copied().collect(),
    }
}

/// A required `(n_samples, per_sample)` array, flattened row major.
fn points(py: Python<'_>, obj: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<f64>> {
    let value = required(obj, key)?;
    let array: PyReadonlyArray2<'_, f64> = value.extract().map_err(|err| at_key(py, key, err))?;
    Ok(c_order(&array))
}

/// An optional `(n_samples, per_sample)` array, flattened row major.
fn optional_points(
    py: Python<'_>,
    obj: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<Vec<f64>>> {
    let Some(value) = optional(obj, key)? else {
        return Ok(None);
    };
    let array: PyReadonlyArray2<'_, f64> = value.extract().map_err(|err| at_key(py, key, err))?;
    Ok(Some(c_order(&array)))
}

/// An optional array of one number per sample.
fn optional_samples(
    py: Python<'_>,
    obj: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<Vec<f64>>> {
    let Some(value) = optional(obj, key)? else {
        return Ok(None);
    };
    let array: PyReadonlyArray1<'_, f64> = value.extract().map_err(|err| at_key(py, key, err))?;
    Ok(Some(c_order(&array)))
}

/// The name `batch/1` spells a route with, the serde name of [`Route`].
///
/// The match is exhaustive, so a route the abi gains fails to compile here instead of being
/// refused at run time as a name this binding does not know.
fn route_name(route: Route) -> &'static str {
    match route {
        Route::IvBolus => "iv_bolus",
        Route::IvInfusion => "iv_infusion",
        Route::Oral => "oral",
    }
}

/// The route of its name in `batch/1`.
fn route_named(key: &str, name: &str) -> PyResult<Route> {
    Route::ALL
        .iter()
        .copied()
        .find(|route| route_name(*route) == name)
        .ok_or_else(|| {
            PyValueError::new_err(format!("{key}: `{name}` is no route of {BATCH_CONTRACT}"))
        })
}

/// The route of the whole batch, `None` when the batch states it per sample or not at all.
fn optional_route(py: Python<'_>, obj: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Route>> {
    let Some(name) = optional_field::<String>(py, obj, key)? else {
        return Ok(None);
    };
    route_named(key, &name).map(Some)
}

/// The route of every sample, `None` when the batch has one route or none.
fn optional_routes(
    py: Python<'_>,
    obj: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<Vec<Route>>> {
    let Some(names) = optional_field::<Vec<String>>(py, obj, key)? else {
        return Ok(None);
    };
    names
        .iter()
        .map(|name| route_named(key, name))
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

/// The non-dimension coordinates, empty when the batch has none.
fn coordinates(
    py: Python<'_>,
    obj: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Vec<SampleCoordinate>> {
    let Some(value) = optional(obj, key)? else {
        return Ok(Vec::new());
    };
    let items = value.try_iter().map_err(|err| at_key(py, key, err))?;
    let mut coords = Vec::new();
    for (index, item) in items.enumerate() {
        let item = item?;
        let at = format!("{key}[{index}]");
        coords.push(coordinate(py, &item).map_err(|err| at_key(py, &at, err))?);
    }
    Ok(coords)
}

/// One sample coordinate: its name, the sample dimensions it lives on and its values.
fn coordinate(py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<SampleCoordinate> {
    let dict = item.cast::<PyDict>()?;
    Ok(SampleCoordinate {
        name: field(py, dict, "name")?,
        dims: field(py, dict, "dims")?,
        values: coordinate_values(py, dict, "values")?,
    })
}

/// The values of a coordinate, tagged with their dtype.
///
/// The tag is a key of a dict of one entry, exactly as serde writes [`Coordinate`]:
/// `{"str": ["A", "B"]}`. It travels because the dtype has to - `nca` carries a coordinate into
/// its result and `stats/bioequivalence.py` reads it back, so a subject labelled `"1"` must not
/// come back as the number 1 - and guessing it from the python values cannot be made to work:
/// a `bool` is an `int` in python and a column of whole numbers is not therefore an integer
/// column.
fn coordinate_values(py: Python<'_>, obj: &Bound<'_, PyDict>, key: &str) -> PyResult<Coordinate> {
    let values = required(obj, key)?;
    let dict = values
        .cast::<PyDict>()
        .map_err(|err| at_key(py, key, err.into()))?;
    if let Some(tagged) = dict.get_item("str")? {
        Ok(Coordinate::Str(
            tagged.extract().map_err(|e| at_key(py, key, e))?,
        ))
    } else if let Some(tagged) = dict.get_item("int")? {
        Ok(Coordinate::Int(
            tagged.extract().map_err(|e| at_key(py, key, e))?,
        ))
    } else if let Some(tagged) = dict.get_item("float")? {
        Ok(Coordinate::Float(
            tagged.extract().map_err(|e| at_key(py, key, e))?,
        ))
    } else if let Some(tagged) = dict.get_item("bool")? {
        Ok(Coordinate::Bool(
            tagged.extract().map_err(|e| at_key(py, key, e))?,
        ))
    } else {
        Err(PyKeyError::new_err(format!(
            "{key}: the values of a coordinate carry their dtype as their key, \
             one of `str`, `int`, `float` or `bool`"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArray2};
    use pyo3::types::PyList;

    /// One coordinate over `subject`, its values under the dtype tag `tag`.
    fn coord_dict<'py, V>(py: Python<'py>, name: &str, tag: &str, values: V) -> Bound<'py, PyDict>
    where
        V: IntoPyObject<'py>,
    {
        let tagged = PyDict::new(py);
        tagged.set_item(tag, values).unwrap();
        let coord = PyDict::new(py);
        coord.set_item("name", name).unwrap();
        coord.set_item("dims", ["subject"]).unwrap();
        coord.set_item("values", tagged).unwrap();
        coord
    }

    /// A `(2, 3)` array of the numbers given row by row.
    fn points_array<'py>(py: Python<'py>, first: [f64; 3], second: [f64; 3]) -> Bound<'py, PyAny> {
        PyArray2::from_vec2(py, &[first.to_vec(), second.to_vec()])
            .unwrap()
            .into_any()
    }

    /// A dict of two subjects of three points and one oral dose each, every required key filled.
    fn batch_dict(py: Python<'_>) -> Bound<'_, PyDict> {
        let coord = coord_dict(py, "subject", "str", ["A", "B"]);

        let dict = PyDict::new(py);
        dict.set_item("sample_dims", ["subject"]).unwrap();
        dict.set_item("sample_shape", [2usize]).unwrap();
        dict.set_item("n_time", 3usize).unwrap();
        dict.set_item("times", points_array(py, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))
            .unwrap();
        dict.set_item(
            "values",
            points_array(py, [10.0, 5.0, 2.0], [12.0, 6.0, f64::NAN]),
        )
        .unwrap();
        dict.set_item("sd", py.None()).unwrap();
        dict.set_item("lloq", PyArray1::from_vec(py, vec![0.1, f64::NAN]))
            .unwrap();
        dict.set_item("max_doses", 1usize).unwrap();
        dict.set_item(
            "dose_amount",
            PyArray2::from_vec2(py, &[vec![100.0], vec![100.0]]).unwrap(),
        )
        .unwrap();
        dict.set_item(
            "dose_time",
            PyArray2::from_vec2(py, &[vec![0.0], vec![0.0]]).unwrap(),
        )
        .unwrap();
        dict.set_item(
            "dose_duration",
            PyArray2::from_vec2(py, &[vec![f64::NAN], vec![f64::NAN]]).unwrap(),
        )
        .unwrap();
        dict.set_item("time_unit", "h").unwrap();
        dict.set_item("unit", "mg/l").unwrap();
        dict.set_item("dose_unit", "mg").unwrap();
        dict.set_item("substance", "caffeine").unwrap();
        dict.set_item("route", "oral").unwrap();
        dict.set_item("tissue", "plasma").unwrap();
        dict.set_item("coords", PyList::new(py, [&coord]).unwrap())
            .unwrap();
        dict
    }

    #[test]
    fn a_batch_dict_becomes_a_batch() {
        let batch = Python::attach(|py| {
            let dict = batch_dict(py);
            batch_from_python(py, &dict).unwrap()
        });

        assert_eq!(batch.contract, BATCH_CONTRACT);
        assert_eq!(batch.sample_dims, ["subject"]);
        assert_eq!(batch.sample_shape, [2]);
        assert_eq!(batch.n_time, 3);
        assert_eq!(batch.times, [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        assert_eq!(batch.values[..5], [10.0, 5.0, 2.0, 12.0, 6.0]);
        assert!(batch.values[5].is_nan());
        assert_eq!(batch.sd, None);
        assert_eq!(batch.se, None);
        let lloq = batch.lloq.as_ref().unwrap();
        assert_eq!(lloq[0], 0.1);
        assert!(lloq[1].is_nan());
        assert_eq!(batch.max_doses, 1);
        assert_eq!(batch.dose_amount, [100.0, 100.0]);
        assert_eq!(batch.dose_time, [0.0, 0.0]);
        assert!(batch.dose_duration.iter().all(|d| d.is_nan()));
        assert_eq!(batch.time_unit, "h");
        assert_eq!(batch.unit, "mg/l");
        assert_eq!(batch.dose_unit.as_deref(), Some("mg"));
        assert_eq!(batch.substance.as_deref(), Some("caffeine"));
        assert_eq!(batch.route, Some(Route::Oral));
        assert_eq!(batch.routes, None);
        assert_eq!(batch.tissue.as_deref(), Some("plasma"));
        assert_eq!(batch.coords.len(), 1);
        assert_eq!(batch.coords[0].name, "subject");
        assert_eq!(batch.coords[0].dims, ["subject"]);
        assert_eq!(
            batch.coords[0].values,
            Coordinate::Str(vec!["A".to_string(), "B".to_string()])
        );
        // the conversion does not validate, so the test does: a dict which converts has to be a
        // batch the kernel accepts, or the conversion built a shape nothing can read
        assert_eq!(batch.validate(), Ok(()));
    }

    #[test]
    fn the_optional_arrays_and_the_per_sample_columns_are_read() {
        let batch = Python::attach(|py| {
            let dict = batch_dict(py);
            dict.set_item("sd", points_array(py, [1.0, 0.5, 0.2], [1.2, 0.6, 0.3]))
                .unwrap();
            dict.set_item("se", points_array(py, [0.5, 0.25, 0.1], [0.6, 0.3, 0.15]))
                .unwrap();
            dict.set_item("n", points_array(py, [4.0, 4.0, 4.0], [4.0, 4.0, 3.0]))
                .unwrap();
            dict.set_item(
                "nominal_time",
                points_array(py, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
            )
            .unwrap();
            // a mixed batch names its substance and its route per sample instead of once
            dict.set_item("substance", py.None()).unwrap();
            dict.set_item("route", py.None()).unwrap();
            dict.set_item("substances", ["caffeine", "paraxanthine"])
                .unwrap();
            dict.set_item("routes", ["oral", "iv_infusion"]).unwrap();
            batch_from_python(py, &dict).unwrap()
        });

        assert_eq!(
            batch.sd.as_deref(),
            Some([1.0, 0.5, 0.2, 1.2, 0.6, 0.3].as_slice())
        );
        assert_eq!(
            batch.se.as_deref(),
            Some([0.5, 0.25, 0.1, 0.6, 0.3, 0.15].as_slice())
        );
        assert_eq!(
            batch.n.as_deref(),
            Some([4.0, 4.0, 4.0, 4.0, 4.0, 3.0].as_slice())
        );
        assert_eq!(
            batch.nominal_time.as_deref(),
            Some([0.0, 1.0, 2.0, 0.0, 1.0, 2.0].as_slice())
        );
        assert_eq!(batch.substance, None);
        assert_eq!(
            batch.substances,
            Some(vec!["caffeine".to_string(), "paraxanthine".to_string()])
        );
        assert_eq!(batch.route, None);
        assert_eq!(batch.routes, Some(vec![Route::Oral, Route::IvInfusion]));
        assert_eq!(batch.validate(), Ok(()));
    }

    #[test]
    fn every_coordinate_tag_keeps_its_dtype() {
        let batch = Python::attach(|py| {
            let dict = batch_dict(py);
            let coords = PyList::new(
                py,
                [
                    coord_dict(py, "subject", "str", ["A", "B"]),
                    coord_dict(py, "period", "int", [1i64, 2]),
                    coord_dict(py, "weight", "float", [70.0, 80.5]),
                    coord_dict(py, "fasted", "bool", [true, false]),
                ],
            )
            .unwrap();
            dict.set_item("coords", coords).unwrap();
            batch_from_python(py, &dict).unwrap()
        });

        let names: Vec<&str> = batch.coords.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["subject", "period", "weight", "fasted"]);
        assert_eq!(
            batch.coords[0].values,
            Coordinate::Str(vec!["A".to_string(), "B".to_string()])
        );
        assert_eq!(batch.coords[1].values, Coordinate::Int(vec![1, 2]));
        assert_eq!(batch.coords[2].values, Coordinate::Float(vec![70.0, 80.5]));
        assert_eq!(batch.coords[3].values, Coordinate::Bool(vec![true, false]));
        assert_eq!(batch.validate(), Ok(()));
    }

    #[test]
    fn an_unknown_route_is_refused_by_name() {
        let message = Python::attach(|py| {
            let dict = batch_dict(py);
            dict.set_item("route", "intramuscular").unwrap();
            let err = batch_from_python(py, &dict).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            err.to_string()
        });

        assert!(message.contains("route"), "{message}");
        assert!(message.contains("intramuscular"), "{message}");
    }

    #[test]
    fn a_coordinate_without_a_dtype_tag_is_refused() {
        let message = Python::attach(|py| {
            let dict = batch_dict(py);
            // `text` is no tag of `batch/1`, so the dtype of the values is unstated
            let coord = coord_dict(py, "subject", "text", ["A", "B"]);
            dict.set_item("coords", PyList::new(py, [coord]).unwrap())
                .unwrap();
            let err = batch_from_python(py, &dict).unwrap_err();
            assert!(err.is_instance_of::<PyKeyError>(py), "{err}");
            err.to_string()
        });

        assert!(message.contains("coords[0]"), "{message}");
        assert!(message.contains("values"), "{message}");
        assert!(message.contains("`str`"), "{message}");
    }

    #[test]
    fn a_float32_values_array_is_refused() {
        let message = Python::attach(|py| {
            let dict = batch_dict(py);
            dict.set_item("values", PyArray2::<f32>::zeros(py, [2, 3], false))
                .unwrap();
            batch_from_python(py, &dict).unwrap_err().to_string()
        });

        assert!(message.contains("values"), "{message}");
        assert!(message.contains("float32"), "{message}");
        assert!(message.contains("float64"), "{message}");
    }

    #[test]
    fn a_non_contiguous_values_array_is_accepted_by_value() {
        let batch = Python::attach(|py| {
            let dict = batch_dict(py);
            let wide = PyArray2::from_vec2(
                py,
                &[
                    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                ],
            )
            .unwrap();
            let locals = PyDict::new(py);
            locals.set_item("wide", &wide).unwrap();
            let strided = py.eval(c"wide[:, ::2]", None, Some(&locals)).unwrap();
            dict.set_item("values", strided).unwrap();
            batch_from_python(py, &dict).unwrap()
        });

        // element by element, so that a stride read as a contiguous buffer cannot pass
        let expected = [0.0, 2.0, 4.0, 10.0, 12.0, 14.0];
        assert_eq!(batch.values.len(), expected.len());
        for (i, (found, want)) in batch.values.iter().zip(expected).enumerate() {
            assert_eq!(*found, want, "values[{i}]");
        }
    }

    #[test]
    fn a_fortran_ordered_values_array_keeps_the_order_of_its_rows() {
        let batch = Python::attach(|py| {
            let dict = batch_dict(py);
            let locals = PyDict::new(py);
            locals.set_item("np", py.import("numpy").unwrap()).unwrap();
            locals
                .set_item(
                    "rows",
                    PyArray2::from_vec2(py, &[vec![0.0, 1.0, 2.0], vec![10.0, 11.0, 12.0]])
                        .unwrap(),
                )
                .unwrap();
            // numpy calls this array contiguous although its buffer is the transpose of it,
            // so a conversion which trusts that flag reads the columns as the rows
            let fortran = py
                .eval(c"np.asfortranarray(rows)", None, Some(&locals))
                .unwrap();
            dict.set_item("values", fortran).unwrap();
            batch_from_python(py, &dict).unwrap()
        });

        let expected = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
        assert_eq!(batch.values.len(), expected.len());
        for (i, (found, want)) in batch.values.iter().zip(expected).enumerate() {
            assert_eq!(*found, want, "values[{i}]");
        }
    }

    #[test]
    fn a_missing_required_key_is_refused() {
        let message = Python::attach(|py| {
            let dict = batch_dict(py);
            dict.del_item("times").unwrap();
            batch_from_python(py, &dict).unwrap_err().to_string()
        });

        assert!(message.contains("times"), "{message}");
    }
}
