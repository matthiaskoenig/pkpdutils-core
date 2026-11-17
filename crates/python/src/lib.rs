//! The python binding of the pkpdutils kernel.
//!
//! This is the only crate of the workspace which knows that python exists. It takes what
//! `pkpdutils` already holds in memory - the numpy buffers of a `Timecourses` batch and the
//! fields of an `NCAOptions` - turns them into the wire shapes of [`pkpdutils_abi`] and hands
//! them to the kernel. Nothing travels as JSON: the layouts of the two sides already agree, so
//! a conversion here is a copy of a buffer and not a parse.
//!
//! The crate binds pyo3 against the stable ABI from python 3.13 on (`abi3-py313`), which is the
//! floor of `pkpdutils` itself, so that one wheel per platform serves every python the package
//! supports. It cannot be built for `wasm32`, which is why continuous integration excludes it
//! from the wasm gates by name; the kernel it binds is checked there as before.

/// The allocator of the extension module.
///
/// A batch of many rows and few points spends about a tenth of its time in the allocator, which
/// is visible from outside the process: turning off glibc's thread cache
/// (`GLIBC_TUNABLES=glibc.malloc.tcache_count=0`) costs eleven percent of the run. mimalloc takes
/// about twenty two percent of the CPU time of such an analysis off, measured through
/// [`analyze_batch`] on 100 000 rows by 12 points, and costs about a third more peak resident set
/// size on that shape; a batch of few rows and many points is unaffected either way. Unlike the
/// instruction set the wheel is compiled for, this is a choice a wheel is allowed to make: it
/// travels inside the artifact and depends on nothing about the machine that installs it.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod convert;
mod options;
mod result;

use std::panic::{AssertUnwindSafe, catch_unwind};

// `::` is not decoration: `#[pymodule]` expands to a module of the same name as the crate, which
// an unqualified path would resolve to instead
use ::pkpdutils_core as kernel;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::{at_key, batch_from_python, field, required};
use crate::options::{options_from_python, unsupported_reason};
use crate::result::result_into_python;

create_exception!(
    pkpdutils_core,
    UnsupportedError,
    PyValueError,
    "An analysis which is well formed and which this build does not carry.\n\n\
     This is the one failure of [`analyze_batch`] a host answers by analysing the batch some \
     other way, and it is a type and not a sentence so that the rule can be written as one \
     line which cannot be got wrong: catch `UnsupportedError` and fall back, let everything \
     else through. The failures it must not be confused with - an options dict carrying a key \
     or an enumeration value this build does not spell, a malformed batch the kernel refuses - \
     mean that the caller and this build disagree about the contract or that the data is \
     malformed, and both of those are defects to fix rather than work to route elsewhere. \
     Several of them are a plain `ValueError`, so a host which caught `ValueError` would \
     swallow exactly the disagreement this design exists to make loud.\n\n\
     A unit string of the batch which the curated registry of this build cannot read (a time in \
     `week`, a concentration in `U/l`) is one of these refusals and not one of those defects: \
     what this build reads is a property of this build, and a host whose own engine reads the \
     string analyses the batch there. It is the one refusal `is_supported` does not see, since \
     answering it means deriving the units of the result.\n\n\
     It subclasses `ValueError`, which is what this refusal was before it had a type of its \
     own, so a caller which catches `ValueError` keeps catching it.\n\n\
     `is_supported` asks the same question without raising, and a host asks it first."
);

/// Analyse a batch of timecourses and give back the parameters of every row.
///
/// `batch` and `options` are the dicts `crate::convert` and `crate::options` describe, and
/// `n_workers` is the shard plan of the row loop: `None` lets the kernel plan it for the shape
/// of the batch, `1` runs the rows in the calling thread and any other number asks for that
/// many shards. The numbers are the same for every plan, down to the bits, and the plan is the
/// most stretches the run may be cut into rather than the fewest. A build of `pkpdutils-core`
/// without its `parallel` feature, which is the default, runs every plan in the calling thread.
///
/// The result is the dict `crate::result` describes, whose arrays are one dimensional of length
/// `prod(sample_shape)` in the row order of the batch.
///
/// The analysis runs without the interpreter: the kernel reads the copies the conversion made
/// and touches no python object, so the threads of the caller run while it does.
///
/// # Errors
///
/// Three outcomes which are not one another, which a caller must not read as one, and which are
/// told apart by their type and not by their text:
///
/// - an [`UnsupportedError`], and nothing else is one. The batch and the options are well
///   formed and this build does not carry the analysis they ask for. This is the one a host
///   answers by taking its own path; [`is_supported`] is the same question without an
///   exception, and a host asks it first. One of these refusals is one the predicate cannot
///   answer: a unit string the curated registry of the kernel cannot read (a time in `week`, a
///   concentration in `U/l`, pint's `mg l`, a spelling pint over-forgives such as `mg//l`). A
///   host which routes on the type, which is what this one is for, sees no difference; a host
///   which took the predicate as the whole answer would raise at its user.
/// - a `KeyError`, a `TypeError` or a `ValueError` naming a key of the batch or of the options.
///   The caller and this build disagree about the contract, which is a defect of one of the
///   two and never a reason to analyse the batch some other way. A field or an enumeration
///   value `NCAOptions` gains without a release of this crate lands here, loudly, in the test
///   suite of the host.
/// - a `ValueError` carrying what the kernel refused the batch with: a malformed batch, doses
///   without a dose unit, options which contradict each other.
///
/// A `RuntimeError` is the fourth and is a bug: see [`guarded`] and [`result_into_python`].
#[pyfunction]
#[pyo3(signature = (batch, options, n_workers = None))]
fn analyze_batch<'py>(
    py: Python<'py>,
    batch: &Bound<'py, PyDict>,
    options: &Bound<'py, PyDict>,
    n_workers: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let batch = batch_from_python(py, batch)?;
    let options = options_from_python(options)?;
    if let Some(what) = unsupported_reason(&batch, &options) {
        return Err(unsupported(&kernel::KernelError::Unsupported { what }));
    }

    let analysed = py.detach(|| {
        guarded(|| match n_workers {
            None => kernel::analyze_batch(&batch, &options),
            Some(shards) => kernel::analyze_batch_in_shards(&batch, &options, shards),
        })
    })?;
    let result = analysed.map_err(|error| match error {
        // the predicate answers for every refusal of the kernel, which
        // `the_predicate_agrees_with_the_kernel` of `crate::options` holds it to, so this arm
        // is reached only if the two have drifted apart. It is the type of the refusal which a
        // host routes on, so a refusal which got past the predicate must still carry it
        kernel::KernelError::Unsupported { .. } => unsupported(&error),
        _ => PyValueError::new_err(error.to_string()),
    })?;
    result_into_python(py, result)
}

/// The summary statistics of raw parameter values, `ParameterResult.summarize` of the host.
///
/// `request` is a dict of the fields of the kernel's `SummaryRequest`: `values` (a float array
/// of any shape, read in C order, `NaN` or infinite for a missing value), `n_vars`, `n_groups`,
/// `n_samples`, `ci_level` and the per-variable lists `lognormal` and `discrete`. The answer is
/// a dict of the fourteen statistics `mean`, `sd`, `se`, `cv`, `ci_low`, `ci_high`, `median`,
/// `q25`, `q75`, `min`, `max`, `count`, `geomean` and `geocv`, each a float array of shape
/// `(n_vars, n_groups)`. The kernel computes every statistic of every variable and the host
/// keeps the ones the classification of the parameter wants.
///
/// # Errors
///
/// A `KeyError` for a missing field and a `TypeError` for a field of the wrong type. A request
/// whose lengths disagree with its dimensions or whose `ci_level` is not strictly between 0 and
/// 1 is a plain `ValueError` and never an `UnsupportedError`: a host answers a refusal by taking
/// its own path, and a request which is wrong must reach the caller instead of being answered
/// by a second implementation.
#[pyfunction]
fn summarize<'py>(py: Python<'py>, request: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let values: PyReadonlyArrayDyn<'_, f64> = required(request, "values")?
        .extract()
        .map_err(|err| at_key(py, "values", err))?;
    let request = kernel::pkpdutils_abi::SummaryRequest {
        values: values.as_array().iter().copied().collect(),
        n_vars: field(py, request, "n_vars")?,
        n_groups: field(py, request, "n_groups")?,
        n_samples: field(py, request, "n_samples")?,
        ci_level: field(py, request, "ci_level")?,
        lognormal: field(py, request, "lognormal")?,
        discrete: field(py, request, "discrete")?,
    };
    let (n_vars, n_groups) = (request.n_vars, request.n_groups);
    let answer = py
        .detach(|| guarded(|| kernel::summarize(&request)))?
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let out = PyDict::new(py);
    for (name, column) in [
        ("mean", answer.mean),
        ("sd", answer.sd),
        ("se", answer.se),
        ("cv", answer.cv),
        ("ci_low", answer.ci_low),
        ("ci_high", answer.ci_high),
        ("median", answer.median),
        ("q25", answer.q25),
        ("q75", answer.q75),
        ("min", answer.min),
        ("max", answer.max),
        ("count", answer.count),
        ("geomean", answer.geomean),
        ("geocv", answer.geocv),
    ] {
        out.set_item(
            name,
            PyArray1::from_vec(py, column).reshape([n_vars, n_groups])?,
        )?;
    }
    Ok(out)
}

/// The time to steady state of the troughs of the dosing intervals, `time_to_steady_state` of
/// the host.
///
/// `request` is a dict of the fields of the kernel's `TssRequest`: `time`, `value`, `start`
/// (float arrays of shape `(n_rows, n_cols)`, read in C order) and `valid` (a bool array of the
/// same shape), `n_rows`, `n_cols`, `method` (`"monoexponential"` or `"stepwise"`), `fraction`
/// and `alpha`. The answer is a dict of `tss`, a float array of `n_rows`, and `c_ss`, the
/// plateau per row for the monoexponential estimate and an empty array for the stepwise one.
///
/// # Errors
///
/// A `KeyError` for a missing field and a `TypeError` for a field of the wrong type. A request
/// whose arrays disagree with its shape, whose troughs are not packed to the front of a row or
/// whose `fraction` or `alpha` is not strictly between 0 and 1 is a plain `ValueError` and never
/// an `UnsupportedError`: a host answers a refusal by taking its own path, and a request which
/// is wrong must reach the caller instead of being answered by a second implementation.
#[pyfunction]
fn time_to_steady_state<'py>(
    py: Python<'py>,
    request: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let floats = |key: &str| -> PyResult<Vec<f64>> {
        let array: PyReadonlyArrayDyn<'_, f64> = required(request, key)?
            .extract()
            .map_err(|err| at_key(py, key, err))?;
        Ok(array.as_array().iter().copied().collect())
    };
    let valid: PyReadonlyArrayDyn<'_, bool> = required(request, "valid")?
        .extract()
        .map_err(|err| at_key(py, "valid", err))?;
    let method: String = field(py, request, "method")?;
    let method = match method.as_str() {
        "monoexponential" => kernel::pkpdutils_abi::TssMethod::Monoexponential,
        "stepwise" => kernel::pkpdutils_abi::TssMethod::Stepwise,
        other => {
            return Err(PyValueError::new_err(format!(
                "method {other:?} is neither \"monoexponential\" nor \"stepwise\""
            )));
        }
    };
    let request = kernel::pkpdutils_abi::TssRequest {
        time: floats("time")?,
        value: floats("value")?,
        start: floats("start")?,
        valid: valid.as_array().iter().copied().collect(),
        n_rows: field(py, request, "n_rows")?,
        n_cols: field(py, request, "n_cols")?,
        method,
        fraction: field(py, request, "fraction")?,
        alpha: field(py, request, "alpha")?,
    };
    let answer = py
        .detach(|| guarded(|| kernel::time_to_steady_state(&request)))?
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let out = PyDict::new(py);
    out.set_item("tss", PyArray1::from_vec(py, answer.tss))?;
    out.set_item("c_ss", PyArray1::from_vec(py, answer.c_ss))?;
    Ok(out)
}

/// The exception of an analysis this build does not carry.
///
/// The sentence is the kernel's own error, so that the text of a refusal the predicate gives
/// and the text of the same refusal from the kernel cannot drift apart; what a host routes on
/// is the type, and the text is for the person reading the traceback.
fn unsupported(refusal: &kernel::KernelError) -> PyErr {
    UnsupportedError::new_err(format!(
        "pkpdutils-core does not carry this analysis: {refusal}. \
         `is_supported` asks the same question without raising."
    ))
}

/// Why `pkpdutils-core` would refuse this batch, `None` when it would analyse it.
///
/// A host asks this before it calls [`analyze_batch`], so that the work this build does not
/// carry goes down its own path instead of reaching the user as an exception. The text is the
/// kernel's own, bare: it names the analysis which is missing and not this build, since the
/// host writes it into a line of its own and [`analyze_batch`] is what raises.
///
/// `None` is "nothing here says this build would refuse it" and not a promise that it will not:
/// one refusal of the kernel is invisible here, a unit string of the batch which its curated
/// registry cannot read (a time in `week`, a concentration in `U/l`). Answering that one means
/// deriving the unit of every parameter of the result, which is the analysis itself, so it is
/// [`analyze_batch`] which reports it - as an [`UnsupportedError`], like every other refusal, so
/// a host which routes on the type is right either way. Mirroring the registry here, or in the
/// host, would be a second copy of a curated list which drifts from the first in silence, and
/// that is worse than the one call which answers it exactly.
///
/// # Errors
///
/// Whatever the conversion of the batch or of the options raises, which is a disagreement about
/// the contract and not an answer to this question: a batch this build cannot read is not a
/// batch it does not support.
#[pyfunction]
fn is_supported(
    py: Python<'_>,
    batch: &Bound<'_, PyDict>,
    options: &Bound<'_, PyDict>,
) -> PyResult<Option<String>> {
    let batch = batch_from_python(py, batch)?;
    let options = options_from_python(options)?;
    Ok(unsupported_reason(&batch, &options))
}

/// Apply `f` to a float or to every element of a 1-d float array, giving back the same shape.
///
/// A python float returns a python float and an array returns a new array, so the three
/// quantile functions serve a scalar call and a vectorised one with the same signature.
///
/// # Errors
///
/// A `TypeError` when the argument is neither a float nor a 1-d array of floats.
fn map_scalar_or_array<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    f: impl Fn(f64) -> f64,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(value) = x.extract::<f64>() {
        let out = guarded(|| f(value))?;
        return Ok(out.into_pyobject(py)?.into_any());
    }
    let array = x
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|_| PyTypeError::new_err("expected a float or a 1-d numpy array of floats"))?;
    // a strided view (a column of a matrix, a reversed array) is read as it is, not copied first
    let view = array.as_array();
    let out = guarded(|| view.iter().map(|&v| f(v)).collect::<Vec<f64>>())?;
    Ok(PyArray1::from_vec(py, out).into_any())
}

/// The standard normal quantile of the upper tail: `x` with `P(Z > x) = p`.
///
/// Accurate to a few ulp over `[1e-300, 1 - 1e-16]`; `0` gives `+inf`, `1` gives `-inf` and a
/// probability outside `[0, 1]` gives `NaN`.
///
/// # Errors
///
/// A `TypeError` when `p` is neither a float nor a 1-d float array.
#[pyfunction]
fn normal_isf<'py>(py: Python<'py>, p: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    map_scalar_or_array(py, p, kernel::normal_isf)
}

/// The Student t quantile of the upper tail with `df` degrees of freedom (real, at least 1,
/// `NaN` below).
///
/// Accurate to a relative `1e-12` for `p` in `[1e-6, 1 - 1e-6]` and `df` in `[1, 1e4]`, `1e-9`
/// down to `1e-300`, strictly monotone below; above `1e12` degrees of freedom it is the normal
/// quantile.
///
/// # Errors
///
/// A `TypeError` when `p` is neither a float nor a 1-d float array.
#[pyfunction]
fn student_t_isf<'py>(
    py: Python<'py>,
    p: &Bound<'py, PyAny>,
    df: f64,
) -> PyResult<Bound<'py, PyAny>> {
    map_scalar_or_array(py, p, move |p| kernel::student_t_isf(p, df))
}

/// The survival function `P(T > t)` of the Student t distribution with `df` degrees of freedom
/// (real, at least 1, `NaN` below), accurate to a relative `1e-12` for `df` in `[1, 1e4]`.
///
/// # Errors
///
/// A `TypeError` when `t` is neither a float nor a 1-d float array.
#[pyfunction]
fn student_t_sf<'py>(
    py: Python<'py>,
    t: &Bound<'py, PyAny>,
    df: f64,
) -> PyResult<Bound<'py, PyAny>> {
    map_scalar_or_array(py, t, move |t| kernel::student_t_sf(t, df))
}

/// Run `f` with the unwind boundary of the extension around it.
///
/// A panic which crosses into pyo3 aborts the process, so the user of a python package would
/// see their interpreter die without a traceback. The kernel is written so that no input
/// panics - it indexes nothing with a number of the caller's and unwraps nothing the caller
/// controls - and its own review found no input which does across the hostile set it was
/// measured on. This is therefore the boundary of something which is not supposed to happen,
/// and the test which reaches it calls this function with a closure which panics, since no
/// batch can.
///
/// The closure is asserted to be unwind safe: a caught panic here ends the call, so nothing
/// which could have been left half written is read afterwards.
///
/// # Errors
///
/// A `RuntimeError` naming this crate, the panic and where to report it.
fn guarded<T>(f: impl FnOnce() -> T) -> PyResult<T> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let what = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "a payload which is no message".to_string());
        PyRuntimeError::new_err(format!(
            "pkpdutils-core panicked: {what}. This is a bug of the kernel and not of the batch \
             it was given; please report it at {}",
            env!("CARGO_PKG_REPOSITORY")
        ))
    })
}

/// The module `pkpdutils` imports: the two entry points, the refusal a host routes on and the
/// version of this build.
///
/// # Errors
///
/// Whatever adding an attribute to a fresh module raises, which is what python raises when it
/// runs out of memory and nothing else.
#[pymodule]
fn pkpdutils_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // by name, so that a host writes `except pkpdutils_core.UnsupportedError` rather than
    // matching on the text of a message
    module.add(
        "UnsupportedError",
        module.py().get_type::<UnsupportedError>(),
    )?;
    module.add_function(wrap_pyfunction!(analyze_batch, module)?)?;
    module.add_function(wrap_pyfunction!(is_supported, module)?)?;
    module.add_function(wrap_pyfunction!(summarize, module)?)?;
    module.add_function(wrap_pyfunction!(time_to_steady_state, module)?)?;
    module.add_function(wrap_pyfunction!(normal_isf, module)?)?;
    module.add_function(wrap_pyfunction!(student_t_isf, module)?)?;
    module.add_function(wrap_pyfunction!(student_t_sf, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Instant;

    use numpy::{PyArray2, PyUntypedArrayMethods};
    use pyo3::exceptions::{PyKeyError, PyRuntimeError};
    use pyo3::types::PyType;

    use super::*;

    /// The sample times of every curve of the test batch.
    const TIMES: [f64; 8] = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    /// The curve of the first sample; every further sample is a multiple of it.
    const VALUES: [f64; 8] = [0.0, 4.0, 6.0, 5.0, 3.0, 1.2, 0.5, 0.04];

    /// A `(rows.len(), rows[0].len())` array of the rows given.
    fn array<'py>(py: Python<'py>, rows: &[Vec<f64>]) -> Bound<'py, PyAny> {
        PyArray2::from_vec2(py, rows).unwrap().into_any()
    }

    /// `n` oral subjects of one 100 mg dose, the `i`-th curve `i + 1` times the first.
    fn batch_dict(py: Python<'_>, n: usize) -> Bound<'_, PyDict> {
        let times: Vec<Vec<f64>> = (0..n).map(|_| TIMES.to_vec()).collect();
        let values: Vec<Vec<f64>> = (0..n)
            .map(|i| VALUES.iter().map(|v| v * (i + 1) as f64).collect())
            .collect();
        let amounts: Vec<Vec<f64>> = (0..n).map(|_| vec![100.0]).collect();
        let dose_times: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0]).collect();
        let durations: Vec<Vec<f64>> = (0..n).map(|_| vec![f64::NAN]).collect();

        let dict = PyDict::new(py);
        dict.set_item("sample_dims", ["subject"]).unwrap();
        dict.set_item("sample_shape", [n]).unwrap();
        dict.set_item("n_time", TIMES.len()).unwrap();
        dict.set_item("times", array(py, &times)).unwrap();
        dict.set_item("values", array(py, &values)).unwrap();
        dict.set_item("max_doses", 1usize).unwrap();
        dict.set_item("dose_amount", array(py, &amounts)).unwrap();
        dict.set_item("dose_time", array(py, &dose_times)).unwrap();
        dict.set_item("dose_duration", array(py, &durations))
            .unwrap();
        dict.set_item("time_unit", "h").unwrap();
        dict.set_item("unit", "mg/l").unwrap();
        dict.set_item("dose_unit", "mg").unwrap();
        dict.set_item("route", "oral").unwrap();
        dict
    }

    /// The terminal regression of `NCAOptions()`, every key of it.
    fn terminal_dict(py: Python<'_>) -> Bound<'_, PyDict> {
        let dict = PyDict::new(py);
        dict.set_item("method", "best_fit").unwrap();
        dict.set_item("min_points", 3u32).unwrap();
        dict.set_item("exclude_cmax", true).unwrap();
        dict.set_item("n_points", py.None()).unwrap();
        dict.set_item("points", py.None()).unwrap();
        dict.set_item("min_adj_r2", py.None()).unwrap();
        dict.set_item("tie_tolerance", 1e-4).unwrap();
        dict.set_item("relative_warning", 1e-8).unwrap();
        dict.set_item("windows", py.None()).unwrap();
        dict.set_item("keep_candidates", false).unwrap();
        dict
    }

    /// The `units` option of one conversion, which this build refuses.
    fn units_dict(py: Python<'_>) -> Bound<'_, PyDict> {
        let units = PyDict::new(py);
        units.set_item("auc_last", "mg*hr/l").unwrap();
        units
    }

    /// `NCAOptions()` as `pkpdutils` hands it over, every key of it at its default.
    fn options_dict(py: Python<'_>) -> Bound<'_, PyDict> {
        let acceptance = PyDict::new(py);
        acceptance.set_item("r2_adj_min", py.None()).unwrap();
        acceptance.set_item("extrapolation_max", py.None()).unwrap();
        acceptance.set_item("span_min", py.None()).unwrap();
        acceptance.set_item("n_points_min", py.None()).unwrap();
        acceptance.set_item("exclude", false).unwrap();

        let dict = PyDict::new(py);
        dict.set_item("kind", "concentration").unwrap();
        dict.set_item("auc_method", "linear_log").unwrap();
        dict.set_item("terminal", terminal_dict(py)).unwrap();
        dict.set_item("lloq", py.None()).unwrap();
        dict.set_item("lloq_relative", py.None()).unwrap();
        dict.set_item("blq", "nan").unwrap();
        dict.set_item("c0_method", "log_back_extrapolation")
            .unwrap();
        dict.set_item("extrapolation_warning", 0.2).unwrap();
        dict.set_item("acceptance", acceptance).unwrap();
        dict.set_item("partial_aucs", PyDict::new(py)).unwrap();
        dict.set_item("tau", py.None()).unwrap();
        dict.set_item("tau_tolerance", 0.1).unwrap();
        dict.set_item("intervals", true).unwrap();
        dict.set_item("units", PyDict::new(py)).unwrap();
        dict.set_item("effect_threshold", py.None()).unwrap();
        dict.set_item("n_workers", py.None()).unwrap();
        dict.set_item("chunk_rows", 5000u32).unwrap();
        dict.set_item("uncertainty", py.None()).unwrap();
        dict.set_item("n_boot", 1000u32).unwrap();
        dict.set_item("seed", py.None()).unwrap();
        dict.set_item("ci_level", 0.95).unwrap();
        dict.set_item("bootstrap_spread", "se").unwrap();
        dict.set_item("bootstrap_distribution", "normal").unwrap();
        dict.set_item("delta_step", 0.01).unwrap();
        dict
    }

    /// The numbers of one `float64` array of a mapping of the result.
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

    /// The truth values of one `bool` array of a mapping of the result.
    fn truths(dict: &Bound<'_, PyDict>, key: &str) -> Vec<bool> {
        dict.get_item(key)
            .unwrap()
            .unwrap_or_else(|| panic!("`{key}` is not in the mapping"))
            .extract::<PyReadonlyArray1<'_, bool>>()
            .unwrap_or_else(|e| panic!("`{key}` is no `bool` array: {e}"))
            .as_slice()
            .unwrap()
            .to_vec()
    }

    /// The keys of a mapping of the result, sorted, so that a set can be compared by name.
    fn names_of(dict: &Bound<'_, PyDict>) -> Vec<String> {
        let mut names: Vec<String> = dict
            .keys()
            .iter()
            .map(|key| key.extract().unwrap())
            .collect();
        names.sort();
        names
    }

    /// One section of the result, as a dict of its own.
    fn section<'py>(dict: &Bound<'py, PyDict>, key: &str) -> Bound<'py, PyDict> {
        dict.get_item(key)
            .unwrap()
            .unwrap_or_else(|| panic!("the result has no section `{key}`"))
            .cast_into::<PyDict>()
            .unwrap()
    }

    #[test]
    fn a_single_dose_batch_returns_its_parameters() {
        Python::attach(|py| {
            let result =
                analyze_batch(py, &batch_dict(py, 2), &options_dict(py), None).expect("analysed");

            let parameters = section(&result, "parameters");
            // the rows keep the order of the batch, which is what the host indexes them by:
            // the second curve is twice the first, so its peak is twice as high
            assert_eq!(numbers(&parameters, "cmax"), [6.0, 12.0]);
            assert_eq!(numbers(&parameters, "tmax"), [1.0, 1.0]);
            assert_eq!(numbers(&parameters, "clast"), [0.04, 0.08]);
            let auc = numbers(&parameters, "auc_last");
            assert_eq!(auc.len(), 2);
            assert!(auc.iter().all(|a| a.is_finite() && *a > 0.0), "{auc:?}");
            // the area is linear in the values for both halves of linear-up/log-down, so the
            // second curve carries twice the area of the first
            assert!((auc[1] / auc[0] - 2.0).abs() < 1e-12, "{auc:?}");
            let thalf = numbers(&parameters, "thalf");
            assert!(thalf.iter().all(|t| t.is_finite() && *t > 0.0), "{thalf:?}");

            let units = section(&result, "units");
            // the unit of a parameter is the one pint derives and spells, not the one the
            // batch spells itself
            for (name, unit) in [
                ("cmax", "milligram / liter"),
                ("tmax", "hour"),
                ("thalf", "hour"),
                ("auc_last", "hour * milligram / liter"),
            ] {
                let found: String = units.get_item(name).unwrap().unwrap().extract().unwrap();
                assert_eq!(found, unit, "the unit of {name}");
            }

            let flags = result
                .get_item("flags")
                .unwrap()
                .unwrap()
                .extract::<PyReadonlyArray1<'_, u32>>()
                .expect("the flags are an array of unsigned integers")
                .as_slice()
                .unwrap()
                .to_vec();
            // the two curves are one curve scaled, so they are flagged alike, and no bit of
            // `NcaFlag` is set on either: the terminal window of this curve spans more than
            // the two half-lives of `SPAN_MINIMUM` and nothing else of the single dose
            // analysis has a flag to raise
            assert_eq!(flags, [0, 0]);

            // `accepted` and `excluded` are on every result of the kernel, by those names, and
            // without a threshold of `NCAOptions.acceptance` every row is accepted and none is
            // excluded
            let booleans = section(&result, "booleans");
            assert_eq!(names_of(&booleans), ["accepted", "excluded"]);
            assert_eq!(truths(&booleans, "accepted"), [true, true]);
            assert_eq!(truths(&booleans, "excluded"), [false, false]);

            // no row was excluded, so there is no text to carry, and the only point variables
            // of this build are of a single curve which asked to keep its candidates
            assert_eq!(section(&result, "texts").len(), 0);
            assert_eq!(section(&result, "point_variables").len(), 0);
        });
    }

    #[test]
    fn the_candidate_windows_of_one_curve_travel_as_point_variables() {
        // the section is not decoration: it is what the host's differential gate compares, and
        // the only analysis of this build which fills it is the one curve which reports the
        // windows its terminal regression could have chosen from
        Python::attach(|py| {
            let options = options_dict(py);
            let terminal = terminal_dict(py);
            terminal.set_item("keep_candidates", true).unwrap();
            options.set_item("terminal", terminal).unwrap();

            let result = analyze_batch(py, &batch_dict(py, 1), &options, None).expect("analysed");
            let point = section(&result, "point_variables");
            let units = section(&result, "units");
            for name in [
                "candidate_t_first",
                "candidate_n_points",
                "candidate_r2_adj",
            ] {
                let variable = point
                    .get_item(name)
                    .unwrap()
                    .unwrap_or_else(|| panic!("the result has no `{name}`"))
                    .cast_into::<PyDict>()
                    .unwrap();
                let dim: String = variable
                    .get_item("dim")
                    .unwrap()
                    .unwrap()
                    .extract()
                    .unwrap();
                let size: usize = variable
                    .get_item("size")
                    .unwrap()
                    .unwrap()
                    .extract()
                    .unwrap();
                assert_eq!(dim, "candidate", "the dimension of {name}");
                // one row, so the values are one per window of the selection
                assert_eq!(numbers(&variable, "values").len(), size, "{name}");
                assert!(size > 0, "{name} covers no window");
                assert!(
                    units.get_item(name).unwrap().is_some(),
                    "the unit of {name}"
                );
            }
        });
    }

    #[test]
    fn the_shard_count_changes_nothing_about_the_numbers() {
        // every count is a plan and no count is an error, and the kernel answers the same bits
        // for all of them, which is why the host may pass its own worker count straight
        // through. The two branches are not told apart by the numbers - nothing can tell them
        // apart by the numbers, by the kernel's own guarantee - so what this pins is that
        // guarantee and that both branches are reached and accepted
        Python::attach(|py| {
            let of = |n_workers| {
                let result = analyze_batch(py, &batch_dict(py, 4), &options_dict(py), n_workers)
                    .expect("analysed");
                numbers(&section(&result, "parameters"), "auc_last")
            };
            let automatic = of(None);
            for shards in [0, 1, 4, 1000] {
                assert_eq!(of(Some(shards)), automatic, "{shards} shards");
            }
        });
    }

    #[test]
    fn the_gil_is_released_during_the_analysis() {
        // three flags in one order, and the order is the whole test. `go` is set by this
        // thread once it holds the interpreter, so the other thread cannot have taken it
        // before the test began; `ready` is set by the other thread just before it asks for
        // the interpreter, so it is at the door when the window opens and nothing of its own
        // start up falls inside the measurement; `attached` is set by it once it has the
        // interpreter, which this thread holds throughout but for the window `analyze_batch`
        // opens
        let go = Arc::new(AtomicBool::new(false));
        let ready = Arc::new(AtomicBool::new(false));
        let attached = Arc::new(AtomicBool::new(false));
        let other = {
            let (go, ready, attached) =
                (Arc::clone(&go), Arc::clone(&ready), Arc::clone(&attached));
            thread::spawn(move || {
                while !go.load(Ordering::SeqCst) {
                    thread::yield_now();
                }
                ready.store(true, Ordering::SeqCst);
                Python::attach(|_| attached.store(true, Ordering::SeqCst));
            })
        };

        // big enough that the window lasts about a fifth of a second in a debug build, which
        // is five orders of magnitude more than a thread which is already waiting for the
        // interpreter needs to be handed it. That is the one thing here which is a margin and
        // not an order; the failure names the window it had, so a machine on which the margin
        // is not enough says so rather than leaving a reader to guess
        let rows = 60_000;
        let (reached, elapsed) = Python::attach(|py| {
            let batch = batch_dict(py, rows);
            let options = options_dict(py);
            go.store(true, Ordering::SeqCst);
            while !ready.load(Ordering::SeqCst) {
                thread::yield_now();
            }
            // the premise, without which the test would pass against anything: the other
            // thread is asking for an interpreter this thread holds, and has not got it
            assert!(
                !attached.load(Ordering::SeqCst),
                "the other thread held the interpreter before the analysis began"
            );
            let started = Instant::now();
            analyze_batch(py, &batch, &options, None).expect("analysed");
            (attached.load(Ordering::SeqCst), started.elapsed())
        });
        // the interpreter is free again here, whatever the answer was, so the other thread
        // finishes and this never hangs
        other.join().unwrap();

        assert!(
            reached,
            "another thread did not reach the interpreter in the {elapsed:?} the analysis of \
             {rows} rows took, so the analysis held it"
        );
    }

    #[test]
    fn a_kernel_error_becomes_a_python_exception() {
        Python::attach(|py| {
            // what the predicate refuses: the message carries the kernel's own text and says
            // that it is the support of this build which is at issue
            let options = options_dict(py);
            options.set_item("units", units_dict(py)).unwrap();
            let err = analyze_batch(py, &batch_dict(py, 2), &options, None).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let message = err.to_string();
            assert!(message.contains("NcaOptions::units"), "{message}");
            assert!(
                message.contains("is not implemented by this build"),
                "{message}"
            );
            assert!(message.contains("is_supported"), "{message}");

            // and what the kernel itself refuses, which the predicate does not answer for: a
            // batch with doses and no dose unit, found while the result is assembled
            let batch = batch_dict(py, 2);
            batch.set_item("dose_unit", py.None()).unwrap();
            let err = analyze_batch(py, &batch, &options_dict(py), None).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert!(!err.is_instance_of::<UnsupportedError>(py), "{err}");
            let message = err.to_string();
            assert!(message.contains("the batch is not analysable"), "{message}");
            assert!(message.contains("cl_f"), "{message}");
        });
    }

    #[test]
    fn a_unit_the_registry_cannot_read_reaches_the_host_as_a_refusal() {
        // The one refusal of the kernel the predicate does not answer for, and the reason this
        // test exists: `week` is the time base of a monoclonal antibody, a depot and an oncology
        // study, `U/l` an ordinary clinical concentration and `mg//l` a typo pint forgives and
        // `pkpdutils` 1.4.0 therefore computes. Reported as a plain `ValueError` every one of
        // them reaches the user of the host, whose fallback catches `UnsupportedError` and,
        // deliberately, nothing broader.
        Python::attach(|py| {
            for (key, spelling) in [
                ("time_unit", "week"),
                ("time_unit", "month"),
                ("unit", "U/l"),
                ("dose_unit", "kilohour"),
                ("unit", "mg l"),
                ("unit", "mg//l"),
                ("time_unit", "hr**--2"),
                ("unit", "mg/"),
            ] {
                let batch = batch_dict(py, 2);
                batch.set_item(key, spelling).unwrap();
                // the predicate reads nothing about the units and says so
                assert_eq!(
                    is_supported(py, &batch, &options_dict(py)).unwrap(),
                    None,
                    "{key} = {spelling}"
                );
                let err = analyze_batch(py, &batch, &options_dict(py), None).unwrap_err();
                assert!(
                    err.is_instance_of::<UnsupportedError>(py),
                    "{key} = {spelling}: {err}"
                );
                let message = err.to_string();
                assert!(message.contains(spelling), "{message}");
                assert!(
                    message.contains("does not carry this analysis"),
                    "{message}"
                );
            }
        });
    }

    #[test]
    fn only_an_unsupported_analysis_carries_the_type_a_host_routes_on() {
        // the host's rule is one line - catch `UnsupportedError` and take the python path, let
        // everything else through - so the whole design rests on no other failure wearing that
        // type. A disagreement about the contract which did would send every batch down the
        // python path for good with nothing to notice it by
        Python::attach(|py| {
            let refused = options_dict(py);
            refused.set_item("units", units_dict(py)).unwrap();
            let err = analyze_batch(py, &batch_dict(py, 2), &refused, None).unwrap_err();
            assert!(err.is_instance_of::<UnsupportedError>(py), "{err}");
            // additive: it subclasses `ValueError`, which this refusal was before it had a
            // type, so a caller which catches `ValueError` keeps catching it
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");

            // an enumeration value `NCAOptions` gains which this build does not spell. This is
            // the case the type exists for: it is a plain `ValueError`, so nothing but the
            // type tells it from the refusal above
            let gained = options_dict(py);
            gained.set_item("auc_method", "trapezoid").unwrap();
            let err = analyze_batch(py, &batch_dict(py, 2), &gained, None).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert!(!err.is_instance_of::<UnsupportedError>(py), "{err}");

            // a key it gains, which is a `KeyError`
            let unknown = options_dict(py);
            unknown.set_item("resolve_lloq", true).unwrap();
            let err = analyze_batch(py, &batch_dict(py, 2), &unknown, None).unwrap_err();
            assert!(!err.is_instance_of::<UnsupportedError>(py), "{err}");

            // and a batch the kernel itself refuses, which is incomplete data and not an
            // analysis of another plan: doses without a dose unit, where a unit this build
            // cannot read is a refusal
            // (`a_unit_the_registry_cannot_read_reaches_the_host_as_a_refusal`)
            let batch = batch_dict(py, 2);
            batch.set_item("dose_unit", py.None()).unwrap();
            let err = analyze_batch(py, &batch, &options_dict(py), None).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert!(!err.is_instance_of::<UnsupportedError>(py), "{err}");
        });
    }

    #[test]
    fn a_panic_does_not_cross_the_boundary() {
        // no input of the kernel is known to panic, so nothing but this test reaches the
        // boundary; a panic which escaped a pyo3 extension would abort the interpreter and the
        // user would see python die without a traceback
        Python::attach(|py| {
            assert_eq!(guarded(|| 1 + 1).unwrap(), 2);

            let err = guarded(|| panic!("the kernel tripped over a row")).unwrap_err();
            assert!(err.is_instance_of::<PyRuntimeError>(py), "{err}");
            let message = err.to_string();
            assert!(message.contains("pkpdutils-core"), "{message}");
            assert!(
                message.contains("the kernel tripped over a row"),
                "{message}"
            );
            assert!(message.contains(env!("CARGO_PKG_REPOSITORY")), "{message}");

            // a payload which is no string is still an exception and not an abort
            let err = guarded(|| std::panic::panic_any(7u8)).unwrap_err();
            assert!(err.is_instance_of::<PyRuntimeError>(py), "{err}");
            assert!(err.to_string().contains("pkpdutils-core"), "{err}");
        });
    }

    #[test]
    fn is_supported_answers_with_the_bare_reason() {
        Python::attach(|py| {
            assert_eq!(
                is_supported(py, &batch_dict(py, 2), &options_dict(py)).unwrap(),
                None
            );

            let options = options_dict(py);
            options.set_item("units", units_dict(py)).unwrap();
            let reason = is_supported(py, &batch_dict(py, 2), &options)
                .unwrap()
                .expect("the analysis is not carried");
            assert_eq!(
                reason,
                "the unit conversion of a result (NcaOptions::units)"
            );
            // the bare `what` and not the whole `Display` of the kernel's error: the host logs
            // this in a line of its own and `analyze_batch` is what raises
            assert!(
                !reason.contains("is not implemented by this build"),
                "{reason}"
            );
        });
    }

    #[test]
    fn a_conversion_error_is_not_a_refusal() {
        // a field python gains which this build does not map is a disagreement about the
        // contract, which both entry points raise over. Were it answered as a reason, every
        // batch would take the python path for good with nothing to notice it by
        Python::attach(|py| {
            let options = options_dict(py);
            options.set_item("resolve_lloq", true).unwrap();

            let err = is_supported(py, &batch_dict(py, 2), &options).unwrap_err();
            assert!(err.is_instance_of::<PyKeyError>(py), "{err}");
            assert!(err.to_string().contains("resolve_lloq"), "{err}");

            let err = analyze_batch(py, &batch_dict(py, 2), &options, None).unwrap_err();
            assert!(err.is_instance_of::<PyKeyError>(py), "{err}");
            assert!(err.to_string().contains("resolve_lloq"), "{err}");
        });
    }

    #[test]
    fn the_module_carries_its_version_and_its_two_entry_points() {
        Python::attach(|py| {
            let module = PyModule::new(py, "pkpdutils_core").unwrap();
            pkpdutils_core(&module).unwrap();

            let version: String = module.getattr("__version__").unwrap().extract().unwrap();
            assert_eq!(version, env!("CARGO_PKG_VERSION"));

            // through the module object and not the rust function, so that the signature
            // python really sees is what is checked: two arguments and a shard plan which may
            // be left out or named
            let analyze = module.getattr("analyze_batch").unwrap();
            let arguments = (batch_dict(py, 2), options_dict(py));
            let result = analyze.call1(arguments.clone()).unwrap();
            let parameters = section(&result.cast_into::<PyDict>().unwrap(), "parameters");
            assert_eq!(numbers(&parameters, "cmax"), [6.0, 12.0]);

            let named = PyDict::new(py);
            named.set_item("n_workers", 2u32).unwrap();
            assert!(analyze.call(arguments, Some(&named)).is_ok());

            let supported = module.getattr("is_supported").unwrap();
            let answer = supported
                .call1((batch_dict(py, 2), options_dict(py)))
                .unwrap();
            assert!(answer.is_none(), "{answer}");

            // the refusal a host routes on is importable by name, and is a `ValueError`, so
            // `except pkpdutils_core.UnsupportedError` is the whole of the host's rule
            let refusal = module
                .getattr("UnsupportedError")
                .unwrap()
                .cast_into::<PyType>()
                .expect("a class");
            assert!(
                refusal.is_subclass_of::<PyValueError>().unwrap(),
                "{refusal}"
            );
            assert!(refusal.is(py.get_type::<UnsupportedError>()), "{refusal}");
        });
    }

    /// The request of one variable and one group over the values given.
    fn summary_request<'py>(
        py: Python<'py>,
        values: &[f64],
        n_samples: usize,
    ) -> Bound<'py, PyDict> {
        let request = PyDict::new(py);
        request
            .set_item("values", PyArray1::from_slice(py, values))
            .unwrap();
        request.set_item("n_vars", 1).unwrap();
        request.set_item("n_groups", 1).unwrap();
        request.set_item("n_samples", n_samples).unwrap();
        request.set_item("ci_level", 0.95).unwrap();
        request.set_item("lognormal", vec![true]).unwrap();
        request.set_item("discrete", vec![false]).unwrap();
        request
    }

    #[test]
    fn summarize_returns_the_statistics_of_a_group_as_arrays_of_variables_by_groups() {
        Python::attach(|py| {
            let answer =
                summarize(py, &summary_request(py, &[1.0, 2.0, 4.0, f64::NAN], 4)).unwrap();
            let get = |name: &str| -> Vec<f64> {
                let array = answer.get_item(name).unwrap().unwrap();
                let array = array.cast::<numpy::PyArray2<f64>>().unwrap();
                assert_eq!(array.shape(), [1, 1], "{name}");
                array.readonly().as_array().iter().copied().collect()
            };
            assert_eq!(get("count"), [3.0]);
            assert!((get("mean")[0] - 7.0 / 3.0).abs() < 1e-15);
            // the geometric mean is the cube root of 1 * 2 * 4, which is 2
            assert!((get("geomean")[0] - 2.0).abs() < 1e-14);
            assert_eq!(get("median"), [2.0]);
            assert_eq!(get("min"), [1.0]);
            assert_eq!(get("max"), [4.0]);
        });
    }

    #[test]
    fn a_malformed_summary_request_is_a_value_error_and_never_a_refusal() {
        Python::attach(|py| {
            // three values for a group of four samples
            let short = summary_request(py, &[1.0, 2.0, 4.0], 4);
            let err = summarize(py, &short).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert!(!err.is_instance_of::<UnsupportedError>(py), "{err}");
            let message = err.value(py).to_string();
            assert!(message.contains("3 values"), "{message}");
            assert!(message.contains("4 samples"), "{message}");

            // a confidence level of 1 is no level
            let level = summary_request(py, &[1.0, 2.0, 4.0], 3);
            level.set_item("ci_level", 1.0).unwrap();
            let err = summarize(py, &level).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert!(!err.is_instance_of::<UnsupportedError>(py), "{err}");

            // a missing field is a KeyError which names it
            let absent = summary_request(py, &[1.0], 1);
            absent.del_item("n_groups").unwrap();
            let err = summarize(py, &absent).unwrap_err();
            assert!(err.is_instance_of::<PyKeyError>(py), "{err}");
            assert!(err.value(py).to_string().contains("n_groups"));
        });
    }

    /// A request of one row of four troughs on a rising curve.
    fn tss_request<'py>(py: Python<'py>, method: &str) -> Bound<'py, PyDict> {
        let request = PyDict::new(py);
        let t = [10.0, 20.0, 30.0, 40.0];
        let v = [3.9, 6.3, 7.8, 8.7];
        let s = [0.0, 10.0, 20.0, 30.0];
        request
            .set_item("time", PyArray1::from_slice(py, &t))
            .unwrap();
        request
            .set_item("value", PyArray1::from_slice(py, &v))
            .unwrap();
        request
            .set_item("start", PyArray1::from_slice(py, &s))
            .unwrap();
        request
            .set_item("valid", PyArray1::from_slice(py, &[true; 4]))
            .unwrap();
        request.set_item("n_rows", 1).unwrap();
        request.set_item("n_cols", 4).unwrap();
        request.set_item("method", method).unwrap();
        request.set_item("fraction", 0.9).unwrap();
        request.set_item("alpha", 0.05).unwrap();
        request
    }

    #[test]
    fn time_to_steady_state_returns_the_estimate_per_row() {
        Python::attach(|py| {
            let answer = time_to_steady_state(py, &tss_request(py, "monoexponential")).unwrap();
            let tss = answer.get_item("tss").unwrap().unwrap();
            let tss = tss.cast::<PyArray1<f64>>().unwrap().readonly();
            assert_eq!(tss.as_array().len(), 1);
            assert!(tss.as_array()[0].is_finite());
            let c_ss = answer.get_item("c_ss").unwrap().unwrap();
            let c_ss = c_ss.cast::<PyArray1<f64>>().unwrap().readonly();
            assert!(c_ss.as_array()[0] > 8.0);
            let answer = time_to_steady_state(py, &tss_request(py, "stepwise")).unwrap();
            let c_ss = answer.get_item("c_ss").unwrap().unwrap();
            assert_eq!(c_ss.cast::<PyArray1<f64>>().unwrap().len(), 0);
        });
    }

    #[test]
    fn a_malformed_tss_request_is_a_value_error_and_never_a_refusal() {
        Python::attach(|py| {
            let short = tss_request(py, "stepwise");
            short.set_item("n_cols", 5).unwrap();
            let err = time_to_steady_state(py, &short).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert!(!err.is_instance_of::<UnsupportedError>(py), "{err}");
            assert!(err.value(py).to_string().contains("time entries"));

            let level = tss_request(py, "monoexponential");
            level.set_item("fraction", 1.0).unwrap();
            let err = time_to_steady_state(py, &level).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert!(!err.is_instance_of::<UnsupportedError>(py), "{err}");

            let unknown = tss_request(py, "bogus");
            let err = time_to_steady_state(py, &unknown).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert!(!err.is_instance_of::<UnsupportedError>(py), "{err}");

            let absent = tss_request(py, "stepwise");
            absent.del_item("alpha").unwrap();
            let err = time_to_steady_state(py, &absent).unwrap_err();
            assert!(err.is_instance_of::<PyKeyError>(py), "{err}");
        });
    }
}
