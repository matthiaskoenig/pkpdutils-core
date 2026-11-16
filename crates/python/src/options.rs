//! The options of the analysis, from python into the wire shape of the abi, and the question a
//! host asks before it calls the kernel at all.
//!
//! [`NcaOptions`] mirrors `pkpdutils.nca.options.NCAOptions` field for field, so the dict
//! `pkpdutils` hands over carries one key per field, spelled as both sides spell it, and the
//! conversion is by name. An enumeration arrives as its serde name (`"linear_log"`, not
//! `AucMethod::LinearLog`), so that the python side and a JSON host agree on one spelling;
//! `blq` is a name or a dict, as python's `BLQHandling | BLQRules` and the untagged `Blq` of
//! the abi both are; and a terminal window arrives as the typed record of `_encode_window_key`,
//! since a python dict key cannot say what it is any more than a JSON key can.
//!
//! A key the abi gives an `Option` may be absent or `None`; the rest must be there. The JSON
//! wire defaults every field of the options instead, which is for a host that writes a partial
//! object; the host here is `pkpdutils`, whose `NCAOptions` always carries all of them, so an
//! absent key is a mistake worth naming rather than a short form.
//!
//! Nothing here validates: [`NcaOptions::validate`] runs on every entry point of
//! `pkpdutils_core` already. [`unsupported_reason`] is the other half of the module, the
//! question the host asks before it converts anything at all, and it is not validation either -
//! it is the list of analyses this build does not carry, which has to be the kernel's list and
//! not a second one that looks like it.

use std::collections::BTreeMap;

use pkpdutils_abi::{
    Acceptance, AucMethod, Batch, Blq, BlqAction, BlqHandling, BlqRules, BlqSetting,
    BootstrapDistribution, BootstrapSpread, C0Method, Kind, NcaOptions, OPTIONS_VERSION,
    TerminalMethod, TerminalPhase, UncertaintyMethod, WindowLabel,
};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::at_key;

/// The options of the analysis, built from the dict `pkpdutils` hands over.
///
/// The module documentation describes the dict. The result is not validated here, so a caller
/// hands it to `pkpdutils_core`, which validates every entry point.
///
/// # Errors
///
/// A `KeyError` naming a required key the dict does not carry, by its path inside the options
/// (`terminal.min_points`), a `TypeError` naming the key whose value has the wrong type, or a
/// `ValueError` naming an enumeration value, a window label type or a duplicate window label
/// the contract does not spell.
#[allow(
    dead_code,
    reason = "the analysis entry point which calls it is the next slice of the binding; the \
              tests of this module are its only caller today"
)]
pub(crate) fn options_from_python(obj: &Bound<'_, PyDict>) -> PyResult<NcaOptions> {
    let options = Fields::root(obj);
    Ok(NcaOptions {
        kind: options.enum_field("kind")?,
        auc_method: options.enum_field("auc_method")?,
        terminal: terminal_phase(&options.nested("terminal")?)?,
        lloq: options.optional_field("lloq")?,
        lloq_relative: options.optional_field("lloq_relative")?,
        blq: blq(&options, "blq")?,
        c0_method: options.enum_field("c0_method")?,
        extrapolation_warning: options.field("extrapolation_warning")?,
        acceptance: acceptance(&options.nested("acceptance")?)?,
        partial_aucs: options.field("partial_aucs")?,
        tau: options.optional_field("tau")?,
        tau_tolerance: options.field("tau_tolerance")?,
        intervals: options.field("intervals")?,
        units: options.field("units")?,
        effect_threshold: options.optional_field("effect_threshold")?,
        n_workers: options.optional_field("n_workers")?,
        chunk_rows: options.field("chunk_rows")?,
        uncertainty: options.optional_enum_field("uncertainty")?,
        n_boot: options.field("n_boot")?,
        seed: options.optional_field("seed")?,
        ci_level: options.field("ci_level")?,
        bootstrap_spread: options.enum_field("bootstrap_spread")?,
        bootstrap_distribution: options.enum_field("bootstrap_distribution")?,
        delta_step: options.field("delta_step")?,
    })
}

/// Why `pkpdutils_core` would refuse this batch, `None` when it would analyse it.
///
/// The host asks this before it converts anything, so that the work it cannot send here goes
/// down the python path instead of reaching the user as an exception from the kernel. The text
/// is the kernel's own, the `what` of its `KernelError::Unsupported`, and the conditions are
/// `refuse_what_is_not_carried` of `crates/core/src/lib.rs` in its order, since a batch may
/// trip over several of them and only the first is reported. Two lists which look alike and
/// are not is what `the_predicate_agrees_with_the_kernel` exists to catch: a predicate wider
/// than the kernel turns into an exception at the user's call site, and a narrower one sends
/// work this build carries down the slow path for good.
#[allow(
    dead_code,
    reason = "the analysis entry point which calls it is the next slice of the binding; the \
              tests of this module are its only caller today"
)]
pub(crate) fn unsupported_reason(batch: &Batch, options: &NcaOptions) -> Option<String> {
    let reason = |what: &str| Some(what.to_string());
    if options.kind == Kind::Effect {
        return reason("the analysis of an effect curve (Kind::Effect)");
    }
    if options.tau.is_some() {
        return reason("a multiple dose analysis (NcaOptions::tau)");
    }
    if (0..batch.n_samples()).any(|row| n_doses_of_row(batch, row) >= 2) {
        return reason("a protocol of more than one dose (the multiple dose analysis)");
    }
    if options.acceptance.any_threshold() {
        return reason("the acceptance criteria (NcaOptions::acceptance)");
    }
    if !options.partial_aucs.is_empty() {
        return reason("the named partial areas (NcaOptions::partial_aucs)");
    }
    if resolved_uncertainty(batch, options) != UncertaintyMethod::None {
        return reason("the propagation of the uncertainty of a group curve");
    }
    if !options.units.is_empty() {
        return reason("the unit conversion of a result (NcaOptions::units)");
    }
    None
}

/// The doses of a row: the columns whose amount and whose time are both a number.
///
/// This is `n_doses_of_row` of the kernel and not [`Batch::n_doses_of_row`] of the abi, which
/// counts the leading amounts which are not `NaN` and so reads a protocol with a hole in it
/// differently. The predicate answers for the kernel, so it counts what the kernel counts.
fn n_doses_of_row(batch: &Batch, row: usize) -> usize {
    let amounts = row_slice(&batch.dose_amount, row, batch.max_doses);
    let times = row_slice(&batch.dose_time, row, batch.max_doses);
    amounts
        .iter()
        .zip(times)
        .filter(|(amount, time)| amount.is_finite() && time.is_finite())
        .count()
}

/// The `width` numbers of a row, empty where the row is no row of the array.
///
/// Nothing is indexed with a number which came from the host: the predicate answers for a batch
/// the kernel has not validated yet, so a shape which does not add up is a count of zero doses
/// and never a panic.
fn row_slice(values: &[f64], row: usize, width: usize) -> &[f64] {
    let Some(start) = row.checked_mul(width) else {
        return &[];
    };
    let Some(end) = start.checked_add(width) else {
        return &[];
    };
    values.get(start..end).unwrap_or(&[])
}

/// The uncertainty the analysis would really propagate, `resolved_uncertainty` of the kernel.
///
/// `None` is the automatic choice: the parametric bootstrap for a batch which carries a spread
/// per point, nothing for a batch of individual curves (`NCAOptions.resolve_uncertainty` of
/// `src/pkpdutils/nca/options.py`). The refusal is therefore of what the options resolve to and
/// not of what they say, in either direction: a batch with `sd` is refused although the options
/// name no method, and one is analysed although it carries `sd` when they name
/// `UncertaintyMethod::None`.
fn resolved_uncertainty(batch: &Batch, options: &NcaOptions) -> UncertaintyMethod {
    options.uncertainty.unwrap_or({
        if batch.sd.is_some() || batch.se.is_some() {
            UncertaintyMethod::Bootstrap
        } else {
            UncertaintyMethod::None
        }
    })
}

/// How an error names the options, as `crate::convert` names the contract of the batch.
///
/// The options have no contract string of their own: `pkpdutils` stamps them with
/// `_nca_options_version`, of which [`OPTIONS_VERSION`] is this build's value.
fn contract() -> String {
    format!("the nca options of version {OPTIONS_VERSION}")
}

/// A dict of the options together with the path it sits at, which names the keys of its errors.
///
/// The path is what tells `terminal.min_points` from `acceptance.n_points_min`: the options are
/// three dicts deep in places, and a key named without its path sends the reader looking for it
/// in the wrong one.
struct Fields<'py> {
    /// The dict.
    dict: Bound<'py, PyDict>,
    /// Its path inside the options, empty at the top level.
    scope: String,
}

impl<'py> Fields<'py> {
    /// The options themselves, the dict at the top level.
    fn root(dict: &Bound<'py, PyDict>) -> Self {
        Self {
            dict: dict.clone(),
            scope: String::new(),
        }
    }

    /// The path of one of its keys.
    fn path(&self, key: &str) -> String {
        if self.scope.is_empty() {
            key.to_string()
        } else {
            format!("{}.{key}", self.scope)
        }
    }

    /// The value of a key the dict must carry.
    fn required(&self, key: &str) -> PyResult<Bound<'py, PyAny>> {
        self.dict.get_item(key)?.ok_or_else(|| {
            PyKeyError::new_err(format!(
                "{} needs the key `{}`, which is absent",
                contract(),
                self.path(key)
            ))
        })
    }

    /// The value of a key the dict may leave out, `None` for an absent key and for `None`.
    fn optional(&self, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(self.dict.get_item(key)?.filter(|value| !value.is_none()))
    }

    /// The value of a required key, extracted into `T` and named on a type mismatch.
    fn field<T>(&self, key: &str) -> PyResult<T>
    where
        T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
    {
        self.extract(key, self.required(key)?)
    }

    /// The value of an optional key, extracted into `T` and named on a type mismatch.
    fn optional_field<T>(&self, key: &str) -> PyResult<Option<T>>
    where
        T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
    {
        self.optional(key)?
            .map(|value| self.extract(key, value))
            .transpose()
    }

    /// One value of the dict, extracted into `T` and named on a type mismatch.
    fn extract<T>(&self, key: &str, value: Bound<'py, PyAny>) -> PyResult<T>
    where
        T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
    {
        value
            .extract::<T>()
            .map_err(|err| at_key(self.dict.py(), &self.path(key), err))
    }

    /// The variant of a required key which carries an enumeration.
    fn enum_field<T: Named>(&self, key: &str) -> PyResult<T> {
        let name: String = self.field(key)?;
        named(&self.path(key), &name)
    }

    /// The variant of an optional key which carries an enumeration.
    fn optional_enum_field<T: Named>(&self, key: &str) -> PyResult<Option<T>> {
        let Some(name) = self.optional_field::<String>(key)? else {
            return Ok(None);
        };
        named(&self.path(key), &name).map(Some)
    }

    /// The dict of a required key, at the path of that key.
    fn nested(&self, key: &str) -> PyResult<Self> {
        self.dict_of(key, &self.required(key)?)
    }

    /// One value of the dict read as a dict of its own, at the path `scope`.
    fn dict_of(&self, scope: &str, value: &Bound<'py, PyAny>) -> PyResult<Self> {
        let dict = value
            .cast::<PyDict>()
            .map_err(|err| at_key(self.dict.py(), &self.path(scope), err.into()))?;
        Ok(Self {
            dict: dict.clone(),
            scope: self.path(scope),
        })
    }
}

/// An enumeration of the abi which travels as its serde name.
trait Named: Copy + Sized + 'static {
    /// Every variant of it.
    const ALL: &'static [Self];
    /// The type of the abi, which an error names.
    const WHAT: &'static str;

    /// The serde name of the variant.
    ///
    /// Every implementation matches exhaustively, so a variant the abi gains fails to compile
    /// here instead of being refused at run time as a name this binding does not know.
    fn name(self) -> &'static str;
}

impl Named for Kind {
    const ALL: &'static [Self] = Self::ALL;
    const WHAT: &'static str = "Kind";

    fn name(self) -> &'static str {
        match self {
            Self::Concentration => "concentration",
            Self::Effect => "effect",
        }
    }
}

impl Named for AucMethod {
    const ALL: &'static [Self] = Self::ALL;
    const WHAT: &'static str = "AucMethod";

    fn name(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::LinearLog => "linear_log",
            Self::Log => "log",
        }
    }
}

impl Named for TerminalMethod {
    const ALL: &'static [Self] = Self::ALL;
    const WHAT: &'static str = "TerminalMethod";

    fn name(self) -> &'static str {
        match self {
            Self::BestFit => "best_fit",
            Self::LastN => "last_n",
            Self::AllAfterTmax => "all_after_tmax",
            Self::Manual => "manual",
        }
    }
}

impl Named for BlqHandling {
    const ALL: &'static [Self] = Self::ALL;
    const WHAT: &'static str = "BlqHandling";

    fn name(self) -> &'static str {
        match self {
            Self::Nan => "nan",
            Self::ZeroBeforeTmax => "zero_before_tmax",
        }
    }
}

impl Named for BlqAction {
    const ALL: &'static [Self] = Self::ALL;
    const WHAT: &'static str = "BlqAction";

    fn name(self) -> &'static str {
        match self {
            Self::Drop => "drop",
            Self::Keep => "keep",
            Self::Zero => "zero",
            Self::Lloq => "lloq",
            Self::HalfLloq => "half_lloq",
        }
    }
}

impl Named for C0Method {
    const ALL: &'static [Self] = Self::ALL;
    const WHAT: &'static str = "C0Method";

    fn name(self) -> &'static str {
        match self {
            Self::LogBackExtrapolation => "log_back_extrapolation",
            Self::FirstValue => "first_value",
            Self::None => "none",
        }
    }
}

impl Named for UncertaintyMethod {
    const ALL: &'static [Self] = Self::ALL;
    const WHAT: &'static str = "UncertaintyMethod";

    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Bootstrap => "bootstrap",
            Self::Delta => "delta",
        }
    }
}

impl Named for BootstrapSpread {
    const ALL: &'static [Self] = Self::ALL;
    const WHAT: &'static str = "BootstrapSpread";

    fn name(self) -> &'static str {
        match self {
            Self::Se => "se",
            Self::Sd => "sd",
        }
    }
}

impl Named for BootstrapDistribution {
    const ALL: &'static [Self] = Self::ALL;
    const WHAT: &'static str = "BootstrapDistribution";

    fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Lognormal => "lognormal",
        }
    }
}

/// The variant of its serde name, named by the key it sits under.
fn named<T: Named>(key: &str, name: &str) -> PyResult<T> {
    T::ALL
        .iter()
        .copied()
        .find(|variant| variant.name() == name)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "{key}: `{name}` is no `{}` of {}",
                T::WHAT,
                contract()
            ))
        })
}

/// The terminal regression, the ten fields of `pkpdutils.nca.options.TerminalPhase`.
fn terminal_phase(terminal: &Fields<'_>) -> PyResult<TerminalPhase> {
    Ok(TerminalPhase {
        method: terminal.enum_field("method")?,
        min_points: terminal.field("min_points")?,
        exclude_cmax: terminal.field("exclude_cmax")?,
        n_points: terminal.optional_field("n_points")?,
        points: terminal.optional_field("points")?,
        min_adj_r2: terminal.optional_field("min_adj_r2")?,
        tie_tolerance: terminal.field("tie_tolerance")?,
        relative_warning: terminal.optional_field("relative_warning")?,
        windows: windows(terminal, "windows")?,
        keep_candidates: terminal.field("keep_candidates")?,
    })
}

/// The terminal window of single samples by label, the records of `_terminal_windows`.
///
/// The labels travel as records rather than as the keys of a dict for the reason the JSON wire
/// has them: a label is a string, a number, a date or a tuple of those, and a python dict key
/// says no more about which of them it is than a JSON key does.
fn windows(
    terminal: &Fields<'_>,
    key: &str,
) -> PyResult<Option<BTreeMap<WindowLabel, (f64, f64)>>> {
    let Some(value) = terminal.optional(key)? else {
        return Ok(None);
    };
    let at = terminal.path(key);
    let py = value.py();
    let items = value.try_iter().map_err(|err| at_key(py, &at, err))?;
    let mut map = BTreeMap::new();
    for (index, item) in items.enumerate() {
        let item = item?;
        let record = terminal.dict_of(&format!("{key}[{index}]"), &item)?;
        let label = window_label(&record.nested("key")?)?;
        let window: (f64, f64) = record.field("window")?;
        if map.insert(label.clone(), window).is_some() {
            return Err(PyValueError::new_err(format!(
                "{}: the terminal window of {label} is given twice",
                record.scope
            )));
        }
    }
    Ok(Some(map))
}

/// One sample label, the `{"type": ..., "value": ...}` of `_encode_window_key`.
fn window_label(label: &Fields<'_>) -> PyResult<WindowLabel> {
    let kind: String = label.field("type")?;
    let value = label.required("value")?;
    Ok(match kind.as_str() {
        "tuple" => {
            let py = value.py();
            let at = label.path("value");
            let items = value.try_iter().map_err(|err| at_key(py, &at, err))?;
            let mut labels = Vec::new();
            for (index, item) in items.enumerate() {
                let item = item?;
                let item = label.dict_of(&format!("value[{index}]"), &item)?;
                labels.push(window_label(&item)?);
            }
            WindowLabel::Tuple(labels)
        }
        "none" => WindowLabel::None,
        "datetime" => WindowLabel::Datetime(label.extract("value", value)?),
        "date" => WindowLabel::Date(label.extract("value", value)?),
        "bool" => WindowLabel::Bool(label.extract("value", value)?),
        "str" => WindowLabel::Str(label.extract("value", value)?),
        "int" => WindowLabel::Int(label.extract("value", value)?),
        "float" => WindowLabel::Float(label.extract("value", value)?),
        _ => {
            return Err(PyValueError::new_err(format!(
                "{}: `{kind}` is no label type of {}, which spells one of {}",
                label.path("type"),
                contract(),
                label_types()
            )));
        }
    })
}

/// Every type a label names itself with, one prototype of every variant of [`WindowLabel`].
///
/// The names come from [`label_type`], whose match is exhaustive, so a label the abi gains is a
/// compile error here and not a type [`window_label`] silently refuses.
fn label_types() -> String {
    [
        WindowLabel::Tuple(Vec::new()),
        WindowLabel::None,
        WindowLabel::Datetime(String::new()),
        WindowLabel::Date(String::new()),
        WindowLabel::Bool(false),
        WindowLabel::Str(String::new()),
        WindowLabel::Int(0),
        WindowLabel::Float(0.0),
    ]
    .iter()
    .map(|label| format!("`{}`", label_type(label)))
    .collect::<Vec<_>>()
    .join(", ")
}

/// The type `_encode_window_key` writes for a label, the serde tag of the abi's wire form.
fn label_type(label: &WindowLabel) -> &'static str {
    match label {
        WindowLabel::Tuple(_) => "tuple",
        WindowLabel::None => "none",
        WindowLabel::Datetime(_) => "datetime",
        WindowLabel::Date(_) => "date",
        WindowLabel::Bool(_) => "bool",
        WindowLabel::Str(_) => "str",
        WindowLabel::Int(_) => "int",
        WindowLabel::Float(_) => "float",
    }
}

/// The handling of the values below the limit, python's `BLQHandling | BLQRules`.
///
/// A name is one of the two classic handlings and a dict is the rules, which is how the
/// untagged [`Blq`] of the abi reads the JSON of the same field.
fn blq(options: &Fields<'_>, key: &str) -> PyResult<Blq> {
    let value = options.required(key)?;
    let at = options.path(key);
    if let Ok(name) = value.extract::<String>() {
        return Ok(Blq::Handling(named(&at, &name)?));
    }
    if value.is_instance_of::<PyDict>() {
        return Ok(Blq::Rules(blq_rules(&options.dict_of(key, &value)?)?));
    }
    Err(PyTypeError::new_err(format!(
        "{at}: the handling below the limit is the name of a `BlqHandling` or the dict of a \
         `BlqRules`"
    )))
}

/// The rules by position or by tmax, the six fields of `pkpdutils.nca.options.BLQRules`.
fn blq_rules(rules: &Fields<'_>) -> PyResult<BlqRules> {
    Ok(BlqRules {
        first: blq_setting(rules, "first")?,
        middle: blq_setting(rules, "middle")?,
        last: blq_setting(rules, "last")?,
        before_tmax: blq_setting(rules, "before_tmax")?,
        after_tmax: blq_setting(rules, "after_tmax")?,
        terminal_regression: rules.field("terminal_regression")?,
    })
}

/// One rule: the name of an action, a number imputed as it is, or nothing.
fn blq_setting(rules: &Fields<'_>, key: &str) -> PyResult<Option<BlqSetting>> {
    let Some(value) = rules.optional(key)? else {
        return Ok(None);
    };
    if let Ok(name) = value.extract::<String>() {
        return Ok(Some(BlqSetting::Action(named(&rules.path(key), &name)?)));
    }
    Ok(Some(BlqSetting::Value(rules.extract(key, value)?)))
}

/// The thresholds of the acceptance criteria of `pkpdutils.nca.options.Acceptance`.
fn acceptance(acceptance: &Fields<'_>) -> PyResult<Acceptance> {
    Ok(Acceptance {
        r2_adj_min: acceptance.optional_field("r2_adj_min")?,
        extrapolation_max: acceptance.optional_field("extrapolation_max")?,
        span_min: acceptance.optional_field("span_min")?,
        n_points_min: acceptance.optional_field("n_points_min")?,
        exclude: acceptance.field("exclude")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use numpy::PyArray2;
    use pkpdutils_core::{KernelError, analyze_batch};
    use pyo3::types::PyList;

    use crate::convert::batch_from_python;

    /// A `(1, n)` array of one row of numbers.
    fn row<'py>(py: Python<'py>, values: Vec<f64>) -> Bound<'py, PyAny> {
        PyArray2::from_vec2(py, &[values]).unwrap().into_any()
    }

    /// One oral subject of five points, dosed `n_doses` times twelve hours apart.
    fn batch_dict(py: Python<'_>, n_doses: usize) -> Bound<'_, PyDict> {
        let amounts = vec![100.0; n_doses];
        let times: Vec<f64> = (0..n_doses).map(|i| 12.0 * i as f64).collect();
        let durations = vec![f64::NAN; n_doses];

        let dict = PyDict::new(py);
        dict.set_item("sample_dims", ["subject"]).unwrap();
        dict.set_item("sample_shape", [1usize]).unwrap();
        dict.set_item("n_time", 5usize).unwrap();
        dict.set_item("times", row(py, vec![0.0, 1.0, 2.0, 4.0, 8.0]))
            .unwrap();
        dict.set_item("values", row(py, vec![0.0, 10.0, 8.0, 4.0, 1.0]))
            .unwrap();
        dict.set_item("max_doses", n_doses).unwrap();
        dict.set_item("dose_amount", row(py, amounts)).unwrap();
        dict.set_item("dose_time", row(py, times)).unwrap();
        dict.set_item("dose_duration", row(py, durations)).unwrap();
        dict.set_item("time_unit", "h").unwrap();
        dict.set_item("unit", "mg/l").unwrap();
        dict.set_item("dose_unit", "mg").unwrap();
        dict.set_item("route", "oral").unwrap();
        dict
    }

    /// The batch of [`batch_dict`], through the conversion the binding really uses.
    fn batch(n_doses: usize) -> Batch {
        Python::attach(|py| batch_from_python(py, &batch_dict(py, n_doses)).unwrap())
    }

    /// The single dose batch with a spread per point, the group curve of several subjects.
    fn batch_with_spread() -> Batch {
        Python::attach(|py| {
            let dict = batch_dict(py, 1);
            dict.set_item("sd", row(py, vec![0.0, 1.0, 0.8, 0.4, 0.1]))
                .unwrap();
            batch_from_python(py, &dict).unwrap()
        })
    }

    /// One typed window label, `{"type": ..., "value": ...}` of `_encode_window_key`.
    fn label<'py, V>(py: Python<'py>, kind: &str, value: V) -> Bound<'py, PyDict>
    where
        V: IntoPyObject<'py>,
    {
        let dict = PyDict::new(py);
        dict.set_item("type", kind).unwrap();
        dict.set_item("value", value).unwrap();
        dict
    }

    /// One window record, the label and its `(t_first, t_last)`.
    fn window<'py>(
        py: Python<'py>,
        key: Bound<'py, PyDict>,
        span: (f64, f64),
    ) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        dict.set_item("key", key).unwrap();
        dict.set_item("window", span).unwrap();
        dict
    }

    /// The terminal regression, every field away from its default.
    fn terminal_dict(py: Python<'_>) -> Bound<'_, PyDict> {
        let windows = PyList::new(
            py,
            [
                window(py, label(py, "str", "A"), (4.0, 24.0)),
                window(py, label(py, "int", 7i64), (1.0, 2.0)),
            ],
        )
        .unwrap();

        let dict = PyDict::new(py);
        dict.set_item("method", "last_n").unwrap();
        dict.set_item("min_points", 4u32).unwrap();
        dict.set_item("exclude_cmax", false).unwrap();
        dict.set_item("n_points", 5u32).unwrap();
        dict.set_item("points", [-3i64, -2, -1]).unwrap();
        dict.set_item("min_adj_r2", 0.9).unwrap();
        dict.set_item("tie_tolerance", 1e-3).unwrap();
        dict.set_item("relative_warning", 1e-6).unwrap();
        dict.set_item("windows", windows).unwrap();
        dict.set_item("keep_candidates", true).unwrap();
        dict
    }

    /// The options dict every key of which carries a value which is not the default.
    ///
    /// The set need not be one the kernel accepts - `kind` and `lloq_relative` contradict each
    /// other in python - since nothing here validates: every field has to differ from its
    /// default for the round trip to say anything, and the kernel checks the combination.
    fn options_dict(py: Python<'_>) -> Bound<'_, PyDict> {
        let blq = PyDict::new(py);
        blq.set_item("first", "zero").unwrap();
        blq.set_item("middle", "drop").unwrap();
        blq.set_item("last", 0.025).unwrap();
        blq.set_item("before_tmax", py.None()).unwrap();
        blq.set_item("after_tmax", py.None()).unwrap();
        blq.set_item("terminal_regression", true).unwrap();

        let acceptance = PyDict::new(py);
        acceptance.set_item("r2_adj_min", 0.95).unwrap();
        acceptance.set_item("extrapolation_max", 0.3).unwrap();
        acceptance.set_item("span_min", 2.5).unwrap();
        acceptance.set_item("n_points_min", 4u32).unwrap();
        acceptance.set_item("exclude", true).unwrap();

        let partial_aucs = PyDict::new(py);
        partial_aucs.set_item("early", (0.0, 4.0)).unwrap();
        partial_aucs.set_item("late", (4.0, 24.0)).unwrap();

        let units = PyDict::new(py);
        units.set_item("auc_last", "mg*hr/l").unwrap();

        let dict = PyDict::new(py);
        dict.set_item("kind", "effect").unwrap();
        dict.set_item("auc_method", "linear").unwrap();
        dict.set_item("terminal", terminal_dict(py)).unwrap();
        dict.set_item("lloq", 0.05).unwrap();
        dict.set_item("lloq_relative", 0.01).unwrap();
        dict.set_item("blq", blq).unwrap();
        dict.set_item("c0_method", "first_value").unwrap();
        dict.set_item("extrapolation_warning", 0.25).unwrap();
        dict.set_item("acceptance", acceptance).unwrap();
        dict.set_item("partial_aucs", partial_aucs).unwrap();
        dict.set_item("tau", 12.0).unwrap();
        dict.set_item("tau_tolerance", 0.2).unwrap();
        dict.set_item("intervals", false).unwrap();
        dict.set_item("units", units).unwrap();
        dict.set_item("effect_threshold", 1.5).unwrap();
        dict.set_item("n_workers", 4u32).unwrap();
        dict.set_item("chunk_rows", 1000u32).unwrap();
        dict.set_item("uncertainty", "delta").unwrap();
        dict.set_item("n_boot", 500u32).unwrap();
        dict.set_item("seed", 7u64).unwrap();
        dict.set_item("ci_level", 0.9).unwrap();
        dict.set_item("bootstrap_spread", "sd").unwrap();
        dict.set_item("bootstrap_distribution", "lognormal")
            .unwrap();
        dict.set_item("delta_step", 0.05).unwrap();
        dict
    }

    /// The options of [`options_dict`] with one key replaced by `value`.
    fn options_with<'py, V>(py: Python<'py>, key: &str, value: V) -> PyResult<NcaOptions>
    where
        V: IntoPyObject<'py>,
    {
        let dict = options_dict(py);
        dict.set_item(key, value).unwrap();
        options_from_python(&dict)
    }

    #[test]
    fn every_option_field_round_trips() {
        let options = Python::attach(|py| options_from_python(&options_dict(py)).unwrap());

        assert_eq!(options.kind, Kind::Effect);
        assert_eq!(options.auc_method, AucMethod::Linear);
        assert_eq!(options.terminal.method, TerminalMethod::LastN);
        assert_eq!(options.terminal.min_points, 4);
        assert!(!options.terminal.exclude_cmax);
        assert_eq!(options.terminal.n_points, Some(5));
        assert_eq!(
            options.terminal.points.as_deref(),
            Some([-3i64, -2, -1].as_slice())
        );
        assert_eq!(options.terminal.min_adj_r2, Some(0.9));
        assert_eq!(options.terminal.tie_tolerance, 1e-3);
        assert_eq!(options.terminal.relative_warning, Some(1e-6));
        let windows = options.terminal.windows.as_ref().unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[&WindowLabel::Str("A".to_string())], (4.0, 24.0));
        assert_eq!(windows[&WindowLabel::Int(7)], (1.0, 2.0));
        assert!(options.terminal.keep_candidates);
        assert_eq!(options.lloq, Some(0.05));
        assert_eq!(options.lloq_relative, Some(0.01));
        assert_eq!(
            options.blq,
            Blq::Rules(BlqRules {
                first: Some(BlqSetting::Action(BlqAction::Zero)),
                middle: Some(BlqSetting::Action(BlqAction::Drop)),
                last: Some(BlqSetting::Value(0.025)),
                before_tmax: None,
                after_tmax: None,
                terminal_regression: true,
            })
        );
        assert_eq!(options.c0_method, C0Method::FirstValue);
        assert_eq!(options.extrapolation_warning, 0.25);
        assert_eq!(
            options.acceptance,
            Acceptance {
                r2_adj_min: Some(0.95),
                extrapolation_max: Some(0.3),
                span_min: Some(2.5),
                n_points_min: Some(4),
                exclude: true,
            }
        );
        assert_eq!(options.partial_aucs.get("early"), Some(&(0.0, 4.0)));
        assert_eq!(options.partial_aucs.get("late"), Some(&(4.0, 24.0)));
        assert_eq!(options.tau, Some(12.0));
        assert_eq!(options.tau_tolerance, 0.2);
        assert!(!options.intervals);
        assert_eq!(
            options.units.get("auc_last").map(String::as_str),
            Some("mg*hr/l")
        );
        assert_eq!(options.effect_threshold, Some(1.5));
        assert_eq!(options.n_workers, Some(4));
        assert_eq!(options.chunk_rows, 1000);
        assert_eq!(options.uncertainty, Some(UncertaintyMethod::Delta));
        assert_eq!(options.n_boot, 500);
        assert_eq!(options.seed, Some(7));
        assert_eq!(options.ci_level, 0.9);
        assert_eq!(options.bootstrap_spread, BootstrapSpread::Sd);
        assert_eq!(
            options.bootstrap_distribution,
            BootstrapDistribution::Lognormal
        );
        assert_eq!(options.delta_step, 0.05);
    }

    #[test]
    fn the_classic_handling_and_an_absent_optional_key_are_read() {
        let options = Python::attach(|py| {
            let dict = options_dict(py);
            // python spells the field `BLQHandling | BLQRules`, so a name stands for the rules
            dict.set_item("blq", "zero_before_tmax").unwrap();
            for key in ["lloq", "tau", "uncertainty", "seed"] {
                dict.del_item(key).unwrap();
            }
            dict.set_item("effect_threshold", py.None()).unwrap();
            let terminal = terminal_dict(py);
            terminal.del_item("windows").unwrap();
            terminal.set_item("points", py.None()).unwrap();
            dict.set_item("terminal", terminal).unwrap();
            options_from_python(&dict).unwrap()
        });

        assert_eq!(options.blq, Blq::Handling(BlqHandling::ZeroBeforeTmax));
        assert_eq!(options.lloq, None);
        assert_eq!(options.tau, None);
        assert_eq!(options.uncertainty, None);
        assert_eq!(options.seed, None);
        assert_eq!(options.effect_threshold, None);
        assert_eq!(options.terminal.windows, None);
        assert_eq!(options.terminal.points, None);
    }

    #[test]
    fn a_missing_required_key_is_refused_by_its_path() {
        let message = Python::attach(|py| {
            let dict = options_dict(py);
            let terminal = terminal_dict(py);
            terminal.del_item("min_points").unwrap();
            dict.set_item("terminal", terminal).unwrap();
            options_from_python(&dict).unwrap_err().to_string()
        });

        assert!(message.contains("terminal.min_points"), "{message}");
    }

    #[test]
    fn an_unknown_enum_name_is_refused_by_its_key() {
        let message = Python::attach(|py| {
            let err = options_with(py, "auc_method", "trapezoid").unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            err.to_string()
        });

        assert!(message.contains("auc_method"), "{message}");
        assert!(message.contains("trapezoid"), "{message}");
        assert!(message.contains("AucMethod"), "{message}");
    }

    #[test]
    fn a_refused_option_is_named_by_the_predicate() {
        let batch = batch(1);
        let reason = |options: NcaOptions| {
            unsupported_reason(&batch, &options).expect("the options are not carried")
        };

        let effect = reason(NcaOptions {
            kind: Kind::Effect,
            ..NcaOptions::default()
        });
        assert!(effect.contains("Kind::Effect"), "{effect}");

        let tau = reason(NcaOptions {
            tau: Some(12.0),
            ..NcaOptions::default()
        });
        assert!(tau.contains("NcaOptions::tau"), "{tau}");

        let acceptance = reason(NcaOptions {
            acceptance: Acceptance {
                span_min: Some(3.0),
                ..Acceptance::default()
            },
            ..NcaOptions::default()
        });
        assert!(
            acceptance.contains("NcaOptions::acceptance"),
            "{acceptance}"
        );

        let partial = reason(NcaOptions {
            partial_aucs: BTreeMap::from([("early".to_string(), (0.0, 4.0))]),
            ..NcaOptions::default()
        });
        assert!(partial.contains("NcaOptions::partial_aucs"), "{partial}");

        let units = reason(NcaOptions {
            units: BTreeMap::from([("auc_last".to_string(), "mg*hr/l".to_string())]),
            ..NcaOptions::default()
        });
        assert!(units.contains("NcaOptions::units"), "{units}");
    }

    #[test]
    fn a_supported_batch_has_no_reason() {
        let batch = batch(1);
        let options = NcaOptions::default();

        assert_eq!(unsupported_reason(&batch, &options), None);
        // the other direction of the agreement: a batch the predicate passes is one the kernel
        // really analyses, so a predicate stricter than the kernel fails here
        assert!(analyze_batch(&batch, &options).is_ok());
    }

    #[test]
    fn the_predicate_agrees_with_the_kernel() {
        let single = batch(1);
        let multiple = batch(2);
        let spread = batch_with_spread();
        let units = BTreeMap::from([("auc_last".to_string(), "mg*hr/l".to_string())]);

        let cases: [(&str, &Batch, NcaOptions); 8] = [
            (
                "an effect curve",
                &single,
                NcaOptions {
                    kind: Kind::Effect,
                    ..NcaOptions::default()
                },
            ),
            (
                "a dosing interval",
                &single,
                NcaOptions {
                    tau: Some(12.0),
                    ..NcaOptions::default()
                },
            ),
            ("a protocol of two doses", &multiple, NcaOptions::default()),
            (
                "the acceptance criteria",
                &single,
                NcaOptions {
                    acceptance: Acceptance {
                        span_min: Some(3.0),
                        ..Acceptance::default()
                    },
                    ..NcaOptions::default()
                },
            ),
            (
                "a named partial area",
                &single,
                NcaOptions {
                    partial_aucs: BTreeMap::from([("early".to_string(), (0.0, 4.0))]),
                    ..NcaOptions::default()
                },
            ),
            (
                "the uncertainty of a group curve",
                &spread,
                NcaOptions::default(),
            ),
            (
                "the unit conversion of a result",
                &single,
                NcaOptions {
                    units: units.clone(),
                    ..NcaOptions::default()
                },
            ),
            // several refusals at once, which pins the order of the two lists and not only
            // what is on them: the kernel names the effect curve and so must the predicate
            (
                "an effect curve of a multiple dose protocol",
                &multiple,
                NcaOptions {
                    kind: Kind::Effect,
                    tau: Some(12.0),
                    units,
                    ..NcaOptions::default()
                },
            ),
        ];

        for (case, batch, options) in cases {
            let error = analyze_batch(batch, &options).expect_err(case);
            let KernelError::Unsupported { what } = error else {
                panic!("{case}: the kernel refused it as {error} and not as unsupported");
            };
            assert_eq!(unsupported_reason(batch, &options), Some(what), "{case}");
        }
    }
}
