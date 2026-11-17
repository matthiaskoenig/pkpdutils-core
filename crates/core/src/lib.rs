//! The non-compartmental analysis kernel of pkpdutils.
//!
//! This crate is the port of `pkpdutils.nca`, the pure numeric core of the python package: it
//! takes the wire shapes of [`pkpdutils_abi`] and gives back the parameters of a curve. It runs
//! natively and on `wasm32`, which decides how it is written. A panic is a trap that kills a
//! wasm instance and takes the host's session with it, so a malformed input is a
//! [`KernelError`] or a `NaN`, never a panic: no slice is indexed where the index comes from
//! the input, nothing a host controls is unwrapped, and the arithmetic is the IEEE arithmetic
//! of numpy, which answers a division by zero with an infinity instead of raising.
//!
//! Agreement with the python package is what this crate is measured by. Every module mirrors
//! one of `src/pkpdutils/nca/`, keeps its order of operations so that the floating point result
//! is the same bit for bit where it can be, and pins an upstream oddity with a test rather than
//! correcting it.

mod assemble;
mod auc;
mod blq;
mod dose;
mod intervals;
mod pack;
mod parameters;
mod percentile;
mod shard;
mod stats;
mod summarize;

pub use crate::stats::{normal_isf, student_t_isf, student_t_sf};
pub use crate::summarize::summarize;
pub use crate::tss::time_to_steady_state;
mod steady_state;
mod terminal;
mod tss;

use pkpdutils_abi::{
    ANALYSIS_CONTRACT, AnalysisRequest, AnalysisResult, BATCH_CONTRACT, Batch, Coordinate, Kind,
    NcaOptions, TerminalMethod, UncertaintyMethod, WindowLabel, check_contract,
};

/// The wire types this crate's whole public API is written in, re-exported so that a host takes
/// one dependency and cannot hold a version of them which disagrees with this build's.
pub use pkpdutils_abi;

pub use crate::shard::{SHARD_WORK_THRESHOLD, automatic_shards};

use crate::assemble::{RowResult, resolved_uncertainty, result_of};
use crate::dose::{ReferenceDose, dose_count, reference_dose};
use crate::parameters::{RowInput, Scratch, compute_row};
use crate::shard::rows_of;
use crate::steady_state::compute_steady_state;
use crate::terminal::Candidate;

/// Why a batch could not be analysed.
///
/// The enumeration is `#[non_exhaustive]`: every [`KernelError::Unsupported`] analysis of this
/// build is a path a later plan implements, and the errors those paths need are variants this
/// one does not have. A host therefore matches with a `_` arm and keeps compiling across the
/// plan which adds one.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum KernelError {
    /// The batch itself is malformed, with the reason `pkpdutils_abi::Batch::validate` gave, or
    /// it carries doses without a dose unit, which that check does not cover
    /// (`crate::assemble`). A unit string this build cannot read is
    /// [`KernelError::Unsupported`] instead, whatever the reason the parser gives: what this
    /// build reads is a property of the build, and the batch is analysable by a host which reads
    /// it.
    #[error("the batch is not analysable: {0}")]
    Batch(String),
    /// The options are malformed or contradict each other.
    #[error("the options are not analysable: {0}")]
    Options(String),
    /// An analysis this build does not carry, named by what was asked for.
    ///
    /// Every one of these but one is decided before a row runs and is therefore also what
    /// `is_supported` of the python binding answers (`unsupported_reason`). The one
    /// which is not is a unit of the batch which the curated registry of `pkpdutils-units`
    /// cannot read: answering it means deriving the unit of every parameter, which is the
    /// analysis, so it is reported where that happens (`crate::assemble::unit_of`) and the
    /// predicate does not see it. A host which routes on the type rather than on the predicate
    /// is unaffected, which is what both are documented to do.
    #[error("{what} is not implemented by this build")]
    Unsupported {
        /// The analysis, the option or the route which is missing.
        what: String,
    },
    /// A summary request whose lengths disagree with its dimensions or whose `ci_level` is not
    /// strictly between 0 and 1, with the reason `SummaryRequest::validate` gave.
    #[error("the summary request is malformed: {0}")]
    Summary(String),
    /// A time to steady state request whose arrays disagree with its dimensions, whose troughs
    /// are not packed to the front or whose `fraction` or `alpha` is not strictly between 0
    /// and 1, with the reason `TssRequest::validate` gave.
    #[error("the time to steady state request is malformed: {0}")]
    Tss(String),
    /// The kernel built a result which is not well formed, which is a bug in this build and
    /// not an answer to anything the caller sent.
    #[error("the kernel built a result which is not readable: {0}")]
    Internal(String),
    /// The peer speaks another contract than this build.
    #[error(transparent)]
    Contract(#[from] pkpdutils_abi::ContractError),
}

/// Analyse the batch of a request, after checking that the peer speaks this contract.
///
/// The contract of the request and the contract of its batch are both checked, since a host may
/// build the two separately; the batch is checked by [`analyze_batch`], which is an entry of its
/// own and does not trust a caller to have done it.
///
/// # Errors
///
/// [`KernelError::Contract`] for a foreign contract, and whatever [`analyze_batch`] refuses.
pub fn analyze(request: &AnalysisRequest) -> Result<AnalysisResult, KernelError> {
    check_contract(&request.contract, ANALYSIS_CONTRACT)?;
    analyze_batch(&request.batch, &request.options)
}

/// The non-compartmental analysis of every row of a batch, single dose and multiple dose.
///
/// The contract, the batch and the options are checked before a row runs, an analysis this build
/// does not carry is refused by name, and only then are the rows computed, each one independent
/// of the others (`parameters::compute_row`). The result carries one column per parameter the
/// rows report in its reported unit, the flags of every row, the acceptance variables and the
/// coordinates of the batch (`assemble::result_of`).
///
/// The rows run in the shards [`automatic_shards`] plans for the shape of the batch: in the
/// calling thread below [`SHARD_WORK_THRESHOLD`] of work and in every build without the
/// `parallel` feature, on the rayon pool above it. [`analyze_batch_in_shards`] takes the plan
/// instead of computing it; the numbers are the same either way, down to the bits.
///
/// # Errors
///
/// [`KernelError::Contract`] for a batch of another contract, [`KernelError::Batch`] and
/// [`KernelError::Options`] for a malformed input, naming the field, and
/// [`KernelError::Unsupported`] for an analysis of another plan: an effect curve, the
/// acceptance criteria, the named partial areas, the propagation of an uncertainty and the unit
/// conversion of the result (the five conditions of [`unsupported_reason`]) - and for a unit
/// string of the batch which the curated registry of this build cannot read (`week`, `U/l`), the
/// one refusal which is decided while the result is assembled rather than before the rows run.
pub fn analyze_batch(batch: &Batch, options: &NcaOptions) -> Result<AnalysisResult, KernelError> {
    let shards = automatic_shards(batch.n_samples(), batch.n_time);
    analyze_batch_in_shards(batch, options, shards)
}

/// [`analyze_batch`] over a number of row shards the caller chooses.
///
/// The rows are cut into contiguous stretches of about `rows / shards` rows, which is what a
/// batch of more work than [`SHARD_WORK_THRESHOLD`] gets from [`automatic_shards`]. `0` and `1`
/// are one shard, a count above the number of rows is one shard per row, and the shards run on
/// the rayon pool with the `parallel` feature and one after the other in the calling thread
/// without it. The count fixes the smallest stretch the pool may cut, `rows / shards` rows, so
/// it is the most stretches the run can have and not the fewest: an idle pool splits them all,
/// a busy one runs longer stretches on fewer threads, and a build without the feature runs the
/// whole plan in the caller. Every shard reads the batch and writes its own rows, so the plan
/// changes nothing about the result: it is the same bits for any count, which
/// `the_parallel_and_the_serial_path_produce_the_same_bits` asserts.
///
/// A host which runs the kernel on a pool of its own - a web worker per instance, a thread per
/// request - takes `1` and parallelizes outside, since a shard inside the pool of the host then
/// competes with the host's own work. The function is public for that host and for
/// `crates/core/examples/kernel_timing.rs`, which times the two paths against each other from
/// outside the crate and therefore cannot reach a private one.
///
/// # Errors
///
/// Whatever [`analyze_batch`] refuses, which is every error of the analysis: the plan itself is
/// no error, since there is no count which is not a plan.
pub fn analyze_batch_in_shards(
    batch: &Batch,
    options: &NcaOptions,
    shards: usize,
) -> Result<AnalysisResult, KernelError> {
    check_contract(&batch.contract, BATCH_CONTRACT)?;
    batch.validate().map_err(KernelError::Batch)?;
    options.validate().map_err(KernelError::Options)?;
    refuse_what_is_not_carried(batch, options)?;
    analyze_checked(batch, options, shards)
}

/// The analysis of a batch whose contract, shape, options and refusals are already checked.
///
/// This is everything [`analyze_batch_in_shards`] does after its checks, so that a test can run
/// a batch on a chosen shard plan.
fn analyze_checked(
    batch: &Batch,
    options: &NcaOptions,
    shards: usize,
) -> Result<AnalysisResult, KernelError> {
    let n_samples = batch.n_samples();
    let rows_of_batch = Rows {
        batch,
        options,
        manual: manual_columns(batch, options)?,
        windows: sample_windows(batch, options)?,
        has_dose: batch.max_doses > 0,
    };
    // only an analysis of a single row reports the windows the selection could have chosen from
    // (`candidate_variables` of `nca.py`), so only such a row is asked for them, and such a row
    // is never a shard of a batch
    let mut candidates: Vec<Candidate> = Vec::new();
    let rows = if options.terminal.keep_candidates && n_samples == 1 {
        let mut scratch = Scratch::new(batch.n_time);
        vec![rows_of_batch.row(0, &mut scratch, Some(&mut candidates))]
    } else {
        rows_of(n_samples, batch.n_time, shards, |row, scratch| {
            rows_of_batch.row(row, scratch, None)
        })
    };

    let result = result_of(batch, options, &rows, &candidates)?;
    // unreachable after `batch.validate`, which is what every length here comes from, and kept
    // because a result the host cannot read is worth a named error rather than a surprise. The
    // batch validated, so the error names this build and not the input.
    result.validate().map_err(KernelError::Internal)?;
    Ok(result)
}

/// The rows of one batch, with everything every row of it needs besides the row itself.
///
/// What is per batch is computed once here and what is per row in [`Rows::row`], which is what a
/// shard calls and the only thing it calls: the struct is read only, so the shards of a batch
/// share it without synchronizing and without copying the batch.
struct Rows<'a> {
    /// The batch.
    batch: &'a Batch,
    /// The options every row is analysed with.
    options: &'a NcaOptions,
    /// The columns `TerminalMethod::Manual` selects, the same for every row.
    manual: Option<Vec<bool>>,
    /// The terminal window of every row, where the options give one.
    windows: Option<RowWindows>,
    /// Whether the batch carries dose arrays at all.
    has_dose: bool,
}

impl Rows<'_> {
    /// The analysis of one row, through the buffers of `scratch`, which it overwrites.
    ///
    /// `candidates`, where it is given, is filled with the windows the terminal selection could
    /// have chosen from; only the single row of a single row batch is ever asked for them.
    fn row(
        &self,
        row: usize,
        scratch: &mut Scratch,
        candidates: Option<&mut Vec<Candidate>>,
    ) -> RowResult {
        let batch = self.batch;
        let route = batch.route_of_row(row);
        // a row of a protocol of several doses, or of an explicit `tau`, is analysed over its
        // dosing intervals from its last dose on (`is_multiple_dose` of `nca.py`); every other
        // row is a single dose curve from its first
        let single_dose = single_dose_of_row(batch, self.options, row);
        let input = RowInput {
            times: row_slice(&batch.times, row, batch.n_time),
            values: row_slice(&batch.values, row, batch.n_time),
            dose: row_dose(batch, row, !single_dose),
            route,
            has_dose: self.has_dose,
            lloq: batch.lloq.as_ref().and_then(|v| v.get(row).copied()),
            window: self
                .windows
                .as_ref()
                .and_then(|w| w.get(row).copied().flatten()),
            manual: self.manual.as_deref(),
            single_dose,
            dose_times: row_slice(&batch.dose_time, row, batch.max_doses),
            dose_amounts: row_slice(&batch.dose_amount, row, batch.max_doses),
        };
        if single_dose {
            let mut parameters = compute_row(&input, self.options, scratch, candidates);
            // the protocol of a row is reported whichever path the row took (`run_analysis_rows`
            // of `nca.py` fills `n_doses` for every row of a batch which has the column)
            parameters.n_doses = dose_count(input.dose_times) as f64;
            return RowResult {
                route,
                dose_amount: input.dose.amount,
                parameters,
                intervals: Vec::new(),
            };
        }
        // the intervals are always computed, since the flags are raised from them; only their
        // emission follows `NcaOptions::intervals` (`if options.intervals` of python)
        let mut intervals = Vec::new();
        let parameters =
            compute_steady_state(&input, self.options, scratch, candidates, &mut intervals);
        if !self.options.intervals {
            intervals.clear();
        }
        RowResult {
            route,
            dose_amount: input.dose.amount,
            parameters,
            intervals,
        }
    }
}

/// Why this build refuses a batch under these options, `None` when nothing here refuses it.
///
/// This is the one statement of the refusals which can be read off a batch and its options,
/// without analysing anything: the effect curve (`Kind::Effect`), the acceptance criteria, the
/// named partial areas, the propagation of the uncertainty of a group curve and the unit
/// conversion of a result. It is what [`analyze_batch`] refuses on and what `is_supported` of
/// the python binding answers, so a host which asks first and the analysis cannot disagree
/// (`the_predicate_and_the_analysis_agree_on_every_refusal`). The text is the `what` of
/// [`KernelError::Unsupported`], and the conditions are tried in the order below, so a batch
/// which trips several is named by the first.
///
/// A protocol of more than one dose and an explicit `NcaOptions::tau` are no refusal: they are
/// the multiple dose analysis (`is_multiple_dose` and `compute_steady_state` of `nca.py`).
///
/// The predicate reads nothing of the shape of the batch, so it is safe on one which
/// [`Batch::validate`] has not seen. It also cannot see the one refusal which needs the
/// registry of `pkpdutils-units`, a unit string it cannot read: answering that means deriving
/// the unit of every parameter, which is the analysis, so it is reported where that happens
/// (`crate::assemble::unit_of`). `None` is therefore "nothing here refuses it" and not a promise
/// that the analysis runs.
#[must_use]
pub fn unsupported_reason(batch: &Batch, options: &NcaOptions) -> Option<String> {
    let reason = |what: &str| Some(what.to_string());
    if options.kind == Kind::Effect {
        return reason("the analysis of an effect curve (Kind::Effect)");
    }
    if options.acceptance.any_threshold() {
        return reason("the acceptance criteria (NcaOptions::acceptance)");
    }
    if !options.partial_aucs.is_empty() {
        return reason("the named partial areas (NcaOptions::partial_aucs)");
    }
    // a batch which carries a spread resolves to the bootstrap on its own, so the refusal is of
    // what the options resolve to and not of what they say (`resolve_uncertainty` of `options.py`)
    if resolved_uncertainty(batch, options) != UncertaintyMethod::None {
        return reason("the propagation of the uncertainty of a group curve");
    }
    if !options.units.is_empty() {
        return reason("the unit conversion of a result (NcaOptions::units)");
    }
    None
}

/// Refuse what this build does not carry, naming it: [`unsupported_reason`] as an error.
///
/// The refusals decided from the batch and the options are all in that one function. The
/// refusal of a unit string the registry cannot read is not added here, contrary to what one
/// might expect: it is raised while the result is assembled (`crate::assemble::unit_of`), the
/// one place which derives the units.
fn refuse_what_is_not_carried(batch: &Batch, options: &NcaOptions) -> Result<(), KernelError> {
    match unsupported_reason(batch, options) {
        Some(what) => Err(KernelError::Unsupported { what }),
        None => Ok(()),
    }
}

/// The doses of a row: the columns whose amount and whose time are both a number.
///
/// This is the count of `reference_dose` and of `is_multiple_dose` of `nca.py`, over every
/// column of the row. `pkpdutils_abi::Batch::n_doses_of_row` counts something else - the leading
/// amounts which are not `NaN` - so a protocol with a hole in it is a different protocol to it;
/// the divergence is recorded against that crate and not worked around here.
fn n_doses_of_row(batch: &Batch, row: usize) -> usize {
    let amounts = row_slice(&batch.dose_amount, row, batch.max_doses);
    let times = row_slice(&batch.dose_time, row, batch.max_doses);
    amounts
        .iter()
        .zip(times)
        .filter(|(amount, time)| amount.is_finite() && time.is_finite())
        .count()
}

/// Whether the dose a row is analysed against is the only one of its curve.
///
/// A protocol of fewer than two doses is a single dose curve, which starts at its dose, so an
/// infusion or an oral dose gets the zero inserted there. An explicit `NcaOptions::tau` says the
/// curve is an interval of a regimen, which has earlier doses whether or not the protocol lists
/// them (a steady state interval is described by one dose and its `tau`), so it makes every row a
/// multiple dose row on its own. This is the one place the rule lives.
fn single_dose_of_row(batch: &Batch, options: &NcaOptions, row: usize) -> bool {
    n_doses_of_row(batch, row) < 2 && options.tau.is_none()
}

/// The dose of a row its analysis runs against: the first of its protocol for a single dose
/// curve and the last (`last`) for a multiple dose row, as `reference_dose` of `nca.py`.
///
/// The durations are read whenever the batch has doses at all, which is what python's
/// `dose_duration is not None` means: `Timecourses` writes `dose_duration` onto every batch with
/// doses and leaves it `NaN` where the route is no infusion.
fn row_dose(batch: &Batch, row: usize, last: bool) -> ReferenceDose {
    let durations =
        (batch.max_doses > 0).then(|| row_slice(&batch.dose_duration, row, batch.max_doses));
    reference_dose(
        row_slice(&batch.dose_amount, row, batch.max_doses),
        row_slice(&batch.dose_time, row, batch.max_doses),
        durations,
        last,
    )
}

/// One row of a flat array of `width` entries per sample, empty where it does not fit.
///
/// The product of a validated batch is the length of the array, so the empty slice is
/// unreachable; it is here because an overflowing index is a trap and a short row is not.
fn row_slice(values: &[f64], row: usize, width: usize) -> &[f64] {
    let Some(start) = row.checked_mul(width) else {
        return &[];
    };
    let Some(end) = start.checked_add(width) else {
        return &[];
    };
    values.get(start..end).unwrap_or(&[])
}

/// The columns `TerminalMethod::Manual` selects, over the columns of a row.
///
/// The indices of `TerminalPhase::points` are indices into the curve as the host sent it, a
/// negative one counting from the end as numpy counts it, and the mask they make is the same for
/// every row; what differs per row is where those columns land once the row is packed, which
/// [`parameters::compute_row`] resolves.
///
/// # Errors
///
/// [`KernelError::Options`] for an index which is no column of the row, which python answers
/// with an `IndexError`.
fn manual_columns(batch: &Batch, options: &NcaOptions) -> Result<Option<Vec<bool>>, KernelError> {
    if options.terminal.method != TerminalMethod::Manual {
        return Ok(None);
    }
    let points = options.terminal.points.as_deref().unwrap_or(&[]);
    let n_time = batch.n_time;
    let mut mask = vec![false; n_time];
    for &index in points {
        let resolved = if index < 0 {
            i64::try_from(n_time).unwrap_or(i64::MAX).checked_add(index)
        } else {
            Some(index)
        };
        let column = resolved
            .and_then(|i| usize::try_from(i).ok())
            .filter(|&i| i < n_time)
            .ok_or_else(|| {
                KernelError::Options(format!(
                    "the terminal point {index} is no column of a curve of {n_time} points"
                ))
            })?;
        mask[column] = true;
    }
    Ok(Some(mask))
}

/// The terminal window of every row of a batch, `None` where a row has none.
type RowWindows = Vec<Option<(f64, f64)>>;

/// The terminal window of every row, `None` for a batch which gives none.
///
/// `TerminalPhase::windows` is keyed by the label of the sample: the value of the dimension
/// coordinate of a batch with one sample dimension, the tuple of them for a batch with several,
/// and the string `"*"` for every sample the mapping does not name. A dimension without a
/// coordinate is labelled by the position of the sample along it, as python labels it.
///
/// # Errors
///
/// [`KernelError::Options`] for a key which names no sample of the batch, which python answers
/// with a `ValueError`.
fn sample_windows(batch: &Batch, options: &NcaOptions) -> Result<Option<RowWindows>, KernelError> {
    let Some(windows) = &options.terminal.windows else {
        return Ok(None);
    };
    if windows.is_empty() {
        return Ok(None);
    }
    let labels = sample_labels(batch);
    for key in windows.keys() {
        let named = matches!(key, WindowLabel::Str(name) if name == WINDOW_DEFAULT_KEY);
        if !named && !labels.iter().any(|label| same_label(label, key)) {
            return Err(KernelError::Options(format!(
                "the terminal window of {key} names no sample of the batch"
            )));
        }
    }
    let default = windows
        .iter()
        .find(|(key, _)| matches!(key, WindowLabel::Str(name) if name == WINDOW_DEFAULT_KEY))
        .map(|(_, window)| *window);
    let of_row = |label: &WindowLabel| {
        windows
            .iter()
            .find(|(key, _)| same_label(label, key))
            .map(|(_, window)| *window)
            .or(default)
            .filter(|(first, last)| first.is_finite() && last.is_finite())
    };
    Ok(Some(labels.iter().map(of_row).collect()))
}

/// The key of `TerminalPhase::windows` which names every sample the mapping does not.
const WINDOW_DEFAULT_KEY: &str = "*";

/// Whether a sample label and a key of the window mapping are the same label.
///
/// `TerminalPhase.windows` is a python `dict` and the check of its keys a python `set`, so what
/// "the same label" means is what `==` means there, and python compares the numbers of its
/// numeric tower across their types: `1 == 1.0`, `True == 1` and `False == 0.0` are all true, and
/// a `dict` keyed `1` is read by `1.0` because the two also hash alike. A column of a batch
/// carries one dtype, so a float coordinate of `[1.0, 2.0]` with a window keyed `1` is a batch
/// `pkpdutils` 1.4.0 analyses and nothing on the host side has anything to flag; this comparison
/// is where the two could disagree, so this is where they are held together
/// (`a_numeric_label_matches_a_key_of_another_type_as_python_does`).
///
/// Numbers are therefore compared by value, with a boolean as the `1` and `0` python makes of it,
/// and exactly: python compares an `int` and a `float` without rounding the `int` to a double, so
/// the float is taken to the integers and not the other way round ([`int_is_float`]). A `NaN`
/// label equals nothing, itself included, as in python.
///
/// `WindowLabel` tells a date and a datetime from a string, where a coordinate of a batch has no
/// dtype for either and carries both as text, so a date key is matched against the text of the
/// coordinate. Text, `None` and a tuple compare as themselves, element by element.
fn same_label(label: &WindowLabel, key: &WindowLabel) -> bool {
    match (label, key) {
        (WindowLabel::Str(text), WindowLabel::Date(key) | WindowLabel::Datetime(key)) => {
            text == key
        }
        (WindowLabel::Tuple(label), WindowLabel::Tuple(key)) => {
            label.len() == key.len() && label.iter().zip(key).all(|(l, k)| same_label(l, k))
        }
        _ => match (numeric_label(label), numeric_label(key)) {
            (Some(label), Some(key)) => same_number(label, key),
            // not both numbers: a number is no string, no date and no `None`, exactly as in
            // python, and two labels of the same kind compare as themselves
            _ => label == key,
        },
    }
}

/// A label as the number python would compare, `None` for a label which is no number.
///
/// A boolean is a number in python, `True` being `1` and `False` being `0`, and it is a label of
/// a batch because a coordinate may be a boolean column.
fn numeric_label(label: &WindowLabel) -> Option<Number> {
    match label {
        WindowLabel::Int(value) => Some(Number::Int(*value)),
        WindowLabel::Bool(value) => Some(Number::Int(i64::from(*value))),
        WindowLabel::Float(value) => Some(Number::Float(*value)),
        WindowLabel::Tuple(_)
        | WindowLabel::None
        | WindowLabel::Datetime(_)
        | WindowLabel::Date(_)
        | WindowLabel::Str(_) => None,
    }
}

/// One label of [`numeric_label`]: a python `int` (a boolean among them) or a python `float`.
#[derive(Clone, Copy)]
enum Number {
    /// An integer, and a boolean as the integer python makes of it.
    Int(i64),
    /// A float.
    Float(f64),
}

/// Whether two numeric labels are equal as python's `==` has them.
fn same_number(label: Number, key: Number) -> bool {
    match (label, key) {
        (Number::Int(label), Number::Int(key)) => label == key,
        // `NaN` equals nothing, itself included, which is what the float comparison already does
        (Number::Float(label), Number::Float(key)) => label == key,
        (Number::Int(int), Number::Float(float)) | (Number::Float(float), Number::Int(int)) => {
            int_is_float(int, float)
        }
    }
}

/// Whether an integer and a float are the same number, as python compares the two.
///
/// Python compares an `int` and a `float` exactly and does not round the `int` to a double, so
/// `10**17 + 1 == 1e17` is false there although both sides are the same `f64`. The float is
/// therefore taken to the integers and never the other way round, and a float which is not an
/// integer, which is not finite or which is outside the range of an `i64` is equal to no `i64`:
/// `fract` of an infinity and of a `NaN` is a `NaN`, which fails the first test, and the range
/// is checked before the cast, which saturates.
fn int_is_float(int: i64, float: f64) -> bool {
    // `i64::MIN` is a power of two and exact as an `f64`, and so is its negation, which is one
    // past `i64::MAX` and therefore an exclusive bound
    let min = i64::MIN as f64;
    float.fract() == 0.0 && float >= min && float < -min && float as i64 == int
}

/// The label of every sample of a batch, in the row order of the analysis.
///
/// Mirrors `sample_keys` of `src/pkpdutils/nca/nca.py`.
fn sample_labels(batch: &Batch) -> Vec<WindowLabel> {
    let per_dim: Vec<Vec<WindowLabel>> = batch
        .sample_dims
        .iter()
        .zip(&batch.sample_shape)
        .map(|(dim, &size)| dimension_labels(batch, dim, size))
        .collect();
    let mut labels: Vec<Vec<WindowLabel>> = vec![Vec::new()];
    for values in &per_dim {
        labels = labels
            .iter()
            .flat_map(|prefix| {
                values.iter().map(move |value| {
                    let mut label = prefix.clone();
                    label.push(value.clone());
                    label
                })
            })
            .collect();
    }
    labels
        .into_iter()
        .map(|mut label| {
            if label.len() == 1 {
                label.remove(0)
            } else {
                WindowLabel::Tuple(label)
            }
        })
        .collect()
}

/// The labels along one sample dimension, the positions where it has no coordinate.
fn dimension_labels(batch: &Batch, dim: &str, size: usize) -> Vec<WindowLabel> {
    let coordinate = batch
        .coords
        .iter()
        .find(|c| c.name == dim && c.dims.len() == 1 && c.dims[0] == dim);
    match coordinate.map(|c| &c.values) {
        Some(Coordinate::Str(values)) => values.iter().cloned().map(WindowLabel::Str).collect(),
        Some(Coordinate::Int(values)) => values.iter().copied().map(WindowLabel::Int).collect(),
        Some(Coordinate::Float(values)) => values.iter().copied().map(WindowLabel::Float).collect(),
        Some(Coordinate::Bool(values)) => values.iter().copied().map(WindowLabel::Bool).collect(),
        None => (0..size)
            .map(|i| WindowLabel::Int(i64::try_from(i).unwrap_or(i64::MAX)))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use pkpdutils_abi::{NcaFlag, Route, SampleCoordinate, TerminalPhase};

    use super::*;

    #[test]
    fn a_row_is_single_dose_with_fewer_than_two_doses_and_no_tau() {
        use crate::parameters::tests::{dosed, linear, many, regimen};
        let curve = || many(&[&[1.0, 2.0, 3.0]], &[&[4.0, 2.0, 1.0]]);
        let with_tau = NcaOptions {
            tau: Some(24.0),
            ..linear()
        };
        let one_dose = dosed(curve(), Route::Oral, 100.0, 0.0);
        let two_doses = regimen(curve(), Route::Oral, 100.0, &[0.0, 24.0]);

        assert!(single_dose_of_row(&one_dose, &linear(), 0));
        assert!(!single_dose_of_row(&one_dose, &with_tau, 0));
        assert!(!single_dose_of_row(&two_doses, &linear(), 0));
        assert!(!single_dose_of_row(&two_doses, &with_tau, 0));
    }

    /// The analysis of a batch through the public entry point a host calls.
    #[track_caller]
    fn analyze_ok(batch: &Batch, options: &NcaOptions) -> AnalysisResult {
        analyze_batch(batch, options).expect("the batch is analysable")
    }

    fn flags_of(result: &AnalysisResult, row: usize) -> NcaFlag {
        NcaFlag::from_bits_truncate(result.flags[row])
    }

    fn interval_names(result: &AnalysisResult) -> Vec<&str> {
        result
            .point_variables
            .iter()
            .map(|v| v.name.as_str())
            .filter(|name| name.starts_with("interval_"))
            .collect()
    }

    #[test]
    fn a_bolus_post_dose_interval_raises_its_flag_whether_or_not_the_intervals_are_emitted() {
        // Carried from Task 6. python raises EXTRAPOLATED_TROUGH from the intervals it computes
        // unconditionally and gates only the emission (`if options.intervals`). For this row
        // python gives flags 1540 = EXTRAPOLATION_HIGH | EXTRAPOLATED_TROUGH | SPAN_LOW with and without the output.
        // A port which computes the intervals only when they are emitted drops the flag
        use crate::parameters::tests::{one, regimen};
        let t = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 13.0, 16.0, 24.0];
        let c = [10.0, 8.0, 6.4, 4.1, 1.7, 9.0, 7.0, 5.0, 2.0];
        let batch = regimen(one(&t, &c), Route::IvBolus, 10.0, &[0.0, 12.0]);
        let want = NcaFlag::EXTRAPOLATION_HIGH | NcaFlag::SPAN_LOW | NcaFlag::EXTRAPOLATED_TROUGH;
        for emit in [false, true] {
            let options = NcaOptions {
                intervals: emit,
                ..NcaOptions::default()
            };
            let result = analyze_ok(&batch, &options);
            assert_eq!(flags_of(&result, 0), want, "intervals: {emit}");
            // the steady state parameters follow from the intervals either way
            let at = |name| crate::parameters::tests::at(&result, name, 0);
            assert_eq!(at("auc_tau"), 51.98279878916171);
            assert_eq!(at("ctrough"), 2.0);
            // the emission is what the option decides
            assert_eq!(
                interval_names(&result).is_empty(),
                !emit,
                "intervals: {emit}"
            );
        }
    }

    #[test]
    fn an_incomplete_last_interval_raises_its_flag_whether_or_not_the_intervals_are_emitted() {
        // python flags 274 = INCOMPLETE_INTERVAL | NO_ABSORPTION | TOO_FEW_POINTS for this row
        // with and without the output, and the area of the last interval is NaN
        use crate::parameters::tests::{one, regimen};
        let t = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 13.0, 16.0];
        let c = [10.0, 8.0, 6.4, 4.1, 1.7, 9.0, 7.0, 5.0];
        let batch = regimen(one(&t, &c), Route::Oral, 10.0, &[0.0, 12.0]);
        let want = NcaFlag::INCOMPLETE_INTERVAL | NcaFlag::NO_ABSORPTION | NcaFlag::TOO_FEW_POINTS;
        for emit in [false, true] {
            let options = NcaOptions {
                intervals: emit,
                ..NcaOptions::default()
            };
            let result = analyze_ok(&batch, &options);
            assert_eq!(flags_of(&result, 0), want, "intervals: {emit}");
            assert!(crate::parameters::tests::at(&result, "auc_tau", 0).is_nan());
            assert_eq!(
                interval_names(&result).is_empty(),
                !emit,
                "intervals: {emit}"
            );
        }
    }

    /// The batch of the mixed protocols: a single dose row, a three dose row, a single dose row
    /// and a three dose row, over the same 16 times. The curves are python's.
    fn mixed_batch() -> Batch {
        use crate::parameters::tests::many;
        let times = [
            0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 13.0, 14.0, 16.0, 20.0, 24.0, 25.0, 26.0, 28.0, 32.0,
            36.0,
        ];
        let rows: [[f64; 16]; 4] = [
            [
                0.0, 6.872, 7.16, 5.156, 2.33, 1.047, 0.857, 0.702, 0.47, 0.211, 0.095, 0.078,
                0.064, 0.043, 0.019, 0.009,
            ],
            [
                0.0, 6.872, 7.16, 5.156, 2.33, 1.047, 7.729, 7.862, 5.626, 2.541, 1.142, 7.807,
                7.925, 5.669, 2.56, 1.15,
            ],
            [
                0.0, 12.224, 14.59, 12.425, 6.972, 3.828, 3.295, 2.836, 2.101, 1.153, 0.633, 0.545,
                0.469, 0.347, 0.191, 0.105,
            ],
            [
                0.0, 3.436, 3.58, 2.578, 1.165, 0.524, 3.864, 3.931, 2.813, 1.27, 0.571, 3.904,
                3.962, 2.834, 1.28, 0.575,
            ],
        ];
        let n = f64::NAN;
        let batch = many(
            &[&times, &times, &times, &times],
            &[&rows[0], &rows[1], &rows[2], &rows[3]],
        );
        Batch {
            max_doses: 3,
            dose_amount: vec![10.0, n, n, 10.0, 20.0, 30.0, 20.0, n, n, 5.0, 5.0, 5.0],
            dose_time: vec![0.0, n, n, 0.0, 12.0, 24.0, 0.0, n, n, 0.0, 12.0, 24.0],
            dose_duration: vec![n; 12],
            dose_unit: Some("mg".to_string()),
            route: Some(Route::Oral),
            ..batch
        }
    }

    /// `mixed_batch` with the protocol of the third row cut to a hole: two dose times, 0 and 12,
    /// of which only the first has an amount (20). The kernel reads one dose there, as
    /// `is_multiple_dose` of `nca.py` does, and the protocol it reports is of two.
    fn mixed_batch_with_a_hole() -> Batch {
        let n = f64::NAN;
        let batch = mixed_batch();
        Batch {
            dose_amount: vec![10.0, n, n, 10.0, 20.0, 30.0, 20.0, n, n, 5.0, 5.0, 5.0],
            dose_time: vec![0.0, n, n, 0.0, 12.0, 24.0, 0.0, 12.0, n, 0.0, 12.0, 24.0],
            ..batch
        }
    }

    /// One parameter column of a result.
    #[track_caller]
    fn column_of(result: &AnalysisResult, name: &str) -> Vec<f64> {
        result
            .parameters
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("no column {name}"))
            .values
            .clone()
    }

    #[test]
    fn a_protocol_of_several_doses_is_no_longer_refused() {
        // The end to end coverage Task 2 deferred: a batch in, a result out, through the entry
        // point a host calls. Every number is python's `nca` (the python path, `_force_python`)
        // on the same four rows: single, three doses, a hole (dose times 0 and 12, amounts 20
        // and NaN) and three doses. The hole row is the one shape which tells the two counting
        // rules apart: the pair rule of `is_multiple_dose` (amount and time) makes it a single
        // dose row, the time rule of `dose_counts` (time only) reports its protocol as 2 doses.
        // A port which analysed it on the multiple dose path would give `auc_tau` and no `cl_f`
        let n = f64::NAN;
        let batch = mixed_batch_with_a_hole();
        let result = analyze_batch(
            &batch,
            &NcaOptions {
                intervals: true,
                ..NcaOptions::default()
            },
        )
        .expect("a protocol of several doses is analysed");
        let column = |name: &str| column_of(&result, name);

        assert_same("n_doses", &column("n_doses"), &[1.0, 3.0, 2.0, 3.0]);
        assert_same("tau", &column("tau"), &[n, 12.0, n, 12.0]);
        assert_same("cmax", &column("cmax"), &[7.16, 7.925, 14.59, 3.962]);
        assert_same("tmax", &column("tmax"), &[2.0; 4]);
        // the single dose rows sum 16 segments, which numpy adds in its own order (`close`)
        assert_near(
            "auc_last",
            &column("auc_last"),
            &[
                48.49655132182141,
                48.49926973218703,
                130.028021765316,
                24.248095168680027,
            ],
        );
        assert_near(
            "auc_inf_obs",
            &column("auc_inf_obs"),
            &[
                48.54171396268703,
                54.26638800635439,
                130.7299413571588,
                27.131973224308183,
            ],
        );
        assert_near(
            "cl_f",
            &column("cl_f"),
            &[0.2060083829690642, n, 0.15298714121931178, n],
        );
        assert_near(
            "vz_f",
            &column("vz_f"),
            &[1.0337647350376231, n, 1.022711159255767, n],
        );
        assert_same("cmax_dn", &column("cmax_dn"), &[0.716, n, 0.7295, n]);
        assert_near(
            "lambda_z",
            &column("lambda_z"),
            &[
                0.19927975484825056,
                0.19940634912087601,
                0.14958978381603016,
                0.19938429743165942,
            ],
        );
        assert_same(
            "auc_tau",
            &column("auc_tau"),
            &[n, 48.49926973218703, n, 24.248095168680027],
        );
        assert_same("ctrough", &column("ctrough"), &[n, 1.15, n, 0.575]);
        assert_same(
            "cavg",
            &column("cavg"),
            &[n, 4.041605811015586, n, 2.0206745973900024],
        );
        assert_same(
            "accumulation_ratio",
            &column("accumulation_ratio"),
            &[n, 1.1005537656433146, n, 1.100583054453734],
        );
        assert_same(
            "cl_ss_f",
            &column("cl_ss_f"),
            &[n, 0.6185660148216664, n, 0.20620176410633004],
        );
        let interval = |name: &str| -> Vec<f64> {
            result
                .point_variables
                .iter()
                .find(|v| v.name == name)
                .unwrap_or_else(|| panic!("no variable {name}"))
                .values
                .clone()
        };
        assert_same(
            "interval_start",
            &interval("interval_start"),
            &[n, n, n, 0.0, 12.0, 24.0, n, n, n, 0.0, 12.0, 24.0],
        );
        assert_same(
            "interval_dose",
            &interval("interval_dose"),
            &[n, n, n, 10.0, 20.0, 30.0, n, n, n, 5.0, 5.0, 5.0],
        );
        assert_same(
            "interval_auc",
            &interval("interval_auc"),
            &[
                n,
                n,
                n,
                43.305541919868325,
                48.06909355615606,
                48.49926973218703,
                n,
                n,
                n,
                21.654100503512577,
                24.032190218178513,
                24.248095168680027,
            ],
        );
        for row in 0..4 {
            assert_eq!(flags_of(&result, row), NcaFlag::NONE, "row {row}");
        }
    }

    #[test]
    fn an_explicit_tau_is_no_longer_refused() {
        // Python's `nca` with `tau=12` on the last interval of the fourth curve alone (the
        // times 24 to 36, one dose of 5 at 24): the curve is an interval of a regimen, so it is
        // analysed over its interval and not as a single dose curve (no `cl_f`, no `cmax_dn`),
        // although its protocol is one dose. A port which only lifted the refusal and kept the
        // row on the single dose path would give `cl_f` and no `auc_tau`
        use crate::parameters::tests::{dosed, one};
        let mixed = mixed_batch();
        let times = &mixed.times[10..16];
        let values = &mixed.values[3 * 16 + 10..3 * 16 + 16];
        let batch = dosed(one(times, values), Route::Oral, 5.0, 24.0);
        let options = NcaOptions {
            tau: Some(12.0),
            intervals: true,
            ..NcaOptions::default()
        };
        let result = analyze_batch(&batch, &options).expect("an explicit tau is analysed");
        let at = |name| crate::parameters::tests::at(&result, name, 0);

        assert_eq!(at("n_doses"), 1.0);
        assert_eq!(at("tau"), 12.0);
        assert_eq!(at("cmax"), 3.962);
        assert_eq!(at("auc_tau"), 24.248095168680027);
        assert_eq!(at("ctrough"), 0.575);
        assert_eq!(at("cavg"), 2.0206745973900024);
        assert_eq!(at("accumulation_ratio"), 1.100583054453734);
        assert_eq!(at("cl_ss_f"), 0.20620176410633004);
        assert_eq!(at("auc_tau_extrap_fraction"), 0.0);
        assert!(at("cl_f").is_nan() && at("cmax_dn").is_nan());
        crate::parameters::tests::close(at("lambda_z"), 0.19938429743165942);
        assert_eq!(flags_of(&result, 0), NcaFlag::NONE);
    }

    #[test]
    fn an_effect_curve_is_still_refused() {
        // out of scope for this module: an effect curve is a different parameter set, and
        // lifting the two refusals of the multiple dose analysis must not make it reachable,
        // on a multiple dose batch or with a tau
        let effect = NcaOptions {
            kind: Kind::Effect,
            ..NcaOptions::default()
        };
        let with_tau = NcaOptions {
            tau: Some(12.0),
            ..effect.clone()
        };
        for (batch, options) in [
            (mixed_batch(), &effect),
            (mixed_batch(), &with_tau),
            (crate::parameters::tests::oral_curve(), &effect),
        ] {
            let error = analyze_batch(&batch, options).unwrap_err();
            assert!(
                matches!(&error, KernelError::Unsupported { what } if what.contains("effect")),
                "{error}"
            );
        }
    }

    #[test]
    fn the_predicate_and_the_analysis_agree_on_every_refusal() {
        // Whatever `unsupported_reason` names, `analyze_batch` refuses with the same text, and
        // whatever it passes, `analyze_batch` analyses: one statement of the refusals, so a host
        // which asks first and the analysis cannot drift apart. The grid is every refusal alone,
        // every pair with the effect curve (which has to win, being first), and the protocols
        // which used to be refused, over a single dose batch, a multiple dose batch, the hole
        // and a group curve
        use std::collections::BTreeMap;

        use pkpdutils_abi::Acceptance;

        let single = crate::parameters::tests::oral_curve();
        let multiple = mixed_batch();
        let hole = mixed_batch_with_a_hole();
        let spread = Batch {
            sd: Some(vec![0.1; single.values.len()]),
            ..single.clone()
        };
        let spread_multiple = Batch {
            sd: Some(vec![0.1; multiple.values.len()]),
            ..multiple.clone()
        };

        let base = NcaOptions::default();
        let variants: Vec<(&str, NcaOptions)> = vec![
            ("default", base.clone()),
            (
                "effect",
                NcaOptions {
                    kind: Kind::Effect,
                    ..base.clone()
                },
            ),
            (
                "tau",
                NcaOptions {
                    tau: Some(12.0),
                    ..base.clone()
                },
            ),
            (
                "acceptance",
                NcaOptions {
                    acceptance: Acceptance {
                        span_min: Some(3.0),
                        ..Acceptance::default()
                    },
                    ..base.clone()
                },
            ),
            (
                "partial",
                NcaOptions {
                    partial_aucs: BTreeMap::from([("early".to_string(), (0.0, 4.0))]),
                    ..base.clone()
                },
            ),
            (
                "units",
                NcaOptions {
                    units: BTreeMap::from([("auc_last".to_string(), "mg*hr/l".to_string())]),
                    ..base.clone()
                },
            ),
            (
                "bootstrap",
                NcaOptions {
                    uncertainty: Some(UncertaintyMethod::Bootstrap),
                    ..base.clone()
                },
            ),
            (
                "delta",
                NcaOptions {
                    uncertainty: Some(UncertaintyMethod::Delta),
                    ..base.clone()
                },
            ),
            (
                "uncertainty off",
                NcaOptions {
                    uncertainty: Some(UncertaintyMethod::None),
                    ..base.clone()
                },
            ),
            (
                "effect and units",
                NcaOptions {
                    kind: Kind::Effect,
                    units: BTreeMap::from([("auc_last".to_string(), "mg*hr/l".to_string())]),
                    ..base.clone()
                },
            ),
            (
                "units and partial",
                NcaOptions {
                    partial_aucs: BTreeMap::from([("early".to_string(), (0.0, 4.0))]),
                    units: BTreeMap::from([("auc_last".to_string(), "mg*hr/l".to_string())]),
                    ..base.clone()
                },
            ),
            (
                "tau and intervals off",
                NcaOptions {
                    tau: Some(12.0),
                    intervals: false,
                    ..base.clone()
                },
            ),
        ];
        let batches = [
            ("single", &single),
            ("multiple", &multiple),
            ("hole", &hole),
            ("spread", &spread),
            ("spread multiple", &spread_multiple),
        ];

        let mut refused = 0;
        let mut accepted = 0;
        for (batch_name, batch) in batches {
            for (options_name, options) in &variants {
                let case = format!("{batch_name} / {options_name}");
                let reason = unsupported_reason(batch, options);
                let outcome = analyze_batch(batch, options);
                match (reason, outcome) {
                    (None, Ok(_)) => accepted += 1,
                    (Some(what), Err(KernelError::Unsupported { what: refused_what })) => {
                        assert_eq!(what, refused_what, "{case}");
                        refused += 1;
                    }
                    (reason, outcome) => panic!(
                        "{case}: the predicate says {reason:?} and the analysis {:?}",
                        outcome.map(|_| "ran")
                    ),
                }
            }
        }
        // neither half is vacuous
        assert_eq!(accepted + refused, batches.len() * variants.len());
        assert!(accepted >= 10 && refused >= 10, "{accepted} / {refused}");

        // and each refusal is named by what it is, on a batch which trips nothing else: the
        // agreement above would hold just as well for a refusal both sides had dropped, so the
        // text is pinned here, including which of two refusals wins
        let expected: [(&str, Option<&str>); 12] = [
            ("default", None),
            ("effect", Some("effect curve")),
            ("tau", None),
            ("acceptance", Some("acceptance criteria")),
            ("partial", Some("partial areas")),
            ("units", Some("unit conversion")),
            ("bootstrap", Some("uncertainty of a group curve")),
            ("delta", Some("uncertainty of a group curve")),
            ("uncertainty off", None),
            ("effect and units", Some("effect curve")),
            ("units and partial", Some("partial areas")),
            ("tau and intervals off", None),
        ];
        for ((name, options), (expected_name, wanted)) in variants.iter().zip(expected) {
            assert_eq!(*name, expected_name);
            let reason = unsupported_reason(&single, options);
            match wanted {
                None => assert_eq!(reason, None, "{name}"),
                Some(text) => {
                    let reason = reason.unwrap_or_else(|| panic!("{name}: not refused"));
                    assert!(reason.contains(text), "{name}: {reason}");
                }
            }
        }
        // a group curve is refused by what the options resolve to, in both directions
        let automatic = unsupported_reason(&spread, &base).expect("the automatic choice");
        assert!(
            automatic.contains("uncertainty of a group curve"),
            "{automatic}"
        );
        assert_eq!(unsupported_reason(&spread, &variants[8].1), None);

        // the protocols which used to be refused are accepted, in every combination of options
        // which refuses nothing else
        for batch in [&multiple, &hole] {
            assert_eq!(unsupported_reason(batch, &base), None);
        }
        assert_eq!(unsupported_reason(&single, &variants[2].1), None);

        // the predicate reads no shape: a batch which does not validate is answered, not
        // indexed, and it is the analysis which names the shape
        let ragged = Batch {
            values: vec![1.0, 2.0],
            ..single.clone()
        };
        assert_eq!(unsupported_reason(&ragged, &base), None);
        assert!(matches!(
            analyze_batch(&ragged, &base),
            Err(KernelError::Batch(_))
        ));
    }

    /// `a == b` with `NaN` equal to `NaN`, element by element.
    #[track_caller]
    fn assert_same(name: &str, actual: &[f64], expected: &[f64]) {
        let equal = actual.len() == expected.len()
            && actual
                .iter()
                .zip(expected)
                .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()));
        assert!(equal, "{name}: {actual:?} is not {expected:?}");
    }

    /// `assert_same` to the tolerance of the point parameters of `compute_row`, which differ from
    /// numpy's in the last place for a regression whose sums are shifted (see `close`).
    #[track_caller]
    fn assert_near(name: &str, actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "{name}");
        for (a, e) in actual.iter().zip(expected) {
            if e.is_nan() {
                assert!(a.is_nan(), "{name}: {actual:?} is not {expected:?}");
            } else {
                crate::parameters::tests::close(*a, *e);
            }
        }
    }

    #[test]
    fn a_batch_of_mixed_protocols_keeps_each_row_in_its_own_path_and_in_input_order() {
        // Review Focus. The rows are single, multiple, single, multiple, and every number is
        // python's `run_analysis_rows`: a single dose row is NaN in every steady state variable
        // and every interval variable (`n_doses` is the exception, the protocol of the row,
        // python gives 1 for a single dose row), a multiple dose row is NaN in the single dose
        // quantities, and the rows come back in the order they went in - every column is
        // different per row, so a merge which concatenated the two paths would show
        let n = f64::NAN;
        let result = analyze_ok(
            &mixed_batch(),
            &NcaOptions {
                intervals: true,
                ..NcaOptions::default()
            },
        );
        let column = |name: &str| -> Vec<f64> {
            result
                .parameters
                .iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| {
                    panic!(
                        "no column {name}: {:?}",
                        crate::parameters::tests::names(&result)
                    )
                })
                .values
                .clone()
        };
        let interval = |name: &str| -> Vec<f64> {
            result
                .point_variables
                .iter()
                .find(|v| v.name == name)
                .unwrap_or_else(|| panic!("no variable {name}"))
                .values
                .clone()
        };
        // the point parameters of each row's own path
        assert_same("cmax", &column("cmax"), &[7.16, 7.925, 14.59, 3.962]);
        assert_near(
            "auc_inf_obs",
            &column("auc_inf_obs"),
            &[
                48.54171396268703,
                54.26638800635439,
                130.7299413571588,
                27.131973224308183,
            ],
        );
        // the single dose quantities: python's for the single rows, NaN for the multiple ones
        assert_near(
            "cl_f",
            &column("cl_f"),
            &[0.2060083829690642, n, 0.15298714121931178, n],
        );
        assert_near(
            "cl_f_pred",
            &column("cl_f_pred"),
            &[0.20601380649550355, n, 0.15298588273287622, n],
        );
        assert_near(
            "auc_inf_dn",
            &column("auc_inf_dn"),
            &[4.854171396268702, n, 6.5364970678579395, n],
        );
        assert_same("cmax_dn", &column("cmax_dn"), &[0.716, n, 0.7295, n]);
        // the steady state quantities: NaN for the single rows
        assert_same(
            "auc_tau",
            &column("auc_tau"),
            &[n, 48.49926973218703, n, 24.248095168680027],
        );
        assert_same(
            "auc_tau_extrap_fraction",
            &column("auc_tau_extrap_fraction"),
            &[n, 0.0, n, 0.0],
        );
        assert_same("ctrough", &column("ctrough"), &[n, 1.15, n, 0.575]);
        assert_same(
            "cavg",
            &column("cavg"),
            &[n, 4.041605811015586, n, 2.0206745973900024],
        );
        assert_same("cmax_ss", &column("cmax_ss"), &[n, 7.925, n, 3.962]);
        assert_same("tau", &column("tau"), &[n, 12.0, n, 12.0]);
        assert_same(
            "accumulation_ratio",
            &column("accumulation_ratio"),
            &[n, 1.1005537656433146, n, 1.100583054453734],
        );
        assert_same(
            "accumulation_ratio_obs",
            &column("accumulation_ratio_obs"),
            &[n, 1.1199321745454445, n, 1.1197923074545013],
        );
        assert_same(
            "cl_ss_f",
            &column("cl_ss_f"),
            &[n, 0.6185660148216664, n, 0.20620176410633004],
        );
        // the protocol of a row is reported whichever path it took
        assert_same("n_doses", &column("n_doses"), &[1.0, 3.0, 1.0, 3.0]);
        // an extravascular batch has no `cl_ss`, and no steady state name is missing
        assert!(!crate::parameters::tests::names(&result).contains(&"cl_ss"));
        // the interval variables: sample major over three intervals, NaN for the single rows
        assert_same(
            "interval_start",
            &interval("interval_start"),
            &[n, n, n, 0.0, 12.0, 24.0, n, n, n, 0.0, 12.0, 24.0],
        );
        assert_same(
            "interval_end",
            &interval("interval_end"),
            &[n, n, n, 12.0, 24.0, 36.0, n, n, n, 12.0, 24.0, 36.0],
        );
        assert_same(
            "interval_dose",
            &interval("interval_dose"),
            &[n, n, n, 10.0, 20.0, 30.0, n, n, n, 5.0, 5.0, 5.0],
        );
        assert_same(
            "interval_n_points",
            &interval("interval_n_points"),
            &[n, n, n, 6.0, 6.0, 6.0, n, n, n, 6.0, 6.0, 6.0],
        );
        assert_same(
            "interval_auc",
            &interval("interval_auc"),
            &[
                n,
                n,
                n,
                43.305541919868325,
                48.06909355615606,
                48.49926973218703,
                n,
                n,
                n,
                21.654100503512577,
                24.032190218178513,
                24.248095168680027,
            ],
        );
        assert_same(
            "interval_cmin",
            &interval("interval_cmin"),
            &[n, n, n, 0.0, 1.047, 1.142, n, n, n, 0.0, 0.524, 0.571],
        );
        for row in 0..4 {
            assert_eq!(flags_of(&result, row), NcaFlag::NONE, "row {row}");
        }
    }

    #[test]
    fn a_mixed_batch_keeps_its_row_order_whatever_the_shard_plan() {
        // Review Focus: the rows come back in input order. Twelve rows, every one different
        // (the four rows of `mixed_batch` in an uneven order, each scaled by its own factor),
        // run as 1, 2, 3, 5 and 12 shards: the result is the same bits for every plan, and the
        // single dose rows stay NaN in the steady state columns at their own positions. The
        // one-shard run is the python-pinned result of the test above for the unscaled rows
        use crate::parameters::tests::many;
        let base = mixed_batch();
        let order = [0usize, 1, 3, 2, 1, 0, 3, 3, 2, 1, 0, 2];
        let width = base.n_time;
        let rows: Vec<Vec<f64>> = order
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                base.values[r * width..(r + 1) * width]
                    .iter()
                    .map(|v| v * (1.0 + 0.05 * i as f64))
                    .collect()
            })
            .collect();
        let times: Vec<&[f64]> = order
            .iter()
            .map(|&r| &base.times[r * width..(r + 1) * width])
            .collect();
        let values: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let pick = |source: &Vec<f64>| -> Vec<f64> {
            order
                .iter()
                .flat_map(|&r| source[r * 3..(r + 1) * 3].to_vec())
                .collect()
        };
        let batch = Batch {
            max_doses: 3,
            dose_amount: pick(&base.dose_amount),
            dose_time: pick(&base.dose_time),
            dose_duration: vec![f64::NAN; 36],
            dose_unit: Some("mg".to_string()),
            route: Some(Route::Oral),
            ..many(&times, &values)
        };
        let options = NcaOptions {
            intervals: true,
            ..NcaOptions::default()
        };
        let run_in = |shards: usize| analyze_batch_in_shards(&batch, &options, shards).unwrap();
        let bits = |result: &AnalysisResult| -> Vec<u64> {
            let columns = result.parameters.iter().flat_map(|p| p.values.iter());
            let intervals = result.point_variables.iter().flat_map(|v| v.values.iter());
            columns
                .chain(intervals)
                .map(|v| v.to_bits())
                .chain(result.flags.iter().map(|&f| u64::from(f)))
                .collect()
        };
        let one = run_in(1);
        let multiple: Vec<bool> = order.iter().map(|&r| r == 1 || r == 3).collect();
        let tau = crate::parameters::tests::names(&one);
        assert!(tau.contains(&"tau"));
        for (row, &is_multiple) in multiple.iter().enumerate() {
            let at = |name| crate::parameters::tests::at(&one, name, row);
            assert_eq!(at("tau").is_nan(), !is_multiple, "row {row}");
            assert_eq!(at("n_doses"), if is_multiple { 3.0 } else { 1.0 });
        }
        for shards in [2, 3, 5, 12] {
            assert_eq!(bits(&run_in(shards)), bits(&one), "{shards} shards");
        }
    }

    #[test]
    fn a_batch_of_single_dose_rows_reports_no_steady_state_column() {
        // the steady state columns belong to a batch with a multiple dose row: the same batch
        // of the two single dose rows alone reports exactly the single dose parameters
        use crate::parameters::tests::names;
        let mixed = mixed_batch();
        let steady_names = [
            "auc_tau",
            "auc_tau_extrap_fraction",
            "cmin_ss",
            "cmax_ss",
            "ctrough",
            "cavg",
            "fluctuation",
            "swing",
            "fluctuation_tau",
            "swing_tau",
            "ptr",
            "accumulation_ratio",
            "accumulation_ratio_obs",
            "accumulation_ratio_cmax_obs",
            "accumulation_ratio_cmin_obs",
            "accumulation_ratio_ctrough_obs",
            "cl_ss_f",
            "n_doses",
            "tau",
        ];
        let result = analyze_ok(&mixed, &NcaOptions::default());
        for name in steady_names {
            assert!(names(&result).contains(&name), "{name}");
        }
        // rows 0 and 2 only: a batch which is all single dose
        let pick = |rows: [usize; 2]| Batch {
            sample_shape: vec![2],
            times: rows
                .iter()
                .flat_map(|&r| mixed.times[r * 16..(r + 1) * 16].to_vec())
                .collect(),
            values: rows
                .iter()
                .flat_map(|&r| mixed.values[r * 16..(r + 1) * 16].to_vec())
                .collect(),
            dose_amount: rows
                .iter()
                .flat_map(|&r| mixed.dose_amount[r * 3..(r + 1) * 3].to_vec())
                .collect(),
            dose_time: rows
                .iter()
                .flat_map(|&r| mixed.dose_time[r * 3..(r + 1) * 3].to_vec())
                .collect(),
            dose_duration: vec![f64::NAN; 6],
            coords: vec![SampleCoordinate {
                name: "subject".to_string(),
                dims: vec!["subject".to_string()],
                values: pkpdutils_abi::Coordinate::Str(vec!["A".to_string(), "B".to_string()]),
            }],
            ..mixed.clone()
        };
        let singles = analyze_ok(&pick([0, 2]), &NcaOptions::default());
        for name in steady_names {
            assert!(!names(&singles).contains(&name), "{name}");
        }
    }

    #[test]
    fn every_kernel_error_names_what_went_wrong() {
        assert_eq!(
            KernelError::Batch("no sample".to_string()).to_string(),
            "the batch is not analysable: no sample"
        );
        assert_eq!(
            KernelError::Options("tau is negative".to_string()).to_string(),
            "the options are not analysable: tau is negative"
        );
        assert_eq!(
            KernelError::Unsupported {
                what: "sparse sampling".to_string()
            }
            .to_string(),
            "sparse sampling is not implemented by this build"
        );
    }

    #[test]
    fn a_numeric_label_matches_a_key_of_another_type_as_python_does() {
        // `TerminalPhase.windows` is a python `dict` and the check of its keys a python `set`, so
        // a label and a key are the same label when python's `==` says so, and python compares
        // its numeric tower across the types: a float coordinate of `[1.0, 2.0]` with a window
        // keyed `1` is a batch `pkpdutils` 1.4.0 analyses. The column of a batch carries one
        // dtype, so nothing on the host side has anything to flag and this comparison is the
        // only place the two engines could disagree.
        let int = WindowLabel::Int;
        let float = WindowLabel::Float;
        let bool_label = WindowLabel::Bool;
        for (label, key) in [
            (float(1.0), int(1)),
            (int(1), float(1.0)),
            (bool_label(true), int(1)),
            (int(1), bool_label(true)),
            (bool_label(false), float(0.0)),
            (float(0.0), bool_label(false)),
            (float(-2.0), int(-2)),
            // exactly representable on both sides, and python says they are equal
            (float(1e17), int(100_000_000_000_000_000)),
            (int(i64::MIN), float(-9_223_372_036_854_775_808.0)),
            // and a label of the same kind, which compared equal before this one did
            (int(7), int(7)),
            (float(1.5), float(1.5)),
            (bool_label(true), bool_label(true)),
            (
                WindowLabel::Tuple(vec![float(1.0), int(2)]),
                WindowLabel::Tuple(vec![int(1), float(2.0)]),
            ),
        ] {
            assert!(same_label(&label, &key), "{label:?} is not {key:?}");
            assert!(same_label(&key, &label), "{key:?} is not {label:?}");
        }

        // and the other half, which a comparison made loose enough to fix the above could start
        // matching: every one of these is `False` in python too
        for (label, key) in [
            (int(2), bool_label(true)),
            (float(1.5), int(1)),
            (float(1.0), int(2)),
            // python compares an int and a float exactly and does not round the int to a double,
            // so this pair is `False` there although both sides are the same `f64`
            (int(100_000_000_000_000_001), float(1e17)),
            // beyond the range of an `i64`, where a saturating cast would match `i64::MAX`
            (int(i64::MAX), float(9.3e18)),
            (int(i64::MIN), float(-9.3e18)),
            // 2^63 exactly, the one float which separates an exclusive upper bound from an
            // inclusive one: it is one above `i64::MAX`, so python says it is no `i64` at all
            (int(i64::MAX), float(9_223_372_036_854_775_808.0)),
            (int(0), float(f64::INFINITY)),
            (int(0), float(f64::NEG_INFINITY)),
            // a `NaN` equals nothing, itself included, as in python
            (float(f64::NAN), float(f64::NAN)),
            (int(0), float(f64::NAN)),
            // a number is no string, no date, no `None` and no tuple
            (int(1), WindowLabel::Str("1".to_string())),
            (float(1.0), WindowLabel::Str("1.0".to_string())),
            (int(1), WindowLabel::None),
            (bool_label(true), WindowLabel::Str("True".to_string())),
            (int(1), WindowLabel::Tuple(vec![int(1)])),
            (
                WindowLabel::Tuple(vec![int(1), int(2)]),
                WindowLabel::Tuple(vec![int(1)]),
            ),
        ] {
            assert!(!same_label(&label, &key), "{label:?} is {key:?}");
            assert!(!same_label(&key, &label), "{key:?} is {label:?}");
        }

        // the reproduction, end to end, and the mutation beside it: a key which genuinely names
        // no sample has to go on being refused, which is what a comparison made loose enough to
        // fix the above would stop doing
        let times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let curve = [1.0, 10.0, 8.0, 6.0, 4.0, 2.0];
        let labelled = |values: Coordinate| Batch {
            coords: vec![SampleCoordinate {
                name: "subject".to_string(),
                dims: vec!["subject".to_string()],
                values,
            }],
            ..parameters::tests::one(&times, &curve)
        };
        let windowed = |key: Option<WindowLabel>| NcaOptions {
            terminal: TerminalPhase {
                windows: key.map(|key| [(key, (1.0, 3.0))].into()),
                ..TerminalPhase::default()
            },
            ..parameters::tests::linear()
        };
        let t_last = |result: &AnalysisResult| parameters::tests::value(result, "lambda_z_t_last");

        // without the window the regression of this curve runs to its last point, so the
        // assertions below say that the window was found and not that the number is 3.0 anyway
        let open = parameters::tests::run(&labelled(Coordinate::Float(vec![1.0])), &windowed(None));
        assert_eq!(t_last(&open), 5.0);

        for (coordinate, key) in [
            (Coordinate::Float(vec![1.0]), int(1)),
            (Coordinate::Int(vec![1]), float(1.0)),
            (Coordinate::Bool(vec![true]), int(1)),
            (Coordinate::Int(vec![1]), bool_label(true)),
        ] {
            let batch = labelled(coordinate.clone());
            let result = parameters::tests::run(&batch, &windowed(Some(key.clone())));
            assert_eq!(
                t_last(&result),
                3.0,
                "the window of {key:?} did not reach {coordinate:?}"
            );
        }

        for (coordinate, key) in [
            (Coordinate::Float(vec![1.0]), int(2)),
            (Coordinate::Float(vec![1.5]), int(1)),
            (Coordinate::Int(vec![2]), bool_label(true)),
            (Coordinate::Int(vec![1]), WindowLabel::Str("1".to_string())),
            (Coordinate::Float(vec![f64::NAN]), float(f64::NAN)),
        ] {
            let batch = labelled(coordinate.clone());
            let error = analyze_batch(&batch, &windowed(Some(key.clone()))).unwrap_err();
            assert!(
                matches!(&error, KernelError::Options(message) if message.contains("names no sample")),
                "the window of {key:?} reached {coordinate:?}: {error}"
            );
        }
    }

    #[test]
    fn a_contract_error_passes_through_unchanged() {
        let contract =
            pkpdutils_abi::check_contract("batch/2", pkpdutils_abi::BATCH_CONTRACT).unwrap_err();
        let error = KernelError::from(contract.clone());
        assert_eq!(error.to_string(), contract.to_string());
        assert_eq!(error, KernelError::Contract(contract));
    }

    #[test]
    fn a_request_is_analysed_only_when_both_its_contracts_are_this_one() {
        // A host may build the request and its batch separately, so `analyze` checks the two
        // before it looks at the batch at all.
        let batch = parameters::tests::oral_curve();
        let options = parameters::tests::linear();
        let request = AnalysisRequest {
            contract: ANALYSIS_CONTRACT.to_string(),
            batch: batch.clone(),
            options: options.clone(),
        };
        let result = analyze(&request).expect("the request is analysable");
        // the oral parameter set of a dosed batch with its count of subjects, and the
        // trapezoids 2 + 6 + 6 + 3 + 1.5 + 0.75 of the canonical curve
        assert_eq!(result.parameters.len(), 44);
        let auc_last = result
            .parameters
            .iter()
            .find(|p| p.name == "auc_last")
            .expect("the result carries the area");
        assert_eq!(auc_last.values, vec![19.25]);

        let foreign = AnalysisRequest {
            contract: "analysis/0".to_string(),
            ..request.clone()
        };
        assert!(matches!(analyze(&foreign), Err(KernelError::Contract(_))));

        let foreign_batch = AnalysisRequest {
            batch: Batch {
                contract: "batch/0".to_string(),
                ..batch
            },
            ..request
        };
        assert!(matches!(
            analyze(&foreign_batch),
            Err(KernelError::Contract(_))
        ));
    }
}
