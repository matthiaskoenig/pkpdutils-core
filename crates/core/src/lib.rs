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
mod pack;
mod parameters;
mod terminal;

use pkpdutils_abi::{
    ANALYSIS_CONTRACT, AnalysisRequest, AnalysisResult, BATCH_CONTRACT, Batch, Coordinate, Kind,
    NcaOptions, TerminalMethod, UncertaintyMethod, WindowLabel, check_contract,
};

use crate::assemble::{resolved_uncertainty, result_of};
use crate::dose::{ReferenceDose, reference_dose};
use crate::parameters::{RowInput, Scratch, compute_row};
use crate::terminal::Candidate;

/// Why a batch could not be analysed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum KernelError {
    /// The batch itself is malformed, with the reason `pkpdutils_abi::Batch::validate` gave,
    /// or one of its units is no unit, which that check does not cover (`crate::assemble`).
    #[error("the batch is not analysable: {0}")]
    Batch(String),
    /// The options are malformed or contradict each other.
    #[error("the options are not analysable: {0}")]
    Options(String),
    /// An analysis this build does not carry, named by what was asked for.
    #[error("{what} is not implemented by this build")]
    Unsupported {
        /// The analysis, the option or the route which is missing.
        what: String,
    },
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

/// The single dose non-compartmental analysis of every row of a batch.
///
/// The contract, the batch and the options are checked before a row runs, an analysis this build
/// does not carry is refused by name, and only then are the rows computed, each one independent
/// of the others (`parameters::compute_row`). The result carries one column per parameter the
/// rows report in its reported unit, the flags of every row, the acceptance variables and the
/// coordinates of the batch (`assemble::result_of`).
///
/// # Errors
///
/// [`KernelError::Contract`] for a batch of another contract, [`KernelError::Batch`] and
/// [`KernelError::Options`] for a malformed input, naming the field, and
/// [`KernelError::Unsupported`] for an analysis of another plan: an effect curve, a multiple
/// dose protocol, a partial area, the acceptance criteria, the propagation of an uncertainty and
/// the unit conversion of the result.
pub fn analyze_batch(batch: &Batch, options: &NcaOptions) -> Result<AnalysisResult, KernelError> {
    check_contract(&batch.contract, BATCH_CONTRACT)?;
    batch.validate().map_err(KernelError::Batch)?;
    options.validate().map_err(KernelError::Options)?;
    refuse_what_is_not_carried(batch, options)?;

    let n_samples = batch.n_samples();
    let manual = manual_columns(batch, options)?;
    let windows = sample_windows(batch, options)?;
    let has_dose = batch.max_doses > 0;
    // only an analysis of a single row reports the windows the selection could have chosen from
    // (`candidate_variables` of `nca.py`), so only such a row is asked for them
    let mut candidates: Vec<Candidate> = Vec::new();
    let wants_candidates = options.terminal.keep_candidates && n_samples == 1;
    let mut scratch = Scratch::new(batch.n_time);
    let mut rows = Vec::with_capacity(n_samples);
    for row in 0..n_samples {
        let route = batch.route_of_row(row);
        let input = RowInput {
            times: row_slice(&batch.times, row, batch.n_time),
            values: row_slice(&batch.values, row, batch.n_time),
            dose: row_dose(batch, row),
            route,
            has_dose,
            lloq: batch.lloq.as_ref().and_then(|v| v.get(row).copied()),
            window: windows.as_ref().and_then(|w| w.get(row).copied().flatten()),
            manual: manual.as_deref(),
        };
        let table = wants_candidates.then_some(&mut candidates);
        rows.push((route, compute_row(&input, options, &mut scratch, table)));
    }

    let result = result_of(batch, options, &rows, &candidates)?;
    // unreachable after `batch.validate`, which is what every length here comes from, and kept
    // because a result the host cannot read is worth a named error rather than a surprise. The
    // batch validated, so the error names this build and not the input.
    result.validate().map_err(KernelError::Internal)?;
    Ok(result)
}

/// Refuse what this build does not carry, naming it.
///
/// Every one of these is a different parameter set or a step beyond the single dose analysis, so
/// a wrong number is the alternative to a named error.
fn refuse_what_is_not_carried(batch: &Batch, options: &NcaOptions) -> Result<(), KernelError> {
    let unsupported = |what: &str| {
        Err(KernelError::Unsupported {
            what: what.to_string(),
        })
    };
    if options.kind == Kind::Effect {
        return unsupported("the analysis of an effect curve (Kind::Effect)");
    }
    if options.tau.is_some() {
        return unsupported("a multiple dose analysis (NcaOptions::tau)");
    }
    if (0..batch.n_samples()).any(|row| n_doses_of_row(batch, row) >= 2) {
        return unsupported("a protocol of more than one dose (the multiple dose analysis)");
    }
    if options.acceptance.any_threshold() {
        return unsupported("the acceptance criteria (NcaOptions::acceptance)");
    }
    if !options.partial_aucs.is_empty() {
        return unsupported("the named partial areas (NcaOptions::partial_aucs)");
    }
    // a batch which carries a spread resolves to the bootstrap on its own, so the refusal is of
    // what the options resolve to and not of what they say (`resolve_uncertainty` of `options.py`)
    if resolved_uncertainty(batch, options) != UncertaintyMethod::None {
        return unsupported("the propagation of the uncertainty of a group curve");
    }
    if !options.units.is_empty() {
        return unsupported("the unit conversion of a result (NcaOptions::units)");
    }
    Ok(())
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

/// The dose of a row the single dose analysis runs against, the first of its protocol.
///
/// The durations are read whenever the batch has doses at all, which is what python's
/// `dose_duration is not None` means: `Timecourses` writes `dose_duration` onto every batch with
/// doses and leaves it `NaN` where the route is no infusion.
fn row_dose(batch: &Batch, row: usize) -> ReferenceDose {
    let durations =
        (batch.max_doses > 0).then(|| row_slice(&batch.dose_duration, row, batch.max_doses));
    reference_dose(
        row_slice(&batch.dose_amount, row, batch.max_doses),
        row_slice(&batch.dose_time, row, batch.max_doses),
        durations,
        false,
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
/// `WindowLabel` tells a date and a datetime from a string, where a coordinate of a batch has no
/// dtype for either and carries both as text, so a date key is matched against the text of the
/// coordinate. Everything else compares as itself.
fn same_label(label: &WindowLabel, key: &WindowLabel) -> bool {
    match (label, key) {
        (WindowLabel::Str(text), WindowLabel::Date(key) | WindowLabel::Datetime(key)) => {
            text == key
        }
        (WindowLabel::Tuple(label), WindowLabel::Tuple(key)) => {
            label.len() == key.len() && label.iter().zip(key).all(|(l, k)| same_label(l, k))
        }
        _ => label == key,
    }
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
    use super::*;

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
        // the oral parameter set of a dosed batch, and the trapezoids
        // 2 + 6 + 6 + 3 + 1.5 + 0.75 of the canonical curve
        assert_eq!(result.parameters.len(), 43);
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
