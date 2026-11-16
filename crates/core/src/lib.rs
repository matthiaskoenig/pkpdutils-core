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

// Each module is one finished piece of the analysis pipeline and the row loop which calls them
// - pack the row, integrate its segments, regress its terminal phase - is not written yet. The
// terminal phase is handed a row which is already packed and never integrates, so it is no
// caller of the two before it. Until that loop exists their own tests are the only caller and
// the lint fires on every item in them; the allow goes away with the first real caller.
#[allow(dead_code)]
mod auc;
#[allow(dead_code)]
mod pack;
#[allow(dead_code)]
mod terminal;

/// Why a batch could not be analysed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum KernelError {
    /// The batch itself is malformed, with the reason `pkpdutils_abi::Batch::validate` gave.
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
    /// The peer speaks another contract than this build.
    #[error(transparent)]
    Contract(#[from] pkpdutils_abi::ContractError),
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
}
