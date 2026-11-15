//! The contract constants every pkpdutils-core crate and every host agrees on.

pub mod batch;
pub mod enums;
pub mod nonfinite;
pub mod options;
pub mod result;
pub mod window;

pub use batch::{Batch, Coordinate, Dosing, SampleCoordinate};
pub use enums::{
    AucMethod, BlqAction, BlqHandling, BootstrapDistribution, BootstrapSpread, C0Method, Kind,
    NcaFlag, Route, TerminalMethod, UncertaintyMethod,
};
pub use options::{Acceptance, Blq, BlqRules, BlqSetting, NcaOptions, TerminalPhase};
pub use result::{AnalysisRequest, AnalysisResult, Attributes, ParameterColumn, PointVariable};
pub use window::WindowLabel;

/// The contract of an analysis request and its result.
pub const ANALYSIS_CONTRACT: &str = "analysis/1";

/// The contract of a batch of timecourses.
pub const BATCH_CONTRACT: &str = "batch/1";

/// The `_nca_options_version` written by `pkpdutils.nca.options` and required on reading.
pub const OPTIONS_VERSION: u32 = 1;

/// A contract string which is not the one this build speaks.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("contract mismatch: found {found}, expected {expected}")]
pub struct ContractError {
    /// The contract the peer announced.
    pub found: String,
    /// The contract this build speaks.
    pub expected: String,
}

/// Accept `found` only when it is exactly `expected`.
pub fn check_contract(found: &str, expected: &str) -> Result<(), ContractError> {
    if found == expected {
        Ok(())
    } else {
        Err(ContractError {
            found: found.to_string(),
            expected: expected.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_constants_are_the_published_strings() {
        assert_eq!(ANALYSIS_CONTRACT, "analysis/1");
        assert_eq!(BATCH_CONTRACT, "batch/1");
        assert_eq!(OPTIONS_VERSION, 1);
    }

    #[test]
    fn check_contract_accepts_an_exact_match() {
        assert!(check_contract("analysis/1", ANALYSIS_CONTRACT).is_ok());
    }

    #[test]
    fn check_contract_names_both_sides_on_a_mismatch() {
        let err = check_contract("analysis/2", ANALYSIS_CONTRACT).unwrap_err();
        assert_eq!(
            err.to_string(),
            "contract mismatch: found analysis/2, expected analysis/1"
        );
    }
}
