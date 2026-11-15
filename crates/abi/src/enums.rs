//! The enumerations every request and result is expressed in; their serde values and flag bits are exact.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

/// Whether the curve is a concentration or an effect.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// `Concentration`.
    Concentration,
    /// `Effect`.
    Effect,
}

/// The trapezoid rule of the areas.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AucMethod {
    /// `Linear`.
    Linear,
    /// `LinearLog`.
    LinearLog,
    /// `Log`.
    Log,
}

/// The rule which picks the terminal regression window.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TerminalMethod {
    /// `BestFit`.
    BestFit,
    /// `LastN`.
    LastN,
    /// `AllAfterTmax`.
    AllAfterTmax,
    /// `Manual`.
    Manual,
}

/// How values below the limit of quantification are treated.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BlqHandling {
    /// `Nan`.
    Nan,
    /// `ZeroBeforeTmax`.
    ZeroBeforeTmax,
}

/// The action on one value below the limit of quantification.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BlqAction {
    /// `Drop`.
    Drop,
    /// `Keep`.
    Keep,
    /// `Zero`.
    Zero,
    /// `Lloq`.
    Lloq,
    /// `HalfLloq`.
    HalfLloq,
}

/// The method which estimates the concentration at the dose time.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum C0Method {
    /// `LogBackExtrapolation`.
    LogBackExtrapolation,
    /// `FirstValue`.
    FirstValue,
    /// `None`.
    None,
}

/// The propagation of the uncertainty of group curves.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyMethod {
    /// `None`.
    None,
    /// `Bootstrap`.
    Bootstrap,
    /// `Delta`.
    Delta,
}

/// The spread the bootstrap resamples with.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BootstrapSpread {
    /// `Se`.
    Se,
    /// `Sd`.
    Sd,
}

/// The distribution the bootstrap draws from.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BootstrapDistribution {
    /// `Normal`.
    Normal,
    /// `Lognormal`.
    Lognormal,
}

/// The route of administration.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    /// `IvBolus`.
    IvBolus,
    /// `IvInfusion`.
    IvInfusion,
    /// `Oral`.
    Oral,
}

impl Route {
    /// Whether the route is intravenous.
    pub fn is_iv(self) -> bool {
        matches!(self, Route::IvBolus | Route::IvInfusion)
    }
}

impl C0Method {
    /// The code of the `c0_method` result variable for no estimate.
    pub const CODE_NONE: i32 = 0;
    /// The code of the `c0_method` result variable for the back-extrapolation.
    pub const CODE_BACK_EXTRAPOLATION: i32 = 1;
    /// The code of the `c0_method` result variable for the first value.
    pub const CODE_FIRST_VALUE: i32 = 2;
}

bitflags! {
    /// The quality flags of a row, the bits of the Python `NCAFlag`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct NcaFlag: u32 {
        /// No flag.
        const NONE = 0;
        /// The terminal slope is positive.
        const POSITIVE_SLOPE = 1;
        /// Too few points for the terminal regression.
        const TOO_FEW_POINTS = 2;
        /// The extrapolated fraction is high.
        const EXTRAPOLATION_HIGH = 4;
        /// No maximum found.
        const NO_MAX = 8;
        /// No absorption found.
        const NO_ABSORPTION = 16;
        /// The curve was truncated at the BLQ values.
        const BLQ_TRUNCATED = 32;
        /// No data.
        const NO_DATA = 64;
        /// The window of the terminal phase changed in the delta method.
        const DELTA_WINDOW_CHANGE = 128;
        /// The last interval is not covered by the data.
        const INCOMPLETE_INTERVAL = 256;
        /// The trough of a bolus interval was regressed.
        const EXTRAPOLATED_TROUGH = 512;
        /// The terminal window covers fewer than two half-lives.
        const SPAN_LOW = 1024;
        /// The row was not accepted.
        const NOT_ACCEPTED = 2048;
        /// A partial area was extrapolated.
        const PARTIAL_EXTRAPOLATED = 4096;
        /// The terminal concentration is low.
        const TERMINAL_LOW_CONCENTRATION = 8192;
    }
}

impl NcaFlag {
    /// The names of the set flags in declaration order, `NONE` never among them.
    pub fn decode(self) -> Vec<&'static str> {
        self.iter_names()
            .filter(|(_, flag)| !flag.is_empty())
            .map(|(name, _)| name)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nca_flag_bits_match_the_python_intflag() {
        assert_eq!(NcaFlag::NONE.bits(), 0);
        assert_eq!(NcaFlag::POSITIVE_SLOPE.bits(), 1);
        assert_eq!(NcaFlag::TOO_FEW_POINTS.bits(), 2);
        assert_eq!(NcaFlag::EXTRAPOLATION_HIGH.bits(), 4);
        assert_eq!(NcaFlag::NO_MAX.bits(), 8);
        assert_eq!(NcaFlag::NO_ABSORPTION.bits(), 16);
        assert_eq!(NcaFlag::BLQ_TRUNCATED.bits(), 32);
        assert_eq!(NcaFlag::NO_DATA.bits(), 64);
        assert_eq!(NcaFlag::DELTA_WINDOW_CHANGE.bits(), 128);
        assert_eq!(NcaFlag::INCOMPLETE_INTERVAL.bits(), 256);
        assert_eq!(NcaFlag::EXTRAPOLATED_TROUGH.bits(), 512);
        assert_eq!(NcaFlag::SPAN_LOW.bits(), 1024);
        assert_eq!(NcaFlag::NOT_ACCEPTED.bits(), 2048);
        assert_eq!(NcaFlag::PARTIAL_EXTRAPOLATED.bits(), 4096);
        assert_eq!(NcaFlag::TERMINAL_LOW_CONCENTRATION.bits(), 8192);
    }

    #[test]
    fn decode_flags_lists_set_names_in_declaration_order() {
        let f = NcaFlag::SPAN_LOW | NcaFlag::POSITIVE_SLOPE;
        assert_eq!(f.decode(), vec!["POSITIVE_SLOPE", "SPAN_LOW"]);
    }

    #[test]
    fn enums_serialize_as_the_python_string_values() {
        assert_eq!(
            serde_json::to_string(&AucMethod::LinearLog).unwrap(),
            "\"linear_log\""
        );
        assert_eq!(
            serde_json::to_string(&Route::IvBolus).unwrap(),
            "\"iv_bolus\""
        );
        assert_eq!(
            serde_json::to_string(&TerminalMethod::AllAfterTmax).unwrap(),
            "\"all_after_tmax\""
        );
        assert_eq!(
            serde_json::to_string(&C0Method::LogBackExtrapolation).unwrap(),
            "\"log_back_extrapolation\""
        );
        assert_eq!(
            serde_json::to_string(&BlqAction::HalfLloq).unwrap(),
            "\"half_lloq\""
        );
    }
}
