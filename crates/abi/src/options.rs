//! The options of the non-compartmental analysis, the JSON of `pkpdutils.nca.options`.
//!
//! The wire format is `serialize_nca_options` of the Python library: the options as a flat
//! object plus `_nca_options_version` and `_terminal_windows`.

use std::collections::BTreeMap;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::enums::{
    AucMethod, BlqAction, BlqHandling, BootstrapDistribution, BootstrapSpread, C0Method, Kind,
    TerminalMethod, UncertaintyMethod,
};
use crate::{OPTIONS_VERSION, nonfinite};

/// A rule for the values below the limit of quantification: an action or a number imputed as is.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(untagged)]
pub enum BlqSetting {
    /// One of the named actions.
    Action(BlqAction),
    /// A value imputed as it is.
    Value(#[serde(with = "nonfinite")] f64),
}

/// Rules for the values below the limit of quantification, by position or by tmax.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(deny_unknown_fields, try_from = "BlqRulesWire", into = "BlqRulesWire")]
pub struct BlqRules {
    /// Rule for the values before the first measurable value.
    pub first: Option<BlqSetting>,
    /// Rule for the values between two measurable values.
    pub middle: Option<BlqSetting>,
    /// Rule for the values after the last measurable value.
    pub last: Option<BlqSetting>,
    /// Rule for the values before the maximum.
    pub before_tmax: Option<BlqSetting>,
    /// Rule for the values at or after the maximum.
    pub after_tmax: Option<BlqSetting>,
    /// Whether an imputed or kept value may enter the terminal regression.
    pub terminal_regression: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct BlqRulesWire {
    first: Option<BlqSetting>,
    middle: Option<BlqSetting>,
    last: Option<BlqSetting>,
    before_tmax: Option<BlqSetting>,
    after_tmax: Option<BlqSetting>,
    terminal_regression: bool,
}

impl Default for BlqRulesWire {
    fn default() -> Self {
        BlqRules::default().into()
    }
}

impl From<BlqRules> for BlqRulesWire {
    fn from(r: BlqRules) -> Self {
        Self {
            first: r.first,
            middle: r.middle,
            last: r.last,
            before_tmax: r.before_tmax,
            after_tmax: r.after_tmax,
            terminal_regression: r.terminal_regression,
        }
    }
}

impl TryFrom<BlqRulesWire> for BlqRules {
    type Error = String;

    fn try_from(w: BlqRulesWire) -> Result<Self, String> {
        let r = BlqRules {
            first: w.first,
            middle: w.middle,
            last: w.last,
            before_tmax: w.before_tmax,
            after_tmax: w.after_tmax,
            terminal_regression: w.terminal_regression,
        };
        r.validate()?;
        Ok(r)
    }
}

impl BlqRules {
    /// Check that only one of the two axes carries rules.
    pub fn validate(&self) -> Result<(), String> {
        let positional = self.first.is_some() || self.middle.is_some() || self.last.is_some();
        if positional && self.by_tmax() {
            return Err(
                "the positional rules (first, middle, last) and the tmax rules \
                        (before_tmax, after_tmax) are two axes of the same values, give one of them"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// Whether the rules split the curve at the maximum instead of by position.
    pub fn by_tmax(&self) -> bool {
        self.before_tmax.is_some() || self.after_tmax.is_some()
    }

    /// The rules of one of the two classic handlings.
    pub fn from_handling(handling: BlqHandling) -> Self {
        let action = |a| Some(BlqSetting::Action(a));
        match handling {
            BlqHandling::ZeroBeforeTmax => Self {
                before_tmax: action(BlqAction::Zero),
                after_tmax: action(BlqAction::Drop),
                ..Self::default()
            },
            BlqHandling::Nan => Self {
                first: action(BlqAction::Drop),
                middle: action(BlqAction::Drop),
                last: action(BlqAction::Drop),
                ..Self::default()
            },
        }
    }

    /// ICH M13A (2024): zero at both ends, dropped in between.
    pub fn ich_m13a() -> Self {
        let action = |a| Some(BlqSetting::Action(a));
        Self {
            first: action(BlqAction::Zero),
            middle: action(BlqAction::Drop),
            last: action(BlqAction::Zero),
            ..Self::default()
        }
    }

    /// PKanalix: 0 before the maximum, `LLOQ/2` after it.
    pub fn pkanalix() -> Self {
        let action = |a| Some(BlqSetting::Action(a));
        Self {
            before_tmax: action(BlqAction::Zero),
            after_tmax: action(BlqAction::HalfLloq),
            ..Self::default()
        }
    }

    /// Pumas: the ends kept, the middle dropped.
    pub fn pumas() -> Self {
        let action = |a| Some(BlqSetting::Action(a));
        Self {
            first: action(BlqAction::Keep),
            middle: action(BlqAction::Drop),
            last: action(BlqAction::Keep),
            ..Self::default()
        }
    }
}

/// The handling of the values below the limit of quantification: a classic handling or rules.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum Blq {
    /// One of the two classic handlings.
    Handling(BlqHandling),
    /// Rules by position or by tmax.
    Rules(BlqRules),
}

impl Blq {
    /// The rule set, the classic handlings included.
    pub fn rules(&self) -> BlqRules {
        match self {
            Blq::Handling(h) => BlqRules::from_handling(*h),
            Blq::Rules(r) => r.clone(),
        }
    }
}

/// Thresholds of the acceptance criteria of a terminal regression.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(
    deny_unknown_fields,
    try_from = "AcceptanceWire",
    into = "AcceptanceWire"
)]
pub struct Acceptance {
    /// Minimum adjusted R squared, in `[0, 1]`.
    pub r2_adj_min: Option<f64>,
    /// Maximum extrapolated fraction of the area, in `(0, 1]`.
    pub extrapolation_max: Option<f64>,
    /// Minimum span in half-lives, positive.
    pub span_min: Option<f64>,
    /// Minimum number of points of the regression, at least 2.
    pub n_points_min: Option<u32>,
    /// Whether a sample which fails a threshold is also excluded.
    pub exclude: bool,
}

#[derive(Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields, default)]
struct AcceptanceWire {
    #[serde(with = "nonfinite::option")]
    r2_adj_min: Option<f64>,
    #[serde(with = "nonfinite::option")]
    extrapolation_max: Option<f64>,
    #[serde(with = "nonfinite::option")]
    span_min: Option<f64>,
    n_points_min: Option<u32>,
    exclude: bool,
}

impl From<Acceptance> for AcceptanceWire {
    fn from(a: Acceptance) -> Self {
        Self {
            r2_adj_min: a.r2_adj_min,
            extrapolation_max: a.extrapolation_max,
            span_min: a.span_min,
            n_points_min: a.n_points_min,
            exclude: a.exclude,
        }
    }
}

impl TryFrom<AcceptanceWire> for Acceptance {
    type Error = String;

    fn try_from(w: AcceptanceWire) -> Result<Self, String> {
        let a = Acceptance {
            r2_adj_min: w.r2_adj_min,
            extrapolation_max: w.extrapolation_max,
            span_min: w.span_min,
            n_points_min: w.n_points_min,
            exclude: w.exclude,
        };
        a.validate()?;
        Ok(a)
    }
}

impl Acceptance {
    /// Check the ranges of the thresholds.
    pub fn validate(&self) -> Result<(), String> {
        check_opt(
            "r2_adj_min",
            self.r2_adj_min,
            |x| (0.0..=1.0).contains(&x),
            "in [0, 1]",
        )?;
        check_opt(
            "extrapolation_max",
            self.extrapolation_max,
            |x| x > 0.0 && x <= 1.0,
            "in (0, 1]",
        )?;
        check_opt("span_min", self.span_min, |x| x > 0.0, "positive")?;
        if self.n_points_min.is_some_and(|n| n < 2) {
            return Err("n_points_min must be at least 2".to_string());
        }
        Ok(())
    }

    /// Whether a threshold is set at all.
    pub fn any_threshold(&self) -> bool {
        self.r2_adj_min.is_some()
            || self.extrapolation_max.is_some()
            || self.span_min.is_some()
            || self.n_points_min.is_some()
    }

    /// The default thresholds of PKanalix.
    pub fn pkanalix() -> Self {
        Self {
            r2_adj_min: Some(0.98),
            extrapolation_max: Some(0.20),
            span_min: Some(3.0),
            n_points_min: Some(3),
            exclude: false,
        }
    }
}

fn check_opt(
    name: &str,
    value: Option<f64>,
    ok: impl Fn(f64) -> bool,
    expected: &str,
) -> Result<(), String> {
    match value {
        Some(x) if !ok(x) => Err(format!("{name} must be {expected}, found {x}")),
        _ => Ok(()),
    }
}

fn check(name: &str, x: f64, ok: bool, expected: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(format!("{name} must be {expected}, found {x}"))
    }
}

/// Selection of the points of the terminal log-linear regression.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(
    deny_unknown_fields,
    try_from = "TerminalPhaseWire",
    into = "TerminalPhaseWire"
)]
pub struct TerminalPhase {
    /// The rule which picks the window.
    pub method: TerminalMethod,
    /// Minimum number of points, at least 3.
    pub min_points: u32,
    /// Whether the windows must start after the point of the maximum.
    pub exclude_cmax: bool,
    /// The number of points of `TerminalMethod::LastN`.
    pub n_points: Option<u32>,
    /// The point indices of `TerminalMethod::Manual`.
    pub points: Option<Vec<u32>>,
    /// Minimum adjusted R squared of a candidate window.
    pub min_adj_r2: Option<f64>,
    /// Tolerance below which two adjusted R squared values tie.
    pub tie_tolerance: f64,
    /// Relative warning threshold of the regression.
    pub relative_warning: Option<f64>,
    /// The window `(t_first, t_last)` of single samples by label, carried separately on the wire.
    pub windows: Option<BTreeMap<String, (f64, f64)>>,
    /// Whether the table of every candidate window is kept.
    pub keep_candidates: bool,
}

impl Default for TerminalPhase {
    fn default() -> Self {
        Self {
            method: TerminalMethod::BestFit,
            min_points: 3,
            exclude_cmax: true,
            n_points: None,
            points: None,
            min_adj_r2: None,
            tie_tolerance: 1e-4,
            relative_warning: Some(1e-8),
            windows: None,
            keep_candidates: false,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct TerminalPhaseWire {
    method: TerminalMethod,
    min_points: u32,
    exclude_cmax: bool,
    n_points: Option<u32>,
    points: Option<Vec<u32>>,
    #[serde(with = "nonfinite::option")]
    min_adj_r2: Option<f64>,
    #[serde(with = "nonfinite")]
    tie_tolerance: f64,
    #[serde(with = "nonfinite::option")]
    relative_warning: Option<f64>,
    #[serde(
        with = "nonfinite::option_pair_map",
        skip_serializing_if = "Option::is_none"
    )]
    windows: Option<BTreeMap<String, (f64, f64)>>,
    keep_candidates: bool,
}

impl Default for TerminalPhaseWire {
    fn default() -> Self {
        TerminalPhase::default().into()
    }
}

impl From<TerminalPhase> for TerminalPhaseWire {
    fn from(t: TerminalPhase) -> Self {
        Self {
            method: t.method,
            min_points: t.min_points,
            exclude_cmax: t.exclude_cmax,
            n_points: t.n_points,
            points: t.points,
            min_adj_r2: t.min_adj_r2,
            tie_tolerance: t.tie_tolerance,
            relative_warning: t.relative_warning,
            windows: t.windows,
            keep_candidates: t.keep_candidates,
        }
    }
}

impl TryFrom<TerminalPhaseWire> for TerminalPhase {
    type Error = String;

    fn try_from(w: TerminalPhaseWire) -> Result<Self, String> {
        let t = TerminalPhase {
            method: w.method,
            min_points: w.min_points,
            exclude_cmax: w.exclude_cmax,
            n_points: w.n_points,
            points: w.points,
            min_adj_r2: w.min_adj_r2,
            tie_tolerance: w.tie_tolerance,
            relative_warning: w.relative_warning,
            windows: w.windows,
            keep_candidates: w.keep_candidates,
        };
        t.validate()?;
        Ok(t)
    }
}

impl TerminalPhase {
    /// Check the fields `method` requires and the ranges.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_points < 3 {
            return Err(format!(
                "min_points must be at least 3, found {}",
                self.min_points
            ));
        }
        if self.n_points.is_some_and(|n| n < 3) {
            return Err("n_points must be at least 3".to_string());
        }
        check(
            "tie_tolerance",
            self.tie_tolerance,
            self.tie_tolerance >= 0.0,
            "non-negative",
        )?;
        check_opt(
            "relative_warning",
            self.relative_warning,
            |x| x > 0.0 && x < 1.0,
            "in (0, 1)",
        )?;
        if self.method == TerminalMethod::LastN && self.n_points.is_none() {
            return Err("TerminalMethod::LastN needs 'n_points'".to_string());
        }
        if self.method == TerminalMethod::Manual {
            let points = self
                .points
                .as_ref()
                .ok_or("TerminalMethod::Manual needs 'points'")?;
            if points.len() < self.min_points as usize {
                return Err(format!(
                    "'points' has {} indices, 'min_points' is {}",
                    points.len(),
                    self.min_points
                ));
            }
        }
        for (key, window) in self.windows.iter().flatten() {
            if window.1.partial_cmp(&window.0) != Some(std::cmp::Ordering::Greater) {
                return Err(format!(
                    "the terminal window {window:?} of '{key}' must be (t_first, t_last) with t_last > t_first"
                ));
            }
        }
        Ok(())
    }
}

/// The options of a non-compartmental analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct NcaOptions {
    /// Concentration or effect curve.
    pub kind: Kind,
    /// The trapezoid rule of the areas.
    pub auc_method: AucMethod,
    /// The terminal regression.
    pub terminal: TerminalPhase,
    /// The lower limit of quantification.
    pub lloq: Option<f64>,
    /// The lower limit of quantification as a fraction of the maximum.
    pub lloq_relative: Option<f64>,
    /// The handling of the values below the limit.
    pub blq: Blq,
    /// The method of the concentration at the dose time.
    pub c0_method: C0Method,
    /// Extrapolated fraction above which a row is flagged.
    pub extrapolation_warning: f64,
    /// The acceptance criteria.
    pub acceptance: Acceptance,
    /// Named partial areas as `(t_start, t_end)`.
    pub partial_aucs: BTreeMap<String, (f64, f64)>,
    /// The dosing interval of a multiple dose analysis.
    pub tau: Option<f64>,
    /// The tolerance of a last interval which falls short of its end.
    pub tau_tolerance: f64,
    /// Whether the per-interval parameters are computed.
    pub intervals: bool,
    /// Units the finished result is converted to, by variable.
    pub units: BTreeMap<String, String>,
    /// The threshold of an effect curve.
    pub effect_threshold: Option<f64>,
    /// The worker threads, `None` automatic.
    pub n_workers: Option<u32>,
    /// The rows per chunk.
    pub chunk_rows: u32,
    /// The propagation of the uncertainty, `None` automatic.
    pub uncertainty: Option<UncertaintyMethod>,
    /// The number of bootstrap replicates.
    pub n_boot: u32,
    /// The seed of the bootstrap.
    pub seed: Option<u64>,
    /// The level of the confidence intervals.
    pub ci_level: f64,
    /// The spread the bootstrap resamples with.
    pub bootstrap_spread: BootstrapSpread,
    /// The distribution the bootstrap draws from.
    pub bootstrap_distribution: BootstrapDistribution,
    /// The relative step of the delta method.
    pub delta_step: f64,
}

impl Default for NcaOptions {
    fn default() -> Self {
        Self {
            kind: Kind::Concentration,
            auc_method: AucMethod::LinearLog,
            terminal: TerminalPhase::default(),
            lloq: None,
            lloq_relative: None,
            blq: Blq::Handling(BlqHandling::Nan),
            c0_method: C0Method::LogBackExtrapolation,
            extrapolation_warning: 0.2,
            acceptance: Acceptance::default(),
            partial_aucs: BTreeMap::new(),
            tau: None,
            tau_tolerance: 0.1,
            intervals: true,
            units: BTreeMap::new(),
            effect_threshold: None,
            n_workers: None,
            chunk_rows: 5000,
            uncertainty: None,
            n_boot: 1000,
            seed: None,
            ci_level: 0.95,
            bootstrap_spread: BootstrapSpread::Se,
            bootstrap_distribution: BootstrapDistribution::Normal,
            delta_step: 0.01,
        }
    }
}

impl NcaOptions {
    /// Check the ranges and the combinations the Python model checks.
    pub fn validate(&self) -> Result<(), String> {
        if self.kind == Kind::Effect && self.lloq_relative.is_some() {
            return Err("'lloq_relative' applies only to concentration curves".to_string());
        }
        check_opt(
            "lloq",
            self.lloq,
            |x| x > 0.0 && x.is_finite(),
            "positive and finite",
        )?;
        check_opt(
            "lloq_relative",
            self.lloq_relative,
            |x| x > 0.0 && x < 1.0,
            "in (0, 1)",
        )?;
        check(
            "extrapolation_warning",
            self.extrapolation_warning,
            self.extrapolation_warning > 0.0 && self.extrapolation_warning < 1.0,
            "in (0, 1)",
        )?;
        check_opt("tau", self.tau, |x| x > 0.0, "positive")?;
        check(
            "tau_tolerance",
            self.tau_tolerance,
            (0.0..1.0).contains(&self.tau_tolerance),
            "in [0, 1)",
        )?;
        if self.n_workers.is_some_and(|n| n < 1) {
            return Err("n_workers must be at least 1".to_string());
        }
        if self.chunk_rows < 1 {
            return Err("chunk_rows must be at least 1".to_string());
        }
        if self.n_boot < 2 {
            return Err(format!("n_boot must be at least 2, found {}", self.n_boot));
        }
        check(
            "ci_level",
            self.ci_level,
            self.ci_level > 0.0 && self.ci_level < 1.0,
            "in (0, 1)",
        )?;
        check(
            "delta_step",
            self.delta_step,
            self.delta_step > 0.0 && self.delta_step < 1.0,
            "in (0, 1)",
        )?;
        self.terminal.validate()?;
        self.acceptance.validate()?;
        if let Blq::Rules(rules) = &self.blq {
            rules.validate()?;
        }
        for (name, (start, end)) in &self.partial_aucs {
            if name.is_empty() {
                return Err("a partial area needs a name".to_string());
            }
            if end.partial_cmp(start) != Some(std::cmp::Ordering::Greater) {
                return Err(format!(
                    "the partial area '{name}' is ({start}, {end}), it must be (t_start, t_end) with t_end > t_start"
                ));
            }
        }
        Ok(())
    }

    /// The rule set of `blq`, the classic handlings included.
    pub fn blq_rules(&self) -> BlqRules {
        self.blq.rules()
    }
}

/// One record of `_terminal_windows`: a typed label and its window.
#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct WindowRecord {
    key: WindowKey,
    #[serde(with = "nonfinite::pair")]
    window: (f64, f64),
}

/// A typed window label; the Rust side carries string labels only.
#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct WindowKey {
    #[serde(rename = "type")]
    kind: String,
    value: String,
}

/// The wire form of [`NcaOptions`]: the options plus the two private keys.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields, default = "wire_default")]
struct NcaOptionsWire {
    kind: Kind,
    auc_method: AucMethod,
    terminal: TerminalPhase,
    #[serde(with = "nonfinite::option")]
    lloq: Option<f64>,
    #[serde(with = "nonfinite::option")]
    lloq_relative: Option<f64>,
    blq: Blq,
    c0_method: C0Method,
    #[serde(with = "nonfinite")]
    extrapolation_warning: f64,
    acceptance: Acceptance,
    #[serde(with = "nonfinite::pair_map")]
    partial_aucs: BTreeMap<String, (f64, f64)>,
    #[serde(with = "nonfinite::option")]
    tau: Option<f64>,
    #[serde(with = "nonfinite")]
    tau_tolerance: f64,
    intervals: bool,
    units: BTreeMap<String, String>,
    #[serde(with = "nonfinite::option")]
    effect_threshold: Option<f64>,
    n_workers: Option<u32>,
    chunk_rows: u32,
    uncertainty: Option<UncertaintyMethod>,
    n_boot: u32,
    seed: Option<u64>,
    #[serde(with = "nonfinite")]
    ci_level: f64,
    bootstrap_spread: BootstrapSpread,
    bootstrap_distribution: BootstrapDistribution,
    #[serde(with = "nonfinite")]
    delta_step: f64,
    #[serde(rename = "_nca_options_version")]
    version: Option<u32>,
    #[serde(rename = "_terminal_windows")]
    terminal_windows: Option<Vec<WindowRecord>>,
}

fn wire_default() -> NcaOptionsWire {
    NcaOptions::default().into()
}

impl From<NcaOptions> for NcaOptionsWire {
    fn from(o: NcaOptions) -> Self {
        let mut terminal = o.terminal;
        // the windows travel in `_terminal_windows`, not inside `terminal`
        let windows = terminal.windows.take().map(|map| {
            map.into_iter()
                .map(|(key, window)| WindowRecord {
                    key: WindowKey {
                        kind: "str".to_string(),
                        value: key,
                    },
                    window,
                })
                .collect()
        });
        Self {
            kind: o.kind,
            auc_method: o.auc_method,
            terminal,
            lloq: o.lloq,
            lloq_relative: o.lloq_relative,
            blq: o.blq,
            c0_method: o.c0_method,
            extrapolation_warning: o.extrapolation_warning,
            acceptance: o.acceptance,
            partial_aucs: o.partial_aucs,
            tau: o.tau,
            tau_tolerance: o.tau_tolerance,
            intervals: o.intervals,
            units: o.units,
            effect_threshold: o.effect_threshold,
            n_workers: o.n_workers,
            chunk_rows: o.chunk_rows,
            uncertainty: o.uncertainty,
            n_boot: o.n_boot,
            seed: o.seed,
            ci_level: o.ci_level,
            bootstrap_spread: o.bootstrap_spread,
            bootstrap_distribution: o.bootstrap_distribution,
            delta_step: o.delta_step,
            version: Some(OPTIONS_VERSION),
            terminal_windows: windows,
        }
    }
}

impl TryFrom<NcaOptionsWire> for NcaOptions {
    type Error = String;

    fn try_from(w: NcaOptionsWire) -> Result<Self, String> {
        if let Some(version) = w.version
            && version != OPTIONS_VERSION
        {
            return Err(format!(
                "unknown NCA options metadata version {version}, expected {OPTIONS_VERSION}"
            ));
        }
        let mut terminal = w.terminal;
        if let Some(records) = w.terminal_windows {
            let mut map = BTreeMap::new();
            for record in records {
                if record.key.kind != "str" {
                    return Err(format!(
                        "terminal window label of type {:?} is not supported, only \"str\"",
                        record.key.kind
                    ));
                }
                map.insert(record.key.value, record.window);
            }
            terminal.windows = Some(map);
            terminal.validate()?;
        }
        let o = NcaOptions {
            kind: w.kind,
            auc_method: w.auc_method,
            terminal,
            lloq: w.lloq,
            lloq_relative: w.lloq_relative,
            blq: w.blq,
            c0_method: w.c0_method,
            extrapolation_warning: w.extrapolation_warning,
            acceptance: w.acceptance,
            partial_aucs: w.partial_aucs,
            tau: w.tau,
            tau_tolerance: w.tau_tolerance,
            intervals: w.intervals,
            units: w.units,
            effect_threshold: w.effect_threshold,
            n_workers: w.n_workers,
            chunk_rows: w.chunk_rows,
            uncertainty: w.uncertainty,
            n_boot: w.n_boot,
            seed: w.seed,
            ci_level: w.ci_level,
            bootstrap_spread: w.bootstrap_spread,
            bootstrap_distribution: w.bootstrap_distribution,
            delta_step: w.delta_step,
        };
        o.validate()?;
        Ok(o)
    }
}

impl Serialize for NcaOptions {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        NcaOptionsWire::from(self.clone()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for NcaOptions {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        NcaOptionsWire::deserialize(deserializer)?
            .try_into()
            .map_err(D::Error::custom)
    }
}
