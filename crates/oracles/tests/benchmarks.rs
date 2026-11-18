//! The kernel reproduces Phoenix WinNonlin, PKNCA and NonCompart on the eight benchmark scenarios.
//!
//! This is the port of `tests/nca/test_benchmark_datasets.py`, the tightest gate of the project:
//! every parameter of every subject of every scenario of
//! `testdata/oracles/docs/data/benchmarks/scenarios.csv`, compared against the result table of
//! each of the three tools at a relative [`RTOL`] of 1e-8 with an absolute [`ATOL`] of 1e-12. The
//! published WinNonlin tables carry 8 to 15 significant digits and the PKNCA and NonCompart
//! tables the full precision of R, so the tolerance is just above the rounding of the oracles
//! and far below anything a formula could get wrong.
//!
//! Every tool writes the one parameter `auc_inf_obs` under a name of its own (`AUCINF_obs`,
//! `aucinf.obs`, `AUCIFO`), and the map which translates them is `src/pkpdutils/crosswalk.py`,
//! which is source and no oracle. `scripts/crosswalk_gate.py` extracts it into
//! `tests/data/crosswalk.json`, which this gate reads: the name maps, the maps of the names
//! which depend on the route, the columns which carry a percentage where `pkpdutils` reports a
//! fraction, the columns no variable of `pkpdutils` corresponds to, and the rules by which the
//! long PKNCA table is read.
//!
//! Nothing is loosened and nothing is quietly left out. A column of a tool which has no name is
//! a failure, a parameter the kernel does not produce is a failure which names it, a subject the
//! table and the dataset do not share is a failure, and the scenario, tool, column and number
//! counts are pinned as literals, so a truncated or a mis-parsed table fails here instead of
//! passing vacuously. The one exception is [`MISSING_ALLOWED`], pinned by scenario, tool and
//! parameter with its reason.

mod common;
// the unit helpers of the numeric gates, which `tests/oracles_present.rs` does not declare
#[path = "common/units.rs"]
mod units;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use pkpdutils_abi::{
    AnalysisResult, AucMethod, BATCH_CONTRACT, Batch, NcaOptions, Route, TerminalMethod,
    TerminalPhase,
};
use pkpdutils_core::analyze_batch;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use common::read_text;
use units::in_unit;

/// The directory of the benchmark oracles, repo-relative as the vendoring mirrors it.
const BENCHMARKS: &str = "docs/data/benchmarks";

/// The crosswalk fixture, next to this gate rather than under `testdata/oracles`: it is
/// extracted from the source of `pkpdutils` and is no vendored oracle.
const CROSSWALK: &str = "crosswalk.json";

/// The scenarios of the oracle.
const SCENARIOS: usize = 8;

/// The tools every scenario is compared against.
const TOOLS: usize = 3;

/// The numbers this gate compares, pinned so that a truncated or mis-parsed table fails here
/// instead of passing vacuously.
const COMPARISONS: usize = 6156;

/// The columns the gate compares, one per scenario, tool and variable, pinned for the same
/// reason as [`COMPARISONS`].
const COLUMNS: usize = 826;

/// Every variable the three tools name in these eight scenarios, sorted.
///
/// The counts above would not notice a column mapped onto another variable the kernel also
/// produces, so the names are pinned as well: the point, area and terminal parameters every
/// tool reports, the predicted family beside the observed one, the `lambda_z_span` of PKNCA,
/// the `tlag` of an extravascular dose and the `c0`, the back-extrapolated fractions, `cl`,
/// `vz` and `vss` of an intravenous one.
const VARIABLES: [&str; 44] = [
    "auc_all",
    "auc_back_extrap_fraction",
    "auc_back_extrap_fraction_pred",
    "auc_extrap_fraction",
    "auc_extrap_fraction_pred",
    "auc_inf_dn",
    "auc_inf_obs",
    "auc_inf_pred",
    "auc_inf_pred_dn",
    "auc_last",
    "aumc_extrap_fraction",
    "aumc_extrap_fraction_pred",
    "aumc_inf",
    "aumc_inf_pred",
    "aumc_last",
    "c0",
    "cl",
    "cl_f",
    "cl_f_pred",
    "cl_pred",
    "clast",
    "clast_pred",
    "cmax",
    "cmax_dn",
    "lambda_z",
    "lambda_z_n_points",
    "lambda_z_r2",
    "lambda_z_r2_adj",
    "lambda_z_span",
    "lambda_z_t_first",
    "lambda_z_t_last",
    "mrt",
    "mrt_last",
    "mrt_pred",
    "thalf",
    "tlag",
    "tlast",
    "tmax",
    "vss",
    "vss_pred",
    "vz",
    "vz_f",
    "vz_f_pred",
    "vz_pred",
];

/// Every scenario of the oracle, in the order `scenarios.csv` writes them, so that a scenario
/// which is renamed or replaced by a copy of another fails here instead of being analysed under
/// the wrong options.
const SCENARIO_IDS: [&str; SCENARIOS] = [
    "theoph-linear",
    "theoph-linear-log",
    "indometh-bolus-linear",
    "indometh-bolus-linear-log",
    "indometh-infusion-linear",
    "indometh-infusion-linear-log",
    "indometh-oral-linear",
    "indometh-oral-linear-log",
];

/// The three tools, sorted as the crosswalk fixture keys them.
const TOOL_IDS: [&str; TOOLS] = ["noncompart", "pknca", "winnonlin"];

/// The relative tolerance, `TOLERANCE` of `test_benchmark_datasets.py:38`.
///
/// One published number is printed with less precision than this, so the headroom there is
/// structurally negative rather than large. `Lambda_z` of theophylline subject 1 is printed as
/// `0.048456997` in both `Final_Parameters_Pivoted_Theoph_Linear.csv` and
/// `..._Theoph_Log.csv`: nine decimals on 0.0485, so half an ulp of that printout is a relative
/// 1.032e-8, and of the nine-decimal values of the eight tables it is the only one above
/// `RTOL`. The kernel's value falls inside that rounding interval and passes - the largest
/// difference of either theophylline pair against WinNonlin is 4.7e-9, on another subject's
/// `Lambda_z` - but a correct kernel could in principle fail on it.
///
/// The asymmetry is in the safe direction: a printout too coarse for the tolerance can fail a
/// right answer and can never pass a wrong one. The tolerance therefore stays at 1e-8, which is
/// what the python gate compares the same file at, and a failure on that one value is to be
/// read as the precision of the oracle rather than as an error of the kernel.
const RTOL: f64 = 1e-8;

/// The absolute tolerance, `atol` of `test_benchmark_datasets.py:97`.
const ATOL: f64 = 1e-12;

/// The smallest number of columns a tool table may contribute,
/// `test_benchmark_datasets.py:93`.
const MIN_COLUMNS: usize = 30;

/// How many deviating numbers the failure names. The gate compares every number before it
/// fails, so a kernel which has drifted lists what drifted rather than the first of it.
const FAILURES_SHOWN: usize = 10;

/// The provenance of the crosswalk fixture, carried here as well.
///
/// The header describes the maps next to it, so a fixture regenerated against a degraded
/// `pkpdutils` would agree with itself and `--check` would agree with the same degraded library.
/// These are the backstop that fires in continuous integration, where there is no `pkpdutils` to
/// compare against at all. Update them deliberately together with the fixture, never to make a
/// failing gate pass.
const PKPDUTILS_VERSION: &str = "1.7.0";

/// The digest of the two sections of the crosswalk fixture, see [`PKPDUTILS_VERSION`].
///
/// [`digest_of`] recomputes it from the sections themselves, so this also pins the fixture
/// against a hand edit which leaves the numbers agreeing: two columns of a tool whose values
/// happen to be equal could be swapped without any comparison noticing.
const CROSSWALK_DIGEST: &str = "3524e6436f407329";

/// The distinct tool columns the crosswalk names, see [`PKPDUTILS_VERSION`].
const CROSSWALK_COLUMNS: usize = 136;

/// The parameters a tool reports which the kernel does not produce, by scenario, tool and name.
///
/// `missing <= {"tlag"}` of `test_benchmark_datasets.py:89-92`: NonCompart reports a lag time of
/// 0 for an infusion, where `pkpdutils` reports no lag time at all, since an intravenous dose
/// has none. The bolus tables carry the column too and leave it `NA` for every subject, so it is
/// dropped as an empty column rather than missing, and [`Tally::dropped`] counts it.
///
/// This is the only parameter of any table this gate does not compare, and it is listed by
/// scenario and tool rather than by name alone: a parameter which stops being produced anywhere
/// else fails instead of joining a set.
const MISSING_ALLOWED: [(&str, &str, &str); 2] = [
    ("indometh-infusion-linear", "noncompart", "tlag"),
    ("indometh-infusion-linear-log", "noncompart", "tlag"),
];

/// The columns a tool reports as `NA` for every subject, which `dropna(axis="columns",
/// how="all")` of `test_benchmark_datasets.py:86` drops before the comparison.
///
/// Counted and pinned, so that a table which loses its numbers shrinks the gate visibly instead
/// of silently.
const DROPPED_ALL_NAN: [(&str, &str, &str); 2] = [
    ("indometh-bolus-linear", "noncompart", "tlag"),
    ("indometh-bolus-linear-log", "noncompart", "tlag"),
];

/// How many duplicate rows the long PKNCA tables drop, over all eight scenarios.
///
/// `duplicates="first"` of `test_benchmark_datasets.py:74-77`: these exports combine the
/// analysis of the observed and of the imputed concentrations at one interval, and the first
/// `tmax` is the observed maximum, which is what this comparison is about. One such row per
/// subject, so twelve for each theophylline scenario and six for each indomethacin one.
///
/// A count alone would not notice a table which gained a duplicate for one subject and lost one
/// for another, so the gate also asserts the identity of what was dropped: for every scenario
/// and tool the dropped rows are either none at all or exactly one [`DUPLICATED_VARIABLE`] per
/// subject of the dataset, and [`Tally::duplicates`] names the subject of each.
const DUPLICATES_DROPPED: usize = 60;

/// The one variable those duplicate rows carry, see [`DUPLICATES_DROPPED`].
const DUPLICATED_VARIABLE: &str = "tmax";

/// The one tool whose table carries duplicate rows at all, see [`DUPLICATES_DROPPED`].
const DUPLICATING_TOOL: &str = "pknca";

/// Which reader of `crosswalk.py` reads a tool's table.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Reader {
    /// `_read_wide`, the pivoted table of WinNonlin and of NonCompart: one row per subject, one
    /// column per parameter.
    Wide,
    /// `read_pknca_results`, the long table of PKNCA: one row per subject, interval and
    /// parameter.
    Long,
}

/// The crosswalk fixture, written by `scripts/crosswalk_gate.py`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Crosswalk {
    /// Which `pkpdutils` the maps were extracted from.
    provenance: Provenance,
    /// The three tools by their key.
    tools: BTreeMap<String, Tool>,
    /// What reading the long PKNCA table needs beyond the names.
    pknca_rules: PkncaRules,
}

/// The provenance header of the crosswalk fixture.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    /// The version of `pkpdutils` the maps were extracted from.
    pkpdutils_version: String,
    /// Whether that `pkpdutils` was a checkout or a release.
    pkpdutils_source: String,
    /// The commit of a clean checkout, `None` for a release or a dirty tree.
    source_commit: Option<String>,
    /// The digest of the two sections below.
    crosswalk_digest: String,
    /// The number of tools.
    n_tools: usize,
    /// The size of the name map of every tool.
    n_names: BTreeMap<String, usize>,
    /// The size of the by-route map of every tool.
    n_names_by_route: BTreeMap<String, usize>,
    /// The number of percentage columns of every tool.
    n_percent: BTreeMap<String, usize>,
    /// The number of columns of every tool which have no variable.
    n_unmapped: BTreeMap<String, usize>,
    /// The distinct columns of all three tools together.
    n_columns: usize,
    /// The number of rules which recognize the route of a PKNCA table.
    n_route_rules: usize,
}

/// The crosswalk of one tool.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Tool {
    /// Which reader of `crosswalk.py` reads its table.
    reader: Reader,
    /// Every variable of `pkpdutils` with the column of the tool, route independent.
    names: BTreeMap<String, String>,
    /// Every variable whose column depends on the route, by the route.
    names_by_route: BTreeMap<String, BTreeMap<String, String>>,
    /// The columns which carry a percentage where `pkpdutils` reports a fraction.
    percent: BTreeSet<String>,
    /// The columns the tool reports which no variable of `pkpdutils` corresponds to.
    unmapped: BTreeSet<String>,
}

/// What reading the long PKNCA table needs beyond the names.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PkncaRules {
    /// The columns of the long table after the grouping columns, `PKNCA_COLUMNS`.
    columns: Vec<String>,
    /// The clearances and volumes of an intravenous dose (`cl`, `vz`, ...), which PKNCA names
    /// like the extravascular ones; dropped when the route is extravascular.
    intravenous_only: BTreeSet<String>,
    /// The clearances and volumes over the bioavailability (`cl_f`, `vz_f`, ...); dropped when
    /// the route is intravenous.
    extravascular_only: BTreeSet<String>,
    /// The rules which recognize the route, in the order `_pknca_route` tries them.
    route_rules: Vec<RouteRule>,
}

/// One rule which recognizes the route from the parameters PKNCA reports.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteRule {
    /// The route the rule recognizes.
    route: Route,
    /// The parameters which name it outright.
    codes: Vec<String>,
    /// The prefixes of the parameters which name it.
    prefixes: Vec<String>,
}

/// One scenario of `scenarios.csv`: the dataset, its columns, the dose and the options.
struct Scenario {
    /// The key of the scenario, which also names its PKNCA and NonCompart tables.
    scenario: String,
    /// The csv of the dataset, next to `scenarios.csv`.
    dataset: String,
    /// The column of the subject.
    subject: String,
    /// The column of the time.
    time: String,
    /// The column of the measured concentration.
    conc: String,
    /// The unit of the times.
    time_unit: String,
    /// The unit of the concentrations.
    unit: String,
    /// The dose every subject received.
    dose: f64,
    /// The unit of the dose.
    dose_unit: String,
    /// The route it was given by.
    route: Route,
    /// The duration of an infusion, `None` for every other route.
    duration: Option<f64>,
    /// The trapezoid rule.
    auc_method: AucMethod,
    /// Whether the maximum may enter a terminal window.
    exclude_cmax: bool,
    /// The published WinNonlin table of the scenario.
    winnonlin: String,
}

/// The observations of a dataset, one block of `n_time` points per subject.
struct Observations {
    /// The subject labels, in the order of their first row of the csv.
    subjects: Vec<String>,
    /// The number of points of every subject.
    n_time: usize,
    /// The times, `subjects.len() * n_time`, row major.
    times: Vec<f64>,
    /// The values, same layout.
    values: Vec<f64>,
}

/// One tool table, read into the variables of `pkpdutils`.
struct Expected {
    /// One map of subject to value per variable, the percentages already fractions.
    values: BTreeMap<String, BTreeMap<String, f64>>,
    /// The variables dropped because the table has no number for any subject.
    dropped: Vec<String>,
    /// The subject and the variable of every duplicate row a long table dropped.
    duplicates: Vec<(String, String)>,
}

/// What the gate did, which the test pins against its literals.
struct Tally {
    /// The scenarios it ran.
    scenarios: usize,
    /// The tools it compared against, by their key.
    tools: BTreeSet<String>,
    /// The scenario and tool pairs it compared.
    pairs: usize,
    /// The columns it compared, over every scenario and tool.
    columns: usize,
    /// The numbers it compared.
    comparisons: usize,
    /// The columns dropped because the table has no number for any subject.
    dropped: Vec<(String, String, String)>,
    /// The parameters a tool reports and the kernel does not produce.
    missing: Vec<(String, String, String)>,
    /// The scenario, tool and subject of every duplicate row the long tables dropped.
    duplicates: Vec<(String, String, String)>,
    /// Every variable it compared, over every scenario and tool.
    variables: BTreeSet<String>,
    /// The largest relative difference of every scenario and tool, with where it was.
    worst: Vec<(String, f64, String)>,
}

/// A csv table: its path, its header and its rows, every field unquoted.
struct Table {
    /// The vendored path, for a failure to name.
    path: String,
    /// The column names.
    header: Vec<String>,
    /// The rows, each as wide as the header.
    rows: Vec<Vec<String>>,
}

/// One field of a csv written by R or by Phoenix: a string is quoted, a number is not.
///
/// The quotes are stripped and a field which still carries one is refused, which is also how a
/// quoted field holding a comma is caught: splitting on the comma would leave the halves with an
/// unbalanced quote.
#[track_caller]
fn field(raw: &str, what: &str) -> String {
    let trimmed = raw.trim();
    let unquoted = trimmed
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap_or(trimmed);
    assert!(
        !unquoted.contains('"'),
        "{what}: the field {raw} is quoted in a way this reader does not handle"
    );
    unquoted.to_string()
}

/// A vendored csv, read strictly: every row is as wide as the header.
#[track_caller]
fn table(rel: &str) -> Table {
    let text = read_text(rel);
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header: Vec<String> = lines
        .next()
        .unwrap_or_else(|| panic!("{rel} is empty"))
        .split(',')
        .map(|cell| field(cell, rel))
        .collect();
    let rows: Vec<Vec<String>> = lines
        .enumerate()
        .map(|(row, line)| {
            let cells: Vec<String> = line
                .split(',')
                .map(|cell| field(cell, &format!("{rel}, row {row}")))
                .collect();
            assert_eq!(
                cells.len(),
                header.len(),
                "{rel}: row {row} is not as wide as the header"
            );
            cells
        })
        .collect();
    assert!(!rows.is_empty(), "{rel} carries no row");
    Table {
        path: rel.to_string(),
        header,
        rows,
    }
}

impl Table {
    /// The index of a column, naming the header when there is none.
    #[track_caller]
    fn column(&self, name: &str) -> usize {
        self.header
            .iter()
            .position(|found| found == name)
            .unwrap_or_else(|| panic!("{}: no column {name}; it has {:?}", self.path, self.header))
    }
}

/// A number of a tool table, `NA` and an empty field being the missing value.
///
/// R writes `NA` where `pandas` reads a `NaN`, and Phoenix leaves the field empty.
#[track_caller]
fn number(raw: &str, what: &str) -> f64 {
    if raw.is_empty() || raw == "NA" || raw == "NaN" {
        return f64::NAN;
    }
    raw.parse::<f64>()
        .unwrap_or_else(|e| panic!("{what}: {raw} is no number: {e}"))
}

/// One enumeration of the wire, read from the spelling an oracle writes.
#[track_caller]
fn enum_of<T: serde::de::DeserializeOwned>(value: &str, what: &str) -> T {
    serde_json::from_value(serde_json::Value::String(value.to_string())).unwrap_or_else(|e| {
        panic!(
            "{what}: {value} is no {}: {e}",
            std::any::type_name::<T>().rsplit("::").next().unwrap_or("")
        )
    })
}

/// The digest of the two map sections of the crosswalk fixture.
///
/// `canonical` of `scripts/crosswalk_gate.py` builds the same text: one sorted line per leaf of
/// the sections, the path of the leaf, a tab and the string. Both sides build that form rather
/// than hashing the JSON text, whose spacing and escaping are a property of its writer.
fn digest_of(sections: &[(&str, &serde_json::Value)]) -> String {
    fn walk(path: &str, node: &serde_json::Value, lines: &mut Vec<String>) {
        match node {
            serde_json::Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                for key in keys {
                    walk(&format!("{path}/{key}"), &map[key], lines);
                }
            }
            serde_json::Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    walk(&format!("{path}/{index}"), item, lines);
                }
            }
            serde_json::Value::String(text) => lines.push(format!("{path}\t{text}")),
            other => panic!("the crosswalk fixture carries {other}, which is no string"),
        }
    }
    let mut lines: Vec<String> = Vec::new();
    for (name, section) in sections {
        walk(name, section, &mut lines);
    }
    lines.sort();
    let mut text = lines.join("\n");
    text.push('\n');
    Sha256::digest(text.as_bytes())
        .iter()
        .take(8)
        .fold(String::new(), |mut out, byte| {
            write!(out, "{byte:02x}").expect("writing into a String cannot fail");
            out
        })
}

/// The crosswalk fixture, with its header checked against the literals of this gate.
fn crosswalk() -> Crosswalk {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(CROSSWALK);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: the crosswalk fixture is missing: {e}", path.display()));
    let raw: serde_json::Value =
        serde_json::from_str(&text).expect("the crosswalk fixture is JSON");
    let own = digest_of(&[
        ("pknca_rules", &raw["pknca_rules"]),
        ("tools", &raw["tools"]),
    ]);
    let crosswalk: Crosswalk =
        serde_json::from_str(&text).expect("the crosswalk fixture is the JSON this gate reads");
    let header = &crosswalk.provenance;
    assert_eq!(
        header.pkpdutils_version, PKPDUTILS_VERSION,
        "the crosswalk fixture was extracted from another pkpdutils than this gate pins"
    );
    assert_eq!(
        header.crosswalk_digest, own,
        "the crosswalk fixture was edited by hand: its header and its maps disagree. Never \
         edit it, regenerate it with scripts/crosswalk_gate.py"
    );
    assert_eq!(
        header.crosswalk_digest, CROSSWALK_DIGEST,
        "the crosswalk fixture does not carry the maps this gate was written against"
    );
    assert_eq!(
        header.pkpdutils_source, "checkout",
        "the maps are extracted from a checkout of pkpdutils"
    );
    assert!(
        header
            .source_commit
            .as_ref()
            .is_some_and(|commit| commit.len() == 40),
        "the crosswalk fixture does not name the commit it was extracted from"
    );
    // the header describes the sections next to it, so it is checked against them as well
    assert_eq!(
        header.n_tools,
        crosswalk.tools.len(),
        "the tools of the header"
    );
    assert_eq!(
        header.n_route_rules,
        crosswalk.pknca_rules.route_rules.len(),
        "the route rules of the header"
    );
    let mut columns: BTreeSet<&str> = BTreeSet::new();
    for (key, tool) in &crosswalk.tools {
        for (field, found, declared) in [
            ("names", tool.names.len(), header.n_names.get(key)),
            (
                "names_by_route",
                tool.names_by_route.len(),
                header.n_names_by_route.get(key),
            ),
            ("percent", tool.percent.len(), header.n_percent.get(key)),
            ("unmapped", tool.unmapped.len(), header.n_unmapped.get(key)),
        ] {
            assert_eq!(
                Some(&found),
                declared,
                "{key}: the header counts other {field} than the section carries"
            );
        }
        columns.extend(tool.names.values().map(String::as_str));
        for routes in tool.names_by_route.values() {
            columns.extend(routes.values().map(String::as_str));
        }
    }
    assert_eq!(
        columns.len(),
        header.n_columns,
        "the header counts other columns than the sections carry"
    );
    assert_eq!(
        columns.len(),
        CROSSWALK_COLUMNS,
        "the crosswalk names other columns than this gate was written against"
    );
    assert_eq!(
        crosswalk
            .tools
            .keys()
            .map(String::as_str)
            .collect::<Vec<&str>>(),
        TOOL_IDS,
        "the crosswalk does not carry the tools this gate was written against"
    );
    crosswalk
}

/// Every scenario of the oracle, in the order the manifest writes them.
fn scenarios() -> Vec<Scenario> {
    let table = table(&format!("{BENCHMARKS}/scenarios.csv"));
    let index: BTreeMap<&str, usize> = [
        "scenario",
        "dataset",
        "subject",
        "time",
        "conc",
        "time_unit",
        "unit",
        "dose",
        "dose_unit",
        "route",
        "duration",
        "auc_method",
        "exclude_cmax",
        "winnonlin",
    ]
    .into_iter()
    .map(|name| (name, table.column(name)))
    .collect();
    let scenarios: Vec<Scenario> = table
        .rows
        .iter()
        .map(|row| {
            let cell = |name: &str| row[index[name]].as_str();
            let what = format!("the scenario {}", cell("scenario"));
            let duration = cell("duration");
            Scenario {
                scenario: cell("scenario").to_string(),
                dataset: cell("dataset").to_string(),
                subject: cell("subject").to_string(),
                time: cell("time").to_string(),
                conc: cell("conc").to_string(),
                time_unit: cell("time_unit").to_string(),
                unit: cell("unit").to_string(),
                dose: {
                    let dose = number(cell("dose"), &what);
                    assert!(dose.is_finite(), "{what}: it names no dose");
                    dose
                },
                dose_unit: cell("dose_unit").to_string(),
                route: enum_of(cell("route"), &what),
                duration: (!duration.is_empty()).then(|| number(duration, &what)),
                auc_method: enum_of(cell("auc_method"), &what),
                exclude_cmax: match cell("exclude_cmax") {
                    "true" => true,
                    "false" => false,
                    other => panic!("{what}: exclude_cmax is {other}, which is no boolean"),
                },
                winnonlin: cell("winnonlin").to_string(),
            }
        })
        .collect();
    assert_eq!(
        scenarios
            .iter()
            .map(|s| s.scenario.as_str())
            .collect::<Vec<&str>>(),
        SCENARIO_IDS,
        "the oracle does not carry the scenarios this gate was written against"
    );
    scenarios
}

/// The observations of a dataset, read from the vendored csv.
///
/// Read as strictly as `tests/validation.rs` reads the same two files: every subject has the
/// same number of points and its times increase, which is what makes the rectangular batch below
/// a faithful reading of the file rather than a reshaping of it.
#[track_caller]
fn observations(scenario: &Scenario) -> Observations {
    let path = format!("{BENCHMARKS}/{}", scenario.dataset);
    let table = table(&path);
    let (subject, time, value) = (
        table.column(&scenario.subject),
        table.column(&scenario.time),
        table.column(&scenario.conc),
    );
    let mut subjects: Vec<String> = Vec::new();
    let mut times: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut values: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (row, fields) in table.rows.iter().enumerate() {
        let what = format!("{path}, row {row}");
        let label = fields[subject].clone();
        if !times.contains_key(&label) {
            subjects.push(label.clone());
        }
        times
            .entry(label.clone())
            .or_default()
            .push(number(&fields[time], &what));
        values
            .entry(label)
            .or_default()
            .push(number(&fields[value], &what));
    }
    let n_time = times[&subjects[0]].len();
    let mut flat_times = Vec::with_capacity(subjects.len() * n_time);
    let mut flat_values = Vec::with_capacity(subjects.len() * n_time);
    for label in &subjects {
        let (t, v) = (&times[label], &values[label]);
        assert_eq!(
            t.len(),
            n_time,
            "{path}: the subject {label} has {} points where the first has {n_time}",
            t.len()
        );
        assert!(
            t.windows(2).all(|pair| pair[0] < pair[1]),
            "{path}: the times of the subject {label} are not increasing: {t:?}"
        );
        flat_times.extend_from_slice(t);
        flat_values.extend_from_slice(v);
    }
    Observations {
        subjects,
        n_time,
        times: flat_times,
        values: flat_values,
    }
}

/// The batch of a scenario: one sample per subject, one dose at time 0.
///
/// `analyse` of `test_benchmark_datasets.py:41-66` builds it from the csv with the flat dose of
/// the scenario and, for the infusion scenarios, its duration.
fn batch_of(scenario: &Scenario, observations: &Observations) -> Batch {
    let n_samples = observations.subjects.len();
    assert_eq!(
        scenario.route == Route::IvInfusion,
        scenario.duration.is_some(),
        "{}: an infusion is the route with a duration and the only one",
        scenario.scenario
    );
    Batch {
        contract: BATCH_CONTRACT.to_string(),
        sample_dims: vec![scenario.subject.clone()],
        sample_shape: vec![n_samples],
        n_time: observations.n_time,
        times: observations.times.clone(),
        values: observations.values.clone(),
        sd: None,
        se: None,
        n: None,
        nominal_time: None,
        lloq: None,
        max_doses: 1,
        dose_amount: vec![scenario.dose; n_samples],
        dose_time: vec![0.0; n_samples],
        dose_duration: vec![scenario.duration.unwrap_or(f64::NAN); n_samples],
        time_unit: scenario.time_unit.clone(),
        unit: scenario.unit.clone(),
        dose_unit: Some(scenario.dose_unit.clone()),
        substance: None,
        substances: None,
        route: Some(scenario.route),
        routes: None,
        tissue: None,
        coords: Vec::new(),
    }
}

/// The options of a scenario, `analyse` of `test_benchmark_datasets.py:58-65`.
///
/// The trapezoid rule and whether the maximum may enter a terminal window come from the
/// scenario; the rule which picks the window and its minimum size are the same everywhere.
fn options_of(scenario: &Scenario) -> NcaOptions {
    NcaOptions {
        auc_method: scenario.auc_method,
        terminal: TerminalPhase {
            method: TerminalMethod::BestFit,
            min_points: 3,
            exclude_cmax: scenario.exclude_cmax,
            ..TerminalPhase::default()
        },
        ..NcaOptions::default()
    }
}

/// The tool tables carry plain numbers, so the kernel's magnitudes have to be in the same units.
///
/// A tool computes from the numeric columns of the data and writes no unit at all: its
/// `Cl_obs` is the dose over the area in whatever `dose / (value * time)` happens to be. The
/// kernel reports a clearance in `liter / hour` and a volume in `liter` (`normalize_clearance`,
/// `normalize_volume` of `units.py`), and every other parameter in the raw unit of its
/// expression. The two sides are therefore comparable as plain numbers exactly when those two
/// normalizations are the identity for the units of the scenario, which is what this asserts.
/// A scenario measured in `ug/mL` with a dose in `g` would be out by a factor of a thousand and
/// fails here instead of in every clearance of the table.
#[track_caller]
fn the_normalizations_are_the_identity(scenario: &Scenario) {
    let what = format!("{}: the units of the scenario", scenario.scenario);
    let raw_clearance = format!(
        "({}) / (({}) * ({}))",
        scenario.dose_unit, scenario.unit, scenario.time_unit
    );
    let raw_volume = format!("({}) / ({})", scenario.dose_unit, scenario.unit);
    for (raw, normalized) in [
        (raw_clearance.as_str(), "liter / hour"),
        (raw_volume.as_str(), "liter"),
    ] {
        let ratio = in_unit(1.0, raw, normalized, &what);
        assert!(
            (ratio - 1.0).abs() <= 1e-12,
            "{what}: {raw} is {ratio} {normalized}, so the numbers of a tool table, which \
             carries no unit, are not the ones the kernel reports"
        );
    }
}

/// The column of a parameter of a result, `None` when the kernel does not produce it.
fn column<'a>(
    result: &'a AnalysisResult,
    name: &str,
) -> Option<&'a pkpdutils_abi::ParameterColumn> {
    result.parameters.iter().find(|column| column.name == name)
}

/// The spelling of a route on the wire, which the crosswalk keys its by-route maps with.
fn route_key(route: Route) -> String {
    serde_json::to_value(route)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .expect("a route is one string on the wire")
}

/// The column of every variable of a wide table, every route together.
///
/// `_read_wide` of `crosswalk.py:330-345` inverts the name map and adds every column of the
/// by-route map under the one variable it names, which is why a wide table needs no route:
/// `MRTEVLST` and `MRTIVLST` are both `mrt_last` and only one of them is ever in a table. The
/// extraction refuses a map which names two variables with one column, so this inverse is a map.
fn reverse_of(tool: &Tool) -> BTreeMap<&str, &str> {
    let pairs = tool.names.iter().chain(
        tool.names_by_route
            .iter()
            .flat_map(|(variable, routes)| routes.values().map(move |column| (variable, column))),
    );
    let mut reverse: BTreeMap<&str, &str> = BTreeMap::new();
    for (variable, column) in pairs {
        let had = reverse.insert(column.as_str(), variable.as_str());
        // the extraction already refuses it; a map which is read here rather than built here is
        // checked where it is read as well
        assert!(
            had.is_none_or(|had| had == variable),
            "the crosswalk names both {} and {variable} {column}, so the table of this tool \
             cannot be read back",
            had.unwrap_or_default()
        );
    }
    reverse
}

/// The pivoted table of WinNonlin or of NonCompart, `_read_wide` of `crosswalk.py:330-366`.
///
/// The columns before the first named one are the grouping columns of the profiles and become
/// the subject; every other column either has a variable or is one of the columns the crosswalk
/// declares unmapped, and anything else is a failure naming it rather than a column dropped in
/// silence.
#[track_caller]
fn read_wide(table: &Table, tool: &Tool, scenario: &Scenario) -> Expected {
    let reverse = reverse_of(tool);
    let known: BTreeSet<&str> = reverse
        .keys()
        .copied()
        .chain(tool.unmapped.iter().map(String::as_str))
        .collect();
    let groups: Vec<&str> = table
        .header
        .iter()
        .map(String::as_str)
        .take_while(|column| !known.contains(column))
        .collect();
    assert_eq!(
        groups,
        [scenario.subject.as_str()],
        "{}: the columns before the first parameter are not the subject alone",
        table.path
    );
    for name in &table.header[groups.len()..] {
        assert!(
            known.contains(name.as_str()),
            "{}: the column {name} has no pkpdutils name and is none of the columns the \
             crosswalk declares unmapped ({:?})",
            table.path,
            tool.unmapped
        );
    }
    let subject = table.column(&scenario.subject);
    let mut values: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    for (index, name) in table.header.iter().enumerate() {
        let Some(variable) = reverse.get(name.as_str()) else {
            continue;
        };
        let percent = tool.percent.contains(name);
        let mut by_subject: BTreeMap<String, f64> = BTreeMap::new();
        for (row, fields) in table.rows.iter().enumerate() {
            let what = format!("{}, row {row}, {name}", table.path);
            let value = number(&fields[index], &what);
            let value = if percent { value / 100.0 } else { value };
            assert!(
                by_subject.insert(fields[subject].clone(), value).is_none(),
                "{}: the subject {} has two rows",
                table.path,
                fields[subject]
            );
        }
        assert!(
            values.insert((*variable).to_string(), by_subject).is_none(),
            "{}: two of its columns are {variable}",
            table.path
        );
    }
    Expected {
        values,
        dropped: Vec::new(),
        duplicates: Vec::new(),
    }
}

/// The route the parameters of a PKNCA table imply, `_pknca_route` of `crosswalk.py:585-593`.
///
/// The rules are tried in the order the crosswalk carries them, which is the order of the
/// branches of that function; `scripts/crosswalk_gate.py` verifies the extracted rules against
/// it for every parameter of the maps and every pair of them.
#[track_caller]
fn recognized_route(rules: &PkncaRules, codes: &BTreeSet<&str>, what: &str) -> Route {
    for rule in &rules.route_rules {
        let named = rule.codes.iter().any(|code| codes.contains(code.as_str()));
        let prefixed = codes
            .iter()
            .any(|code| rule.prefixes.iter().any(|prefix| code.starts_with(prefix)));
        if named || prefixed {
            return rule.route;
        }
    }
    panic!("{what}: the route cannot be recognized from the PKNCA parameters {codes:?}")
}

/// The long table of PKNCA, `read_pknca_results` of `crosswalk.py:596-687`.
///
/// One row per subject, interval and parameter. A row PKNCA excluded is dropped, the route
/// decides whether `cl.obs` is `cl` or `cl_f`, and a profile which carries the same parameter
/// twice keeps the first of them, which `test_benchmark_datasets.py:74-77` asks for: these
/// exports combine the analysis of the observed and of the imputed concentrations at one
/// interval and the first `tmax` is the observed maximum.
#[track_caller]
fn read_long(table: &Table, tool: &Tool, rules: &PkncaRules, scenario: &Scenario) -> Expected {
    assert_eq!(
        rules.columns,
        ["start", "end", "PPTESTCD", "PPORRES", "exclude"],
        "the crosswalk carries other PKNCA columns than this reader was written against"
    );
    let first = table.column(&rules.columns[0]);
    let groups: Vec<&str> = table.header[..first].iter().map(String::as_str).collect();
    assert_eq!(
        groups,
        [scenario.subject.as_str()],
        "{}: the columns before `start` are not the subject alone",
        table.path
    );
    assert_eq!(
        table.header[first..],
        rules.columns[..],
        "{}: the columns from `start` on are not the ones of a PKNCA result table",
        table.path
    );
    let subject = table.column(&scenario.subject);
    let [start, end, code, value, exclude]: [usize; 5] = std::array::from_fn(|i| first + i);

    // `crosswalk.py:636-640` drops a row whose `exclude` is neither missing nor blank, where
    // missing is whatever `pandas` read as a `NaN`. The three spellings below are the ones R
    // writes and the ones all eight vendored tables carry; `pandas` would also read `null`,
    // `N/A` and `None` as missing, so a table written by another tool with one of those would
    // fail here as a number which is no number rather than be dropped. Widen the set when such
    // a table appears, never silently.
    let kept: Vec<&Vec<String>> = table
        .rows
        .iter()
        .filter(|fields| matches!(fields[exclude].as_str(), "" | "NA" | "NaN"))
        .collect();
    assert!(!kept.is_empty(), "{}: PKNCA excluded every row", table.path);
    let interval = (kept[0][start].as_str(), kept[0][end].as_str());
    for fields in &kept {
        assert_eq!(
            (fields[start].as_str(), fields[end].as_str()),
            interval,
            "{}: it carries more than the one interval {interval:?}, which the comparison \
             against a result of one row per subject drops",
            table.path
        );
    }

    let codes: BTreeSet<&str> = kept.iter().map(|fields| fields[code].as_str()).collect();
    let route = recognized_route(rules, &codes, &table.path);
    assert_eq!(
        route, scenario.route,
        "{}: its parameters are those of a {route:?} dose, where the scenario is {:?}",
        table.path, scenario.route
    );
    let key = route_key(route);
    let mut resolved: BTreeMap<&str, &str> = tool
        .names
        .iter()
        .map(|(variable, column)| (variable.as_str(), column.as_str()))
        .collect();
    for (variable, routes) in &tool.names_by_route {
        if let Some(column) = routes.get(&key) {
            resolved.insert(variable.as_str(), column.as_str());
        }
    }
    // `cl.obs` is `cl` after an intravenous dose and `cl_f` otherwise, so the variables of the
    // other route are dropped before the map is inverted (`crosswalk.py:649-657`)
    let other = if route.is_iv() {
        &rules.extravascular_only
    } else {
        &rules.intravenous_only
    };
    let mut reverse: BTreeMap<&str, &str> = BTreeMap::new();
    for (variable, column) in resolved {
        if other.contains(variable) {
            continue;
        }
        assert!(
            reverse.insert(column, variable).is_none(),
            "after a {route:?} dose PKNCA's {column} is more than one variable"
        );
    }
    for name in &codes {
        assert!(
            reverse.contains_key(name),
            "{}: the parameter {name} has no pkpdutils name",
            table.path
        );
    }

    let mut values: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    let mut duplicates: Vec<(String, String)> = Vec::new();
    for (row, fields) in kept.iter().enumerate() {
        let name = fields[code].as_str();
        let variable = reverse[name];
        let what = format!("{}, row {row}, {name}", table.path);
        let number = number(&fields[value], &what);
        let number = if tool.percent.contains(name) {
            number / 100.0
        } else {
            number
        };
        let by_subject = values.entry(variable.to_string()).or_default();
        if by_subject.contains_key(&fields[subject]) {
            duplicates.push((fields[subject].clone(), variable.to_string()));
            continue;
        }
        by_subject.insert(fields[subject].clone(), number);
    }
    Expected {
        values,
        dropped: Vec::new(),
        duplicates,
    }
}

/// Drop the variables the table has no number for at all.
///
/// `dropna(axis="columns", how="all")` of `test_benchmark_datasets.py:86`: NonCompart reports a
/// lag time column after a bolus and leaves it `NA` for every subject. A column with some `NaN`
/// stays and its `NaN` has to be one in the result as well.
fn drop_empty(mut read: Expected) -> Expected {
    let dropped: Vec<String> = read
        .values
        .iter()
        .filter(|(_, by_subject)| by_subject.values().all(|value| value.is_nan()))
        .map(|(variable, _)| variable.clone())
        .collect();
    for variable in &dropped {
        read.values.remove(variable);
    }
    read.dropped = dropped;
    read
}

/// The published table of one tool for one scenario, in the variables of `pkpdutils`.
///
/// `reference` of `test_benchmark_datasets.py:68-80`: the WinNonlin table is the one the
/// scenario names, the PKNCA and NonCompart tables are named after the scenario itself.
#[track_caller]
fn expected(tool: &str, scenario: &Scenario, crosswalk: &Crosswalk) -> Expected {
    let map = crosswalk
        .tools
        .get(tool)
        .unwrap_or_else(|| panic!("the crosswalk carries no map for {tool}"));
    let path = if tool == "winnonlin" {
        format!("{BENCHMARKS}/{tool}/{}", scenario.winnonlin)
    } else {
        format!("{BENCHMARKS}/{tool}/{}.csv", scenario.scenario)
    };
    let table = table(&path);
    drop_empty(match map.reader {
        Reader::Wide => read_wide(&table, map, scenario),
        Reader::Long => read_long(&table, map, &crosswalk.pknca_rules, scenario),
    })
}

/// Whether two numbers agree, `np.testing.assert_allclose` with `equal_nan`.
///
/// `|value - expected| <= atol + rtol * |expected|`, and two `NaN` agree while one does not: the
/// kernel either reports a parameter the tool leaves out or leaves out one the tool reports.
fn agree(expected: f64, value: f64) -> bool {
    if expected.is_nan() || value.is_nan() {
        return expected.is_nan() && value.is_nan();
    }
    (value - expected).abs() <= ATOL + RTOL * expected.abs()
}

/// The relative difference from a published number, absolute where that number is 0.
///
/// This is only reported, never compared against a tolerance; [`agree`] decides. The comparison
/// with zero is reached only once neither side is a `NaN`.
fn difference(expected: f64, value: f64) -> f64 {
    if expected.is_nan() && value.is_nan() {
        return 0.0;
    }
    if expected.is_nan() || value.is_nan() {
        return f64::INFINITY;
    }
    if expected == 0.0 {
        return value.abs();
    }
    (value - expected).abs() / expected.abs()
}

/// Run every scenario of the oracle against every tool.
fn run(crosswalk: &Crosswalk) -> Tally {
    let mut tally = Tally {
        scenarios: 0,
        tools: BTreeSet::new(),
        pairs: 0,
        columns: 0,
        comparisons: 0,
        dropped: Vec::new(),
        missing: Vec::new(),
        duplicates: Vec::new(),
        variables: BTreeSet::new(),
        worst: Vec::new(),
    };
    // every number which deviates, not just the first: a systematic divergence is then one run
    // and not one edit and run per number
    let mut failures: Vec<String> = Vec::new();
    for scenario in scenarios() {
        the_normalizations_are_the_identity(&scenario);
        let observations = observations(&scenario);
        let result = analyze_batch(&batch_of(&scenario, &observations), &options_of(&scenario))
            .unwrap_or_else(|e| panic!("{}: the kernel refused the batch: {e}", scenario.scenario));
        let rows: BTreeMap<&str, usize> = observations
            .subjects
            .iter()
            .enumerate()
            .map(|(row, label)| (label.as_str(), row))
            .collect();
        for tool in TOOL_IDS {
            let table = expected(tool, &scenario, crosswalk);
            let where_ = format!("{}, {tool}", scenario.scenario);
            assert!(
                table.values.len() >= MIN_COLUMNS,
                "{where_}: {} of its columns have a pkpdutils name, at least {MIN_COLUMNS} are \
                 required",
                table.values.len()
            );
            for name in &table.dropped {
                tally
                    .dropped
                    .push((scenario.scenario.clone(), tool.to_string(), name.clone()));
            }
            // the identity of the dropped duplicates, not only how many there were: the long
            // PKNCA tables drop one `tmax` per subject and the wide tables drop nothing, so
            // anything in between is a table which has changed shape
            let duplicated: BTreeSet<(&str, &str)> = table
                .duplicates
                .iter()
                .map(|(subject, variable)| (subject.as_str(), variable.as_str()))
                .collect();
            assert_eq!(
                duplicated.len(),
                table.duplicates.len(),
                "{where_}: a profile carries the same parameter three times"
            );
            let one_each: BTreeSet<(&str, &str)> = rows
                .keys()
                .map(|subject| (*subject, DUPLICATED_VARIABLE))
                .collect();
            assert!(
                duplicated.is_empty() || duplicated == one_each,
                "{where_}: it drops the duplicate rows {duplicated:?}, which is neither none \
                 nor one {DUPLICATED_VARIABLE} per subject"
            );
            for (subject, _) in &table.duplicates {
                tally.duplicates.push((
                    scenario.scenario.clone(),
                    tool.to_string(),
                    subject.clone(),
                ));
            }
            let mut worst = (0.0f64, "nothing was compared".to_string());
            for (variable, by_subject) in &table.values {
                let Some(column) = column(&result, variable) else {
                    tally.missing.push((
                        scenario.scenario.clone(),
                        tool.to_string(),
                        variable.clone(),
                    ));
                    continue;
                };
                assert_eq!(
                    by_subject
                        .keys()
                        .map(String::as_str)
                        .collect::<BTreeSet<&str>>(),
                    rows.keys().copied().collect::<BTreeSet<&str>>(),
                    "{where_}: the table and {} do not carry the same subjects",
                    scenario.dataset
                );
                for (subject, published) in by_subject {
                    let row = rows[subject.as_str()];
                    let value = *column.values.get(row).unwrap_or_else(|| {
                        panic!("{where_}: {variable} has no value for the sample {row}")
                    });
                    let relative = difference(*published, value);
                    if !agree(*published, value) {
                        failures.push(format!(
                            "{where_}, subject {subject}, {variable}: {value:?} against the \
                             published {published:?}, relative difference {relative:?}"
                        ));
                    }
                    if relative > worst.0 {
                        worst = (relative, format!("subject {subject}, {variable}"));
                    }
                    tally.comparisons += 1;
                }
                tally.columns += 1;
                tally.variables.insert(variable.clone());
            }
            tally.worst.push((where_, worst.0, worst.1));
            tally.tools.insert(tool.to_string());
            tally.pairs += 1;
        }
        tally.scenarios += 1;
    }
    assert!(
        failures.is_empty(),
        "{} of {} published numbers deviate by more than the tolerance{}:\n{}",
        failures.len(),
        tally.comparisons,
        if failures.len() > FAILURES_SHOWN {
            format!(", the first {FAILURES_SHOWN} of them")
        } else {
            String::new()
        },
        failures
            .iter()
            .take(FAILURES_SHOWN)
            .cloned()
            .collect::<Vec<String>>()
            .join("\n")
    );
    tally
}

#[test]
fn every_scenario_matches_all_three_tools() {
    let crosswalk = crosswalk();
    let tally = run(&crosswalk);
    assert_eq!(tally.scenarios, SCENARIOS, "the scenarios the gate ran");
    assert_eq!(
        tally
            .tools
            .iter()
            .map(String::as_str)
            .collect::<Vec<&str>>(),
        TOOL_IDS,
        "the tools the gate compared against"
    );
    assert_eq!(
        tally.tools.len(),
        TOOLS,
        "the tools the gate compared against"
    );
    assert_eq!(
        tally.pairs,
        SCENARIOS * TOOLS,
        "every scenario is compared against every tool"
    );
    assert_eq!(
        tally.missing,
        MISSING_ALLOWED
            .iter()
            .map(|(s, t, p)| (s.to_string(), t.to_string(), p.to_string()))
            .collect::<Vec<(String, String, String)>>(),
        "the parameters of a tool table the kernel does not produce"
    );
    assert_eq!(
        tally.dropped,
        DROPPED_ALL_NAN
            .iter()
            .map(|(s, t, p)| (s.to_string(), t.to_string(), p.to_string()))
            .collect::<Vec<(String, String, String)>>(),
        "the columns a tool reports as NA for every subject"
    );
    assert_eq!(
        tally.duplicates.len(),
        DUPLICATES_DROPPED,
        "the duplicate rows of the long PKNCA tables"
    );
    // which scenario and subject each of them was is asserted in `run`, against the subjects of
    // the dataset; here only that no other tool's table carries one at all
    assert_eq!(
        tally
            .duplicates
            .iter()
            .map(|(_, tool, _)| tool.as_str())
            .collect::<BTreeSet<&str>>(),
        BTreeSet::from([DUPLICATING_TOOL]),
        "a table other than the long {DUPLICATING_TOOL} one carries duplicate rows"
    );
    assert_eq!(
        tally
            .variables
            .iter()
            .map(String::as_str)
            .collect::<Vec<&str>>(),
        VARIABLES,
        "the three tools do not name the variables this gate was written against"
    );
    assert_eq!(tally.columns, COLUMNS, "the columns the gate compared");
    assert_eq!(
        tally.comparisons, COMPARISONS,
        "the numbers the gate compared"
    );
    // cargo captures the output of a test which passes, so this is read on a failure or under
    // `cargo test -- --nocapture`, and nothing the gate decides depends on it being seen: what
    // has to fail does so in an assertion above
    println!(
        "{COMPARISONS} numbers over {COLUMNS} columns and {} variables of {SCENARIOS} \
         scenarios and {TOOLS} tools:",
        tally.variables.len()
    );
    for (where_, worst, what) in &tally.worst {
        println!("  {where_}: worst relative difference {worst:e} on {what}");
    }
}
