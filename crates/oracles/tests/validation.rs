//! The kernel reproduces the published Phoenix WinNonlin and PKNCA values.
//!
//! The oracle is `testdata/oracles/tests/data/validation/reference.json` together with the two
//! datasets it names, and this is the port of `tests/nca/test_validation.py` and of the
//! comparison of `scripts/validation.py`: eight reference groups over the theophylline and the
//! indomethacin profiles, every number compared at the tolerance the oracle carries for it.
//!
//! What this gate adds to the one of `tests/reference.rs` is the terminal selection. That gate
//! runs `TerminalMethod::AllAfterTmax`, where the window is not chosen at all; every group here
//! runs `TerminalMethod::BestFit`, so the published `lambda_z`, `lambda_z_n_points`,
//! `lambda_z_t_first` and everything derived from them pin the selection rule: the candidates
//! within an absolute `tie_tolerance` of the best adjusted R squared tie, and of those the
//! earliest start wins. A strict comparison on the adjusted R squared instead reproduces most
//! of these numbers and not all of them.
//!
//! Nothing is skipped except the two groups named in [`GROUPS_SKIPPED`], which need
//! `ParameterResult.summarize` and `partial_auc`, neither of which this kernel carries. A
//! parameter the kernel does not produce, a subject the dataset does not have and a number
//! which deviates by more than its tolerance are each a failure which names what was missing or
//! by how much it differed.

mod common;
// the unit helpers of the numeric gates, which `tests/oracles_present.rs` does not declare
#[path = "common/units.rs"]
mod units;

use std::collections::{BTreeMap, BTreeSet};

use pkpdutils_abi::{
    AnalysisResult, AucMethod, BATCH_CONTRACT, Batch, NcaOptions, Route, TerminalMethod,
    TerminalPhase,
};
use pkpdutils_core::analyze_batch;
use serde::Deserialize;

use common::read_text;
use units::in_unit;

/// The oracle, repo-relative as the vendoring mirrors it.
const REFERENCE: &str = "tests/data/validation/reference.json";

/// The reference groups of the oracle, the three sections of the file together.
const GROUPS: usize = 8;

/// Every group of the oracle, sorted, so that a group which is renamed or replaced by a copy of
/// another fails here instead of being compared under the wrong options.
const GROUP_IDS: [&str; GROUPS] = [
    "indometh-winnonlin-linear",
    "indometh-winnonlin-linear-infusion",
    "indometh-winnonlin-linear-log",
    "theoph-pknca",
    "theoph-pknca-partial",
    "theoph-pknca-summary",
    "theoph-winnonlin-linear",
    "theoph-winnonlin-linear-log",
];

/// The groups this gate runs.
const GROUPS_COMPARED: usize = 6;

/// The two groups out of the scope of a single dose kernel, skipped by their key.
///
/// `theoph-pknca-summary` is the summary of `pk.nca` over the twelve subjects, which needs the
/// geometric statistics of `ParameterResult.summarize`, and `theoph-pknca-partial` is the area
/// over the window `[0, 12.12]`, which needs `partial_auc`. Neither is part of this kernel, and
/// both are skipped here by name with their numbers counted, which is the only skip this gate
/// allows: anything else which does not line up is a failure.
const GROUPS_SKIPPED: [&str; 2] = ["theoph-pknca-partial", "theoph-pknca-summary"];

/// The numbers of the skipped groups: the one partial area and the eleven statistics.
const NUMBERS_SKIPPED: usize = 12;

/// The numbers this gate compares, pinned so that a truncated or mis-parsed oracle fails here
/// instead of passing vacuously.
const NUMBERS_COMPARED: usize = 1043;

/// Every reference of the file, `TOTAL_REFERENCES` of `tests/nca/test_validation.py:127` and of
/// `docs/python-reference/gates.md` section 1b, written as the number both of them name rather
/// than derived from the two literals above.
const TOTAL_REFERENCES: usize = 1055;

/// The compared and the skipped numbers have to account for every reference of the file, which
/// holds when the gate is built and not only when it has read the oracle.
const _: () = assert!(NUMBERS_COMPARED + NUMBERS_SKIPPED == TOTAL_REFERENCES);

/// The subjects of the compared groups: twelve theophylline profiles twice, six indomethacin
/// profiles three times and the two subjects the PKNCA vignette prints.
const SUBJECTS_COMPARED: usize = 44;

/// How many deviating numbers the failure names. The gate compares every number before it
/// fails, so a kernel which has drifted lists what drifted rather than the first of it, and a
/// kernel which is broken does not print a thousand lines to say so.
const FAILURES_SHOWN: usize = 10;

/// The tolerance every reference of a group carries, which the gate asserts rather than reads,
/// so that a loosened oracle fails instead of hiding a deviation.
///
/// 1e-6 for the five Phoenix WinNonlin groups, whose tables carry 8 to 15 significant digits,
/// 1e-4 for the per-subject numbers of the PKNCA vignette and 1e-2 for its summary, which is
/// printed to three digits.
const TOLERANCES: [(&str, f64); GROUPS] = [
    ("indometh-winnonlin-linear", 1e-6),
    ("indometh-winnonlin-linear-infusion", 1e-6),
    ("indometh-winnonlin-linear-log", 1e-6),
    ("theoph-pknca", 1e-4),
    ("theoph-pknca-partial", 1e-4),
    ("theoph-pknca-summary", 1e-2),
    ("theoph-winnonlin-linear", 1e-6),
    ("theoph-winnonlin-linear-log", 1e-6),
];

/// Every parameter the compared groups reference, sorted.
///
/// The counts above would not notice a parameter renamed into another the kernel also produces,
/// so the names are pinned as well. The twenty-one of `WINNONLIN_PARAMETERS`
/// (`test_validation.py:34-56`), the `tlag`, `cl_f` and `vz_f` of the extravascular groups and
/// the `c0`, `auc_back_extrap_fraction`, `cl`, `vz` and `vss` of the intravenous ones.
const PARAMETERS: [&str; 29] = [
    "auc_all",
    "auc_back_extrap_fraction",
    "auc_extrap_fraction",
    "auc_inf_dn",
    "auc_inf_obs",
    "auc_inf_pred",
    "auc_last",
    "aumc_inf",
    "aumc_last",
    "c0",
    "cl",
    "cl_f",
    "clast",
    "cmax",
    "cmax_dn",
    "lambda_z",
    "lambda_z_n_points",
    "lambda_z_r2",
    "lambda_z_r2_adj",
    "lambda_z_t_first",
    "lambda_z_t_last",
    "mrt",
    "thalf",
    "tlag",
    "tlast",
    "tmax",
    "vss",
    "vz",
    "vz_f",
];

/// The groups which carry every subject of their dataset, `FULL_CASES` of
/// `test_validation.py:129-135`.
const FULL_GROUPS: [&str; 5] = [
    "indometh-winnonlin-linear",
    "indometh-winnonlin-linear-infusion",
    "indometh-winnonlin-linear-log",
    "theoph-winnonlin-linear",
    "theoph-winnonlin-linear-log",
];

/// The oracle, read into the shape this gate was written against.
///
/// Every object the comparison depends on is `deny_unknown_fields`, so a key the file gains
/// fails here rather than being ignored.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    /// What the file is, asserted to be there and not read further.
    description: String,
    /// The publications the numbers were transcribed from, by their key.
    sources: BTreeMap<String, Source>,
    /// The two datasets, by their key.
    datasets: BTreeMap<String, Dataset>,
    /// The groups of one value per subject and parameter.
    cases: BTreeMap<String, Group>,
    /// The group of a partial area.
    partial_cases: BTreeMap<String, Group>,
    /// The group of the statistics over a whole batch.
    summary_cases: BTreeMap<String, Group>,
}

/// The provenance of a published number.
///
/// The fields named here are the ones every source has to fill; the `date` and the `file` of
/// the oracle are not, since the PKNCA vignette is a living web page with neither, and they are
/// left out rather than carried and never read. This is the one object of the oracle which is
/// not `deny_unknown_fields`: a field added to the provenance of a source is no reason to fail
/// a numeric gate.
#[derive(Deserialize)]
struct Source {
    /// The tool the numbers were produced with.
    tool: String,
    /// The author of the publication.
    author: String,
    /// Its title.
    title: String,
    /// Where it is published.
    url: String,
    /// When it was retrieved.
    retrieved: String,
    /// How many digits the published numbers carry.
    precision: String,
    /// The settings of the analysis which produced them.
    settings: String,
}

/// One dataset of the oracle: the file, its columns and the dose every subject received.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    /// The vendored csv, relative to the root of the python repository.
    file: String,
    /// The number of subjects it carries.
    n_subjects: usize,
    /// The column of the subject.
    subject_column: String,
    /// The column of the time.
    time_column: String,
    /// The column of the measured value.
    value_column: String,
    /// The unit of the times.
    time_unit: String,
    /// The unit of the values.
    unit: String,
    /// The substance.
    substance: String,
    /// The route of the dataset, which a group may override.
    route: Route,
    /// The dose every subject is analysed against.
    dose: f64,
    /// The unit of the dose.
    dose_unit: String,
    /// Why that dose, asserted to be there and not read further.
    dose_note: String,
}

/// One reference group: the dataset, the options and the published numbers.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Group {
    /// The key of its dataset.
    dataset: String,
    /// The tool the numbers come from.
    comparator: String,
    /// The key of the source of the group.
    source: String,
    /// What the group is, asserted to be there and not read further.
    note: String,
    /// The options the published analysis ran with.
    options: OptionsSpec,
    /// The route the dataset is analysed under, when it is not the one of the dataset.
    #[serde(default)]
    route: Option<Route>,
    /// The infusion duration, with an `iv_infusion` route.
    #[serde(default)]
    dose_duration: Option<f64>,
    /// The window of a partial area, which only the partial group has.
    #[serde(default)]
    window: Option<[f64; 2]>,
    /// The dimension a summary group reduces over, which only the summary group has, and which
    /// says that its references are flat rather than one map per subject
    /// (`entries` of `scripts/validation.py:225-233`).
    #[serde(default)]
    dim: Option<String>,
    /// Differences which are known and accepted. Every list of the oracle is empty and this
    /// gate has no machinery for an expected failure, so a difference which appears here is a
    /// failure and the entries are never read.
    #[serde(default)]
    known_differences: Vec<serde_json::Value>,
    /// The published numbers, one map per subject or, for a summary group, flat.
    references: BTreeMap<String, serde_json::Value>,
}

/// The options of a group, `options_of` of `scripts/validation.py:128-137`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OptionsSpec {
    /// The trapezoid rule.
    auc_method: AucMethod,
    /// The terminal regression.
    terminal: TerminalSpec,
}

/// The terminal regression of a group.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TerminalSpec {
    /// The rule which picks the window.
    method: TerminalMethod,
    /// The minimum number of points of a window.
    min_points: u32,
    /// Whether the maximum may enter a window.
    exclude_cmax: bool,
}

/// One published number.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    /// The published value, in `unit`.
    value: f64,
    /// The unit it is written in.
    unit: String,
    /// The key of the source it was transcribed from.
    source: String,
    /// The tolerance it is compared at.
    tolerance: f64,
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

/// What the gate did, which the test pins against its literals.
struct Tally {
    /// The groups it ran.
    groups: usize,
    /// The groups it skipped, by their key.
    skipped: Vec<String>,
    /// The subject and group pairs it ran.
    subjects: usize,
    /// The numbers it compared.
    compared: usize,
    /// The numbers of the skipped groups.
    skipped_numbers: usize,
    /// The largest deviation of every group it ran, with the subject and the parameter it was
    /// on, which the test prints: a gate whose numbers are reported is one whose next drift is
    /// visible before it crosses a tolerance.
    worst: Vec<(String, f64, String)>,
}

/// The relative deviation from a published number, absolute where that number is 0.
///
/// `deviation` of `scripts/validation.py:211-223`, which the python gate asserts against the
/// tolerance of the reference. A `NaN` on one side only is an infinite deviation and never a
/// pass: the analysis either reports a parameter the comparator does not or misses one it
/// reports. The comparison with zero is reached only once neither side is a `NaN`.
fn deviation(expected: f64, value: f64) -> f64 {
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

/// The oracle, with the structure this gate was written against asserted on the way in.
fn document() -> Document {
    let document: Document =
        serde_json::from_str(&read_text(REFERENCE)).expect("the validation oracle is JSON");
    assert!(
        !document.description.is_empty(),
        "the oracle says what it is"
    );
    for (key, source) in &document.sources {
        for (field, value) in [
            ("tool", &source.tool),
            ("author", &source.author),
            ("title", &source.title),
            ("retrieved", &source.retrieved),
            ("precision", &source.precision),
            ("settings", &source.settings),
        ] {
            assert!(
                !value.is_empty(),
                "the source {key} does not name its {field}"
            );
        }
        assert!(
            source.url.starts_with("https://"),
            "the source {key} is published at {}, which is no web address",
            source.url
        );
    }
    for (key, dataset) in &document.datasets {
        assert!(
            !dataset.dose_note.is_empty(),
            "the dataset {key} does not say why its dose is {}",
            dataset.dose
        );
    }
    document
}

/// Every group of the oracle by its key, the three sections of the file together.
///
/// `SECTIONS` of `scripts/validation.py:60`. The sections differ in the shape of their
/// references and in nothing else, so the gate keeps one map and reads the shape from `dim`, as
/// `entries` does.
fn groups(document: &Document) -> BTreeMap<&str, &Group> {
    let mut groups: BTreeMap<&str, &Group> = BTreeMap::new();
    for section in [
        &document.cases,
        &document.partial_cases,
        &document.summary_cases,
    ] {
        for (key, group) in section {
            assert!(
                groups.insert(key.as_str(), group).is_none(),
                "the group {key} is in two sections of the oracle"
            );
        }
    }
    assert_eq!(
        groups.keys().copied().collect::<Vec<&str>>(),
        GROUP_IDS,
        "the oracle does not carry the groups this gate was written against"
    );
    groups
}

/// The references of a group as `(subject, parameter, reference)`, a summary group under the
/// subject `all`.
///
/// `entries` of `scripts/validation.py:225-233`.
#[track_caller]
fn entries(key: &str, group: &Group) -> Vec<(String, String, Reference)> {
    let mut entries = Vec::new();
    for (name, value) in &group.references {
        if group.dim.is_some() {
            let reference: Reference = serde_json::from_value(value.clone())
                .unwrap_or_else(|e| panic!("{key}: the statistic {name} is no reference: {e}"));
            entries.push(("all".to_string(), name.clone(), reference));
            continue;
        }
        let parameters: BTreeMap<String, Reference> = serde_json::from_value(value.clone())
            .unwrap_or_else(|e| panic!("{key}: the subject {name} is no map of references: {e}"));
        for (parameter, reference) in parameters {
            entries.push((name.clone(), parameter, reference));
        }
    }
    entries
}

/// The observations of a dataset, read from the vendored csv.
///
/// The csv is read strictly: no quoting, every row the width of the header and every subject
/// the same number of points, which is what makes the rectangular batch below a faithful
/// reading of the file rather than a reshaping of it.
#[track_caller]
fn observations(dataset: &Dataset) -> Observations {
    let text = read_text(&dataset.file);
    assert!(
        !text.contains('"'),
        "{}: the reader of this gate does not handle a quoted field",
        dataset.file
    );
    let mut lines = text.lines().filter(|line| !line.is_empty());
    let header: Vec<&str> = lines
        .next()
        .unwrap_or_else(|| panic!("{} is empty", dataset.file))
        .split(',')
        .collect();
    let column = |name: &str| {
        header
            .iter()
            .position(|found| *found == name)
            .unwrap_or_else(|| panic!("{}: no column {name}; it has {header:?}", dataset.file))
    };
    let (subject, time, value) = (
        column(&dataset.subject_column),
        column(&dataset.time_column),
        column(&dataset.value_column),
    );
    let mut subjects: Vec<String> = Vec::new();
    let mut times: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut values: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (row, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(
            fields.len(),
            header.len(),
            "{}: row {row} is not as wide as the header",
            dataset.file
        );
        let number = |index: usize| {
            fields[index].parse::<f64>().unwrap_or_else(|e| {
                panic!(
                    "{}: row {row}, {}: {} is no number: {e}",
                    dataset.file, header[index], fields[index]
                )
            })
        };
        let label = fields[subject].to_string();
        if !times.contains_key(&label) {
            subjects.push(label.clone());
        }
        times.entry(label.clone()).or_default().push(number(time));
        values.entry(label).or_default().push(number(value));
    }
    assert_eq!(
        subjects.len(),
        dataset.n_subjects,
        "{} carries other subjects than the oracle says",
        dataset.file
    );
    let n_time = times[&subjects[0]].len();
    let mut flat_times = Vec::with_capacity(subjects.len() * n_time);
    let mut flat_values = Vec::with_capacity(subjects.len() * n_time);
    for label in &subjects {
        let (t, v) = (&times[label], &values[label]);
        assert_eq!(
            t.len(),
            n_time,
            "{}: the subject {label} has {} points where the first has {n_time}",
            dataset.file,
            t.len()
        );
        assert!(
            t.windows(2).all(|pair| pair[0] < pair[1]),
            "{}: the times of the subject {label} are not increasing: {t:?}",
            dataset.file
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

/// The batch of a group: one sample per subject, one dose at time 0.
///
/// `batch_of` of `scripts/validation.py:100-126` builds the batch from the csv with a flat
/// dose per subject and, for the infusion group, a duration; a group may analyse the dataset
/// under another route than the one it was measured with.
///
/// Where python differs is only in the time axis: `Timecourses.from_dataframe` pivots the
/// subjects onto the union of their times and pads the values it has none for with `NaN`, while
/// the batch of the wire carries the times of every sample. The two reach the kernel as the
/// same curve, since `pack_valid` keeps the points whose time and value are both finite and
/// drops the rest, so the padded grid packs down to exactly the points of the csv.
fn batch_of(dataset: &Dataset, group: &Group, observations: &Observations) -> Batch {
    let n_samples = observations.subjects.len();
    let route = group.route.unwrap_or(dataset.route);
    let duration = group.dose_duration.unwrap_or(f64::NAN);
    assert_eq!(
        route == Route::IvInfusion,
        group.dose_duration.is_some(),
        "an infusion is the route with a duration and the only one"
    );
    Batch {
        contract: BATCH_CONTRACT.to_string(),
        sample_dims: vec![dataset.subject_column.clone()],
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
        dose_amount: vec![dataset.dose; n_samples],
        dose_time: vec![0.0; n_samples],
        dose_duration: vec![duration; n_samples],
        time_unit: dataset.time_unit.clone(),
        unit: dataset.unit.clone(),
        dose_unit: Some(dataset.dose_unit.clone()),
        substance: Some(dataset.substance.clone()),
        substances: None,
        route: Some(route),
        routes: None,
        tissue: None,
        coords: Vec::new(),
    }
}

/// The options of a group, as the oracle writes them.
fn options_of(group: &Group) -> NcaOptions {
    NcaOptions {
        auc_method: group.options.auc_method,
        terminal: TerminalPhase {
            method: group.options.terminal.method,
            min_points: group.options.terminal.min_points,
            exclude_cmax: group.options.terminal.exclude_cmax,
            ..TerminalPhase::default()
        },
        ..NcaOptions::default()
    }
}

/// One parameter of one sample of a result, converted into `unit`.
///
/// A parameter the kernel does not produce is a failure which names it and lists what the
/// kernel did produce, never a skip.
#[track_caller]
fn converted(result: &AnalysisResult, name: &str, row: usize, unit: &str, what: &str) -> f64 {
    let column = result
        .parameters
        .iter()
        .find(|column| column.name == name)
        .unwrap_or_else(|| {
            let produced: Vec<&str> = result
                .parameters
                .iter()
                .map(|column| column.name.as_str())
                .collect();
            panic!("{what}: the kernel produces no parameter {name}; it produces {produced:?}")
        });
    let value = *column
        .values
        .get(row)
        .unwrap_or_else(|| panic!("{what}: {name} has no value for the sample {row}"));
    in_unit(value, &column.unit, unit, what)
}

/// Run every compared group of the oracle against the kernel.
fn run(document: &Document) -> Tally {
    let groups = groups(document);
    let mut tally = Tally {
        groups: 0,
        // filled by the groups the loop really skips, not from the constant it is checked
        // against, so that a group which stops being skipped is noticed
        skipped: Vec::new(),
        subjects: 0,
        compared: 0,
        skipped_numbers: 0,
        worst: Vec::new(),
    };
    let tolerances: BTreeMap<&str, f64> = TOLERANCES.into_iter().collect();
    let mut parameters: BTreeSet<String> = BTreeSet::new();
    let mut total = 0usize;
    // every number which deviates, not just the first: a systematic divergence is then one run
    // and not one edit and run per number. Only the numbers are collected this way; a malformed
    // oracle still fails where it is read, since nothing sensible follows it.
    let mut failures: Vec<String> = Vec::new();
    for (key, group) in groups {
        let entries = entries(key, group);
        total += entries.len();
        assert!(
            group.known_differences.is_empty(),
            "{key}: the oracle accepts a known difference, which this gate has no machinery for"
        );
        let tolerance = *tolerances
            .get(key)
            .unwrap_or_else(|| panic!("{key}: the gate pins no tolerance for this group"));
        for (subject, parameter, reference) in &entries {
            assert!(
                document.sources.contains_key(&reference.source),
                "{key}, subject {subject}, {parameter}: the source {} is not one of the file",
                reference.source
            );
            // exact equality of two doubles on purpose: the tolerance the oracle carries has
            // to be the one this gate pins and not merely close to it, and JSON cannot write
            // the NaN which would make the comparison meaningless
            assert_eq!(
                reference.tolerance, tolerance,
                "{key}, subject {subject}, {parameter}: the oracle compares at {} where this \
                 gate pins {tolerance}",
                reference.tolerance
            );
        }
        if GROUPS_SKIPPED.contains(&key) {
            // the two groups out of the scope of the kernel, skipped by name with their numbers
            // counted; the reason is in the doc comment of GROUPS_SKIPPED
            assert!(
                group.window.is_some() || group.dim.is_some(),
                "{key} is skipped although it is an ordinary group of one value per subject"
            );
            tally.skipped_numbers += entries.len();
            tally.skipped.push(key.to_string());
            continue;
        }
        assert!(
            group.window.is_none() && group.dim.is_none(),
            "{key} is compared although it is a partial or a summary group"
        );
        assert!(!group.note.is_empty(), "{key} does not say what it is");
        assert!(
            !group.comparator.is_empty(),
            "{key} does not name its comparator"
        );
        assert!(
            document.sources.contains_key(&group.source),
            "{key}: the source {} is not one of the file",
            group.source
        );
        let dataset = document
            .datasets
            .get(&group.dataset)
            .unwrap_or_else(|| panic!("{key}: the oracle has no dataset {}", group.dataset));
        let observations = observations(dataset);
        let result = analyze_batch(&batch_of(dataset, group, &observations), &options_of(group))
            .unwrap_or_else(|e| panic!("{key}: the kernel refused the batch: {e}"));
        let rows: BTreeMap<&str, usize> = observations
            .subjects
            .iter()
            .enumerate()
            .map(|(row, label)| (label.as_str(), row))
            .collect();
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut by_subject: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        let mut worst = (0.0f64, "nothing was compared".to_string());
        for (subject, parameter, reference) in &entries {
            let row = *rows.get(subject.as_str()).unwrap_or_else(|| {
                panic!(
                    "{key}: the reference names the subject {subject}, which {} does not carry",
                    dataset.file
                )
            });
            let what = format!("{key}, subject {subject}, {parameter}");
            let value = converted(&result, parameter, row, &reference.unit, &what);
            let relative = deviation(reference.value, value);
            // `deviation` returns 0, an infinity or a finite non-negative number and never a
            // NaN, so this is the exact negation of the `<=` the python gate asserts
            if relative > tolerance {
                failures.push(format!(
                    "{what}: {value:?} against {} {} of {}, deviation {relative:?} above \
                     {tolerance:e}",
                    reference.value, reference.unit, reference.source
                ));
            }
            if relative > worst.0 {
                worst = (relative, format!("subject {subject}, {parameter}"));
            }
            parameters.insert(parameter.clone());
            seen.insert(subject.as_str());
            by_subject
                .entry(subject.as_str())
                .or_default()
                .insert(parameter.as_str());
            tally.compared += 1;
        }
        // `test_full_cases_cover_every_subject` of `test_validation.py:198-212`: a WinNonlin
        // group carries every subject of its dataset with the same parameters. The PKNCA
        // vignette prints two subjects and one more parameter for the second of them, so
        // neither holds there.
        if FULL_GROUPS.contains(&key) {
            assert_eq!(
                seen.len(),
                dataset.n_subjects,
                "{key} is a full group and must carry every subject of {}",
                dataset.file
            );
            let shapes: BTreeSet<&BTreeSet<&str>> = by_subject.values().collect();
            assert_eq!(
                shapes.len(),
                1,
                "{key}: its subjects do not carry the same parameters"
            );
        }
        tally.groups += 1;
        tally.subjects += seen.len();
        tally.worst.push((key.to_string(), worst.0, worst.1));
    }
    assert_eq!(
        total, TOTAL_REFERENCES,
        "the oracle must carry {TOTAL_REFERENCES} references"
    );
    assert_eq!(
        parameters.iter().map(String::as_str).collect::<Vec<&str>>(),
        PARAMETERS,
        "the compared groups do not reference the parameters this gate was written against"
    );
    assert!(
        failures.is_empty(),
        "{} of {} published numbers deviate by more than their tolerance{}:\n{}",
        failures.len(),
        tally.compared,
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
fn every_published_parameter_of_the_validation_datasets_is_reproduced() {
    let document = document();
    let tally = run(&document);
    assert_eq!(
        tally.groups + tally.skipped.len(),
        GROUPS,
        "every group of the oracle is either run or skipped by name"
    );
    assert_eq!(tally.groups, GROUPS_COMPARED, "the groups the gate ran");
    assert_eq!(tally.skipped, GROUPS_SKIPPED, "the groups it skipped");
    assert_eq!(
        tally.skipped_numbers, NUMBERS_SKIPPED,
        "the numbers of the skipped groups"
    );
    assert_eq!(
        tally.subjects, SUBJECTS_COMPARED,
        "the subject and group pairs the gate ran"
    );
    assert_eq!(
        tally.compared, NUMBERS_COMPARED,
        "the numbers the gate compared"
    );
    assert_eq!(
        tally.worst.len(),
        GROUPS_COMPARED,
        "every group the gate ran reports its largest deviation"
    );
    // cargo captures the output of a test which passes, so this is read on a failure or under
    // `cargo test -- --nocapture`, and nothing the gate decides depends on it being seen: what
    // has to fail does so in an assertion above
    println!("{NUMBERS_COMPARED} numbers compared over {GROUPS_COMPARED} groups:");
    for (key, worst, what) in &tally.worst {
        println!("  {key}: worst deviation {worst:e} on {what}");
    }
}
