//! Timing of the kernel on synthetic batches, serial and over row shards.
//!
//! What it measures is one call of [`pkpdutils_core::analyze_batch_in_shards`]: the contract and
//! option checks, the row loop and the assembly of the result, which is everything a host pays
//! for after it has built the batch. Building the batch is not timed, and neither is any
//! serialization, since no host of this crate speaks JSON to it yet.
//!
//! The protocol is the one of `spike/RESULTS.md` and of `BENCHMARKS.md`, which records what this
//! example printed: three warm up calls, then three rounds of twenty timed calls each, the
//! median of every round and the median of the three round medians as the number of the shape.
//! The peak resident set size of the whole process (`VmHWM`) and the load average of the machine
//! are printed with it, since a measurement taken on a busy machine is worth nothing: compare
//! the load average of two runs before comparing their times.
//!
//! ```bash
//! # the serial build, which is what a wasm host runs
//! cargo run --release --example kernel_timing
//! # the sharded build, with the plan `automatic_shards` chooses
//! cargo run --release --example kernel_timing --features parallel
//! # the crossover: every shape over a sweep of explicit shard counts
//! cargo run --release --example kernel_timing --features parallel -- sweep
//! ```
//!
//! The batches are generated here and the same generator is written in python for the comparison
//! of `BENCHMARKS.md`, down to the order of the arithmetic, so that both sides analyse the same
//! numbers: a xorshift64 stream of uniform values, a geometric decline per row and a
//! multiplicative noise per point, an intravenous bolus of 100 mg at time zero. The checksum
//! every run prints - the sum of `auc_last` over the rows - is what says that they did.

use std::time::Instant;

use pkpdutils_abi::{BATCH_CONTRACT, Batch, NcaOptions, Route};
use pkpdutils_core::{SHARD_WORK_THRESHOLD, analyze_batch_in_shards, automatic_shards};

/// The four shapes of `spike/RESULTS.md`, which `BENCHMARKS.md` reports.
///
/// One curve as a host of a single analysis sends it, a thousand and ten thousand dense curves
/// as a parameter scan produces them, and a hundred thousand short curves as a study of a
/// population has them.
const SHAPES: [(usize, usize); 4] = [(1, 12), (1_000, 200), (10_000, 200), (100_000, 12)];

/// The shapes of the sweep which the threshold is chosen from.
///
/// The work of a shape is its product, so the sweep walks from 12 to 2 000 000 observations with
/// both a short and a dense curve at every size: the threshold is work and not rows, and a sweep
/// of one curve length could not show that.
const SWEEP: [(usize, usize); 20] = [
    (1, 12),
    (10, 12),
    (20, 12),
    (50, 12),
    (100, 12),
    (200, 12),
    (500, 12),
    (1_000, 12),
    (2_000, 12),
    (5_000, 12),
    (10_000, 12),
    (100_000, 12),
    (1, 200),
    (5, 200),
    (10, 200),
    (20, 200),
    (50, 200),
    (100, 200),
    (1_000, 200),
    (10_000, 200),
];

/// The shard counts the sweep times every shape over, before the number of rows caps them.
///
/// `1` is the serial loop, `2` and `4` are the small plans a rule which grew with the work would
/// ask for, and the rest is the pool: as many shards as it has threads and four times that,
/// which is what `automatic_shards` asks for above its threshold.
fn shard_counts(rows: usize) -> Vec<usize> {
    let pool = threads();
    let mut counts = vec![1, 2, 4, pool, pool * 4];
    counts.retain(|&s| s <= rows);
    counts.sort_unstable();
    counts.dedup();
    if counts.is_empty() {
        counts.push(1);
    }
    counts
}

/// The number of timed calls of one round.
const RUNS: usize = 20;

/// The number of rounds.
const ROUNDS: usize = 3;

/// The number of calls before the first round, whose times are thrown away.
const WARM_UPS: usize = 3;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sweep = args.iter().any(|a| a == "sweep");
    println!(
        "# pkpdutils-core kernel timing\n\n\
         feature parallel: {}\nlogical CPUs: {}\nshard work threshold: {SHARD_WORK_THRESHOLD}\n\
         load average: {}\n",
        cfg!(feature = "parallel"),
        threads(),
        load_average()
    );
    if sweep {
        run_sweep();
    } else {
        run_shapes(&shapes(&args));
    }
    // of the whole process, so a peak which belongs to one shape alone needs a run of that shape
    // alone: `kernel_timing 1000x200`
    println!("\npeak RSS: {}", peak_rss());
}

/// The shapes of the arguments, [`SHAPES`] where they name none.
///
/// A shape is `<rows>x<points>`, as the table prints it without the spaces; an argument which is
/// no shape is ignored, since the only other argument is the mode.
fn shapes(args: &[String]) -> Vec<(usize, usize)> {
    let given: Vec<(usize, usize)> = args
        .iter()
        .filter_map(|a| a.split_once('x'))
        .filter_map(|(rows, points)| Some((rows.parse().ok()?, points.parse().ok()?)))
        .collect();
    if given.is_empty() {
        SHAPES.to_vec()
    } else {
        given
    }
}

/// Every shape, serial and with the plan the kernel would choose by itself.
fn run_shapes(shapes: &[(usize, usize)]) {
    println!("| rows x points | work | shards | median ms | rounds ms | us per row | checksum |");
    println!("|---|---:|---:|---:|---|---:|---:|");
    for &(rows, points) in shapes {
        let batch = batch_of(rows, points);
        let automatic = automatic_shards(rows, points);
        let mut plans = vec![1, automatic];
        plans.dedup();
        for shards in plans {
            report(&batch, rows, points, shards);
        }
    }
}

/// Every shape of the sweep over every shard count, which is where the threshold comes from.
fn run_sweep() {
    println!("| rows x points | work | shards | median ms | rounds ms | us per row | checksum |");
    println!("|---|---:|---:|---:|---|---:|---:|");
    for (rows, points) in SWEEP {
        let batch = batch_of(rows, points);
        for shards in shard_counts(rows) {
            report(&batch, rows, points, shards);
        }
    }
}

/// Time one batch over one plan and print the row of the table.
fn report(batch: &Batch, rows: usize, points: usize, shards: usize) {
    let (median, rounds, checksum) = timing(batch, shards);
    let per_row = median * 1_000.0 / rows as f64;
    let rounds = rounds
        .iter()
        .map(|t| format!("{t:.4}"))
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "| {rows} x {points} | {} | {shards} | {median:.4} | {rounds} | {per_row:.3} | {checksum:.12e} |",
        rows.saturating_mul(points)
    );
}

/// The median of the medians of [`ROUNDS`] rounds of [`RUNS`] calls, in milliseconds.
///
/// The checksum of the result is returned with it and printed, so that nothing of the call can
/// be optimized away and so that a run of another build, or of the python package on the same
/// data, is known to have analysed the same numbers.
fn timing(batch: &Batch, shards: usize) -> (f64, Vec<f64>, f64) {
    let options = NcaOptions::default();
    let mut checksum = 0.0;
    let call = || {
        let result = analyze_batch_in_shards(batch, &options, shards).expect("the batch runs");
        result
            .parameters
            .iter()
            .find(|p| p.name == "auc_last")
            .map_or(f64::NAN, |p| {
                p.values.iter().filter(|v| v.is_finite()).sum()
            })
    };
    for _ in 0..WARM_UPS {
        checksum = call();
    }
    let mut medians = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let mut times = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let start = Instant::now();
            checksum = call();
            times.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        medians.push(median(&mut times));
    }
    let mut copy = medians.clone();
    (median(&mut copy), medians, checksum)
}

/// The median of a sample, which it sorts.
///
/// The sample of a round is twenty times and of a run three round medians, so the middle of an
/// even sample is the mean of the two around it and of an odd one the one in the middle.
fn median(times: &mut [f64]) -> f64 {
    times.sort_by(f64::total_cmp);
    match times.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => times[n / 2],
        n => (times[n / 2 - 1] + times[n / 2]) / 2.0,
    }
}

/// A batch of `n_samples` curves of `n_time` points, the same batch the python script builds.
///
/// Every number here is the result of a fixed sequence of IEEE operations on the xorshift
/// stream, with no transcendental function anywhere, so that the python side computes the same
/// batch bit for bit: a times array of `0.25 + 23.75 j / (n - 1)`, a geometric decline of a per
/// row ratio and a multiplicative noise of five percent per point.
///
/// The decline is four to six time constants over the whole curve, whatever the number of
/// points, so that a dense curve is a dense sampling of the same profile and not a deeper one:
/// the terminal values of a two hundred point curve would otherwise fall below the `1e-8` of
/// `TerminalPhase.relative_warning` and the python side would log a line per call, which is time
/// the kernel does not spend.
fn batch_of(n_samples: usize, n_time: usize) -> Batch {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut uniform = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11;
        bits as f64 / 9_007_199_254_740_992.0
    };
    let work = n_samples.saturating_mul(n_time);
    let mut times = Vec::with_capacity(work);
    let mut values = Vec::with_capacity(work);
    for _ in 0..n_samples {
        let mut c = 100.0 / (20.0 + 40.0 * uniform());
        let q = 1.0 - ((4.0 + 2.0 * uniform()) / (n_time - 1).max(1) as f64).min(0.5);
        for j in 0..n_time {
            let t = if n_time > 1 {
                0.25 + 23.75 * j as f64 / (n_time - 1) as f64
            } else {
                0.25
            };
            times.push(t);
            values.push(c * (0.95 + 0.1 * uniform()));
            c *= q;
        }
    }
    Batch {
        contract: BATCH_CONTRACT.to_string(),
        sample_dims: vec!["individual".to_string()],
        sample_shape: vec![n_samples],
        n_time,
        times,
        values,
        sd: None,
        se: None,
        n: None,
        nominal_time: None,
        lloq: None,
        max_doses: 1,
        dose_amount: vec![100.0; n_samples],
        dose_time: vec![0.0; n_samples],
        dose_duration: vec![f64::NAN; n_samples],
        time_unit: "hr".to_string(),
        unit: "mg/l".to_string(),
        dose_unit: Some("mg".to_string()),
        substance: Some("drug".to_string()),
        substances: None,
        route: Some(Route::IvBolus),
        routes: None,
        tissue: None,
        coords: Vec::new(),
    }
}

/// The size of the pool the shards would run on.
///
/// What rayon starts its global pool with, and what the kernel plans against, unless a host
/// installed a pool of its own or `RAYON_NUM_THREADS` says otherwise; a build without the
/// `parallel` feature runs every shard here, one after the other, however many this is.
fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
}

/// The peak resident set size of this process, from `/proc/self/status`.
fn peak_rss() -> String {
    read_proc("/proc/self/status")
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .map_or_else(
            || "unknown".to_string(),
            |line| line.trim_start_matches("VmHWM:").trim().to_string(),
        )
}

/// The load average of the machine over the last minute, from `/proc/loadavg`.
fn load_average() -> String {
    read_proc("/proc/loadavg")
        .split_whitespace()
        .next()
        .unwrap_or("unknown")
        .to_string()
}

/// One file of `/proc`, empty where there is no `/proc` to read.
fn read_proc(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}
