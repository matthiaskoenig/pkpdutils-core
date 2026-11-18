//! The row shards of a batch analysis: how many, and what runs them.
//!
//! The rows of a batch are independent - [`crate::parameters::compute_row`] reads one row and
//! writes one row of parameters - so the row loop is cut into contiguous shards and the shards
//! are handed to a thread pool. Nothing is shared but the batch, which is read only; every shard
//! has its own [`Scratch`] and writes its own stretch of the output, in the row order of the
//! batch. A shard boundary therefore cannot change a number, which
//! `the_parallel_and_the_serial_path_produce_the_same_bits` asserts bit for bit.
//!
//! The pool is [rayon](https://docs.rs/rayon) and it is behind the `parallel` feature, which is
//! off by default. A wasm instance has no threads to run a shard on, so a wasm build takes the
//! kernel without the feature (`cargo build --no-default-features`) and the host runs one kernel
//! instance per Web Worker instead; with the feature on a wasm target rayon reports one thread
//! and runs every shard in the calling thread, which is measured in `BENCHMARKS.md`. That is
//! rayon's own fallback: a pool which cannot spawn a thread, which is what a wasm runtime
//! answers with `io::ErrorKind::Unsupported`, is replaced by a one-thread pool that runs the
//! work in the caller (`rayon-core/src/registry.rs`). Rayon's comment on it warns that the
//! fallback "probably won't work for non-blocking calls like `spawn` or `broadcast_spawn`",
//! which is the boundary this module stays inside: [`in_parallel`] only ever calls
//! `into_par_iter().collect_into_vec()`, a blocking call which has finished before
//! [`rows_of`] returns, and the kernel never spawns anything which outlives it. Without the
//! feature the shards of a plan run one after the other here, which computes the same numbers
//! and is what the identity assertion compares against in a build that has no pool at all.
//!
//! Python's analogue is `src/pkpdutils/parallel.py`, whose `NCA_WORKER_THRESHOLD` counts rows
//! alone: 20 000 of them, so a thousand dense curves of two hundred points stay in the calling
//! thread. [`SHARD_WORK_THRESHOLD`] counts the work instead, rows times time points, and the
//! figure comes from the measurement in `BENCHMARKS.md`.

use crate::parameters::Scratch;

/// The work of a batch below which the automatic plan is a single shard.
///
/// The work of a batch is its number of samples times its number of time points, which is the
/// number of observations the row loop walks. Below this figure the dispatch of the pool costs
/// more than the rows do, so [`automatic_shards`] answers `1` and the rows run in the calling
/// thread; above it the kernel asks for the whole pool.
///
/// The figure is measured, not guessed: `crates/core/examples/kernel_timing.rs` times the
/// sharded against the serial path over a sweep of shapes and `BENCHMARKS.md` records the run it
/// comes from. On that machine the shards lose below about 1 200 observations, win by a fifth at
/// 2 400, by a third at 4 000 and by more than two from 10 000 on. 4 096 is above that crossing
/// and not at it, and the reason is the size of what is given up: between 2 400 and 4 096
/// observations the shards do win, but by 0.027 and 0.049 ms, while a batch of 1 200
/// observations pays 0.035 ms for a pool it cannot fill. Shapes that small move by about that
/// much between runs - the sweep reads 10 x 12 at 0.131 ms where a fresh pinned process reads
/// 0.056 - so a threshold above the noisy part of the ramp costs a batch just below it a few
/// hundredths of a millisecond and buys every batch below that the certainty of never starting
/// the pool at all. Python's `NCA_WORKER_THRESHOLD` is 20 000 rows and counts no points at all,
/// which leaves a thousand curves of two hundred points - 56 ms of work there, 6x off what the
/// shards do here - in a single thread.
pub const SHARD_WORK_THRESHOLD: usize = 4_096;

/// How many shards the automatic plan asks for per thread of the pool.
///
/// One shard per thread balances nothing: a row whose terminal regression has many candidate
/// windows costs several times what a short row costs, and the thread which drew a shard of them
/// finishes last. A few shards per thread let the pool steal the rest, and the only cost of a
/// shard is one set of row buffers, which is a handful of vectors the width of a row.
///
/// Measured like the threshold, in the shard sweep of `BENCHMARKS.md`. It is not public: a host
/// chooses a plan by calling [`automatic_shards`] or by passing a count of its own, and the
/// figure a count of its own is compared against is the one in `BENCHMARKS.md`, not a constant
/// to compute with.
pub(crate) const SHARDS_PER_THREAD: usize = 4;

/// The number of threads the shards of a plan would run on.
///
/// `rayon::current_num_threads` is the size of the pool the shards would land in, which is the
/// global pool, or the pool of a host which installed its own, or what `RAYON_NUM_THREADS` says.
/// A build without the `parallel` feature has no pool and answers `1`.
pub(crate) fn threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads().max(1)
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// The number of shards [`crate::analyze_batch`] cuts a batch of this shape into.
///
/// `n_samples` and `n_time` are `pkpdutils_abi::Batch::n_samples` and
/// `pkpdutils_abi::Batch::n_time`, so their product is the work of the batch. A batch whose work
/// is below [`SHARD_WORK_THRESHOLD`] is one shard, and so is every batch in a build without the
/// `parallel` feature, which has no second thread to run one on.
///
/// Above the threshold it is the whole pool at once, four shards per thread of it
/// (`SHARDS_PER_THREAD`) and never more shards than there are rows. A plan which grows with the
/// work instead - two shards at the threshold, twenty at ten times it - was measured against
/// this one and was slower everywhere in between: a batch of twelve thousand observations runs
/// in 0.47 ms over four shards and in 0.26 ms over eighty, because the shards are not equally
/// expensive and the pool steals what a thread has left over.
///
/// The work is tested before the pool is asked how large it is, so a batch below the threshold
/// never starts the pool: `rayon::current_num_threads` builds the global pool, and a host which
/// only ever sends small batches should not find twenty threads in its process because it asked
/// for a plan.
///
/// ```
/// # use pkpdutils_core::automatic_shards;
/// assert_eq!(automatic_shards(1, 12), 1);
/// ```
pub fn automatic_shards(n_samples: usize, n_time: usize) -> usize {
    if n_samples.saturating_mul(n_time) < SHARD_WORK_THRESHOLD {
        return 1;
    }
    let threads = threads();
    if threads <= 1 {
        return 1;
    }
    threads
        .saturating_mul(SHARDS_PER_THREAD)
        .min(n_samples)
        .max(1)
}

/// The rows of one shard of a plan of `shards` shards over `rows` rows.
///
/// The last shard of a plan is the short one, since the others take the ceiling; a plan of no
/// shards at all is a plan of one, and so is a plan of more shards than rows. Never `0`, which
/// `Iterator::step_by` answers with a panic, and a panic in the kernel is a wasm trap.
pub(crate) fn shard_rows(rows: usize, shards: usize) -> usize {
    rows.div_ceil(shards.max(1)).max(1)
}

/// What `row` answers for every row of a batch, in the row order of the batch.
///
/// The answer of a row is whatever `row` returns: the parameters of the row for an analysis, the
/// parameters of a replicate for the bootstrap and the propagated standard errors of a row for
/// the delta method, which all shard the same way and differ only in what one row costs.
///
/// `row` computes one row into the [`Scratch`] it is given, which it may overwrite: a shard
/// reuses one set of buffers for every row it holds, so the kernel allocates per shard and not
/// per row. A plan of more than one shard runs them on the rayon pool where the build has one
/// ([`in_parallel`]) and one after the other where it has not ([`in_order`]); a plan of one shard
/// never touches the pool at all, so a host which parallelizes outside the kernel does not even
/// start it. Row `i` of the answer is `row(i, ..)` in every one of those cases, which is what
/// keeps them bit identical.
pub(crate) fn rows_of<T, F>(n_rows: usize, n_time: usize, shards: usize, row: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize, &mut Scratch) -> T + Send + Sync,
{
    let per_shard = shard_rows(n_rows, shards);
    #[cfg(feature = "parallel")]
    if shards > 1 && per_shard < n_rows {
        return in_parallel(n_rows, n_time, per_shard, &row);
    }
    in_order(n_rows, n_time, per_shard, &row)
}

/// The shards of a plan, one after the other in the calling thread.
fn in_order<T, F>(n_rows: usize, n_time: usize, per_shard: usize, row: &F) -> Vec<T>
where
    T: Send,
    F: Fn(usize, &mut Scratch) -> T + Send + Sync,
{
    let mut out = Vec::with_capacity(n_rows);
    for first in (0..n_rows).step_by(per_shard.max(1)) {
        let mut scratch = Scratch::new(n_time);
        for i in first..n_rows.min(first.saturating_add(per_shard)) {
            out.push(row(i, &mut scratch));
        }
    }
    out
}

/// The shards of a plan on the rayon pool.
///
/// `collect_into_vec` takes an `IndexedParallelIterator` and writes row `i` to position `i`, so
/// the row order is the type system's here and not a convention of this function; the vector is
/// filled without being initialized first, which matters at a hundred thousand rows. The two
/// adapters are the shards: `with_min_len` keeps the pool from splitting one further and
/// `map_init` gives each one the single [`Scratch`] it then reuses for its rows. The pool may
/// still run several shards on one thread, which is the point of cutting more of them than there
/// are threads.
#[cfg(feature = "parallel")]
fn in_parallel<T, F>(n_rows: usize, n_time: usize, per_shard: usize, row: &F) -> Vec<T>
where
    T: Send,
    F: Fn(usize, &mut Scratch) -> T + Send + Sync,
{
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    let mut out = Vec::new();
    (0..n_rows)
        .into_par_iter()
        .with_min_len(per_shard.max(1))
        .map_init(|| Scratch::new(n_time), |scratch, i| row(i, scratch))
        .collect_into_vec(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use pkpdutils_abi::{
        AnalysisResult, AucMethod, BATCH_CONTRACT, Batch, Blq, BlqHandling, Coordinate, NcaOptions,
        Route, SampleCoordinate, TerminalMethod, WindowLabel,
    };

    use crate::assemble::RowResult;
    use crate::parameters::RowParameters;
    use crate::shard::{
        SHARD_WORK_THRESHOLD, SHARDS_PER_THREAD, automatic_shards, rows_of, shard_rows, threads,
    };
    use crate::{analyze_batch, analyze_batch_in_shards};

    /// A batch of `n_samples` curves of `n_time` points, built to exercise every row path.
    ///
    /// The curves are mono-exponential with a deterministic pseudo-random volume, rate and
    /// noise, so the shape of the analysis of a row - where its peak falls, how many points its
    /// terminal regression may choose from - differs from row to row. Every fourth row carries a
    /// hole, every seventh a limit of quantification above its own tail, every eleventh no dose,
    /// the first row is empty and the second has a single point: a shard boundary which falls on
    /// one of them must not change anything. The routes cycle, so the rows of one shard do not
    /// all report the same parameters.
    fn mixed_batch(n_samples: usize, n_time: usize) -> Batch {
        let mut times = Vec::with_capacity(n_samples * n_time);
        let mut values = Vec::with_capacity(n_samples * n_time);
        let mut lloq = Vec::with_capacity(n_samples);
        let mut routes = Vec::with_capacity(n_samples);
        let mut amounts = Vec::with_capacity(n_samples);
        let mut durations = Vec::with_capacity(n_samples);
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            // xorshift64, so that the batch is the same on every machine and in every build
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11;
            bits as f64 / (1u64 << 53) as f64
        };
        for row in 0..n_samples {
            let v = 20.0 + 40.0 * next();
            let k = 0.05 + 0.25 * next();
            for j in 0..n_time {
                let t = 0.25 + 23.75 * (j as f64) / ((n_time - 1).max(1) as f64);
                let value = 100.0 / v * (-k * t).exp() * (0.95 + 0.1 * next());
                let gone = row % 4 == 3 && j == n_time / 2 || row == 0 || (row == 1 && j > 0);
                times.push(if row == 0 { f64::NAN } else { t });
                values.push(if gone { f64::NAN } else { value });
            }
            lloq.push(if row % 7 == 6 { 0.5 } else { f64::NAN });
            let route = match row % 3 {
                0 => Route::IvBolus,
                1 => Route::Oral,
                _ => Route::IvInfusion,
            };
            routes.push(route);
            amounts.push(if row % 11 == 10 { f64::NAN } else { 100.0 });
            durations.push(if route == Route::IvInfusion {
                0.5
            } else {
                f64::NAN
            });
        }
        Batch {
            contract: BATCH_CONTRACT.to_string(),
            sample_dims: vec!["subject".to_string()],
            sample_shape: vec![n_samples],
            n_time,
            times,
            values,
            sd: None,
            se: None,
            n: None,
            nominal_time: None,
            lloq: Some(lloq),
            max_doses: 1,
            dose_amount: amounts,
            dose_time: vec![0.0; n_samples],
            dose_duration: durations,
            time_unit: "h".to_string(),
            unit: "mg/l".to_string(),
            dose_unit: Some("mg".to_string()),
            substance: Some("caffeine".to_string()),
            substances: None,
            route: None,
            routes: Some(routes),
            tissue: Some("plasma".to_string()),
            coords: vec![SampleCoordinate {
                name: "subject".to_string(),
                dims: vec!["subject".to_string()],
                values: Coordinate::Int(
                    (0..n_samples)
                        .map(|i| i64::try_from(i).unwrap_or(-1))
                        .collect(),
                ),
            }],
        }
    }

    /// The number of points of the batches of these tests.
    const N_TIME: usize = 24;

    /// A batch whose work is above the threshold, so that the automatic plan shards it.
    fn above_the_threshold() -> Batch {
        let rows = SHARD_WORK_THRESHOLD
            .div_ceil(N_TIME)
            .saturating_add(1)
            .max(64);
        mixed_batch(rows, N_TIME)
    }

    /// Every float of a result, column by column, named by where it sits.
    fn floats(result: &AnalysisResult) -> Vec<(String, usize, f64)> {
        let mut out = Vec::new();
        for column in &result.parameters {
            for (row, &value) in column.values.iter().enumerate() {
                out.push((column.name.clone(), row, value));
            }
        }
        for variable in &result.point_variables {
            for (row, &value) in variable.values.iter().enumerate() {
                out.push((variable.name.clone(), row, value));
            }
        }
        for coordinate in &result.coords {
            if let Coordinate::Float(values) = &coordinate.values {
                for (row, &value) in values.iter().enumerate() {
                    out.push((coordinate.name.clone(), row, value));
                }
            }
        }
        out
    }

    /// The option sets the identity assertion runs over, named.
    ///
    /// The default is the best fit regression of a batch with no limit of quantification; the
    /// others take the four other paths through a row which depend on something outside the
    /// row's own numbers, since those are the ones a shard boundary could get wrong: the manual
    /// point indices and the last n points, which are resolved against the length of each row,
    /// the window of a single sample, which is looked up by the label of the row, and a limit of
    /// quantification, whose rule walks the row against its peak. Each set changes exactly one
    /// of those row dependent paths, so the assertion that a set computes something other than
    /// the default pins that path and not a bundle of them. The linear up log down rule rides
    /// along with the last n set because it is row independent: every row takes it the same way,
    /// which is not something a shard boundary could get wrong.
    fn option_sets(n_samples: usize) -> Vec<(&'static str, NcaOptions)> {
        let mut manual = NcaOptions::default();
        manual.terminal.method = TerminalMethod::Manual;
        manual.terminal.points = Some(vec![-5, -4, -3, -2, -1]);

        let mut last_n = NcaOptions::default();
        last_n.terminal.method = TerminalMethod::LastN;
        last_n.terminal.n_points = Some(4);
        last_n.terminal.exclude_cmax = false;
        last_n.auc_method = AucMethod::LinearLog;

        let mut windows = BTreeMap::new();
        for row in (0..n_samples).step_by(3) {
            let first = if row % 2 == 0 { 6.0 } else { 10.0 };
            windows.insert(
                WindowLabel::Int(i64::try_from(row).unwrap_or(-1)),
                (first, 24.0),
            );
        }
        let mut by_label = NcaOptions::default();
        by_label.terminal.windows = Some(windows);

        let below_limit = NcaOptions {
            lloq: Some(1.0),
            blq: Blq::Handling(BlqHandling::ZeroBeforeTmax),
            ..NcaOptions::default()
        };

        vec![
            ("the default options", NcaOptions::default()),
            ("manual point indices", manual),
            ("the last four points, linear up log down", last_n),
            ("a window per sample", by_label),
            ("a limit of quantification, zero before tmax", below_limit),
        ]
    }

    #[test]
    fn the_parallel_and_the_serial_path_produce_the_same_bits() {
        // Compared by `to_bits` deliberately: a parameter of a row whose route has none is a
        // NaN, and `==` would call two of them different, where this asks what the sharded run
        // really wrote. The row order is part of it, since `floats` carries the position of
        // every value: a shard which wrote its rows anywhere but where they belong fails here.
        let batch = above_the_threshold();
        let n_samples = batch.sample_shape[0];
        let mut reference: Option<Vec<(String, usize, f64)>> = None;
        for (set, options) in option_sets(n_samples) {
            let serial = analyze_batch_in_shards(&batch, &options, 1)
                .unwrap_or_else(|err| panic!("one shard runs {set}: {err:?}"));
            let left = floats(&serial);
            assert!(!left.is_empty(), "{set}: the serial run reports no numbers");
            // a set which computes what the default computes would widen nothing
            match &reference {
                None => reference = Some(left.clone()),
                Some(first) => assert!(
                    first.len() != left.len()
                        || first
                            .iter()
                            .zip(&left)
                            .any(|((_, _, a), (_, _, b))| a.to_bits() != b.to_bits()),
                    "{set} analyses the batch exactly as the default options do"
                ),
            }
            for shards in [2usize, 3, 7, 16, 64, n_samples] {
                let sharded = analyze_batch_in_shards(&batch, &options, shards)
                    .unwrap_or_else(|err| panic!("{shards} shards run {set}: {err:?}"));
                let right = floats(&sharded);
                assert_eq!(
                    left.len(),
                    right.len(),
                    "{set}: {shards} shards report another number of values"
                );
                for ((name, row, a), (_, _, b)) in left.iter().zip(&right) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{set}: {name} of row {row} is {b} over {shards} shards and {a} serially"
                    );
                }
                // the flags, the boolean and the text sections, the coordinates and the
                // attributes; `AnalysisResult` compares the floats of its columns by the bits
                assert_eq!(
                    serial.flags, sharded.flags,
                    "{set}: {shards} shards, the flags"
                );
                assert_eq!(
                    serial, sharded,
                    "{set}: {shards} shards report another result"
                );
            }
        }
    }

    #[test]
    fn the_automatic_plan_is_what_an_explicit_one_computes() {
        // `analyze_batch` is `analyze_batch_in_shards` with the automatic plan, and the batch is
        // above the threshold, so this is the entry point every host calls against the serial
        // result of the same batch.
        let batch = above_the_threshold();
        let options = NcaOptions::default();
        assert!(automatic_shards(batch.sample_shape[0], batch.n_time) > 1 || threads() == 1);
        let serial = analyze_batch_in_shards(&batch, &options, 1).expect("one shard runs");
        assert_eq!(
            analyze_batch(&batch, &options).expect("the batch runs"),
            serial
        );
    }

    #[test]
    fn a_batch_below_the_threshold_runs_serially() {
        // the plan is one shard, whatever the build and however many threads it has: the
        // dispatch of a small batch costs more than its rows do
        assert_eq!(automatic_shards(1, 12), 1);
        assert_eq!(automatic_shards(1, 0), 1);
        assert_eq!(automatic_shards(0, 12), 1);
        let rows = (SHARD_WORK_THRESHOLD - 1) / N_TIME;
        assert_eq!(automatic_shards(rows, N_TIME), 1);
        // and a batch whose work overflows is one shard per thread and not one per row
        assert!(automatic_shards(usize::MAX, usize::MAX) <= threads() * SHARDS_PER_THREAD);

        // and so is the path, which this observes rather than infers: every row of that plan
        // records the thread it ran on, and all of them are the thread which asked. In a build
        // with the feature this is the assertion that the pool is not reached at all below the
        // threshold; in a build without one there is no pool to reach.
        let here = std::thread::current().id();
        let elsewhere = AtomicUsize::new(0);
        let computed = rows_of(rows, N_TIME, automatic_shards(rows, N_TIME), |_, _| {
            if std::thread::current().id() != here {
                elsewhere.fetch_add(1, Ordering::Relaxed);
            }
            RowResult {
                route: None,
                dose_amount: f64::NAN,
                parameters: RowParameters::EMPTY,
                intervals: Vec::new(),
            }
        });
        assert_eq!(computed.len(), rows, "a row of the plan was not computed");
        assert_eq!(
            elsewhere.load(Ordering::Relaxed),
            0,
            "rows of a one shard plan ran on another thread"
        );
    }

    #[test]
    fn a_batch_above_the_threshold_is_cut_into_more_than_one_shard() {
        let rows = SHARD_WORK_THRESHOLD.div_ceil(N_TIME).saturating_add(1);
        let shards = automatic_shards(rows, N_TIME);
        if threads() > 1 {
            assert!(
                shards > 1,
                "{rows} x {N_TIME} is above the threshold and was planned as {shards} shards"
            );
            assert!(shards <= rows, "{shards} shards for {rows} rows");
        } else {
            // without the feature, and on a machine of one core, there is nothing to run a
            // second shard on
            assert_eq!(shards, 1);
        }
        // a batch large enough to saturate the machine is cut into no more shards than the pool
        // can run at once, so that the row loop does not pay for shards nobody runs
        assert!(automatic_shards(1_000_000, 200) <= threads() * SHARDS_PER_THREAD);
    }

    #[test]
    fn a_shard_plan_never_asks_for_a_slice_of_no_rows() {
        // `Iterator::step_by(0)` is a panic, and a panic in the kernel is a wasm trap, so the
        // row count of a shard is at least one for every input, including the impossible ones
        for rows in [0usize, 1, 2, 7, 1000] {
            for shards in [0usize, 1, 3, 64, usize::MAX] {
                let per_shard = shard_rows(rows, shards);
                assert!(per_shard >= 1, "{rows} rows over {shards} shards");
                assert!(
                    per_shard.saturating_mul(shards.max(1)) >= rows,
                    "{per_shard} rows per shard leave rows of {rows} unanalysed"
                );
            }
        }
        assert_eq!(shard_rows(1000, 4), 250);
        assert_eq!(shard_rows(1001, 4), 251);
        assert_eq!(shard_rows(3, 4), 1);
    }
}
