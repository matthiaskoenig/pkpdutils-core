/**
 * A pool of workers, one instance of the module in each.
 *
 * A wasm instance is single threaded and the module imports nothing, so a page which wants
 * several cores runs several instances. This pool compiles the module once, posts it to every
 * worker, cuts the rows of a batch into one shard per worker and merges the answers:
 *
 *     import { createPool } from "pkpdutils-core/pool";
 *     const pool = await createPool({ module: bytes });
 *     const answer = await pool.analyze(request);
 *     pool.close();
 *
 * The analysis of a row does not depend on any other row, which is what makes the cut sound: the
 * python host shards the same way, over its thread pool. A summary is a reduction over the
 * samples, so `summarize` runs whole on one worker.
 */

import { KernelError, RefusedError } from "./loader.mjs";

/** The URL of the worker of this package, for a page which builds its own workers. */
export const WORKER_URL = new URL("./worker.mjs", import.meta.url);

/**
 * How many workers to run by default: the cores of the machine, at most eight.
 *
 * Eight is the cap of `resolve_workers` of the python host, for the same reason: the shards of a
 * batch stop paying before the cores of a big machine run out, and a page has a user interface
 * to keep responsive.
 *
 * @returns {number}
 */
export function defaultWorkers() {
  const cores = globalThis.navigator?.hardwareConcurrency ?? 4;
  return Math.max(1, Math.min(8, cores));
}

/** The arrays of a batch which carry one block per sample over the time axis. */
const TIME_ARRAYS = ["times", "values", "sd", "se", "n", "nominal_time", "lloq"];

/** The arrays of a batch which carry one block per sample over the dose axis. */
const DOSE_ARRAYS = ["dose_amount", "dose_time", "dose_duration"];

/** The arrays of a batch which carry one value per sample. */
const SAMPLE_ARRAYS = ["substances", "routes"];

/**
 * Whether the rows of a request can be cut, and why not when they cannot.
 *
 * A batch of one sample dimension is a list of rows and cuts along it. A batch of several
 * dimensions would have to be cut along the first and its coordinates rewritten per dimension,
 * and a coordinate of another dimension than the one being cut cannot be sliced at all; both run
 * whole on one worker rather than coming back subtly wrong.
 *
 * @param {object} batch the batch of a request
 * @returns {string | null} the reason it cannot be cut, or `null`
 */
export function shardable(batch) {
  if (batch.sample_dims.length !== 1) {
    return `a batch of ${batch.sample_dims.length} sample dimensions is not cut into shards`;
  }
  const [dim] = batch.sample_dims;
  for (const coord of batch.coords ?? []) {
    if (coord.dims.length !== 0 && (coord.dims.length !== 1 || coord.dims[0] !== dim)) {
      return `the coordinate ${coord.name} lives on ${coord.dims.join(", ")} and not on ${dim}`;
    }
  }
  return null;
}

/**
 * What a row which does not report a column contributes to it.
 *
 * The module declares it per column (`fill`), a `NaN` as the string of the wire encoding. A
 * header from a build which declares none leaves a missing number, which is what a parameter
 * fills with.
 *
 * @param {object} column one column of a header
 * @returns {number}
 */
function fillOf(column) {
  return typeof column?.fill === "number" ? column.fill : Number.NaN;
}

/** The one key of a coordinate's values, which is its dtype. */
function dtypeOf(values) {
  return Object.keys(values)[0];
}

/**
 * The request of the rows `[start, end)` of a request, every per-sample array sliced.
 *
 * @param {object} request an `analysis/1` request
 * @param {number} start the first row
 * @param {number} end one past the last row
 * @returns {object} the request of those rows
 */
export function shardRequest(request, start, end) {
  const batch = { ...request.batch };
  const rows = end - start;
  for (const name of TIME_ARRAYS) {
    if (Array.isArray(batch[name])) {
      batch[name] = batch[name].slice(start * batch.n_time, end * batch.n_time);
    }
  }
  for (const name of DOSE_ARRAYS) {
    if (Array.isArray(batch[name])) {
      batch[name] = batch[name].slice(start * batch.max_doses, end * batch.max_doses);
    }
  }
  for (const name of SAMPLE_ARRAYS) {
    if (Array.isArray(batch[name])) {
      batch[name] = batch[name].slice(start, end);
    }
  }
  batch.sample_shape = [rows];
  batch.coords = (batch.coords ?? []).map((coord) =>
    coord.dims.length === 0
      ? coord
      : {
          ...coord,
          values: { [dtypeOf(coord.values)]: coord.values[dtypeOf(coord.values)].slice(start, end) },
        },
  );
  return { ...request, batch };
}

/**
 * The answers of the shards of one batch, merged into the answer of the whole batch.
 *
 * A shard reports the columns of its own rows, and two shards need not report the same ones: a
 * batch of an intravenous and an extravascular row reports `cl` for the one and `cl_f` for the
 * other, and a shard whose rows have no terminal phase reports no candidate table. So the merge
 * is the union of the columns in the order they first appear, and every row of a shard which
 * does not carry a column is filled with what the module says that row contributes: the `fill`
 * of the column of the header, which is a `NaN` for a parameter and 0 for a variable carrying an
 * integer code (`c0_method` means "no rule was applied" by 0, not by a missing number). That is
 * what makes the answer of a pool the answer of one instance, which fills a mixed batch the same
 * way from the same table. A point variable takes the widest width of any shard and the short
 * rows are padded.
 *
 * @param {object[]} answers the answers of the shards, in the order of their rows
 * @returns {{header: object, columns: object[], values: Float64Array, column: (name: string) => Float64Array | undefined}}
 */
export function mergeAnswers(answers) {
  if (answers.length === 1) {
    return answers[0];
  }
  const first = answers[0].header;
  for (const { header } of answers.slice(1)) {
    if (header.contract !== first.contract) {
      throw new KernelError(
        "internal",
        `a shard answered ${header.contract} and another ${first.contract}`,
      );
    }
  }
  const described = new Map();
  for (const { columns } of answers) {
    for (const column of columns) {
      const seen = described.get(column.name);
      if (seen === undefined) {
        described.set(column.name, { ...column, width: column.width ?? 1 });
      } else {
        seen.width = Math.max(seen.width, column.width ?? 1);
      }
    }
  }
  const rows = answers.reduce((total, { header }) => total + header.n_rows, 0);
  const columns = [...described.values()];
  const values = new Float64Array(columns.reduce((total, column) => total + column.width * rows, 0));
  let offset = 0;
  const at = new Map();
  for (const column of columns) {
    at.set(column.name, [offset, offset + column.width * rows]);
    let row = 0;
    for (const answer of answers) {
      const source = answer.column(column.name);
      const width = answer.columns.find((one) => one.name === column.name)?.width ?? 1;
      const fill = fillOf(column);
      for (let shardRow = 0; shardRow < answer.header.n_rows; shardRow += 1, row += 1) {
        for (let item = 0; item < column.width; item += 1) {
          values[offset + row * column.width + item] =
            source === undefined || item >= width ? fill : source[shardRow * width + item];
        }
      }
    }
    offset += column.width * rows;
  }
  const header = { ...first, n_rows: rows, columns, coords: mergeCoords(answers) };
  const texts = mergeTexts(answers);
  if (texts !== undefined) {
    header.text_variables = texts;
  }
  return {
    header,
    columns,
    values,
    column(name) {
      const range = at.get(name);
      return range === undefined ? undefined : values.subarray(range[0], range[1]);
    },
  };
}

/** The coordinates of the shards, concatenated along the dimension which was cut. */
function mergeCoords(answers) {
  const [dim] = answers[0].header.sample_dims;
  return (answers[0].header.coords ?? []).map((coord) => {
    if (coord.dims.length === 0) {
      return coord;
    }
    const dtype = dtypeOf(coord.values);
    const values = answers.flatMap(
      ({ header }) => header.coords.find((one) => one.name === coord.name)?.values[dtype] ?? [],
    );
    return { ...coord, dims: [dim], values: { [dtype]: values } };
  });
}

/** The text variables of the shards, concatenated row by row. */
function mergeTexts(answers) {
  if (answers.every(({ header }) => (header.text_variables ?? []).length === 0)) {
    return undefined;
  }
  const names = new Map();
  for (const { header } of answers) {
    for (const variable of header.text_variables ?? []) {
      names.set(variable.name, variable.unit);
    }
  }
  return [...names].map(([name, unit]) => ({
    name,
    unit,
    values: answers.flatMap(({ header }) => {
      const variable = (header.text_variables ?? []).find((one) => one.name === name);
      return variable === undefined ? new Array(header.n_rows).fill(null) : variable.values;
    }),
  }));
}

/** The error of one `{kind, message}` of a worker. */
function errorOf(error) {
  return error.kind === "refused"
    ? new RefusedError(error.message)
    : new KernelError(error.kind, error.message);
}

/** One worker of a browser or of Node, behind the three calls this pool makes. */
async function spawn(url) {
  if (typeof Worker === "function") {
    const worker = new Worker(url, { type: "module" });
    return {
      post: (message, transfer) => worker.postMessage(message, transfer),
      onMessage: (fn) => worker.addEventListener("message", (event) => fn(event.data)),
      onError: (fn) =>
        worker.addEventListener("error", (event) => fn(event.error ?? new Error(event.message))),
      terminate: () => worker.terminate(),
      ref: () => {},
      unref: () => {},
    };
  }
  const { Worker: NodeWorker } = await import("node:worker_threads");
  const worker = new NodeWorker(url);
  return {
    post: (message, transfer) => worker.postMessage(message, transfer),
    onMessage: (fn) => worker.on("message", fn),
    onError: (fn) => {
      worker.on("error", fn);
      worker.on("messageerror", fn);
    },
    terminate: () => worker.terminate(),
    // an idle worker does not keep a node process alive, so a script which forgets `close()`
    // still exits; a worker with a job in flight does, so an answer is never lost to an exit
    ref: () => worker.ref(),
    unref: () => worker.unref(),
  };
}

/**
 * A pool of workers over one module.
 *
 * @param {{module: WebAssembly.Module | BufferSource, workers?: number, createWorker?: () => Promise<object>, maxQueuedJobs?: number}} options
 *   the module (compiled once here and posted to every worker), how many workers to run,
 *   a factory for a page which builds its workers itself (bundlers rewrite a worker URL), and
 *   how many jobs may wait before the pool refuses more
 * @returns {Promise<{about: object, workers: number, analyze: (request: object, options?: {signal?: AbortSignal}) => Promise<object>, summarize: (request: object, options?: {signal?: AbortSignal}) => Promise<object>, close: () => void}>}
 */
export async function createPool({
  module,
  workers = defaultWorkers(),
  createWorker = () => spawn(WORKER_URL),
  maxQueuedJobs = 1024,
}) {
  const compiled =
    module instanceof WebAssembly.Module ? module : await WebAssembly.compile(module);
  /** The jobs which wait for a worker. */
  const queue = [];
  /** The job every worker is running, `null` when it is idle. */
  const running = new Map();
  /** The workers which failed, which are out of the rotation. */
  const dead = new Set();
  let closed = false;
  let about = null;

  const seats = await Promise.all(
    Array.from({ length: Math.max(1, workers) }, async () => {
      const worker = await createWorker();
      const ready = new Promise((resolve, reject) => {
        worker.onMessage((message) => {
          if (message.kind === "ready") {
            about = message.about;
            resolve();
            return;
          }
          const job = running.get(worker);
          running.delete(worker);
          if (job !== undefined) {
            if (message.kind === "error") {
              job.reject(errorOf(message.error));
            } else {
              job.resolve(answerOf(message));
            }
          }
          pump();
        });
        worker.onError((bad) => {
          // a worker which failed is out of the rotation: its instance is gone and a job posted
          // to it would wait for an answer which cannot come. The pool runs on what is left,
          // and a pool whose last worker failed says so on the next call rather than hanging.
          const job = running.get(worker);
          running.delete(worker);
          dead.add(worker);
          reject(bad);
          job?.reject(new KernelError("internal", `the worker failed: ${bad.message ?? bad}`));
          pump();
        });
      });
      worker.post({ kind: "init", module: compiled });
      await ready;
      return worker;
    }),
  );

  /** One answer of a worker, with the payload it transferred. */
  function answerOf(message) {
    const { header, columns } = message;
    const values = message.values instanceof Float64Array
      ? message.values
      : new Float64Array(message.values);
    const at = new Map();
    let offset = 0;
    for (const column of columns) {
      const width = (column.width ?? 1) * (header.n_rows ?? header.width ?? 0);
      at.set(column.name, [offset, offset + width]);
      offset += width;
    }
    return {
      header,
      columns,
      values,
      column(name) {
        const range = at.get(name);
        return range === undefined ? undefined : values.subarray(range[0], range[1]);
      },
    };
  }

  /** Hand every idle worker the next job of the queue. */
  function pump() {
    for (const worker of seats) {
      if (queue.length === 0) {
        break;
      }
      if (!running.has(worker) && !dead.has(worker)) {
        const job = queue.shift();
        if (job.signal?.aborted) {
          job.reject(job.signal.reason ?? new Error("aborted"));
          continue;
        }
        running.set(worker, job);
        worker.ref?.();
        worker.post({ kind: job.kind, id: job.id, request: job.request });
      }
    }
    if (running.size === 0) {
      for (const worker of seats) {
        worker.unref?.();
      }
    }
  }

  let next = 0;

  /** Queue one call and answer when its worker has. */
  function submit(kind, request, signal) {
    if (closed) {
      throw new KernelError("internal", "this pool is closed");
    }
    if (queue.length >= maxQueuedJobs) {
      throw new KernelError("internal", `${queue.length} jobs are already waiting`);
    }
    if (dead.size === seats.length) {
      throw new KernelError("internal", `every one of the ${seats.length} workers of this pool failed`);
    }
    next += 1;
    const id = next;
    return new Promise((resolve, reject) => {
      queue.push({ id, kind, request, resolve, reject, signal });
      signal?.addEventListener("abort", () => {
        const waiting = queue.findIndex((job) => job.id === id);
        if (waiting !== -1) {
          queue.splice(waiting, 1);
          reject(signal.reason ?? new Error("aborted"));
        }
      });
      pump();
    });
  }

  return {
    get about() {
      return about;
    },
    get workers() {
      return seats.length - dead.size;
    },
    /**
     * Analyse a batch over the workers of this pool.
     *
     * @param {object} request an `analysis/1` request
     * @param {{signal?: AbortSignal}} [options] a signal which drops the call while it waits
     * @returns {Promise<object>} the merged answer
     */
    async analyze(request, { signal } = {}) {
      const batch = request.batch;
      const rows = (batch.sample_shape ?? []).reduce((all, one) => all * one, 1);
      const why = shardable(batch);
      if (rows <= 1 || why !== null || seats.length === 1) {
        return submit("analyze", request, signal);
      }
      const shards = Math.min(seats.length, rows);
      const size = Math.ceil(rows / shards);
      const ranges = [];
      for (let start = 0; start < rows; start += size) {
        ranges.push([start, Math.min(rows, start + size)]);
      }
      const answers = await Promise.all(
        ranges.map(([start, end]) => submit("analyze", shardRequest(request, start, end), signal)),
      );
      const merged = mergeAnswers(answers);
      merged.header.sample_dims = batch.sample_dims;
      merged.header.sample_shape = batch.sample_shape;
      return merged;
    },
    /**
     * Reduce a parameter table on one worker.
     *
     * @param {object} request a summary request
     * @param {{signal?: AbortSignal}} [options] a signal which drops the call while it waits
     * @returns {Promise<object>} the answer
     */
    summarize(request, { signal } = {}) {
      return submit("summarize", request, signal);
    },
    /** End every worker; a call after this throws. */
    close() {
      closed = true;
      for (const job of queue.splice(0)) {
        job.reject(new KernelError("internal", "this pool was closed before the job ran"));
      }
      for (const worker of seats) {
        worker.terminate();
      }
    },
  };
}
