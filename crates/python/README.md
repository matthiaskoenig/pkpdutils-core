# pkpdutils-core

The compiled non-compartmental analysis kernel of [pkpdutils](https://github.com/matthiaskoenig/pkpdutils), as a python extension module.

This distribution is not a library to program against. It is the compiled engine of [`pkpdutils`](https://pypi.org/project/pkpdutils/), which will take it as a dependency and send it the batches this build can analyse; install `pkpdutils` rather than this package. Its two functions take and return the shapes `pkpdutils` already holds in memory. The source is in [pkpdutils-core](https://github.com/matthiaskoenig/pkpdutils-core), a rust workspace, and the numbers it computes are held to Phoenix WinNonlin, PKNCA and NonCompart at a relative 1e-8 by the oracle gates of that repository.

```python
import pkpdutils_core

# the version of the crate this wheel was built from
version = pkpdutils_core.__version__
# None when this build would analyse the batch, the reason as a string when it would not
reason = pkpdutils_core.is_supported(batch, options)
result = pkpdutils_core.analyze_batch(batch, options, n_workers=None)
```

`batch` and `options` are plain dicts of numpy arrays, numbers and strings, one key per field of the `batch/1` contract and of version 1 of the serialized options; the result is a dict of numpy arrays. Both are described in the rust documentation of the crate, and both are the internals of `pkpdutils` rather than a public interface of this package: they change with the contract version and with nothing else.

## What this build does not carry

`analyze_batch` is the non-compartmental analysis of a batch, single dose and multiple dose (a protocol of more than one dose, or an explicit `tau`). An analysis which is well formed and which this build does not carry raises `pkpdutils_core.UnsupportedError`, a subclass of `ValueError`, and that one exception is what a host answers by analysing the batch its own way. Every other failure keeps its natural type: a `KeyError`, a `TypeError` or a `ValueError` naming a key means the caller and this build disagree about the contract, which is a defect and never a reason to take another path. `is_supported` asks the same question without raising and returns the reason as a string, and a host asks it first.

Not carried today: the analysis of an effect curve, the acceptance criteria, the named partial areas, the propagation of the uncertainty of a group curve and the unit conversion of a result. One more is not carried and `is_supported` does not report it: a unit string the curated unit registry of this build cannot read. That registry is narrower than pint, deliberately, so it refuses a time in `week`, `month` or `year`, a concentration in `U/l`, a prefixed `kilohour`, pint's implicit product `mg l` and its fractional exponent `hr ** 0.5`, and it also refuses a spelling pint over-forgives, such as `mg//l`, which pint reads as `mg / l`. What this build reads is a property of this build, so every one of them is an `UnsupportedError` and a host whose own engine reads the string analyses the batch there. Answering it means deriving the unit of every parameter of the result, which is the analysis itself, so `analyze_batch` is what reports it and a host which routes on the type rather than on the predicate takes its own path for it as for any other refusal.

## Threads, and a warning about `fork`

The wheels are built with the `parallel` feature of the kernel, so the row loop of a large batch is cut into contiguous shards which run on a [rayon](https://docs.rs/rayon) thread pool. The shard plan is `n_workers`: `None` lets the kernel plan it for the shape of the batch, `1` runs every row in the calling thread, and any other number is the most stretches the run may be cut into. The numbers are identical for every plan, down to the bit. The analysis releases the interpreter while it runs, so the threads of the caller run with it.

That pool is why **a process forked after an analysis must not be expected to work**. `os.fork`, and a `multiprocessing` context or a pool with the `fork` start method, copy the parent's memory without its threads, so the rayon workers do not exist in the child while every lock they held is still locked there. A child which then analyses a batch can deadlock. Use the `forkserver` or the `spawn` start method and the question does not arise; `pkpdutils` already starts its own process pools that way (`PROCESS_START_METHOD` of `pkpdutils.parallel`), to keep its own threads out of a `fork`. A child which never calls into this module is unaffected.

## Requirements

Python 3.13 or newer and numpy 2.2.2 or newer. One wheel per platform serves every python from 3.13 on: the extension is built against the stable ABI (`abi3`), so a new python release needs no new wheel.

## License

MIT, see `LICENSE` in the repository.
