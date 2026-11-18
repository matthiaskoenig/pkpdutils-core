"""The gate of the built artifact: what a user gets when they install the wheel.

The rust tests of this crate run against a library which links libpython, which is not what a
wheel is: a wheel is built with `extension-module` and leaves libpython to the interpreter that
loads it, and it carries the `parallel` feature of the kernel, which the rust tests do not ask
for. The module name, the `PyInit_` symbol it exports, the abi tag of the wheel and the version
its metadata claims are all decided by the build and by nothing the rust tests can see, so this
file is run against the installed distribution and never against the source tree.

Every number asserted here is written out from the formula it comes from, so that the test
states what the analysis should produce rather than what it did produce.
"""

import importlib.machinery
import importlib.metadata
import math
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pkpdutils_core
import pytest

# the manifest the version of the crate is written in, when the tests are run from a checkout
WORKSPACE_MANIFEST = Path(__file__).resolve().parents[3] / "Cargo.toml"

# What `test_the_wheel_carries_the_row_shards` runs in an interpreter of its own, so that the
# thread count it reads is not whatever the tests before it left behind. A plan of one shard
# never touches the pool, so the first analysis is the baseline and the second is the one which
# starts it; a wheel without the `parallel` feature of the kernel runs both in the calling
# thread and the two counts are then the same.
SHARD_PROBE = """
import os
import sys

sys.path.insert(0, sys.argv[1])
import pkpdutils_core
from test_bindings import linear_options, replicated_batch


def threads():
    return len(os.listdir("/proc/self/task"))


batch = replicated_batch(4000)
options = linear_options()
pkpdutils_core.analyze_batch(batch, options, n_workers=1)
serial = threads()
pkpdutils_core.analyze_batch(batch, options, n_workers=8)
print(serial, threads())
"""

# the curve of `an_oral_row_reports_the_oral_parameter_set_and_no_iv_parameter` of
# `crates/core/src/parameters.rs`: an extravascular 100 mg dose at t = 0, a peak of 8 mg/l at
# t = 2 h, and four points from t = 3 h on which halve every hour
TIMES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
VALUES = [0.0, 4.0, 8.0, 4.0, 2.0, 1.0, 0.5]
DOSE = 100.0


def oral_batch() -> dict:
    """One subject, one extravascular dose, the curve above.

    Returns:
        The batch dict of the `batch/1` contract, its arrays as numpy buffers.
    """
    return {
        "sample_dims": ["subject"],
        "sample_shape": [1],
        "n_time": len(TIMES),
        "times": np.array([TIMES]),
        "values": np.array([VALUES]),
        "max_doses": 1,
        "dose_amount": np.array([[DOSE]]),
        "dose_time": np.array([[0.0]]),
        "dose_duration": np.array([[np.nan]]),
        "time_unit": "h",
        "unit": "mg/l",
        "dose_unit": "mg",
        "substance": "caffeine",
        "route": "oral",
        "tissue": "plasma",
        "coords": [{"name": "subject", "dims": ["subject"], "values": {"str": ["A"]}}],
    }


def replicated_batch(n_samples: int) -> dict:
    """The curve above, copied into `n_samples` rows.

    Args:
        n_samples: The number of subjects of the batch.

    Returns:
        The batch dict, every row the same curve and the same dose.
    """
    batch = oral_batch()
    batch["sample_shape"] = [n_samples]
    batch["times"] = np.tile(np.array([TIMES]), (n_samples, 1))
    batch["values"] = np.tile(np.array([VALUES]), (n_samples, 1))
    batch["dose_amount"] = np.full((n_samples, 1), DOSE)
    batch["dose_time"] = np.zeros((n_samples, 1))
    batch["dose_duration"] = np.full((n_samples, 1), np.nan)
    batch["coords"] = [
        {
            "name": "subject",
            "dims": ["subject"],
            "values": {"int": list(range(n_samples))},
        }
    ]
    return batch


def linear_options() -> dict:
    """The options of version 1, every field at its default but the trapezoid rule.

    The linear rule is taken so that the areas below can be summed by hand. Every key is
    written out: the options are closed on both sides, a key too few is a `KeyError` and a key
    too many a `KeyError` as well, so this dict is also what pins the field set of this build.

    Returns:
        The options dict `analyze_batch` and `is_supported` take.
    """
    return {
        "kind": "concentration",
        "auc_method": "linear",
        "terminal": {
            "method": "best_fit",
            "min_points": 3,
            "exclude_cmax": True,
            "n_points": None,
            "points": None,
            "min_adj_r2": None,
            "tie_tolerance": 1e-4,
            "relative_warning": 1e-8,
            "windows": None,
            "keep_candidates": False,
        },
        "lloq": None,
        "lloq_relative": None,
        "blq": "nan",
        "c0_method": "log_back_extrapolation",
        "extrapolation_warning": 0.2,
        "acceptance": {
            "r2_adj_min": None,
            "extrapolation_max": None,
            "span_min": None,
            "n_points_min": None,
            "exclude": False,
        },
        "partial_aucs": {},
        "tau": None,
        "tau_tolerance": 0.1,
        "intervals": True,
        "units": {},
        "effect_threshold": None,
        "n_workers": None,
        "chunk_rows": 5000,
        "uncertainty": None,
        "n_boot": 1000,
        "seed": None,
        "ci_level": 0.95,
        "bootstrap_spread": "se",
        "bootstrap_distribution": "normal",
        "delta_step": 0.01,
    }


def value(result: dict, name: str) -> float:
    """The parameter `name` of the one sample of `result`.

    Args:
        result: What `analyze_batch` returned.
        name: The parameter column.

    Returns:
        The number of the first row.
    """
    column = result["parameters"][name]
    assert column.dtype == np.float64, f"{name} is no float64 column"
    assert column.shape == (1,), f"{name} has the shape {column.shape}"
    return float(column[0])


def test_a_single_dose_batch_analyses() -> None:
    """The wheel computes the parameters of one extravascular curve.

    The expected numbers are the formulas of the analysis evaluated here: the linear
    trapezoids, the terminal slope of four points which halve every hour, and the
    extrapolation to infinity off the last measurable value.
    """
    result = pkpdutils_core.analyze_batch(oral_batch(), linear_options())

    assert set(result) == {
        "parameters",
        "units",
        "flags",
        "point_variables",
        "booleans",
        "texts",
    }

    # the peak, the trough and the last measurable point are read off the curve
    assert value(result, "cmax") == 8.0
    assert value(result, "tmax") == 2.0
    assert value(result, "clast") == 0.5
    assert value(result, "tlast") == 6.0

    # the linear trapezoid over the six segments of the curve
    auc_last = sum(
        (VALUES[i] + VALUES[i + 1]) / 2.0 * (TIMES[i + 1] - TIMES[i])
        for i in range(len(TIMES) - 1)
    )
    assert auc_last == 19.25
    assert value(result, "auc_last") == auc_last
    # no trailing zero, so there is nothing for `auc_all` to add
    assert value(result, "auc_all") == auc_last

    # the regression takes the four points from t = 3 h, each half of the one before, so the
    # slope of the log concentration is exactly -ln 2 and the half-life is one hour
    assert value(result, "lambda_z_n_points") == 4.0
    assert value(result, "lambda_z_t_first") == 3.0
    assert value(result, "lambda_z_t_last") == 6.0
    lambda_z = math.log(2.0)
    assert value(result, "lambda_z") == pytest.approx(lambda_z, rel=1e-12)
    assert value(result, "thalf") == pytest.approx(1.0, rel=1e-12)

    # the observed extrapolation, and the clearance and the volume it carries. The route is
    # extravascular, so they are reported as `cl_f` and `vz_f` and the intravenous `cl` and
    # `vz` are not columns of this result at all
    auc_inf_obs = auc_last + 0.5 / lambda_z
    assert value(result, "auc_inf_obs") == pytest.approx(auc_inf_obs, rel=1e-12)
    assert value(result, "auc_extrap_fraction") == pytest.approx(
        (auc_inf_obs - auc_last) / auc_inf_obs, rel=1e-12
    )
    assert value(result, "cl_f") == pytest.approx(DOSE / auc_inf_obs, rel=1e-12)
    assert value(result, "vz_f") == pytest.approx(
        DOSE / auc_inf_obs / lambda_z, rel=1e-12
    )
    assert "cl" not in result["parameters"]
    assert "vz" not in result["parameters"]

    # the units travel beside the numbers, one mapping over every section, derived from the
    # units of the batch and written as pint writes them
    assert result["units"]["cmax"] == "milligram / liter"
    assert result["units"]["auc_last"] == "hour * milligram / liter"
    assert result["units"]["cl_f"] == "liter / hour"

    # nothing about this curve is worth a flag
    assert result["flags"].dtype == np.uint32
    assert result["flags"].tolist() == [0]

    # the acceptance variables are booleans and travel in their own section, and this build
    # reports no point variable and no text for a single dose row
    assert sorted(result["booleans"]) == ["accepted", "excluded"]
    assert result["booleans"]["accepted"].tolist() == [True]
    assert result["point_variables"] == {}
    assert result["texts"] == {}


def test_a_multiple_dose_batch_analyses() -> None:
    """The wheel analyses a three dose protocol over its dosing intervals.

    The expected numbers are those of the python core of `pkpdutils` (`nca` on the same
    curve, which runs `compute_steady_state` for a protocol of more than one dose), read from
    a run of it and not from this wheel. The curve is 10 mg every 12 h, sampled over 36 h.
    """
    times = [0, 1, 2, 4, 8, 12, 13, 14, 16, 20, 24, 25, 26, 28, 32, 36.0]
    values = [
        0,
        6.8,
        7.1,
        5.1,
        2.3,
        1.0,
        7.7,
        7.8,
        5.6,
        2.5,
        1.1,
        7.8,
        7.9,
        5.6,
        2.5,
        1.1,
    ]
    batch = oral_batch()
    batch.update(
        n_time=len(times),
        times=np.array([times]),
        values=np.array([values]),
        max_doses=3,
        dose_amount=np.array([[10.0, 10.0, 10.0]]),
        dose_time=np.array([[0.0, 12.0, 24.0]]),
        dose_duration=np.full((1, 3), np.nan),
    )
    options = linear_options()
    assert pkpdutils_core.is_supported(batch, options) is None
    result = pkpdutils_core.analyze_batch(batch, options)

    # the protocol, and the steady state of the last interval (24 h to 36 h)
    assert value(result, "n_doses") == 3.0
    assert value(result, "tau") == 12.0
    assert value(result, "auc_tau") == pytest.approx(49.2, rel=1e-12)
    assert value(result, "cmax_ss") == 7.9
    assert value(result, "cmin_ss") == 1.1
    assert value(result, "ctrough") == 1.1
    assert value(result, "cavg") == pytest.approx(4.1, rel=1e-12)
    assert value(result, "accumulation_ratio") == pytest.approx(
        1.0953594281312835, rel=1e-12
    )
    assert result["flags"].tolist() == [0]

    # the areas of the three intervals, over the dimension `interval`
    interval_auc = result["point_variables"]["interval_auc"]
    assert interval_auc["dim"] == "interval"
    assert interval_auc["size"] == 3
    assert interval_auc["values"] == pytest.approx(
        [43.949999999999996, 48.900000000000006, 49.2], rel=1e-12
    )


def test_an_explicit_tau_analyses_a_single_dose_as_an_interval() -> None:
    """An explicit `tau` is analysed, not refused, and takes the dose as the last of its regimen.

    The curve is that of `test_a_single_dose_batch_analyses` with `tau` of 6 h; the expected
    numbers are those of the python core (`NCAOptions(tau=6.0)`), read from a run of it.
    """
    options = {**linear_options(), "tau": 6.0}
    assert pkpdutils_core.is_supported(oral_batch(), options) is None
    result = pkpdutils_core.analyze_batch(oral_batch(), options)

    assert value(result, "n_doses") == 1.0
    assert value(result, "tau") == 6.0
    assert value(result, "auc_tau") == 19.25
    assert value(result, "cmax_ss") == 8.0
    assert value(result, "ctrough") == 0.5
    assert value(result, "cavg") == pytest.approx(3.2083333333333335, rel=1e-12)
    assert value(result, "accumulation_ratio") == pytest.approx(
        1.0158730158730158, rel=1e-12
    )
    assert result["point_variables"]["interval_auc"]["values"].tolist() == [19.25]


def test_the_shard_plan_does_not_change_the_numbers() -> None:
    """A sharded run gives the bits of a serial one.

    What this proves is that `n_workers` changes no number: the three plans agree down to the
    bit, `NaN` included. What it cannot prove is that the shards ran on a pool at all, because
    a wheel built without the `parallel` feature of the kernel answers every plan from the
    calling thread and passes this test too. `test_the_wheel_carries_the_row_shards` is what
    covers that half.

    The batch is 4000 rows rather than one on purpose. A plan is only a plan if it can be cut:
    a single row is one shard whatever `n_workers` says, and the automatic plan runs a batch
    of less work than the kernel's threshold in the calling thread, so a one row batch would
    compare three serial runs with each other and say nothing about the sharded path.
    """
    batch = replicated_batch(4000)
    options = linear_options()
    serial = pkpdutils_core.analyze_batch(batch, options, n_workers=1)
    sharded = pkpdutils_core.analyze_batch(batch, options, n_workers=8)
    planned = pkpdutils_core.analyze_batch(batch, options)

    assert serial["parameters"], "nothing to compare: the result carries no parameter"
    assert set(serial["parameters"]) == set(sharded["parameters"])
    for name, column in serial["parameters"].items():
        # bit for bit, `NaN` where the serial run has `NaN`: `array_equal` with `equal_nan`
        # compares the values and not a tolerance
        assert np.array_equal(column, sharded["parameters"][name], equal_nan=True), name
        assert np.array_equal(column, planned["parameters"][name], equal_nan=True), name


def test_the_wheel_carries_the_row_shards() -> None:
    """A sharded plan really runs on a pool, which is what the wheel is built for.

    Nothing in the numbers can show this, since a build without the `parallel` feature of the
    kernel answers a shard plan with the same bits from the calling thread. What shows it is
    that the pool exists: the analysis of a plan of more than one shard starts rayon's threads
    and a serial one does not touch it. Without the feature `n_workers` would be inert and the
    speedup this wheel exists for would be most of itself smaller, in silence.
    """
    if not Path("/proc/self/task").is_dir():
        pytest.skip("the thread count is read from /proc, which this platform has not")

    probe = subprocess.run(
        [sys.executable, "-I", "-c", SHARD_PROBE, str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        check=True,
    )
    serial, sharded = (int(count) for count in probe.stdout.split())
    assert sharded > serial, (
        f"a plan of eight shards left the thread count at {serial}, so this build ran it in "
        "the calling thread: the wheel was built without the `parallel` feature of the kernel"
    )


def test_an_unsupported_batch_raises() -> None:
    """An analysis this build does not carry is refused by a type a host can route on.

    `UnsupportedError` subclasses `ValueError`, so a caller which catches `ValueError` keeps
    catching it, and the message names the option which this build does not carry.
    """
    options = linear_options()
    options["units"] = {"auc_last": "mg*hr/l"}

    assert issubclass(pkpdutils_core.UnsupportedError, ValueError)
    with pytest.raises(pkpdutils_core.UnsupportedError) as raised:
        pkpdutils_core.analyze_batch(oral_batch(), options)
    assert "units" in str(raised.value)

    # and the refusals which are not that one keep their own type: a key no field of the
    # options spells is a disagreement about the contract, never work to route elsewhere
    stray = linear_options()
    stray["engine"] = "rust"
    with pytest.raises(KeyError) as mistyped:
        pkpdutils_core.analyze_batch(oral_batch(), stray)
    assert not isinstance(mistyped.value, pkpdutils_core.UnsupportedError)
    assert "engine" in str(mistyped.value)


def test_is_supported_reports_the_reason() -> None:
    """The same question without an exception, which a host asks first."""
    assert pkpdutils_core.is_supported(oral_batch(), linear_options()) is None

    options = linear_options()
    options["units"] = {"auc_last": "mg*hr/l"}
    reason = pkpdutils_core.is_supported(oral_batch(), options)
    assert isinstance(reason, str)
    assert "units" in reason


def test_the_version_matches_the_crate() -> None:
    """The module, the metadata of the distribution and the manifest agree.

    `__version__` is `CARGO_PKG_VERSION` baked into the extension at compile time and the
    metadata version is what maturin read from the same manifest, so a wheel whose compiled
    code and whose metadata disagree has been assembled from two different trees.
    """
    assert pkpdutils_core.__version__ == importlib.metadata.version("pkpdutils-core")

    if not WORKSPACE_MANIFEST.is_file():
        pytest.skip(f"no checkout at {WORKSPACE_MANIFEST} to compare the crate with")
    manifest = tomllib.loads(WORKSPACE_MANIFEST.read_text(encoding="utf-8"))
    assert pkpdutils_core.__version__ == manifest["workspace"]["package"]["version"]


def test_the_wheel_is_built_against_the_stable_abi() -> None:
    """The installed distribution carries a `cp313-abi3` tag.

    One wheel per platform serves every python from 3.13 on, which is what `abi3-py313` of
    pyo3 buys and what `requires-python = ">=3.13"` promises. A build which lost that feature
    would be tagged for the one interpreter it was built against, and the promise would be
    broken for every later one.
    """
    wheel = importlib.metadata.distribution("pkpdutils-core").read_text("WHEEL")
    assert wheel is not None, "the installed distribution carries no WHEEL metadata"
    tags = [
        line.split(":", 1)[1].strip()
        for line in wheel.splitlines()
        if line.startswith("Tag:")
    ]
    assert tags, wheel
    for tag in tags:
        assert tag.startswith("cp313-abi3-"), f"{tag} is no abi3 tag of the 3.13 floor"


def test_the_macos_wheel_carries_no_fixed_slot_allocator() -> None:
    """The macOS wheel allocates through the system allocator, every other one through mimalloc.

    On macOS mimalloc keeps the heap of a thread in two slots of the pthread TLS block which it
    claims by compiled in number, so two extension modules which each link a copy of it hand
    each other their heaps and corrupt them. The test suite of `sbmlsim`, which loads this module
    next to `sbmlsim_core`, failed on macOS and only on macOS until this wheel stopped linking
    one (sbmlsim#565); the `mod allocator` comment of `crates/python/src/lib.rs` is the long
    version.

    Both halves are asserted, so the test fails if a build loses mimalloc where it is wanted as
    well as if it gains one where it corrupts: the attribute is what the compiled code says about
    itself, and the bytes of the extension are what it actually links. mimalloc's symbols are
    stripped from a release build but the texts of its own error messages are not. The two halves
    are not equally strong: finding the string proves it is linked, while not finding it only
    fails to prove it, so the macOS half rests on `__allocator__`, which is exact, and the byte
    search is what makes the other platforms assert anything at all.
    """
    linked = sys.platform != "darwin"
    assert pkpdutils_core.__allocator__ == ("mimalloc" if linked else "system")

    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    extensions = [
        path
        for directory in pkpdutils_core.__path__
        for path in sorted(Path(directory).iterdir())
        if path.name.endswith(suffixes)
    ]
    assert len(extensions) == 1, f"one extension module, not {extensions}"
    assert (b"mimalloc" in extensions[0].read_bytes()) is linked, (
        f"{extensions[0].name} and __allocator__ disagree about mimalloc"
    )


def test_the_wheel_summarizes_and_refuses_a_malformed_request_as_a_value_error() -> (
    None
):
    """The installed wheel summarizes, and answers a malformed request with a `ValueError`."""
    request = {
        "values": np.array([[[1.0, 2.0, 4.0, np.nan]]]),
        "n_vars": 1,
        "n_groups": 1,
        "n_samples": 4,
        "ci_level": 0.95,
        "lognormal": [True],
        "discrete": [False],
    }
    answer = pkpdutils_core.summarize(request)
    assert answer["count"].shape == (1, 1)
    assert answer["count"][0, 0] == 3.0
    # the geometric mean of 1, 2 and 4 is 2
    assert answer["geomean"][0, 0] == pytest.approx(2.0, rel=1e-14)
    # a request of the wrong size is a ValueError, and not the refusal a host answers
    # by analysing in python
    request["n_samples"] = 5
    with pytest.raises(
        ValueError, match="4 values for 1 variables x 1 groups x 5"
    ) as raised:
        pkpdutils_core.summarize(request)
    assert not isinstance(raised.value, pkpdutils_core.UnsupportedError)
