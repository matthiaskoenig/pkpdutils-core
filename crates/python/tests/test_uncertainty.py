"""The uncertainty of a result through the built wheel.

The kernel's own tests cover the numbers of the bootstrap and of the delta method against the
python engine (`crates/core/tests/uncertainty_oracle.rs`). What they cannot cover is the way out:
an uncertainty variable is a column of the result like any other, so it travels through the dict
the binding builds, and `bootstrap_normals` is a function of the module a host imports. Both are
decided by the build and are what this file asserts.

Every number asserted here is a relation the uncertainty has to satisfy - an interval brackets
its estimate, a standard error is positive - and never a value copied out of a run.
"""

import math

import numpy as np
import pkpdutils_core
import pytest
from test_bindings import linear_options, replicated_batch

# the subjects of a group curve, which is what makes the spread a standard error of a mean
N_SUBJECTS = 9.0

# the replicates of the bootstrap of these tests, enough for a finite interval and small enough
# to keep the wheel's test suite quick
N_BOOT = 64


def group_batch(n_samples: int = 2) -> dict:
    """The replicated curve with a spread and a subject count, a batch of group curves.

    Args:
        n_samples: The number of group curves.

    Returns:
        The batch dict, its `se` a twelfth of every value and its `n` the subject count.
    """
    batch = replicated_batch(n_samples)
    values = np.asarray(batch["values"], dtype=np.float64)
    batch["se"] = 0.12 * values
    batch["n"] = np.full(values.shape, N_SUBJECTS)
    return batch


def uncertainty_options(method: str, **changes: object) -> dict:
    """The options of an analysis which carries an uncertainty.

    Args:
        method: `"bootstrap"` or `"delta"`.
        **changes: Options to override, e.g. `seed` or `ci_level`.

    Returns:
        The options dict.
    """
    options = linear_options()
    options["uncertainty"] = method
    options["n_boot"] = N_BOOT
    options["seed"] = 20261118
    options["ci_level"] = 0.90
    options.update(changes)
    return options


def column(result: dict, name: str) -> np.ndarray:
    """The values of one variable of a result.

    Args:
        result: The result dict of `analyze_batch`.
        name: The name of the variable.

    Returns:
        The values of the variable, one per sample.

    Raises:
        AssertionError: If the result carries no such variable.
    """
    columns = result["parameters"]
    assert name in columns, f"the result carries no {name}"
    values = columns[name]
    assert values.dtype == np.float64, f"{name} is no float64 column"
    return values


@pytest.mark.parametrize("method", ["bootstrap", "delta"])
def test_the_uncertainty_of_a_parameter_travels_through_the_wheel(method: str) -> None:
    """Every uncertainty variable of a parameter arrives through the dict of the binding.

    The numbers themselves are held against the python library in the kernel's own tests;
    what is asserted here are the relations which have to hold whatever they are: a standard
    error is positive, an interval brackets its estimate, the spread over the subjects is the
    standard error times the square root of their count, and the unit of a variable is the
    unit of the parameter it belongs to.
    """
    batch = group_batch(2)
    result = pkpdutils_core.analyze_batch(batch, uncertainty_options(method))
    estimate = column(result, "auc_inf_obs")
    standard_error = column(result, "auc_inf_obs_se")
    low = column(result, "auc_inf_obs_ci_low")
    high = column(result, "auc_inf_obs_ci_high")
    # the spread over the subjects of the group, which needs the count the batch carries
    spread = column(result, "auc_inf_obs_sd")
    for row in range(estimate.size):
        assert standard_error[row] > 0.0
        assert low[row] < estimate[row] < high[row], f"row {row}"
        assert spread[row] == pytest.approx(
            standard_error[row] * math.sqrt(N_SUBJECTS), rel=1e-12
        )
    # the exposure is log-normal, so its geometric statistics travel as well, and the geometric
    # mean of a log-normal parameter is positive wherever its estimate is
    assert np.all(column(result, "auc_inf_obs_geomean") > 0.0)
    assert np.all(column(result, "auc_inf_obs_geocv") > 0.0)
    # the units of the result are the ones of the parameter the variable belongs to, which is
    # what makes a standard error readable at all; a coefficient of variation is a fraction and
    # carries the spelling pint gives a dimensionless quantity, the one python writes as well
    units = result["units"]
    assert units["auc_inf_obs_se"] == units["auc_inf_obs"] == "hour * milligram / liter"
    assert units["auc_inf_obs_geocv"] == "dimensionless"


def test_a_discrete_parameter_carries_no_uncertainty() -> None:
    """A count has no uncertainty, in the wheel as in the library."""
    # the number of points of the terminal regression is a count and not an estimate: python
    # reports no uncertainty for it and neither does the wheel
    result = pkpdutils_core.analyze_batch(
        group_batch(1), uncertainty_options("bootstrap")
    )
    columns = result["parameters"]
    assert "lambda_z_n_points" in columns
    assert "lambda_z_n_points_se" not in columns


def test_a_bootstrap_without_a_seed_is_unsupported_and_the_delta_is_not() -> None:
    """The one part of a bootstrap the kernel cannot do is draw the seed."""
    # the kernel reads no entropy, so a seed is the host's to draw; the predicate says so, which
    # is how a host knows to take its own path instead of being handed an error
    batch = group_batch(1)
    seedless = uncertainty_options("bootstrap", seed=None)
    reason = pkpdutils_core.is_supported(batch, seedless)
    assert reason is not None and "seed" in reason
    with pytest.raises(ValueError, match="seed"):
        pkpdutils_core.analyze_batch(batch, seedless)
    # the delta method draws nothing, so it needs no seed
    assert (
        pkpdutils_core.is_supported(batch, uncertainty_options("delta", seed=None))
        is None
    )
    without_seed = pkpdutils_core.analyze_batch(
        batch, uncertainty_options("delta", seed=None)
    )
    assert column(without_seed, "auc_inf_obs_se")[0] > 0.0


def test_the_draws_of_the_wheel_are_a_pure_function_of_the_seed_and_the_index() -> None:
    """`bootstrap_normals` is a function of the module and a function of its arguments.

    A host streams the draws of a large batch in pieces and a block of rows asks for its own,
    so the same request twice and a range in pieces both have to be the range itself.
    """
    draws = pkpdutils_core.bootstrap_normals
    whole = draws(7, 0, 32)
    assert whole.dtype == np.float64 and whole.shape == (32,)
    # the same request twice is the same numbers, and a range in pieces is the range: a host
    # streams the draws of a large batch in chunks and a block of rows asks for its own
    assert np.array_equal(whole, draws(7, 0, 32))
    pieces = np.concatenate([draws(7, 0, 13), draws(7, 13, 7), draws(7, 20, 12)])
    assert np.array_equal(pieces, whole)
    assert not np.array_equal(draws(8, 0, 32), whole)
    assert np.all(np.isfinite(whole))
    assert draws(7, 0, 0).shape == (0,)
