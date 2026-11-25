"""Shared fixtures and configuration for the pltstat test suite.

The suite is a compatibility smoke net: it checks that every public function
runs and returns the expected kind of object on every supported Python and
dependency version. It is not a validation of the statistical results beyond
the closed-form cases asserted in ``test_stat_methods.py``.
"""

import matplotlib

# Select a headless backend before pyplot is imported anywhere in the suite.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import pltstat

ENGINES = ("matplotlib", "plotly")

N_ROWS = 60
RANDOM_SEED = 0


@pytest.fixture(autouse=True)
def close_figures():
    """Close every matplotlib figure left open by a test."""
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def no_auto_show():
    """Keep plotly from opening a renderer while the suite runs."""
    previous = pltstat.config.get_auto_show()
    pltstat.config.set_auto_show(False)
    yield
    pltstat.config.set_auto_show(previous)


@pytest.fixture(autouse=True)
def default_backend():
    """Restore the global engine after a test changes it."""
    previous = pltstat.get_backend()
    yield
    pltstat.set_backend(previous)


@pytest.fixture(scope="session")
def df():
    """Return a mixed numeric and categorical DataFrame with missing values.

    The frame deliberately mixes column kinds so that a single fixture feeds
    the numeric, categorical and mixed variants of the plotting functions.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    data = pd.DataFrame(
        {
            "num1": rng.normal(size=N_ROWS),
            "num2": rng.normal(size=N_ROWS),
            "num3": rng.exponential(size=N_ROWS),
            "cat1": rng.choice(["a", "b"], size=N_ROWS),
            "cat2": rng.choice(["x", "y", "z"], size=N_ROWS),
            "bin1": rng.integers(0, 2, size=N_ROWS),
        }
    )
    # A block of missing values exercises the dropna branches and gives
    # `multfeats.nulls` something to draw.
    data.loc[:5, "num3"] = np.nan
    return data


@pytest.fixture(scope="session")
def num_cols():
    """Return the names of the numeric columns of the `df` fixture."""
    return ["num1", "num2", "num3"]


@pytest.fixture(scope="session")
def cat_cols():
    """Return the names of the categorical columns of the `df` fixture."""
    return ["cat1", "cat2", "bin1"]


@pytest.fixture(scope="session")
def hours():
    """Return circular data expressed in hours, for the `circle` module."""
    rng = np.random.default_rng(RANDOM_SEED + 1)
    return rng.uniform(0, 24, size=N_ROWS)
