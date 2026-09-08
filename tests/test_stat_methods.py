"""Tests for the computational core in ``pltstat.stat_methods``.

This module owns the numeric assertions of the suite. Every expected value is
a closed-form result of the underlying statistic, so the tests stay valid
across NumPy, pandas and SciPy versions.
"""

import numpy as np
import pandas as pd
import pytest

from pltstat.stat_methods import (
    chi2_fisher_by_cat,
    cramer_v,
    cramer_v_by_obs,
    kde_curve,
    kruskal_by_cat,
    mannwhitneyu_by_cat,
    matthews,
)


class TestMatthews:
    """Tests for the Matthews correlation coefficient."""

    def test_perfect_positive_correlation(self):
        x = ["yes", "no", "yes", "no"]
        assert matthews(x, x) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        x = ["yes", "no", "yes", "no"]
        y = ["no", "yes", "no", "yes"]
        assert matthews(x, y) == pytest.approx(-1.0)

    def test_known_value(self):
        x = np.array(["yes", "no", "yes", "yes", "no", "no"])
        y = np.array(["no", "no", "yes", "yes", "no", "no"])
        assert matthews(x, y) == pytest.approx(0.7071067811865476)

    def test_missing_values_are_dropped(self):
        x = [1, 0, 1, 0, None]
        y = [1, 0, 1, 0, 1]
        assert matthews(x, y) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "x, y",
        [
            (["a", "b", "c"], ["x", "y", "z"]),  # more than two categories
            (["a", "a", "a"], ["x", "y", "x"]),  # fewer than two categories
        ],
    )
    def test_non_binary_input_raises(self, x, y):
        with pytest.raises(ValueError):
            matthews(x, y)


class TestCramerV:
    """Tests for Cramer's V, by contingency table and by raw observations."""

    def test_independent_table_is_zero(self):
        obs = pd.DataFrame([[10, 20], [20, 40]])
        assert cramer_v_by_obs(obs) == pytest.approx(0.0)

    def test_perfectly_associated_table_is_one(self):
        obs = pd.DataFrame([[50, 0], [0, 50]])
        assert cramer_v_by_obs(obs) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "obs",
        [
            pd.DataFrame([[10], [20]]),  # single column
            pd.DataFrame([[10, 20]]),  # single row
        ],
    )
    def test_degenerate_table_returns_zero(self, obs):
        """A table with one row or one column has no association to measure.

        Guards the ``min_dim == 0`` branch added in 0.10.1, which previously
        divided by zero.
        """
        assert cramer_v_by_obs(obs) == 0.0

    def test_returns_builtin_float(self):
        """The result is a plain float, not a NumPy scalar."""
        assert type(cramer_v_by_obs(pd.DataFrame([[50, 0], [0, 50]]))) is float

    def test_from_raw_observations(self):
        """Cramer's V of a 3x2 table, checked against the closed form.

        The table is [[2, 1], [1, 1], [0, 1]] with n = 6, so chi2 = 4/3 and
        min_dim = 1, giving V = sqrt(2/9).
        """
        data1 = ["A", "A", "A", "B", "B", "C"]
        data2 = ["X", "X", "Y", "X", "Y", "Y"]
        assert cramer_v(data1, data2) == pytest.approx(np.sqrt(2 / 9))

    def test_independent_raw_observations_is_zero(self):
        data1 = ["A", "A", "B", "B", "C", "C"]
        data2 = ["X", "Y", "X", "Y", "X", "Y"]
        assert cramer_v(data1, data2) == pytest.approx(0.0)


class TestMannWhitneyUByCat:
    """Tests for the Mann-Whitney U wrapper."""

    def test_separated_groups_are_significant(self):
        data = pd.DataFrame(
            {
                "group": ["A"] * 6 + ["B"] * 6,
                "value": [1, 2, 3, 4, 5, 6, 21, 22, 23, 24, 25, 26],
            }
        )
        statistic, p_value = mannwhitneyu_by_cat(data, "group", "value")
        assert statistic == pytest.approx(0.0)
        assert p_value < 0.05

    def test_wrong_number_of_categories_warns_and_returns_nan(self):
        data = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B", "C", "C"],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        with pytest.warns(UserWarning, match="exactly two unique categories"):
            statistic, p_value = mannwhitneyu_by_cat(data, "group", "value")
        assert np.isnan(statistic)
        assert np.isnan(p_value)


class TestKruskalByCat:
    """Tests for the Kruskal-Wallis wrapper."""

    def test_separated_groups_are_significant(self):
        data = pd.DataFrame(
            {
                "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
                "value": [1, 2, 3, 4, 5, 21, 22, 23, 24, 25, 41, 42, 43, 44, 45],
            }
        )
        statistic, p_value = kruskal_by_cat(data, "group", "value")
        assert statistic > 0
        assert p_value < 0.05

    def test_single_category_warns_and_returns_nan(self):
        data = pd.DataFrame({"group": ["A"] * 5, "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        with pytest.warns(UserWarning):
            statistic, p_value = kruskal_by_cat(data, "group", "value")
        assert np.isnan(p_value)


class TestChi2FisherByCat:
    """Tests for the combined chi-squared and Fisher exact wrapper."""

    def test_associated_2x2_table_is_significant(self):
        data = pd.DataFrame(
            {
                "left": ["a"] * 20 + ["b"] * 20,
                "right": ["x"] * 20 + ["y"] * 20,
            }
        )
        _, p_value, _ = chi2_fisher_by_cat(data, "left", "right")
        assert p_value < 0.05

    def test_independent_table_is_not_significant(self):
        data = pd.DataFrame(
            {
                "left": ["a", "a", "b", "b"] * 10,
                "right": ["x", "y", "x", "y"] * 10,
            }
        )
        _, p_value, _ = chi2_fisher_by_cat(data, "left", "right")
        assert p_value > 0.05

    def test_invalid_method_raises(self):
        data = pd.DataFrame({"left": ["a", "b"], "right": ["x", "y"]})
        with pytest.raises(ValueError):
            chi2_fisher_by_cat(data, "left", "right", method="not_a_method")


class TestKdeCurve:
    """Tests for the engine-independent density curve helper."""

    def test_returns_matching_grid_and_density(self):
        values = np.concatenate([np.zeros(10), np.ones(10)])
        grid, density = kde_curve(values, n_points=50)
        assert len(grid) == len(density) == 50
        assert np.all(density >= 0)

    def test_clip_bounds_the_grid(self):
        values = np.linspace(0, 10, 50)
        grid, _ = kde_curve(values, clip=(0, 10), n_points=20)
        assert grid.min() >= 0
        assert grid.max() <= 10
