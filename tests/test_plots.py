"""Smoke tests for every public plotting function, on both rendering engines.

Each test calls a function the way the README does and checks that it returns
the kind of object the engine is expected to produce. The point is to catch
API drift in matplotlib, seaborn, pandas or plotly on a new Python version,
not to validate the appearance of the figures.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import pltstat
from pltstat import circle
from pltstat import multfeats as mf
from pltstat import singlefeat as sf
from pltstat import twofeats as tf

from conftest import ENGINES

pytestmark = pytest.mark.parametrize("engine", ENGINES)


def assert_rendered(result, engine):
    """Assert that a plotting call produced something for the given engine.

    The matplotlib engine draws on the current figure and usually returns
    None, while the plotly engine returns a figure object. Both are accepted;
    what matters is that the call completed and, for matplotlib, that a figure
    was actually created.
    """
    if engine == "plotly":
        import plotly.graph_objects as go

        assert isinstance(result, go.Figure)
        assert result.data or result.layout is not None
    else:
        assert plt.get_fignums(), "no matplotlib figure was created"


class TestSingleFeat:
    """One-feature visualizations."""

    def test_pie(self, df, engine):
        assert_rendered(sf.pie(df["cat2"], engine=engine), engine)

    def test_countplot(self, df, engine):
        assert_rendered(sf.countplot(df["cat2"], engine=engine), engine)

    def test_histplot(self, df, engine):
        assert_rendered(sf.histplot(df["num1"], engine=engine), engine)

    def test_histplot_without_kde(self, df, engine):
        assert_rendered(sf.histplot(df["num1"], kde=False, engine=engine), engine)

    def test_histplot_with_missing_values(self, df, engine):
        assert_rendered(sf.histplot(df["num3"], engine=engine), engine)

    def test_auto_naive_plot_categorical(self, df, engine):
        assert_rendered(sf.auto_naive_plot(df["cat2"], engine=engine), engine)

    def test_auto_naive_plot_numeric(self, df, engine):
        assert_rendered(sf.auto_naive_plot(df["num1"], engine=engine), engine)


class TestTwoFeats:
    """Two-feature visualizations."""

    def test_crosstab(self, df, engine):
        assert_rendered(tf.crosstab(df, "cat1", "cat2", engine=engine), engine)

    @pytest.mark.parametrize("is_norm", [False, None])
    def test_crosstab_absolute_panel_only(self, df, engine, is_norm):
        """Switching off the normalized panel leaves the absolute one."""
        result = tf.crosstab(df, "cat1", "cat2", is_norm=is_norm, engine=engine)
        assert_rendered(result, engine)

    @pytest.mark.parametrize("is_abs", [False, None])
    def test_crosstab_normalized_panel_only(self, df, engine, is_abs):
        """Switching off the absolute panel leaves the normalized one."""
        result = tf.crosstab(df, "cat1", "cat2", is_abs=is_abs, engine=engine)
        assert_rendered(result, engine)

    def test_crosstab_without_any_panel_raises(self, df, engine):
        with pytest.raises(ValueError, match="At least one of"):
            tf.crosstab(df, "cat1", "cat2", is_abs=False, is_norm=False, engine=engine)

    def test_crosstab_without_any_panel_creates_no_figure(self, df, engine):
        """The call is rejected before a figure is built."""
        plt.close("all")
        with pytest.raises(ValueError):
            tf.crosstab(df, "cat1", "cat2", is_abs=False, is_norm=False, engine=engine)
        assert not plt.get_fignums()

    @pytest.mark.parametrize("method", ["fisher", "chi2"])
    def test_crosstab_explicit_method(self, df, engine, method):
        result = tf.crosstab(df, "cat1", "cat2", method=method, engine=engine)
        assert_rendered(result, engine)

    def test_corr(self, df, engine):
        assert_rendered(tf.corr(df, "num1", "num2", engine=engine), engine)

    def test_corr_without_regression(self, df, engine):
        result = tf.corr(df, "num1", "num2", show_regression=False, engine=engine)
        assert_rendered(result, engine)

    def test_boxplot(self, df, engine):
        assert_rendered(tf.boxplot(df, "cat2", "num1", engine=engine), engine)

    def test_boxplot_with_explicit_order(self, df, engine):
        result = tf.boxplot(df, "cat2", "num1", cat_order=["z", "y", "x"], engine=engine)
        assert_rendered(result, engine)

    def test_dis_box_plot(self, df, engine):
        assert_rendered(tf.dis_box_plot(df, "cat2", "num1", engine=engine), engine)


class TestMultFeats:
    """Multi-feature heatmaps and diagnostics."""

    def test_nulls(self, df, engine):
        assert_rendered(mf.nulls(df, engine=engine), engine)

    def test_dist_qq_plot(self, df, num_cols, engine):
        result = mf.dist_qq_plot(df[num_cols], (10, 6), engine=engine)
        assert result is not None

    def test_heatmap_corr(self, df, num_cols, engine):
        assert_rendered(mf.heatmap_corr(df[num_cols], engine=engine), engine)

    def test_heatmap_corr_with_threshold(self, df, num_cols, engine):
        """Exercises the ``cm.get_corr_thr_cmap`` path of the heatmap."""
        result = mf.heatmap_corr(df[num_cols], threshold=0.5, engine=engine)
        assert_rendered(result, engine)

    @pytest.mark.parametrize("corr_type", ["pearson", "spearman"])
    def test_heatmap_corr_types(self, df, num_cols, engine, corr_type):
        result = mf.heatmap_corr(df[num_cols], corr_type=corr_type, engine=engine)
        assert_rendered(result, engine)

    def test_pvals_num(self, df, num_cols, engine):
        result = mf.pvals_num(df[num_cols], engine=engine)
        assert isinstance(result, pd.DataFrame)

    def test_pvals_cat(self, df, cat_cols, engine):
        result = mf.pvals_cat(df[cat_cols], engine=engine)
        assert isinstance(result, pd.DataFrame)

    def test_pvals_num_cat(self, df, engine):
        result = mf.pvals_num_cat(df, ["cat1", "cat2"], ["num1", "num2"], engine=engine)
        assert isinstance(result, pd.DataFrame)

    def test_pvals_num_cat_transposed(self, df, engine):
        result = mf.pvals_num_cat(
            df, ["cat1", "cat2"], ["num1", "num2"], is_T=True, engine=engine
        )
        assert isinstance(result, pd.DataFrame)

    def test_phik_corrs(self, df, engine):
        result = mf.phik_corrs(
            df[["num1", "cat1", "cat2"]], interval_cols=["num1"], engine=engine
        )
        assert_rendered(result, engine)


class TestCircle:
    """Circular and directional visualizations."""

    def test_hist(self, hours, engine):
        assert_rendered(circle.hist(hours, engine=engine), engine)

    def test_hist_with_title(self, hours, engine):
        assert_rendered(circle.hist(hours, title="hours", engine=engine), engine)

    def test_scatter(self, hours, df, engine):
        assert_rendered(circle.scatter(hours, df["num1"], engine=engine), engine)


class TestGlobalBackend:
    """The engine can be selected globally instead of per call."""

    def test_set_backend_is_honored(self, df, engine):
        pltstat.set_backend(engine)
        assert pltstat.get_backend() == engine
        assert_rendered(sf.pie(df["cat2"]), engine)

    def test_backend_context_manager_restores_previous(self, df, engine):
        pltstat.set_backend("matplotlib")
        with pltstat.backend(engine):
            assert pltstat.get_backend() == engine
            assert_rendered(sf.pie(df["cat2"]), engine)
        assert pltstat.get_backend() == "matplotlib"


class TestMatplotlibOnly:
    """Behaviour specific to the matplotlib engine."""

    def test_pie_draws_on_the_given_axes(self, df, engine):
        """Passing ``ax`` must reuse that axes rather than create a figure."""
        if engine != "matplotlib":
            pytest.skip("`ax` is a matplotlib-only parameter")

        fig, ax = plt.subplots()
        sf.pie(df["cat2"], ax=ax)
        assert ax.patches or ax.texts
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
