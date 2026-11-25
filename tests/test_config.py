"""Tests for the backend configuration helpers in ``pltstat.config``."""

import pytest

import pltstat
from pltstat import singlefeat as sf


class TestBackendSelection:
    """Global backend getters, setters, and context manager."""

    def test_set_backend_is_case_insensitive(self):
        pltstat.set_backend("Plotly")
        assert pltstat.get_backend() == "plotly"
        pltstat.set_backend("MATPLOTLIB")
        assert pltstat.get_backend() == "matplotlib"

    def test_set_backend_invalid_engine_raises(self):
        with pytest.raises(ValueError, match="Invalid `engine`"):
            pltstat.set_backend("bokeh")

    def test_per_call_invalid_engine_raises(self, df):
        with pytest.raises(ValueError, match="Invalid `engine`"):
            sf.pie(df["cat2"], engine="bokeh")

    def test_backend_context_manager_restores_on_exception(self):
        pltstat.set_backend("matplotlib")
        with pytest.raises(RuntimeError, match="boom"):
            with pltstat.backend("plotly"):
                assert pltstat.get_backend() == "plotly"
                raise RuntimeError("boom")
        assert pltstat.get_backend() == "matplotlib"


class TestAutoShow:
    """Toggle for automatic ``fig.show()`` on data-returning plotly calls."""

    def test_set_and_get_auto_show(self):
        pltstat.config.set_auto_show(True)
        assert pltstat.config.get_auto_show() is True
        pltstat.config.set_auto_show(False)
        assert pltstat.config.get_auto_show() is False
