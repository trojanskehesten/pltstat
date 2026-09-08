"""Tests for the file helpers in ``pltstat.in_out``."""

import matplotlib.pyplot as plt
import pytest

from pltstat import in_out


@pytest.fixture
def mpl_figure():
    """Return a minimal matplotlib figure to save."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


@pytest.fixture
def plotly_figure():
    """Return a minimal plotly figure to save."""
    import plotly.graph_objects as go

    return go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))


class TestSavePlt:
    """The matplotlib-only helper kept for backward compatibility."""

    def test_creates_the_file(self, mpl_figure, tmp_path):
        filepath = tmp_path / "figure.png"
        in_out.save_plt(mpl_figure, filepath)
        assert filepath.is_file()
        assert filepath.stat().st_size > 0

    def test_overwrites_an_existing_file(self, mpl_figure, tmp_path):
        """The helper removes the file first, so a stale figure cannot remain."""
        filepath = tmp_path / "figure.png"
        filepath.write_bytes(b"stale content")
        in_out.save_plt(mpl_figure, filepath)
        assert filepath.read_bytes() != b"stale content"

    def test_honors_dpi(self, mpl_figure, tmp_path):
        low = tmp_path / "low.png"
        high = tmp_path / "high.png"
        in_out.save_plt(mpl_figure, low, dpi=50)
        in_out.save_plt(mpl_figure, high, dpi=150)
        assert high.stat().st_size > low.stat().st_size


class TestSave:
    """The polymorphic helper which accepts either kind of figure."""

    def test_saves_a_matplotlib_figure(self, mpl_figure, tmp_path):
        filepath = tmp_path / "figure.png"
        in_out.save(mpl_figure, filepath)
        assert filepath.is_file()
        assert filepath.stat().st_size > 0

    def test_saves_a_plotly_figure_as_html(self, plotly_figure, tmp_path):
        filepath = tmp_path / "figure.html"
        in_out.save(plotly_figure, filepath)
        assert filepath.is_file()
        assert filepath.stat().st_size > 0

    def test_overwrites_an_existing_file(self, mpl_figure, tmp_path):
        filepath = tmp_path / "figure.png"
        filepath.write_bytes(b"stale content")
        in_out.save(mpl_figure, filepath)
        assert filepath.read_bytes() != b"stale content"

    def test_unsupported_figure_type_raises(self, tmp_path):
        with pytest.raises(TypeError, match="matplotlib or a plotly figure"):
            in_out.save("not a figure", tmp_path / "figure.png")
