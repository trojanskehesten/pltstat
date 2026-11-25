"""Tests for the colormap and colorscale helpers in ``pltstat.cm``."""

import pytest
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

from pltstat import cm


class TestPvalLegendThrCmap:
    """Discrete two-color map used by the p-value heatmaps."""

    def test_returns_cmap_norm_and_cbar_kws(self):
        cmap, norm, cbar_kws = cm.get_pval_legend_thr_cmap()
        assert isinstance(cmap, ListedColormap)
        assert isinstance(norm, BoundaryNorm)
        assert cbar_kws["ticks"] == [0.0, 0.05, 1.0]

    def test_alpha_sets_the_boundary(self):
        _, norm, cbar_kws = cm.get_pval_legend_thr_cmap(alpha=0.01)
        assert list(norm.boundaries) == [0, 0.01, 1]
        assert cbar_kws["ticks"] == [0.0, 0.01, 1.0]

    def test_custom_colors_are_used(self):
        cmap, _, _ = cm.get_pval_legend_thr_cmap(
            color_signif="green", color_non_signif="red"
        )
        assert cmap.colors == ["green", "red"]


class TestCorrThrCmap:
    """Continuous colormap used by the correlation heatmaps."""

    def test_returns_a_linear_segmented_colormap(self):
        """Guards the ``LinearSegmentedColormap`` import in ``cm``.

        The import was briefly dropped, which made this function and every
        caller of it raise NameError.
        """
        assert isinstance(cm.get_corr_thr_cmap(), LinearSegmentedColormap)

    @pytest.mark.parametrize("vmin", [-1, 0])
    def test_supported_vmin_values(self, vmin):
        assert isinstance(cm.get_corr_thr_cmap(vmin=vmin), LinearSegmentedColormap)

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_threshold_outside_unit_interval_raises(self, threshold):
        with pytest.raises(ValueError, match="thresholds must be from 0 to 1"):
            cm.get_corr_thr_cmap(threshold=threshold)

    def test_invalid_vmin_raises(self):
        with pytest.raises(ValueError, match="'vmin' must be -1 or 0"):
            cm.get_corr_thr_cmap(vmin=0.5)


class TestColorscales:
    """Plotly equivalents of the matplotlib colormaps."""

    @pytest.mark.parametrize("vmin", [-1, 0])
    def test_corr_colorscale_is_well_formed(self, vmin):
        scale = cm.get_corr_thr_colorscale(vmin=vmin)
        positions = [position for position, _ in scale]
        assert positions[0] == 0
        assert positions[-1] == 1
        assert positions == sorted(positions)

    def test_pval_colorscale_is_well_formed(self):
        scale = cm.get_pval_thr_colorscale()
        positions = [position for position, _ in scale]
        assert positions[0] == 0
        assert positions[-1] == 1
        assert positions == sorted(positions)


class TestPaletteAndFormatting:
    """Helpers shared by both engines."""

    def test_get_palette_hex_length(self):
        assert len(cm.get_palette_hex(n_colors=4)) == 4

    def test_get_palette_hex_returns_hex_strings(self):
        assert all(color.startswith("#") for color in cm.get_palette_hex(n_colors=3))

    def test_format_matrix(self):
        formatted = cm.format_matrix([[0.12345, 0.6]], fmt=".2f")
        assert formatted == [["0.12", "0.60"]]
