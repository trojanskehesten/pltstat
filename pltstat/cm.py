"""
Contains custom colormap utilities for visualizations, such as rendering correlation matrices
or creating two-colored maps for p-values with a threshold (e.g., alpha).

Also provides plotly-compatible colorscale adapters that mirror the matplotlib
colormaps so heatmaps can be rendered through either backend.
"""

import colorsys

from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NAMED_COLORS = {
    "palegreen": (152, 251, 152),
    "lightcoral": (240, 128, 128),
    "white": (255, 255, 255),
    "blue": (0, 0, 255),
    "red": (255, 0, 0),
}


def _rgb_str(color):
    """Return ``"rgb(r,g,b)"`` for a named color or an (r,g,b) tuple."""
    if isinstance(color, str):
        color = color.lower()
        r, g, b = _NAMED_COLORS.get(color, (128, 128, 128))
    else:
        r, g, b = color
    return f"rgb({r},{g},{b})"


def _hsl_lighten(color, amount=0.12):
    """Lighten *color* (named string or ``(r,g,b)``) by *amount* in HSL space."""
    if isinstance(color, str):
        r, g, b = _NAMED_COLORS.get(color.lower(), (128, 128, 128))
    else:
        r, g, b = color
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    l = min(1.0, l + amount)
    nr, ng, nb = colorsys.hls_to_rgb(h, l, s)
    return f"rgb({round(nr*255)},{round(ng*255)},{round(nb*255)})"


def get_pval_legend_thr_cmap(alpha=0.05):
    """
    Get Red-Green LinearSegmentedColormap for plot of p-values with ``alpha`` threshold and ``cbar_kws`` for legend.
    It is green when a value is less than threshold and red in another case

    Parameters
    ----------
    alpha : float, default: 0.05
        Significance level. Must be in (0, 1)

    Returns
    -------
    cmap : :class:`matplotlib.colors.LinearSegmentedColormap`
        Colormap instance for p-values
    cbar_kws : dict([(str, list)])
        Dictionary with list of legend ticks

    Example
    --------
    >>> from pltstat.cm import get_pval_legend_thr_cmap
    >>> from numpy.random import random
    >>> import seaborn as sns
    >>> from matplotlib import pyplot as plt
    >>>
    >>> pvals = random((30, 4))
    >>> cmap, cbar_kws = get_pval_legend_thr_cmap()
    >>> plt.figure(figsize=(14, 8))
    >>> sns.heatmap(pvals, vmin=0, vmax=1, annot=True, fmt='.2f', linewidth=1, cmap=cmap, cbar_kws=cbar_kws);
    """
    green = "palegreen"
    red = "lightcoral"
    cmap = [
        (0, green),
        (alpha, green),
        (alpha, red),
        (1, red),
    ]
    cmap = LinearSegmentedColormap.from_list("custom", cmap)
    cbar_kws = {"ticks": [0.0, alpha, 1.0]}
    return cmap, cbar_kws


def get_corr_thr_cmap(threshold=0.8, vmin=-1):
    """
    Get Blue-Red cmap for plot correlations with ``|threshold|``
    It works for correlations from 'vmin' to +1, where vmin is 0 or -1

    Parameters
    ----------
    threshold : float, default: 0.08
        Level for colouring the correlation. Must be in (0, 1)
    vmin : int, default: -1
        Minimum value of correlations. Must be in -1 or 0.

    Returns
    -------
    cmap : :class:``matplotlib.colors.LinearSegmentedColormap``
        Colormap instance for correlations with specific ``threshold``

    Example
    --------
    >>> from pltstat.cm import get_corr_thr_cmap
    >>> from numpy.random import random
    >>> import seaborn as sns
    >>> from matplotlib import pyplot as plt
    >>>
    >>> pvals = random((30, 4))
    >>> cmap = get_corr_thr_cmap(vmin=0)
    >>> plt.figure(figsize=(14, 8))
    >>> sns.heatmap(pvals, vmin=0, vmax=1, annot=True, fmt='.2f', linewidth=1, cmap=cmap);
    """
    if (threshold > 1) or (threshold < 0):
        raise ValueError("thresholds must be from 0 to 1")

    white = "White"
    blue = "Blue"
    red = "Red"

    if vmin == -1:
        threshold = 1 - threshold
        threshold = threshold / 2
        cmap = [
            (0, blue),
            (threshold, white),
            (1 - threshold, white),
            (1, red),
        ]
    elif vmin == 0:
        cmap = [
            (0, white),
            (threshold, white),
            (1, red),
        ]
    else:
        raise ValueError("'vmin' must be -1 or 0")

    cmap = LinearSegmentedColormap.from_list("custom", cmap)
    return cmap


# ---------------------------------------------------------------------------
# Plotly colorscale adapters (no plotly import - pure Python)
# ---------------------------------------------------------------------------


def get_pval_colorscale(alpha=0.05):
    """Return a plotly-compatible colorscale for p-value heatmaps.

    Mirrors the logic of `get_pval_legend_thr_cmap`: green for values below
    *alpha*, red for values at or above *alpha*.

    Parameters
    ----------
    alpha : float, default: 0.05
        Significance level. Must be in (0, 1).

    Returns
    -------
    colorscale : list of (float, str)
        Plotly colorscale as ``[(pos, "rgb(r,g,b)"), ...]``.
    vmin : float
        Suggested minimum (0).
    vmax : float
        Suggested maximum (1).
    colorbar_ticks : list of float
        Tick positions for the colour bar.

    Examples
    --------
    >>> from pltstat.cm import get_pval_colorscale
    >>> colorscale, vmin, vmax, ticks = get_pval_colorscale(0.05)
    >>> vmin, vmax
    (0, 1)
    """
    g = _rgb_str("palegreen")
    r = _rgb_str("lightcoral")
    colorscale = [
        [0.0, g],
        [alpha, g],
        [alpha, r],
        [1.0, r],
    ]
    return colorscale, 0.0, 1.0, [0.0, alpha, 1.0]


def get_corr_colorscale(threshold=0.8, vmin=-1):
    """Return a plotly-compatible colorscale for correlation heatmaps.

    Mirrors the logic of `get_corr_thr_cmap`: blue-white-red for *vmin*=-1,
    white-red for *vmin*=0, with *threshold* controlling the transition.

    Parameters
    ----------
    threshold : float, default: 0.8
        Level for colouring the correlation. Must be in (0, 1).
    vmin : int, default: -1
        Minimum value of correlations. Must be -1 or 0.

    Returns
    -------
    colorscale : list of (float, str)
        Plotly colorscale as ``[(pos, "rgb(r,g,b)"), ...]``.
    vmin : float
        Suggested minimum for the colours.
    vmax : float
        Suggested maximum (1).

    Raises
    ------
    ValueError
        If *threshold* or *vmin* are invalid.

    Examples
    --------
    >>> from pltstat.cm import get_corr_colorscale
    >>> colorscale, vmin, vmax = get_corr_colorscale(vmin=0)
    >>> vmin, vmax
    (0, 1)
    """
    if (threshold > 1) or (threshold < 0):
        raise ValueError("thresholds must be from 0 to 1")

    w = _rgb_str("white")
    b = _rgb_str("blue")
    r = _rgb_str("red")

    if vmin == -1:
        t = (1 - threshold) / 2
        colorscale = [
            [0.0, b],
            [t, w],
            [1 - t, w],
            [1.0, r],
        ]
    elif vmin == 0:
        colorscale = [
            [0.0, w],
            [threshold, w],
            [1.0, r],
        ]
    else:
        raise ValueError("'vmin' must be -1 or 0")

    return colorscale, float(vmin), 1.0
