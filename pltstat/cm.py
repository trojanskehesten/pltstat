"""
Contains custom colormap utilities for visualizations, such as rendering correlation matrices
or creating two-colored maps for p-values with a threshold (e.g., alpha).
"""

from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, to_hex

import seaborn as sns


def get_pval_legend_thr_cmap(alpha=0.05, color_signif="palegreen", color_non_signif="lightcoral"):
    """
    Get Red-Green ListedColormap for plot of p-values with ``alpha`` threshold,
    ``norm`` and ``cbar_kws`` for legend. It is green when a value is less than
    threshold and red in another case.

    Parameters
    ----------
    alpha : float, default: 0.05
        Significance level. Must be in (0, 1)
    color_signif : str, default: "palegreen"
        Color of the cells with p-value less than ``alpha`` (significant)
    color_non_signif : str, default: "lightcoral"
        Color of the cells with p-value greater or equal to ``alpha``
        (not significant)

    Returns
    -------
    cmap : :class:`matplotlib.colors.ListedColormap`
        Colormap instance for p-values
    norm : :class:`matplotlib.colors.BoundaryNorm`
        Boundary normalization instance for p-values
    cbar_kws : dict[str, list]
        Dictionary with list of legend ticks

    Example
    --------
    >>> from pltstat.cm import get_pval_legend_thr_cmap
    >>> from numpy.random import random
    >>> import seaborn as sns
    >>> from matplotlib import pyplot as plt
    >>>
    >>> pvals = random((30, 4))
    >>> cmap, norm, cbar_kws = get_pval_legend_thr_cmap()
    >>> plt.figure(figsize=(14, 8))
    >>> sns.heatmap(pvals, vmin=0, vmax=1, annot=True, fmt='.2f', linewidth=1,
    ...             cmap=cmap, norm=norm, cbar_kws=cbar_kws);
    """
    bounds = [0, alpha, 1]
    cmap = ListedColormap([color_signif, color_non_signif])
    norm = BoundaryNorm(bounds, cmap.N)
    cbar_kws = {"ticks": [0.0, alpha, 1.0]}
    return cmap, norm, cbar_kws


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
    stops = _corr_thr_stops(threshold, vmin)
    cmap = LinearSegmentedColormap.from_list("custom", stops)
    return cmap


def _corr_thr_stops(threshold=0.8, vmin=-1):
    """
    Get the colormap stops used to plot correlations with a ``threshold``.

    The stops are shared by the matplotlib colormap and by the plotly
    colorscale, so that both engines colour a correlation heatmap identically.

    Parameters
    ----------
    threshold : float, default: 0.8
        Level for colouring the correlation. Must be in (0, 1)
    vmin : int, default: -1
        Minimum value of correlations. Must be in -1 or 0.

    Returns
    -------
    stops : list[tuple[float, str]]
        List of ``(position, color)`` pairs, where the position is in [0, 1]

    Raises
    ------
    ValueError
        If ``threshold`` is not in (0, 1) or ``vmin`` is neither -1 nor 0

    Examples
    --------
    >>> from pltstat.cm import _corr_thr_stops
    >>> _corr_thr_stops(threshold=0.8, vmin=0)
    [(0, 'White'), (0.8, 'White'), (1, 'Red')]
    """
    if (threshold > 1) or (threshold < 0):
        raise ValueError("thresholds must be from 0 to 1")

    white = "White"
    blue = "Blue"
    red = "Red"

    if vmin == -1:
        threshold = 1 - threshold
        threshold = threshold / 2
        stops = [
            (0, blue),
            (threshold, white),
            (1 - threshold, white),
            (1, red),
        ]
    elif vmin == 0:
        stops = [
            (0, white),
            (threshold, white),
            (1, red),
        ]
    else:
        raise ValueError("'vmin' must be -1 or 0")

    return stops


def get_corr_thr_colorscale(threshold=0.8, vmin=-1):
    """
    Get Blue-Red plotly colorscale for plot correlations with ``|threshold|``

    This is the plotly counterpart of :func:`get_corr_thr_cmap`. Both build the
    colours from the same stops, so a correlation heatmap looks the same with
    either engine.

    Parameters
    ----------
    threshold : float, default: 0.8
        Level for colouring the correlation. Must be in (0, 1)
    vmin : int, default: -1
        Minimum value of correlations. Must be in -1 or 0.

    Returns
    -------
    colorscale : list[list]
        Plotly colorscale as a list of ``[position, color]`` pairs, where the
        position is in [0, 1] and the colour is a hexadecimal string

    Raises
    ------
    ValueError
        If ``threshold`` is not in (0, 1) or ``vmin`` is neither -1 nor 0

    Examples
    --------
    >>> from pltstat.cm import get_corr_thr_colorscale
    >>> get_corr_thr_colorscale(threshold=0.8, vmin=0)
    [[0, '#ffffff'], [0.8, '#ffffff'], [1, '#ff0000']]
    """
    stops = _corr_thr_stops(threshold, vmin)
    return [[position, to_hex(color)] for position, color in stops]


def get_pval_thr_colorscale(alpha=0.05, color_signif="palegreen", color_non_signif="lightcoral"):
    """
    Get Red-Green plotly colorscale for p-values with ``alpha`` threshold

    This is the plotly counterpart of :func:`get_pval_legend_thr_cmap`. The
    colour changes abruptly at ``alpha`` instead of blending, which reproduces
    the discrete matplotlib colormap.

    Parameters
    ----------
    alpha : float, default: 0.05
        Significance level. Must be in (0, 1)
    color_signif : str, default: "palegreen"
        Color of the cells with p-value less than ``alpha`` (significant)
    color_non_signif : str, default: "lightcoral"
        Color of the cells with p-value greater or equal to ``alpha``
        (not significant)

    Returns
    -------
    colorscale : list[list]
        Plotly colorscale as a list of ``[position, color]`` pairs. The colour
        at ``alpha`` is repeated to make the transition sharp.

    Notes
    -----
    The heatmap must be drawn with ``zmin=0`` and ``zmax=1`` for the threshold
    to fall at ``alpha``.

    Examples
    --------
    >>> from pltstat.cm import get_pval_thr_colorscale
    >>> get_pval_thr_colorscale(alpha=0.05)
    [[0.0, '#98fb98'], [0.05, '#98fb98'], [0.05, '#f08080'], [1.0, '#f08080']]
    """
    signif = to_hex(color_signif)
    non_signif = to_hex(color_non_signif)
    return [
        [0.0, signif],
        [alpha, signif],
        [alpha, non_signif],
        [1.0, non_signif],
    ]


def get_palette_hex(palette="muted", n_colors=None):
    """
    Get a seaborn palette as a list of hexadecimal colours for plotly.

    Parameters
    ----------
    palette : str, default: "muted"
        Name of the seaborn palette, for example "muted" or "pastel"
    n_colors : int or None, default: None
        Number of colours to return. None returns the whole palette.

    Returns
    -------
    colors : list[str]
        List of colours as hexadecimal strings

    Notes
    -----
    Resolving the palette through seaborn keeps the colours of a plot the same
    with either engine.

    Examples
    --------
    >>> from pltstat.cm import get_palette_hex
    >>> get_palette_hex("muted", 2)
    ['#4878d0', '#ee854a']
    """
    return sns.color_palette(palette, n_colors).as_hex()
