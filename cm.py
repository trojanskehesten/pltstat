from matplotlib.colors import LinearSegmentedColormap


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
