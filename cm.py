from matplotlib.colors import LinearSegmentedColormap


def get_pval_legend_thr_cmap(threshold=0.05):
    """Get Red-Green cmap for plot p-values with alpha threshold and cbar_kws for legend
    It is green when a value is less than threshold and red in another case"""
    cmap = [
        (0, "palegreen"),
        (threshold, "palegreen"),
        (threshold, "lightcoral"),
        (1, "lightcoral"),
    ]
    cmap = LinearSegmentedColormap.from_list("custom", cmap)
    cbar_kws = {"ticks": [0.0, threshold, 1.0]}
    return cmap, cbar_kws


def get_corr_thr_cmap(threshold=0.8, vmin=-1):
    """Get Blue-Red cmap for plot correlations with |threshold|
    It works for correlations from 'vmin' to +1, where vmin is 0 or -1"""
    if (threshold > 1) or (threshold < 0):
        raise ValueError("thresholds must be from 0 to 1")

    if vmin == -1:
        threshold = 1 - threshold
        threshold = threshold / 2
        # cmap for correlations with |0.5| threshold:
        cmap = [
            (0, "Blue"),
            (threshold, "White"),
            (1 - threshold, "White"),
            (1, "Red"),
        ]
    elif vmin == 0:
        cmap = [
            (0, "White"),
            (threshold, "White"),
            (1, "Red"),
        ]
    else:
        raise ValueError("'vmin' must be -1 or 0")

    cmap = LinearSegmentedColormap.from_list("custom", cmap)
    return cmap
