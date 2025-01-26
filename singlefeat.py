import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np


def pie(df_column, ax=None, **kwargs):
    """
    Plot a pie chart of the value counts of a DataFrame column with enhanced settings.

    This function generates a pie chart displaying the proportions of unique values
    in a pandas Series. The chart includes both percentage and absolute count for each slice.
    Additional arguments can be passed to `ax.pie()` for further customization of the chart.

    Parameters
    ----------
    df_column : pd.Series
        The pandas Series containing the data to plot. It is expected to contain
        categorical data (either strings or numbers).
    ax : matplotlib.axes.Axes, optional, default=None
        The Matplotlib Axes object to plot on. If None, the pie chart will be
        created on the current active plot.
    **kwargs : keyword arguments, optional
        Additional arguments to pass to `ax.pie()` to further customize the pie chart,
        such as `colors`, `startangle`, `shadow`, etc.

    Returns
    -------
    None
        The function modifies the plot in place and does not return any value.

    Notes
    -----
    - The pie chart includes custom labels showing both the percentage and the
      absolute count of each category.
    - The `autopct` function formats the labels to display one decimal point
      for the percentage and the exact count.
    - The slices are slightly exploded (offset) to enhance visualization.
    - Additional keyword arguments can be used to modify the pie chart, such as
      `colors`, `startangle`, `shadow`, etc.

    Examples
    --------
    >>> import pandas as pd
    >>> from pltstat.singlefeat import hist
    >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'B', 'B'])
    >>> pie(data)
    """
    value_counts = df_column.value_counts()
    value_counts_norm = df_column.value_counts(normalize=True)
    cat_number = value_counts.shape[0]

    def func(pct, allvals_norm, allvals):
        TOL = 0.01
        absolute = allvals[
            np.where(np.abs(round(pct, 2) - 100 * np.round(allvals_norm, 4)) < TOL)[
                0
            ][0]
        ]

        # absolute = int(pct*np.sum(allvals)/100)
        return f"{pct:.1f}% ({absolute:d} )"

    if ax is None:
        plt.title(df_column.name)
        ax = plt
    else:
        ax.set_title(df_column.name)

    ax.pie(
        value_counts.values,
        labels=value_counts.index,
        autopct=lambda pct: func(pct, value_counts_norm.values, value_counts.values),
        # autopct='%.1f %%',
        # startangle= 120,
        explode=[0.02] * cat_number,
        **kwargs,
    )


def countplot(df_column, is_count_order=True, x_rotation=90, figsize=(18, 6), **kwargs):
    """
    Plot a count plot for a DataFrame column with additional information on the bars.

    This function creates a count plot (bar plot) showing the distribution of
    categorical data. It can optionally order the bars by the count of occurrences
    and display percentages and raw counts on top of the bars. The x-axis labels
    can also be rotated for better readability. The figure size can be customized.

    Parameters
    ----------
    df_column : pd.Series
        The pandas Series containing the categorical data to plot. It can be
        any Series with categorical or object-type data.
    is_count_order : bool, optional, default=True
        If True, the bars will be ordered by the count of occurrences in descending order.
        If False, the bars will be ordered according to the original order of the values.
    x_rotation : int, optional, default=90
        The rotation angle of the x-axis labels, in degrees.
    figsize : tuple of (float, float), optional, default=(18, 6)
        The size of the figure in inches.
    **kwargs : keyword arguments, optional
        Additional arguments passed to `sns.countplot()` for further customization
        of the plot, such as `hue`, `palette`, `ax`, etc.

    Returns
    -------
    None
        The function modifies the plot in place and does not return any value.

    Notes
    -----
    - The percentage and raw count are displayed above each bar for better visualization.
    - The `order` argument of `sns.countplot()` is modified when `is_count_order=True`
      to display the categories in descending order of frequency.

    Examples
    --------
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> from pltstat.singlefeat import countplot
    >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'B', 'B'])
    >>> countplot(data)
    """
    plt.figure(figsize=figsize)
    if is_count_order:
        ax = sns.countplot(df_column, order=df_column.value_counts().index, **kwargs)
    else:
        ax = sns.countplot(df_column, **kwargs)
    total = len(df_column)
    for p in ax.patches:
        text = f"{100 * p.get_height() / total:.1f}% \n ({p.get_height()})"
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.text(
            x,
            y,
            text,
            fontsize=12,
            horizontalalignment="center",
            verticalalignment="center",
        )
        # ax.annotate(text, (x, y), ha='center', va='center', xytext=(x, y))
    plt.xticks(rotation=x_rotation)


def histplot(df_column, is_limits=False, bins=None, **kwargs):  # , n_modes=0):
    """
    Plot a histogram with a Kernel Density Estimation (KDE) overlay and additional statistics.

    This function creates a histogram for a DataFrame column with the option to overlay
    a KDE plot. It can also display the mode (most frequent value) of the data and annotate
    it with both its value and count. Optionally, the KDE plot's limits can be set based on
    the minimum and maximum values of the data.

    Parameters
    ----------
    df_column : pd.Series
        The pandas Series containing the data to plot. It can be any numerical data.
    is_limits : bool, optional, default=False
        If True, the limits for the KDE plot are set based on the minimum and maximum values
        of the data. If False, the KDE plot is drawn without limits.
    bins : int, optional, default=None
        The number of bins to use for the histogram. If None, an automatic binning strategy is used.
    **kwargs : keyword arguments, optional
        Additional keyword arguments passed to `sns.histplot()`, such as `hue`, `palette`, `ax`, etc.

    Returns
    -------
    None
        The function modifies the plot in place and does not return any value.

    Notes
    -----
    - The histogram is annotated with the mode (the most frequent value) of the data.
    - The mode's value and count are displayed on the plot.
    - The histogram is overlaid with a KDE curve, and the mode is indicated by a red vertical line.

    Examples
    --------
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> from pltstat.singlefeat import histplot
    >>> data = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> histplot(data)

    >>> # Using specific settings for limits and bins
    >>> histplot(data, is_limits=True, bins=5)
    """
    if is_limits:
        min_val = df_column.min()
        max_val = df_column.max()
        ax = sns.histplot(
            df_column, kde=True, kde_kws={"clip": (min_val, max_val)}, bins=bins
        )
    else:
        ax = sns.histplot(df_column, kde=True, bins=bins, **kwargs)

    top_values = df_column.value_counts().index.to_numpy()
    top_counts = df_column.value_counts().values

    mode = top_values[0]
    bins_most_height = np.array([p.get_height() for p in ax.patches]).max()
    # coef = bins_most_height / top_counts[0]

    plt.vlines(mode, 0, bins_most_height, colors="r", label="mode")
    plt.text(
        mode,
        bins_most_height,
        "mode=%.2f" % mode,  # '%.2f (mode)' % mode,
        fontsize=12,
        horizontalalignment="right",
        verticalalignment="top",
        rotation="vertical",
    )
    plt.text(
        mode,
        bins_most_height,
        "count=%d" % top_counts[0],
        fontsize=12,
        horizontalalignment="left",
        verticalalignment="top",
        rotation="vertical",
    )

    plt.legend()


def auto_naive_plot(
    df_column,
    is_ordinal=False,
    is_limits=None,
    is_show_average=True,
    is_count_order=True,
):
    """
    Plot information about a DataFrame column based on the type of values.
    The function performs a naive analysis of the column and plots appropriate graphs
    based on whether the feature is categorical or continuous. The function can automatically
    detect feature types, but you need to check the obtained information.

    Parameters
    ----------
    df_column : pd.Series
        The pandas Series containing the data to analyze. It can be a categorical or continuous feature.
    is_ordinal : bool, optional, default=False
        If True, treats the feature as ordinal. The function can't automatically detect ordinal features.
    is_limits : bool or None, optional, default=None
        If None, the function will automatically choose whether to apply limits for continuous features.
        If True, applies limits for continuous features.
    is_show_average : bool, optional, default=True
        If True, the mean and median values will be displayed for continuous features.
    is_count_order : bool, optional, default=True
        If True, orders categorical features by the count of their unique values for plotting.

    Returns
    -------
    None
        The function modifies the plot in place and does not return any value.

    Notes
    -----
    - The function automatically detects whether a feature is categorical or continuous.
    - It applies appropriate plots such as pie charts, count plots, or histograms based on the feature type.
    - For continuous features, the function shows the min, max, mean, and median values if requested.

    Examples
    --------
    >>> import pandas as pd
    >>> from pltstat.singlefeat import auto_naive_plot
    >>> data = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> auto_naive_plot(data)

    >>> # For ordinal features
    >>> auto_naive_plot(data, is_ordinal=True)
    """
    threshold_cont = 20  # If categories more than 20 and numbers - continuous
    threshold_big_cat = 5  # If categories more than 5 - too many categories, don't plot pie plot

    n_nan = np.sum(df_column.isnull())
    n_unique = len(df_column.dropna().unique())
    top5 = df_column.head().to_string()

    # Drop NaN values if present
    if n_nan > 0:
        df_column = df_column.dropna()

    dtype = "Categorical"  # Initial assumption

    try:
        # If number of unique values > threshold, treat as continuous
        assert n_unique > threshold_cont
        df_column = df_column.astype("float64")   # Try converting to float
        dtype = "Continuous"
    except Exception:
        # Otherwise, treat as categorical
        # It is impossible to use np.array_equal, because df_column can have float features:
        if n_unique == 2:  # and _np.allclose(_np.sort(df_column.unique()), [0, 1]):  # Likely a binary feature
            dtype += ":Boolean"
        elif is_ordinal:
            dtype += ":Ordinal"
        else:
            dtype += ":Nominal"

    # Further refinement for continuous types
    if dtype == "Continuous":
        if np.allclose(df_column, df_column.astype("int64")):
            dtype += ":Integer"
        elif (df_column.max() <= 1) and (df_column.max() > 0) and (df_column.min() >= -1):
            dtype += ":Proportion"
        else:
            dtype += ":Float"

    # Auto-decide for continuous feature limits
    if (is_limits is None) and dtype.startswith("Continuous"):
        if dtype.endswith("Proportion") or (df_column.min() == 0) or (df_column.min() == 1):  # prop or counter
            is_limits = True
        else:
            is_limits = False

    # Print feature summary
    print(f"Name of feature: '{df_column.name}'")
    print(f"Feature type: {dtype}")
    # if dtype.startswith('Categorical'):
    print(f"Number of unique values: {n_unique}")
    if n_nan > 0:
        print(f"Number of NaN values: {n_nan}")
    else:
        print("No NaN values found")
    print("\nFirst 5 values:")
    print(top5)
    print()

    # If continuous, show min, max, mean, and median
    if dtype.startswith("Continuous"):
        print(f"Min / Max values: {df_column.min():.3f} / {df_column.max():.3f}")
        if is_show_average:
            print(f"Mean / Median values: {df_column.mean():.3f} / {df_column.median():.3f}")
        print()

    # Plot the data depending on the type
    try:
        if n_unique < 1:
            print("Error: Number of unique values is less than 1")
        elif n_unique == 1:
            print(f"Only one unique value: {df_column[0]}")
        elif n_unique < threshold_big_cat:
            pie(df_column)
        elif n_unique < threshold_cont:
            countplot(df_column, is_count_order)
        else:
            histplot(df_column, is_limits=is_limits)
    except Exception:
        print("Unable to determine feature format or plot it")
    # print('un', n_unique)
    # print('# nan:', n_nan)


# TODO:Not circle but heatmap with HDE and values of bins!
def plot_hist_man_ing(hours, title, y_step, h_step=2, height=6, aspect=2.5):
    """
    Plot histogram of ingestion time
    """
    bins = np.arange(0, 25, h_step)
    bin_values = np.histogram(hours, bins=bins)
    max_y = bin_values[0].max()
    max_y = max_y + y_step - max_y % y_step

    sns.displot(hours, kde=True, bins=bins, height=height, aspect=aspect)
    if (h_step - int(h_step)) != 0:
        plt.xticks(bins, bins, rotation=45);
    else:
        plt.xticks(bins, bins)
    plt.yticks(range(0, max_y+1, y_step), range(0, max_y+1, y_step))
    plt.ylabel('Count of ingestions')
    plt.xlim(0, 24)
    plt.ylim(0, max_y)
    for x, y in zip(bin_values[1], bin_values[0]):
        plt.text(x+h_step/2, y, y, horizontalalignment='center', verticalalignment='bottom')
    plt.grid()
    plt.title(title);
