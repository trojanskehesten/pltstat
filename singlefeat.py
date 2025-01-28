import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def pie(df_column, ax=None, figsize=None, **kwargs):
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
    figsize : tuple or None, optional, default=None
        The size of the figure. If None, default sizes are used.
        Note: `figsize` is ignored if the input parameter `ax` is not None.
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

    Example
    --------
    >>> import pandas as pd
    >>> from pltstat.singlefeat import pie
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
        fig, ax = plt.subplots(1, 1, figsize=figsize)

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


def countplot(df_column, is_count_order=True, is_color=True, ax=None, figsize=(18, 6), **kwargs):
    """
    Plot a count plot for a DataFrame column with additional information on the bars.

    This function creates a count plot (bar plot) showing the distribution of
    categorical data. It can optionally order the bars by the count of occurrences
    and display percentages and raw counts on top of the bars. Note that the figure
    size (`figsize`) is ignored if an existing matplotlib Axes (`ax`) is provided.

    Parameters
    ----------
    df_column : pd.Series
        The pandas Series containing the categorical data to plot. It can be
        any Series with categorical or object-type data.
    is_count_order : bool, optional, default=True
        If True, the bars will be ordered by the count of occurrences in descending order.
        If False, the bars will be ordered according to the original order of the values.
    is_color : bool, optional, default=True
        If True, bars are colored using the column's unique values with the "muted" palette.
        If False, a default single color is used.
    ax : matplotlib.axes.Axes, optional, default=None
        An existing matplotlib Axes to plot on. If None, a new figure and Axes are created.
    figsize : tuple of (float, float), optional, default=(18, 6)
        The size of the figure in inches. Ignored if `ax` is not None.
    **kwargs : keyword arguments, optional
        Additional arguments passed to `sns.countplot()` for further customization of the plot.

    Returns
    -------
    None
        The function modifies the plot in place and does not return any value.

    Notes
    -----
    - The percentage and raw count are displayed above each bar for better visualization.
    - If `is_count_order=True`, the `order` argument of `sns.countplot()` is modified
      to display the categories in descending order of frequency.
    - The `figsize` parameter has no effect if `ax` is not None.

    Example
    --------
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> from pltstat.singlefeat import countplot
    >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'B', 'B'])
    >>> countplot(data, is_count_order=True, is_color=True, figsize=(12, 4))
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    colname = df_column.name
    if colname is None:
        temp_name = 'Values'
        df_column.name = temp_name
        colname = temp_name

    if is_color:
        hue = colname
        palette = "muted"
    else:
        hue = None
        palette = None

    if is_count_order:
        order = df_column.value_counts().index
    else:
        order = None

    sns.countplot(
        df_column.to_frame(),
        x=colname,
        order=order,
        palette=palette,
        hue=hue,
        legend=False,
        ax=ax,
        **kwargs,
    )

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

    # Text should not be above the limits:
    max_height = np.array([p.get_height() for p in ax.patches]).max()
    ax.set_ylim(0, max_height + 1)


def histplot(df_column, is_limits=False, bins='auto', kde=True, show_mode=False, ax=None, figsize=(18, 6), **kwargs):  # , n_modes=0):
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

    Example
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.singlefeat import histplot
    >>>
    >>> # Generate 40 normally distributed float values
    >>> np.random.seed(42)
    >>> normal_floats = np.random.normal(loc=50, scale=20, size=40)
    >>>
    >>> # Clip values to be within the range [0, 100]
    >>> clipped_values = np.clip(np.round(normal_floats), 0, 100).astype(int)
    >>>
    >>> # Convert to a pandas Series
    >>> s = pd.Series(clipped_values)
    >>> histplot(s, show_mode=True, bins=np.arange(-5, 106, 10))
    """
    if is_limits:
        kde_kws = {"clip": (df_column.min(), df_column.max())}
    else:
        kde_kws = None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.histplot(df_column, kde=kde, kde_kws=kde_kws, bins=bins, ax=ax)

    # Get coordinates for the texts (bin heights)
    for p in ax.patches:
        height = p.get_height()
        x = p.get_x() + p.get_width() / 2
        # Add text at the top of each bin
        ax.text(x, height + 0.1, str(int(height)), ha='center', va='bottom', fontsize=10)

    top_values = df_column.value_counts().index.to_numpy()
    top_counts = df_column.value_counts().values

    max_height = np.array([p.get_height() for p in ax.patches]).max()

    ax.set_ylim(0, max_height + 1)
    plt.legend()

    if show_mode is False:
        return

    mode = top_values[0]
    plt.vlines(mode, 0, max_height, colors="r", label="mode")
    plt.text(
        mode,
        max_height,
        f"mode={mode:.2f}",
        fontsize=12,
        horizontalalignment="right",
        verticalalignment="top",
        rotation="vertical",
    )
    plt.text(
        mode,
        max_height,
        f"count={top_counts[0]:d}",
        fontsize=12,
        horizontalalignment="left",
        verticalalignment="top",
        rotation="vertical",
    )


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

    Example
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
