"""
Dedicated to the analysis and visualization of single-variable features,
including plotting functions such as pie charts, count plots, and histograms.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from . import cm
from .config import _figsize_to_px, _import_plotly, _resolve_engine, _warn_ignored_mpl_params
from .stat_methods import kde_curve

THRESHOLD_MANY_CATS = 20  # If categories more than 20 and numbers - group small categories to "Other" group
THRESHOLD_BIG_CAT = 5  # If categories more than 5 - too many categories, don't plot pie plot


# --- Specs: plot-ready data shared by both engines ---


@dataclass(frozen=True)
class _PieSpec:
    """
    Plot-ready data of a pie chart.

    Attributes
    ----------
    categories : list
        Names of the slices, in the order they are drawn.
    values : list[int]
        Absolute count of every slice, aligned with ``categories``.
    labels : list[str]
        Rendered labels of the slices, such as "12.5% (3)".
    title : str or None
        Title of the chart, taken from the name of the Series.
    """

    categories: list
    values: list
    labels: list
    title: object


@dataclass(frozen=True)
class _CountplotSpec:
    """
    Plot-ready data of a count plot.

    Attributes
    ----------
    series : pd.Series
        Observations of the feature, with the small categories already
        grouped into the "Other" category.
    colname : str
        Name of the plotted feature, used as the label of the x axis.
    order : list
        Categories in the order they are drawn.
    counts : np.ndarray
        Number of observations of every category, aligned with ``order``.
    labels : list[str]
        Rendered labels of the bars.
    """

    series: object
    colname: str
    order: list
    counts: object
    labels: list


@dataclass(frozen=True)
class _HistplotSpec:
    """
    Plot-ready data of a histogram.

    Attributes
    ----------
    series : pd.Series
        Observations of the feature, as given by the caller.
    values : np.ndarray
        Observations of the feature, without the missing values.
    bin_edges : np.ndarray
        Edges of the bins, of length ``len(counts) + 1``.
    counts : np.ndarray
        Number of observations in every bin.
    mode : float or None
        Most frequent value, or None when the mode is not displayed.
    mode_count : int
        Number of observations equal to the mode.
    title : str or None
        Title of the chart, taken from the name of the Series.
    """

    series: object
    values: object
    bin_edges: object
    counts: object
    mode: object
    mode_count: int
    title: object


def _pie_spec(df_column, is_count_order=True):
    """
    Compute the plot-ready data of a pie chart.

    Parameters
    ----------
    df_column : pd.Series
        The pandas Series containing the categorical data to plot.
    is_count_order : bool, default: True
        If True, the slices are ordered by the count of occurrences.

    Returns
    -------
    spec : _PieSpec
        Categories, counts and rendered labels of the slices.

    Notes
    -----
    The label of every slice is rendered here, so that both engines show the
    same percentage and the same count even when two categories have the same
    number of observations.

    Examples
    --------
    >>> import pandas as pd
    >>> from pltstat.singlefeat import _pie_spec
    >>> _pie_spec(pd.Series(["A", "B", "B"])).labels
    ['33.3% (1)', '66.7% (2)']
    """
    value_counts = df_column.value_counts()

    if is_count_order is True:
        value_counts = value_counts.sort_values()
    else:
        value_counts = value_counts.sort_index()

    total = int(value_counts.sum())
    values = [int(value) for value in value_counts.values]
    labels = [f"{100 * value / total:.1f}% ({value:d})" for value in values]

    return _PieSpec(
        categories=list(value_counts.index),
        values=values,
        labels=labels,
        title=df_column.name,
    )


def _countplot_spec(df_column, is_count_order=True, is_group_small_cats=True):
    """
    Compute the plot-ready data of a count plot.

    Categories which are not among the ``THRESHOLD_MANY_CATS`` most frequent
    ones are grouped into an "Other" category, which is drawn last.

    Parameters
    ----------
    df_column : pd.Series
        The pandas Series containing the categorical data to plot.
    is_count_order : bool, default: True
        If True, the bars are ordered by the count of occurrences.
    is_group_small_cats : bool, default: True
        If True, small categories are grouped into an "Other" category.

    Returns
    -------
    spec : _CountplotSpec
        Name, order, counts and rendered labels of the bars.

    Examples
    --------
    >>> import pandas as pd
    >>> from pltstat.singlefeat import _countplot_spec
    >>> _countplot_spec(pd.Series(["A", "B", "B"], name="f")).order
    ['B', 'A']
    """
    colname = df_column.name
    if colname is None:
        # Rename a copy, so that the Series of the caller is left unchanged
        colname = "Values"
        df_column = df_column.rename(colname)

    other_val = "Other"
    put_other_at_the_end = False

    # Change non-top THRESHOLD_MANY_CATS categories to "Other" group:
    n_unique = len(df_column.dropna().unique())
    if (n_unique > THRESHOLD_MANY_CATS) and (is_group_small_cats is True):
        value_counts = df_column.dropna().value_counts()
        min_top_count = value_counts.iloc[THRESHOLD_MANY_CATS]  # Count of the 21st category

        # Find categories for Other group:
        other_mask = value_counts <= min_top_count
        other_categories = value_counts[other_mask].index

        # Replace small categories to Other group:
        other_replacer = dict(zip(other_categories, [other_val] * len(other_categories)))
        df_column = df_column.replace(other_replacer)
        put_other_at_the_end = True

    if is_count_order:
        order = df_column.value_counts().index
    else:
        order = df_column.unique()
        order = np.sort(order)

    # Put Other category at the end of the order:
    if put_other_at_the_end is True:
        order = order[order != other_val]
        order = np.append(order, other_val)

    counts = df_column.value_counts().reindex(order, fill_value=0).values
    total = len(df_column)
    labels = [f"{100 * count / total:.1f}% \n ({count:.0f})" for count in counts]

    return _CountplotSpec(
        series=df_column,
        colname=colname,
        order=list(order),
        counts=counts,
        labels=labels,
    )


def _histplot_spec(df_column, bins="auto", show_mode=False):
    """
    Compute the plot-ready data of a histogram.

    Parameters
    ----------
    df_column : pd.Series
        The pandas Series containing the numerical data to plot.
    bins : int, str or array-like, default: "auto"
        Binning passed to :func:`numpy.histogram`.
    show_mode : bool, default: False
        If True, the mode and its count are computed.

    Returns
    -------
    spec : _HistplotSpec
        Observations, bins, counts and mode of the feature.

    Notes
    -----
    The bins are computed here, so that both engines split the observations at
    the same edges.

    Examples
    --------
    >>> import pandas as pd
    >>> from pltstat.singlefeat import _histplot_spec
    >>> float(_histplot_spec(pd.Series([1.0, 2.0, 2.0]), bins=2, show_mode=True).mode)
    2.0
    """
    values = df_column.dropna().to_numpy()
    counts, bin_edges = np.histogram(values, bins=bins)

    mode = None
    mode_count = 0
    if show_mode:
        value_counts = df_column.value_counts()
        if len(value_counts) > 0:
            mode = value_counts.index[0]
            mode_count = int(value_counts.values[0])

    return _HistplotSpec(
        series=df_column,
        values=values,
        bin_edges=bin_edges,
        counts=counts,
        mode=mode,
        mode_count=mode_count,
        title=df_column.name,
    )


# --- Matplotlib renderers ---


def _pie_mpl(spec, ax=None, figsize=None, **kwargs):
    """
    Draw a pie chart with matplotlib.

    Parameters
    ----------
    spec : _PieSpec
        Plot-ready data built by :func:`_pie_spec`.
    ax : matplotlib.axes.Axes or None, default: None
        Axes to draw on. If None, a new figure and Axes are created.
    figsize : tuple or None, default: None
        Size of the figure in inches. Ignored if ``ax`` is not None.
    **kwargs : keyword arguments, optional
        Additional arguments passed to ``ax.pie()``.

    Returns
    -------
    None
        The function modifies the plot in place and does not return any value.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(spec.title)

    # The labels are precomputed, so they are consumed in the order matplotlib
    # draws the slices instead of being looked up by percentage.
    labels = iter(spec.labels)
    ax.pie(
        spec.values,
        labels=spec.categories,
        autopct=lambda pct: next(labels),
        explode=[0.02] * len(spec.values),
        **kwargs,
    )


def _countplot_mpl(spec, is_color=True, ax=None, figsize=(18, 6), **kwargs):
    """
    Draw a count plot with matplotlib.

    Parameters
    ----------
    spec : _CountplotSpec
        Plot-ready data built by :func:`_countplot_spec`.
    is_color : bool, default: True
        If True, the bars are coloured with the "muted" palette.
    ax : matplotlib.axes.Axes or None, default: None
        Axes to draw on. If None, a new figure and Axes are created.
    figsize : tuple, default: (18, 6)
        Size of the figure in inches. Ignored if ``ax`` is not None.
    **kwargs : keyword arguments, optional
        Additional arguments passed to ``sns.countplot()``.

    Returns
    -------
    None
        The function modifies the plot in place and does not return any value.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if is_color:
        hue = spec.colname
        palette = "muted"
    else:
        hue = None
        palette = None

    sns.countplot(
        spec.series.to_frame(),
        x=spec.colname,
        order=spec.order,
        palette=palette,
        hue=hue,
        legend=False,
        ax=ax,
        **kwargs,
    )

    for position, (count, label) in enumerate(zip(spec.counts, spec.labels)):
        ax.text(
            position,
            count,
            label,
            fontsize=12,
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Text should not be above the limits:
    max_height = spec.counts.max()
    ax.set_ylim(0, max_height + 1)


def _histplot_mpl(spec, is_limits=False, bins="auto", kde=True, show_mode=False,
                  ax=None, figsize=(18, 6), **kwargs):
    """
    Draw a histogram with matplotlib.

    Parameters
    ----------
    spec : _HistplotSpec
        Plot-ready data built by :func:`_histplot_spec`.
    is_limits : bool, default: False
        If True, the KDE curve is clipped to the range of the data.
    bins : int, str or array-like, default: "auto"
        Binning passed to ``sns.histplot()``.
    kde : bool, default: True
        If True, a kernel density estimate is overlaid on the histogram.
    show_mode : bool, default: False
        If True, the mode of the data is displayed with its count.
    ax : matplotlib.axes.Axes or None, default: None
        Axes to draw on. If None, a new figure and Axes are created.
    figsize : tuple, default: (18, 6)
        Size of the figure in inches. Ignored if ``ax`` is not None.
    **kwargs : keyword arguments, optional
        Additional arguments passed to ``sns.histplot()``.

    Returns
    -------
    None
        The function modifies the plot in place and does not return any value.
    """
    if is_limits:
        kde_kws = {"clip": (spec.series.min(), spec.series.max())}
    else:
        kde_kws = None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.histplot(spec.series, kde=kde, kde_kws=kde_kws, bins=bins, ax=ax, **kwargs)

    # Get coordinates for the texts (bin heights)
    for p in ax.patches:
        height = p.get_height()
        x = p.get_x() + p.get_width() / 2
        # Add text at the top of each bin
        ax.text(x, height + 0.1, str(int(height)), ha="center", va="bottom", fontsize=10)

    max_height = np.array([p.get_height() for p in ax.patches]).max()

    ax.set_ylim(0, max_height + 1)

    if show_mode is False:
        return

    mode = spec.mode
    ax.vlines(mode, 0, max_height, colors="r", label="mode")
    ax.text(
        mode,
        max_height,
        f"mode={mode:.2f}",
        fontsize=12,
        horizontalalignment="right",
        verticalalignment="top",
        rotation="vertical",
    )
    ax.text(
        mode,
        max_height,
        f"count={spec.mode_count:d}",
        fontsize=12,
        horizontalalignment="left",
        verticalalignment="top",
        rotation="vertical",
    )


# --- Plotly renderers ---


def _pie_plotly(spec, figsize=None, **kwargs):
    """
    Draw a pie chart with plotly.

    Parameters
    ----------
    spec : _PieSpec
        Plot-ready data built by :func:`_pie_spec`.
    figsize : tuple or None, default: None
        Size of the figure in inches, converted to pixels.
    **kwargs : keyword arguments, optional
        Additional arguments passed to ``plotly.graph_objects.Pie``.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The pie chart.
    """
    go, _ = _import_plotly()
    width, height = _figsize_to_px(figsize)

    fig = go.Figure(
        go.Pie(
            labels=[str(category) for category in spec.categories],
            values=spec.values,
            text=spec.labels,
            textinfo="label+text",
            # Plotly reorders the slices by value unless sorting is disabled
            sort=False,
            pull=0.02,
            hovertemplate="%{label}<br>%{text}<extra></extra>",
            **kwargs,
        )
    )
    fig.update_layout(title=spec.title, width=width, height=height)

    return fig


def _countplot_plotly(spec, is_color=True, figsize=(18, 6), **kwargs):
    """
    Draw a count plot with plotly.

    Parameters
    ----------
    spec : _CountplotSpec
        Plot-ready data built by :func:`_countplot_spec`.
    is_color : bool, default: True
        If True, the bars are coloured with the "muted" palette.
    figsize : tuple, default: (18, 6)
        Size of the figure in inches, converted to pixels.
    **kwargs : keyword arguments, optional
        Additional arguments passed to ``plotly.graph_objects.Bar``.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The count plot.
    """
    go, _ = _import_plotly()
    width, height = _figsize_to_px(figsize)

    if is_color:
        marker_color = cm.get_palette_hex("muted", len(spec.order))
    else:
        marker_color = None

    fig = go.Figure(
        go.Bar(
            x=[str(category) for category in spec.order],
            y=spec.counts,
            text=[label.replace("\n", "<br>") for label in spec.labels],
            textposition="outside",
            marker_color=marker_color,
            hovertemplate="%{x}<br>count = %{y}<extra></extra>",
            **kwargs,
        )
    )
    fig.update_layout(
        width=width,
        height=height,
        showlegend=False,
        xaxis_title=spec.colname,
        yaxis_title="count",
        yaxis_range=[0, spec.counts.max() + 1],
    )

    return fig


def _histplot_plotly(spec, is_limits=False, kde=True, show_mode=False,
                     figsize=(18, 6), **kwargs):
    """
    Draw a histogram with plotly.

    Parameters
    ----------
    spec : _HistplotSpec
        Plot-ready data built by :func:`_histplot_spec`.
    is_limits : bool, default: False
        If True, the KDE curve is clipped to the range of the data.
    kde : bool, default: True
        If True, a kernel density estimate is overlaid on the histogram.
    show_mode : bool, default: False
        If True, the mode of the data is displayed with its count.
    figsize : tuple, default: (18, 6)
        Size of the figure in inches, converted to pixels.
    **kwargs : keyword arguments, optional
        Additional arguments passed to ``plotly.graph_objects.Bar``.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The histogram.
    """
    go, _ = _import_plotly()
    width, height = _figsize_to_px(figsize)

    edges = spec.bin_edges
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    fig = go.Figure(
        go.Bar(
            x=centers,
            y=spec.counts,
            width=widths,
            text=[str(int(count)) for count in spec.counts],
            textposition="outside",
            marker_line_width=1,
            marker_line_color="white",
            name="count",
            hovertemplate="count = %{y}<extra></extra>",
            **kwargs,
        )
    )

    if kde and (np.unique(spec.values).size > 1):
        # Scale the density to the histogram of counts, as seaborn does
        clip = (spec.values.min(), spec.values.max()) if is_limits else None
        scale = len(spec.values) * widths.mean()
        grid, density = kde_curve(spec.values, clip=clip, scale=scale)
        fig.add_trace(
            go.Scatter(x=grid, y=density, mode="lines", name="kde", hoverinfo="skip")
        )

    if show_mode and (spec.mode is not None):
        fig.add_vline(
            x=spec.mode,
            line_color="red",
            annotation_text=f"mode={spec.mode:.2f} count={spec.mode_count:d}",
            annotation_textangle=-90,
        )

    fig.update_layout(
        title=spec.title,
        width=width,
        height=height,
        showlegend=False,
        yaxis_range=[0, spec.counts.max() + 1],
    )

    return fig


# --- Public plotting functions ---


def pie(df_column, ax=None, figsize=None, is_count_order=True, engine=None, **kwargs):
    """
    Plot a pie chart of the value counts of a DataFrame column with enhanced settings.

    This function generates a pie chart displaying the proportions of unique values
    in a pandas Series. The chart includes both percentage and absolute count for each slice.
    Additional arguments can be passed to the underlying plotting call for further
    customization of the chart.

    Parameters
    ----------
    df_column : pd.Series
        The pandas Series containing the data to plot. It is expected to contain
        categorical data (either strings or numbers).
    ax : matplotlib.axes.Axes, optional, default=None
        The Matplotlib Axes object to plot on. If None, the pie chart will be
        created on the current active plot.
        Ignored when ``engine="plotly"``.
    figsize : tuple or None, optional, default=None
        The size of the figure. If None, default sizes are used.
        Note: `figsize` is ignored if the input parameter `ax` is not None.
        With ``engine="plotly"`` it is converted to pixels at 100 dpi.
    is_count_order : bool, optional, default=True
        If True, the bars will be ordered by the count of occurrences in descending order.
        If False, the bars will be ordered according to the original order of the values.
    engine : {"matplotlib", "plotly"} or None, optional, default=None
        The rendering engine. If None, the engine set by
        :func:`pltstat.set_backend` is used.
    **kwargs : keyword arguments, optional
        Additional arguments to further customize the pie chart. They are passed
        to ``ax.pie()`` with matplotlib and to ``plotly.graph_objects.Pie`` with
        plotly, so they are engine specific.

    Returns
    -------
    fig : plotly.graph_objects.Figure or None
        The figure when ``engine="plotly"``. With matplotlib the function
        modifies the plot in place and returns None.

    Notes
    -----
    - The pie chart includes custom labels showing both the percentage and the
      absolute count of each category.
    - The slices are slightly exploded (offset) to enhance visualization.
    - The labels are computed before drawing, so two categories with the same
      count are labelled with their own count.

    Example
    --------
    >>> import pandas as pd
    >>> from pltstat.singlefeat import pie
    >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'B', 'B'])
    >>> pie(data)
    """
    engine = _resolve_engine(engine)
    spec = _pie_spec(df_column, is_count_order=is_count_order)

    if engine == "matplotlib":
        return _pie_mpl(spec, ax=ax, figsize=figsize, **kwargs)

    _warn_ignored_mpl_params(engine, ax=ax)
    return _pie_plotly(spec, figsize=figsize, **kwargs)


def countplot(
        df_column,
        is_count_order=True,
        is_color=True,
        ax=None,
        figsize=(18, 6),
        is_group_small_cats=True,
        engine=None,
        **kwargs
):
    """
    Plot a count plot for a DataFrame column with additional information on the bars.

    This function creates a count plot (bar plot) showing the distribution of
    categorical data. It can optionally order the bars by the count of occurrences
    and display percentages and raw counts on top of the bars. Note that the figure
    size (`figsize`) is ignored if an existing matplotlib Axes (`ax`) is provided.

    Parameters
    ----------
    df_column : pd.Series
        The pandas Series containing the categorical data to plot.
    is_count_order : bool, optional, default=True
        If True, the bars will be ordered by the count of occurrences in descending order.
        If False, the bars will be ordered according to the sorted order of the values.
    is_color : bool, optional, default=True
        If True, the bars are coloured with the "muted" palette.
    ax : matplotlib.axes.Axes, optional, default=None
        An existing matplotlib Axes to plot on. If None, a new figure and Axes are created.
        Ignored when ``engine="plotly"``.
    figsize : tuple of (float, float), optional, default=(18, 6)
        The size of the figure in inches. Ignored if `ax` is not None.
        With ``engine="plotly"`` it is converted to pixels at 100 dpi.
    is_group_small_cats : bool, optional, default=True
        If True, groups categories which are not among the 20 most frequent ones
        into an "Other" category.
    engine : {"matplotlib", "plotly"} or None, optional, default=None
        The rendering engine. If None, the engine set by
        :func:`pltstat.set_backend` is used.
    **kwargs : keyword arguments, optional
        Additional arguments to further customize the plot. They are passed to
        ``sns.countplot()`` with matplotlib and to ``plotly.graph_objects.Bar``
        with plotly, so they are engine specific.

    Returns
    -------
    fig : plotly.graph_objects.Figure or None
        The figure when ``engine="plotly"``. With matplotlib the function
        modifies the plot in place and returns None.

    Notes
    -----
    - Each bar is annotated with the percentage and the count of the category.
    - The "Other" category is always drawn last.

    Example
    --------
    >>> import pandas as pd
    >>> from pltstat.singlefeat import countplot
    >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'B', 'B'])
    >>> countplot(data, is_count_order=True, is_color=True, figsize=(12, 4))
    """
    engine = _resolve_engine(engine)
    spec = _countplot_spec(
        df_column,
        is_count_order=is_count_order,
        is_group_small_cats=is_group_small_cats,
    )

    if engine == "matplotlib":
        return _countplot_mpl(spec, is_color=is_color, ax=ax, figsize=figsize, **kwargs)

    _warn_ignored_mpl_params(engine, ax=ax)
    return _countplot_plotly(spec, is_color=is_color, figsize=figsize, **kwargs)


def histplot(df_column, is_limits=False, bins='auto', kde=True, show_mode=False, ax=None,
             figsize=(18, 6), engine=None, **kwargs):
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
    bins : int, str or array-like, optional, default='auto'
        The bins to use for the histogram, as accepted by :func:`numpy.histogram`.
    kde : bool, optional, default=True
        If True, a Kernel Density Estimation (KDE) plot is overlaid on the histogram.
    show_mode : bool, optional, default=False
        If True, the mode (most frequent value) of the data is displayed on the plot with its count.
    ax : matplotlib.axes.Axes, optional, default=None
        An existing matplotlib Axes to plot on. If None, a new figure and Axes are created.
        Ignored when ``engine="plotly"``.
    figsize : tuple of (float, float), optional, default=(18, 6)
        The size of the figure in inches. Ignored if `ax` is not None.
        With ``engine="plotly"`` it is converted to pixels at 100 dpi.
    engine : {"matplotlib", "plotly"} or None, optional, default=None
        The rendering engine. If None, the engine set by
        :func:`pltstat.set_backend` is used.
    **kwargs : keyword arguments, optional
        Additional arguments to further customize the plot. They are passed to
        ``sns.histplot()`` with matplotlib and to ``plotly.graph_objects.Bar``
        with plotly, so they are engine specific.

    Returns
    -------
    fig : plotly.graph_objects.Figure or None
        The figure when ``engine="plotly"``. With matplotlib the function
        modifies the plot in place and returns None.

    Notes
    -----
    - The histogram is annotated with the mode (the most frequent value) of the data if `show_mode=True`.
    - The mode's value and count are displayed on the plot, indicated by a red vertical line.
    - The KDE plot is overlaid on the histogram, and the KDE limits are adjusted if `is_limits=True`.
    - With plotly the density is estimated by :func:`pltstat.stat_methods.kde_curve`,
      which uses the same bandwidth rule as seaborn.


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
    engine = _resolve_engine(engine)
    spec = _histplot_spec(df_column, bins=bins, show_mode=show_mode)

    if engine == "matplotlib":
        return _histplot_mpl(
            spec,
            is_limits=is_limits,
            bins=bins,
            kde=kde,
            show_mode=show_mode,
            ax=ax,
            figsize=figsize,
            **kwargs,
        )

    _warn_ignored_mpl_params(engine, ax=ax)
    return _histplot_plotly(
        spec,
        is_limits=is_limits,
        kde=kde,
        show_mode=show_mode,
        figsize=figsize,
        **kwargs,
    )


def auto_naive_plot(
    df_column,
    is_ordinal=False,
    is_limits=None,
    is_show_average=True,
    is_count_order=None,
    is_group_small_cats=True,
    engine=None,
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
    is_count_order : bool, optional, default=None
        If True, orders categorical features by the count of their unique values for plotting.
    is_group_small_cats : bool, optional, default=True
        If True, groups small categories into an "Other" category when there are too many unique values of a categorical
        feature.
    engine : {"matplotlib", "plotly"} or None, optional, default=None
        The rendering engine. If None, the engine set by
        :func:`pltstat.set_backend` is used.

    Returns
    -------
    fig : plotly.graph_objects.Figure or None
        The figure of the chosen plot when ``engine="plotly"``. With matplotlib
        the function modifies the plot in place and returns None. None is also
        returned when the feature cannot be plotted.

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

    n_nan = np.sum(df_column.isnull())
    n_unique = len(df_column.dropna().unique())
    top5 = df_column.head().to_string()

    # Drop NaN values if present
    if n_nan > 0:
        df_column = df_column.dropna()

    dtype = "Categorical"  # Initial assumption

    try:
        # If number of unique values > threshold, treat as continuous
        # assert n_unique > threshold_cont
        df_column = df_column.astype("float64")   # Try converting to float
        dtype = "Numerical"
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
    if dtype == "Numerical":
        if np.allclose(df_column, df_column.astype("int64")):
            dtype += ":Discrete"
        elif (df_column.max() <= 1) and (df_column.max() > 0) and (df_column.min() >= -1):
            dtype += ":Proportion"
        else:
            dtype += ":Continuous"

    if dtype.startswith("Numerical") and (n_unique == 2):
        # if df_column.unique.min() == 0 and df_column.max() == 1:
        if bool(np.all(np.sort(df_column.unique()) == np.array([0, 1]))):
            dtype = "Categorical:Boolean"

    # Auto-decide for continuous feature limits
    if (is_limits is None) and dtype.startswith("Numerical"):
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
    if dtype.startswith("Numerical"):
        print(f"Min / Max values: {df_column.min():.3f} / {df_column.max():.3f}")
        if is_show_average:
            print(f"Mean / Median values: {df_column.mean():.3f} / {df_column.median():.3f}")
        print()

    # Plot the data depending on the type

    if is_count_order is None:
        if dtype.startswith("Continuous"):
            is_count_order = False
        if dtype.startswith("Categorical"):
            is_count_order = True

    try:
        if n_unique < 1:
            print("Error: Number of unique values is less than 1")
        elif n_unique == 1:
            print(f"Only one unique value: {df_column[0]}")
        elif n_unique < THRESHOLD_BIG_CAT:
            return pie(df_column, is_count_order=is_count_order, engine=engine)
        elif dtype.startswith("Categorical"):
            return countplot(
                df_column,
                is_count_order=is_count_order,
                is_group_small_cats=is_group_small_cats,
                engine=engine,
            )
        else:
            return histplot(df_column, is_limits=is_limits, engine=engine)

    except Exception:
        print("Unable to determine feature format or plot it")

    return None
    # print('un', n_unique)
    # print('# nan:', n_nan)
