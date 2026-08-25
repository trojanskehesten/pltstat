"""
Dedicated to the analysis and visualization of single-variable features,
including plotting functions such as pie charts, count plots, and histograms.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .config import _resolve_engine, _ensure_plotly

THRESHOLD_MANY_CATS = 20  # If categories more than 20 and numbers - group small categories to "Other" group
THRESHOLD_BIG_CAT = 5  # If categories more than 5 - too many categories, don't plot pie plot


# ====================================================================
# Shared spec helpers (pure computation, no plotting)
# ====================================================================

def _pie_spec(df_column, is_count_order):
    """Compute the data needed for a pie chart.

    Returns a dict with ``values``, ``labels``, ``pct_texts`` (a list of
    ``"pct% (count)"`` strings), and ``cat_number``.
    """
    value_counts = df_column.value_counts()
    value_counts_norm = df_column.value_counts(normalize=True)

    if is_count_order:
        value_counts = value_counts.sort_values()
        value_counts_norm = value_counts_norm.sort_values()
    else:
        value_counts = value_counts.sort_index()
        value_counts_norm = value_counts_norm.sort_index()

    pct_texts = []
    for i in range(len(value_counts)):
        pct = 100 * value_counts_norm.values[i]
        pct_texts.append(f"{pct:.1f}% ({value_counts.values[i]:d})")

    return {
        "values": value_counts.values,
        "labels": value_counts.index.tolist(),
        "pct_texts": pct_texts,
        "cat_number": value_counts.shape[0],
        "title": str(df_column.name),
    }


def _countplot_spec(df_column, is_count_order, is_group_small_cats):
    """Compute the data needed for a count plot.

    Returns a dict with ``colname``, ``order``, ``counts``, ``percentages``,
    ``total``, ``n_unique``, and a boolean ``put_other_at_the_end``.
    """
    colname = df_column.name
    if colname is None:
        colname = "Values"

    put_other_at_the_end = False
    n_unique = len(df_column.dropna().unique())

    if (n_unique > THRESHOLD_MANY_CATS) and is_group_small_cats:
        other_val = "Other"
        value_counts_raw = df_column.dropna().value_counts()
        min_top_count = value_counts_raw.iloc[THRESHOLD_MANY_CATS]
        other_mask = value_counts_raw <= min_top_count
        other_categories = value_counts_raw[other_mask].index
        other_replacer = dict(zip(
            other_categories, [other_val] * len(other_categories)
        ))
        df_column = df_column.replace(other_replacer)
        put_other_at_the_end = True

    if is_count_order:
        order = df_column.value_counts().index
    else:
        order = df_column.unique()
        order = np.sort(order)

    if put_other_at_the_end:
        order = order[order != "Other"]
        order = np.append(order, "Other")

    counts = df_column.value_counts().reindex(order, fill_value=0)
    total = len(df_column)
    percentages = [f"{100 * c / total:.1f}%\n({c:.0f})" for c in counts.values]

    return {
        "colname": colname,
        "order": list(order),
        "counts": counts.values,
        "percentages": percentages,
        "total": total,
        "n_unique": n_unique,
        "put_other_at_the_end": put_other_at_the_end,
    }


def _histplot_spec(df_column, show_mode):
    """Compute the data needed for a histogram.

    Returns a dict with ``mode``, ``mode_count``, ``top_values``,
    ``top_counts``, and ``series_name``.
    """
    top_values = df_column.value_counts().index.to_numpy()
    top_counts = df_column.value_counts().values
    mode = top_values[0] if len(top_values) > 0 else None
    mode_count = top_counts[0] if len(top_counts) > 0 else 0
    return {
        "mode": mode,
        "mode_count": mode_count,
        "top_values": top_values,
        "top_counts": top_counts,
        "series_name": str(df_column.name),
        "show_mode": show_mode,
    }


# ====================================================================
# Matplotlib renderers (preserve exact current behaviour)
# ====================================================================

def _pie_mpl(spec, ax, figsize, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(spec["title"])
    ax.pie(
        spec["values"],
        labels=spec["labels"],
        autopct=lambda pct: _format_pie_label(
            pct, spec["pct_texts"], spec["cat_number"]
        ),
        explode=[0.02] * spec["cat_number"],
        **kwargs,
    )


def _format_pie_label(pct, pct_texts, cat_number):
    """Match the original pie label formatter using precomputed texts."""
    idx = int(round(pct / 100 * cat_number))
    idx = max(0, min(idx, len(pct_texts) - 1))
    return pct_texts[idx]


def _countplot_mpl(spec, is_color, ax, figsize, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if is_color:
        hue = spec["colname"]
        palette = "muted"
    else:
        hue = None
        palette = None

    import pandas as pd
    df_plot = pd.DataFrame({
        spec["colname"]: spec["order"],
        "_cnt": spec["counts"],
    })
    sns.barplot(
        data=df_plot,
        x=spec["colname"],
        y="_cnt",
        order=spec["order"],
        palette=palette,
        hue=hue,
        legend=False,
        ax=ax,
        **kwargs,
    )

    total = spec["total"]
    for i, (label, cnt) in enumerate(zip(spec["order"], spec["counts"])):
        text = f"{100 * cnt / total:.1f}% \n ({cnt:.0f})"
        ax.text(
            i, cnt, text,
            fontsize=12,
            horizontalalignment="center",
            verticalalignment="center",
        )

    max_height = max(spec["counts"].max(), 1)
    ax.set_ylim(0, max_height + 1)


def _histplot_mpl(spec, df_column, is_limits, bins, kde, ax, figsize, **kwargs):
    if is_limits:
        kde_kws = {"clip": (df_column.min(), df_column.max())}
    else:
        kde_kws = None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.histplot(df_column, kde=kde, kde_kws=kde_kws, bins=bins, ax=ax)

    for p in ax.patches:
        height = p.get_height()
        x = p.get_x() + p.get_width() / 2
        ax.text(x, height + 0.1, str(int(height)), ha="center",
                va="bottom", fontsize=10)

    max_height = np.array([p.get_height() for p in ax.patches]).max()
    ax.set_ylim(0, max_height + 1)

    if not spec["show_mode"]:
        return

    mode = spec["mode"]
    if mode is not None:
        ax.vlines(mode, 0, max_height, colors="r", label="mode")
        ax.text(
            mode, max_height, f"mode={mode:.2f}",
            fontsize=12, horizontalalignment="right",
            verticalalignment="top", rotation="vertical",
        )
        ax.text(
            mode, max_height, f"count={spec['mode_count']:d}",
            fontsize=12, horizontalalignment="left",
            verticalalignment="top", rotation="vertical",
        )


# ====================================================================
# Plotly renderers
# ====================================================================

def _pie_plotly(spec, **kwargs):
    _ensure_plotly()
    import plotly.express as px

    data = {
        "labels": spec["labels"],
        "values": spec["values"],
        "text": spec["pct_texts"],
    }
    fig = px.pie(
        data,
        names="labels",
        values="values",
        title=spec["title"],
        **kwargs,
    )
    fig.update_traces(
        textinfo="text",
        pull=[0.02] * spec["cat_number"],
    )
    return fig


def _countplot_plotly(spec, is_color, **kwargs):
    _ensure_plotly()
    import plotly.express as px

    data = {
        spec["colname"]: spec["order"],
        "count": spec["counts"],
        "text": spec["percentages"],
    }
    color = spec["colname"] if is_color else None
    fig = px.bar(
        data,
        x=spec["colname"],
        y="count",
        text="text",
        color=color,
        title=spec["colname"],
        category_orders={spec["colname"]: spec["order"]},
        **kwargs,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title="count", showlegend=False)
    return fig


def _histplot_plotly(spec, df_column, is_limits, bins, kde, **kwargs):
    _ensure_plotly()
    import plotly.express as px

    marginal = "violin" if kde else None
    hist_kwargs = dict(
        title=spec["series_name"],
        marginal=marginal,
        text_auto=True,
        **kwargs,
    )
    if isinstance(bins, (int, float)):
        hist_kwargs["nbins"] = int(bins)
    fig = px.histogram(df_column.values, **hist_kwargs)
    if is_limits:
        xmin = df_column.min()
        xmax = df_column.max()
        fig.update_xaxes(range=[xmin, xmax])

    if spec["show_mode"] and spec["mode"] is not None:
        fig.add_vline(
            x=spec["mode"], line_color="red",
            annotation_text=f"mode={spec['mode']:.2f}\n"
                            f"count={spec['mode_count']:d}",
        )

    return fig


# ====================================================================
# Public API (dispatchers)
# ====================================================================

def pie(df_column, ax=None, figsize=None, is_count_order=True, *,
        engine=None, **kwargs):
    """
    Plot a pie chart of the value counts of a DataFrame column.

    Parameters
    ----------
    df_column : pd.Series
        The pandas Series containing categorical data.
    ax : matplotlib.axes.Axes, optional, default=None
        Matplotlib Axes to plot on. Ignored when ``engine="plotly"``.
    figsize : tuple or None, optional, default=None
        Figure size (matplotlib only).
    is_count_order : bool, optional, default=True
        If True, order slices by descending count.
    engine : {"matplotlib", "plotly"} or None, default=None
        The rendering backend. ``None`` uses the global ``config.engine``
        (default ``"matplotlib"``).
    **kwargs
        Passed to ``ax.pie`` (matplotlib) or ``px.pie`` (plotly).

    Returns
    -------
    plotly.graph_objects.Figure or None
        A plotly Figure by default; ``None`` for matplotlib.

    Examples
    --------
    >>> import pandas as pd
    >>> from pltstat.singlefeat import pie
    >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'B', 'B'])
    >>> pie(data)
    """
    engine = _resolve_engine(engine)
    spec = _pie_spec(df_column, is_count_order)

    if engine == "plotly":
        return _pie_plotly(spec, **kwargs)

    _pie_mpl(spec, ax=ax, figsize=figsize, **kwargs)
    return None


def countplot(
        df_column,
        is_count_order=True,
        is_color=True,
        ax=None,
        figsize=(18, 6),
        is_group_small_cats=True,
        *,
        engine=None,
        **kwargs
):
    """
    Plot a count plot for a DataFrame column with percentage and count labels.

    Parameters
    ----------
    df_column : pd.Series
        Categorical data to plot.
    is_count_order : bool, optional, default=True
        Order bars by descending count.
    is_color : bool, optional, default=True
        Colour bars by category using the "muted" palette.
    ax : matplotlib.axes.Axes, optional, default=None
        Axes to draw on (matplotlib only).
    figsize : tuple, optional, default=(18, 6)
        Figure size in inches (matplotlib only).
    is_group_small_cats : bool, optional, default=True
        Group small categories into "Other" when >20 unique values.
    engine : {"matplotlib", "plotly"} or None, default=None
        The rendering backend.
    **kwargs
        Passed to ``sns.barplot`` (matplotlib) or ``px.bar`` (plotly).

    Returns
    -------
    plotly.graph_objects.Figure or None

    Examples
    --------
    >>> import pandas as pd
    >>> from pltstat.singlefeat import countplot
    >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'B', 'B'])
    >>> countplot(data)
    """
    engine = _resolve_engine(engine)
    spec = _countplot_spec(df_column, is_count_order, is_group_small_cats)

    if engine == "plotly":
        return _countplot_plotly(spec, is_color=is_color, **kwargs)

    _countplot_mpl(spec, is_color=is_color, ax=ax, figsize=figsize, **kwargs)
    return None


def histplot(df_column, is_limits=False, bins='auto', kde=True,
             show_mode=False, ax=None, figsize=(18, 6), *,
             engine=None, **kwargs):
    """
    Plot a histogram with optional KDE overlay and mode annotation.

    Parameters
    ----------
    df_column : pd.Series
        Numerical data to plot.
    is_limits : bool, optional, default=False
        Clip KDE to data range (matplotlib) / set x-axis range (plotly).
    bins : int or str, optional, default='auto'
        Number of bins or binning strategy.
    kde : bool, optional, default=True
        Overlay a kernel density estimate.
    show_mode : bool, optional, default=False
        Display the mode with a red vertical line.
    ax : matplotlib.axes.Axes, optional, default=None
        Axes to draw on (matplotlib only).
    figsize : tuple, optional, default=(18, 6)
        Figure size in inches (matplotlib only).
    engine : {"matplotlib", "plotly"} or None, default=None
        The rendering backend.
    **kwargs
        Passed to ``sns.histplot`` (matplotlib) or ``px.histogram`` (plotly).

    Returns
    -------
    plotly.graph_objects.Figure or None

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.singlefeat import histplot
    >>>
    >>> np.random.seed(42)
    >>> normal_floats = np.random.normal(loc=50, scale=20, size=40)
    >>> clipped_values = np.clip(np.round(normal_floats), 0, 100).astype(int)
    >>> s = pd.Series(clipped_values)
    >>> histplot(s, show_mode=True, bins=np.arange(-5, 106, 10))
    """
    engine = _resolve_engine(engine)
    spec = _histplot_spec(df_column, show_mode)

    if engine == "plotly":
        return _histplot_plotly(spec, df_column=df_column,
                                is_limits=is_limits, bins=bins, kde=kde,
                                **kwargs)

    _histplot_mpl(spec, df_column=df_column, is_limits=is_limits,
                  bins=bins, kde=kde, ax=ax, figsize=figsize, **kwargs)
    return None


def auto_naive_plot(
    df_column,
    is_ordinal=False,
    is_limits=None,
    is_show_average=True,
    is_count_order=None,
    is_group_small_cats=True,
    *,
    engine=None,
):
    """
    Auto-detect feature type and plot an appropriate chart.

    Parameters
    ----------
    df_column : pd.Series
        Data to analyse.
    is_ordinal : bool, optional, default=False
        Treat as ordinal if not numerical.
    is_limits : bool or None, optional, default=None
        Apply axis limits for continuous features.
    is_show_average : bool, optional, default=True
        Print mean/median for numerical features.
    is_count_order : bool, optional, default=None
        Order categorical bars by count.
    is_group_small_cats : bool, optional, default=True
        Group small categories into "Other".
    engine : {"matplotlib", "plotly"} or None, default=None
        The rendering backend, forwarded to the underlying plot function.

    Returns
    -------
    plotly.graph_objects.Figure or None

    Examples
    --------
    >>> import pandas as pd
    >>> from pltstat.singlefeat import auto_naive_plot
    >>> data = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> auto_naive_plot(data)

    >>> # For ordinal features
    >>> auto_naive_plot(data, is_ordinal=True)
    """
    engine = _resolve_engine(engine)

    n_nan = np.sum(df_column.isnull())
    n_unique = len(df_column.dropna().unique())
    top5 = df_column.head().to_string()

    if n_nan > 0:
        df_column = df_column.dropna()

    dtype = "Categorical"
    try:
        df_column = df_column.astype("float64")
        dtype = "Numerical"
    except Exception:
        if n_unique == 2:
            dtype += ":Boolean"
        elif is_ordinal:
            dtype += ":Ordinal"
        else:
            dtype += ":Nominal"

    if dtype == "Numerical":
        if np.allclose(df_column, df_column.astype("int64")):
            dtype += ":Discrete"
        elif (df_column.max() <= 1) and (df_column.max() > 0) and \
                (df_column.min() >= -1):
            dtype += ":Proportion"
        else:
            dtype += ":Continuous"

    if dtype.startswith("Numerical") and (n_unique == 2):
        if bool(np.all(np.sort(df_column.unique()) == np.array([0, 1]))):
            dtype = "Categorical:Boolean"

    if (is_limits is None) and dtype.startswith("Numerical"):
        if dtype.endswith("Proportion") or (df_column.min() == 0) or \
                (df_column.min() == 1):
            is_limits = True
        else:
            is_limits = False

    print(f"Name of feature: '{df_column.name}'")
    print(f"Feature type: {dtype}")
    print(f"Number of unique values: {n_unique}")
    if n_nan > 0:
        print(f"Number of NaN values: {n_nan}")
    else:
        print("No NaN values found")
    print("\nFirst 5 values:")
    print(top5)
    print()

    if dtype.startswith("Numerical"):
        print(f"Min / Max values: {df_column.min():.3f} / "
              f"{df_column.max():.3f}")
        if is_show_average:
            print(f"Mean / Median values: {df_column.mean():.3f} / "
                  f"{df_column.median():.3f}")
        print()

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
            pie(df_column, is_count_order=is_count_order, engine=engine)
        elif dtype.startswith("Categorical"):
            countplot(df_column, is_count_order=is_count_order,
                      is_group_small_cats=is_group_small_cats, engine=engine)
        else:
            histplot(df_column, is_limits=is_limits, engine=engine)
    except Exception:
        print("Unable to determine feature format or plot it")