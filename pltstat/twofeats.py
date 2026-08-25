"""
Provides tools for analyzing interactions between two features.
Includes functions for creating crosstabs, computing correlations,
and visualizing results using violin plots, boxplots,
and distribution box plots. These functions also display p-values
and other statistical metrics to summarize relationships between
the two features.
"""

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from scipy import stats

from sklearn.linear_model import LinearRegression

from .stat_methods import cramer_v_by_obs, chi2_fisher_by_cat, matthews
from .stat_methods import kruskal_by_cat, mannwhitneyu_by_cat
from .config import _resolve_engine, _ensure_plotly


# ====================================================================
# Shared spec helpers (pure computation, no plotting)
# ====================================================================

def _boxplot_spec(df, cat_feat, num_feat, size, cat_order, alpha):
    """Compute the data needed for a boxplot.

    Returns a dict with ``cat_order``, ``n_cat_feat``, ``p_value``,
    ``test_type``, ``title_color``, ``counts``, and ``df``.
    """
    try:
        coef = {
            "tiny": 2.5,
            "compact": 2,
            "normal": 1.5,
            "huge": 1,
        }[size]
    except KeyError:
        raise ValueError(
            f"Value of size must be 'tiny', 'compact', 'normal', or "
            f"'huge' but given: {size}"
        )

    df = df[[cat_feat, num_feat]].dropna()
    if df.shape[0] == 0:
        return None

    df[cat_feat] = df[cat_feat].astype("str")
    if cat_order is not None:
        cat_order = np.array(cat_order).astype("str")
        cat_order = cat_order[np.isin(cat_order, df[cat_feat].unique())]
    else:
        cat_order = df[cat_feat].astype("str").unique()
        cat_order = np.sort(cat_order)

    n_cat_feat = len(df[cat_feat].unique())

    if n_cat_feat == 2:
        p = mannwhitneyu_by_cat(df, cat_feat=cat_feat, num_feat=num_feat)[1]
        test_type = "Mann-Whitney U-test"
    elif (n_cat_feat > 2) and (n_cat_feat <= 16):
        p = kruskal_by_cat(df, cat_feat=cat_feat, num_feat=num_feat)[1]
        test_type = "Kruskal-Wallis H-test"
    elif n_cat_feat > 16:
        raise ValueError(
            f"Too many unique values of categorical feature "
            f"'{cat_feat}': {n_cat_feat}"
        )
    else:
        raise ValueError(
            f"The categorical feature '{cat_feat}' has {n_cat_feat} "
            f"unique value(s), which seems unusual for this function."
        )

    title_color = "g" if p <= alpha else "r"
    counts = df.groupby(cat_feat, observed=False)[num_feat].count()

    return {
        "cat_order": cat_order,
        "n_cat_feat": n_cat_feat,
        "p_value": p,
        "test_type": test_type,
        "title_color": title_color,
        "counts": counts,
        "df": df,
        "cat_feat": cat_feat,
        "num_feat": num_feat,
        "palette": "pastel",
    }


# ====================================================================
# Matplotlib renderers (preserve exact current behaviour)
# ====================================================================

def _crosstab_mpl(df, x_col, y_col, values, aggfunc, title, color_title,
                  is_abs, is_norm, figsize, method, alpha, **kwargs):
    df_subset = df[[x_col, y_col]].dropna()

    def plot_crosstab_abs(method, ax_count):
        if df_subset.shape[0] == 0:
            print(f"Number of dataframe rows for columns "
                  f"{x_col} and {y_col} is zero")
            return
        crosstab_df = pd.crosstab(
            df_subset[x_col], df_subset[y_col],
            values=values, aggfunc=aggfunc,
        )
        _, p_value, method = chi2_fisher_by_cat(
            df_subset, x_col, y_col, method=method
        )
        test_type = "Exact Fisher" if method == "fisher" else "$chi^2$"

        if crosstab_df.shape == (2, 2):
            corr_type = "Matthews"
            correlation = matthews(df_subset[x_col], df_subset[y_col])
        else:
            corr_type = "Cramer V"
            correlation = cramer_v_by_obs(crosstab_df)

        if title is None:
            ax_count.set_title(
                "Crosstab. Absolute values\n%s p_value = %.3f; "
                "%s corr = %.3f" % (test_type, p_value, corr_type,
                                   correlation),
                color=color_title or ("g" if p_value <= alpha else "r"),
            )
        else:
            ax_count.set_title(title, color=color_title or "k")
        sns.heatmap(crosstab_df, annot=True, fmt=".0f", linewidths=1,
                    cmap="coolwarm", ax=ax_count, **kwargs)

    def plot_crosstab_norm(ax_norm):
        crosstab_df = pd.crosstab(
            df_subset[x_col],
            df_subset[y_col],
            normalize="index",
            values=values,
            aggfunc=aggfunc,
        )
        ax_norm.set_title("Crosstab. Normalized by index",
                          color=color_title or "k")
        sns.heatmap(
            crosstab_df, annot=True, fmt=".2f", vmin=0, vmax=1,
            linewidths=1, cmap="coolwarm", ax=ax_norm, **kwargs,
        )

    if is_abs and is_norm:
        figsize = figsize or (10, 3)
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        plot_crosstab_abs(method, ax_count=ax[0])
        plot_crosstab_norm(ax_norm=ax[1])
    else:
        figsize = figsize or (5, 3)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if is_abs is None:
            plot_crosstab_norm(ax_norm=ax)
        elif is_norm is None:
            plot_crosstab_abs(method, ax_count=ax)
        else:
            raise ValueError(
                "Not less that one of is_abs or is_norm must be True"
            )


def _corr_mpl(df, col_x, col_y, ax, show_means, show_regression, **kwargs):
    data = df[[col_x, col_y]].dropna()
    x = data[col_x].values
    y = data[col_y].values

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(x, y, c="green", s=2, label="Data Points", **kwargs)

    if show_means:
        ax.plot(
            [x.mean()] * 2, [y.min(), y.max()], "--r",
            label=r"$\overline{%s}$" % col_x.replace("_", r"\_"),
        )
        ax.plot(
            [x.min(), x.max()], [y.mean()] * 2, "--b",
            label=r"$\overline{%s}$" % col_y.replace("_", r"\_"),
        )

    if show_regression:
        lr = LinearRegression().fit(x.reshape((-1, 1)), y)
        r, p_value = stats.pearsonr(x, y)
        x_th = np.array([x.min(), x.max()])
        y_th = lr.predict(x_th.reshape((-1, 1)))
        ax.plot(x_th, y_th,
                label=f"{lr.intercept_:.3f} + {lr.coef_[0]:.3f}x")
        ax.set_title(f"r = {r:.3f}, p_value = {p_value:.3f}")

    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.legend()


def _boxplot_mpl(spec, ax, fig_return, palette, **kwargs):
    cat_order = spec["cat_order"]
    df = spec["df"]
    cat_feat = spec["cat_feat"]
    num_feat = spec["num_feat"]

    if ax is None:
        h_size = spec["n_cat_feat"] / 2
        w_size = 18
        fig, ax = plt.subplots(1, 1, figsize=(w_size, h_size))

    fig = sns.boxplot(
        data=df, x=num_feat, y=cat_feat, hue=cat_feat,
        legend=False, orient="h", fliersize=1, showmeans=True,
        order=cat_order, hue_order=cat_order, ax=ax,
        palette=palette,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
        **kwargs,
    )

    if cat_order is None:
        cat_order = fig.axes.get_yticklabels()
        cat_order = [plt_text.get_text() for plt_text in cat_order]
        dtype = df[cat_feat].dtype
        cat_order = np.array(cat_order).astype(dtype)

    ax.set_title(
        f"{spec['test_type']} p-value={spec['p_value']:.3f}",
        color=spec["title_color"],
    )

    counts = spec["counts"].loc[cat_order]
    xmin, xmax, ymin, ymax = ax.axis()
    x = (xmin + xmax) / 2
    for y, val in enumerate(counts):
        color = "r" if val < 30 else "k"
        ax.text(x, y, "count=" + str(val), color=color, fontweight="bold")

    if fig_return:
        return fig
    return None


def _dis_box_plot_mpl(df, cat_feat, num_feat, cat_order, stat, figsize,
                      palette, alpha, ax_return):
    df_subset = df[[cat_feat, num_feat]].dropna()
    if df_subset.shape[0] == 0:
        print(f"Number of dataframe rows for columns "
              f"{cat_feat} and {num_feat} is zero")
        return

    df_subset[cat_feat] = df_subset[cat_feat].astype("str")
    if cat_order is not None:
        cat_order = np.array(cat_order).astype("str")
        cat_order = cat_order[
            np.isin(cat_order, df_subset[cat_feat].unique())
        ]
    else:
        cat_order = df_subset[cat_feat].astype("str").unique()
        cat_order = np.sort(cat_order)

    _, ax = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [1, 2]},
    )
    fig = boxplot(
        df_subset, cat_feat, num_feat, fig_return=True,
        cat_order=cat_order, ax=ax[0], palette=palette, alpha=alpha,
        engine="matplotlib",
    )

    title = fig.axes.get_title()
    ax[0].set_title(f"{num_feat}\n{title}")

    sns.histplot(
        data=df_subset, x=num_feat, hue=cat_feat, kde=True,
        hue_order=cat_order, stat=stat, common_norm=False,
        ax=ax[1], palette=palette,
    )
    if ax_return:
        return ax
    return None


# ====================================================================
# Plotly renderers
# ====================================================================

def _crosstab_plotly(df, x_col, y_col, values, aggfunc, title, color_title,
                     is_abs, is_norm, figsize, method, alpha, **kwargs):
    _ensure_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df_subset = df[[x_col, y_col]].dropna()

    if df_subset.shape[0] == 0:
        print(f"Number of dataframe rows for columns "
              f"{x_col} and {y_col} is zero")
        return None

    show_both = is_abs and is_norm
    if show_both:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Absolute values",
                                             "Normalized by index"))
    else:
        fig = make_subplots(rows=1, cols=1)

    col = 1

    if is_abs:
        crosstab_df = pd.crosstab(
            df_subset[x_col], df_subset[y_col],
            values=values, aggfunc=aggfunc,
        )
        _, p_value, resolved_method = chi2_fisher_by_cat(
            df_subset, x_col, y_col, method=method
        )
        test_type = "Exact Fisher" if resolved_method == "fisher" \
            else "chi^2"

        if crosstab_df.shape == (2, 2):
            corr_type = "Matthews"
            correlation = matthews(df_subset[x_col], df_subset[y_col])
        else:
            corr_type = "Cramer V"
            correlation = cramer_v_by_obs(crosstab_df)

        if title is None:
            subtitle = (
                "Crosstab. Absolute values\n%s p_value = %.3f; "
                "%s corr = %.3f" % (test_type, p_value, corr_type,
                                    correlation)
            )
        else:
            subtitle = title

        heatmap = go.Heatmap(
            z=crosstab_df.values,
            x=crosstab_df.columns.astype(str),
            y=crosstab_df.index.astype(str),
            colorscale="RdBu",
            text=crosstab_df.values.round(0).astype(str),
            texttemplate="%{text}",
            **kwargs,
        )
        if show_both:
            fig.add_trace(heatmap, row=1, col=1)
            fig.update_xaxes(title_text=subtitle, row=1, col=1)
        else:
            fig.add_trace(heatmap)
            fig.update_layout(title=subtitle)
        col = 2

    if is_norm:
        crosstab_norm = pd.crosstab(
            df_subset[x_col], df_subset[y_col],
            normalize="index", values=values, aggfunc=aggfunc,
        )
        heatmap_norm = go.Heatmap(
            z=crosstab_norm.values,
            x=crosstab_norm.columns.astype(str),
            y=crosstab_norm.index.astype(str),
            colorscale="RdBu", zmin=0, zmax=1,
            text=crosstab_norm.values.round(2),
            texttemplate="%{text}",
            **kwargs,
        )
        if show_both:
            fig.add_trace(heatmap_norm, row=1, col=2)
        else:
            fig.add_trace(heatmap_norm)
            fig.update_layout(title="Crosstab. Normalized by index")

    return fig


def _corr_plotly(df, col_x, col_y, show_means, show_regression, **kwargs):
    _ensure_plotly()
    import plotly.express as px

    data = df[[col_x, col_y]].dropna()
    x = data[col_x].values
    y = data[col_y].values

    fig = px.scatter(
        data, x=col_x, y=col_y, **kwargs,
    )
    fig.update_traces(marker=dict(color="green", size=3))

    if show_means:
        fig.add_vline(
            x=x.mean(), line_dash="dash", line_color="red",
            annotation_text=f"mean({col_x})",
        )
        fig.add_hline(
            y=y.mean(), line_dash="dash", line_color="blue",
            annotation_text=f"mean({col_y})",
        )

    if show_regression:
        lr = LinearRegression().fit(x.reshape((-1, 1)), y)
        r, p_value = stats.pearsonr(x, y)
        x_th = np.array([x.min(), x.max()])
        y_th = lr.predict(x_th.reshape((-1, 1)))
        fig.add_scatter(
            x=x_th, y=y_th, mode="lines",
            name=f"{lr.intercept_:.3f} + {lr.coef_[0]:.3f}x",
            line=dict(color="orange"),
        )
        fig.update_layout(
            title=f"r = {r:.3f}, p_value = {p_value:.3f}"
        )

    return fig


def _boxplot_plotly(spec, palette, **kwargs):
    _ensure_plotly()
    import plotly.express as px

    df = spec["df"]
    cat_feat = spec["cat_feat"]
    num_feat = spec["num_feat"]
    cat_order = spec["cat_order"]

    color = cat_feat if palette is not None else None
    fig = px.box(
        df, x=num_feat, y=cat_feat, color=color,
        category_orders={cat_feat: list(cat_order)},
        orientation="h",
        **kwargs,
    )
    fig.update_layout(
        title=f"{spec['test_type']} p-value={spec['p_value']:.3f}",
    )
    return fig


def _dis_box_plot_plotly(df, cat_feat, num_feat, cat_order, stat,
                         figsize, palette, alpha):
    _ensure_plotly()
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df_subset = df[[cat_feat, num_feat]].dropna()
    if df_subset.shape[0] == 0:
        print(f"Number of dataframe rows for columns "
              f"{cat_feat} and {num_feat} is zero")
        return None

    df_subset[cat_feat] = df_subset[cat_feat].astype("str")
    if cat_order is not None:
        cat_order = np.array(cat_order).astype("str")
        cat_order = cat_order[
            np.isin(cat_order, df_subset[cat_feat].unique())
        ]
    else:
        cat_order = df_subset[cat_feat].astype("str").unique()
        cat_order = np.sort(cat_order)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[1, 2],
        vertical_spacing=0.15,
        subplot_titles=(f"{num_feat}", ""),
    )

    box_fig = px.box(
        df_subset, x=num_feat, y=cat_feat, color=cat_feat,
        category_orders={cat_feat: list(cat_order)},
        orientation="h",
    )
    for trace in box_fig.data:
        fig.add_trace(trace, row=1, col=1)

    hist_fig = px.histogram(
        df_subset, x=num_feat, color=cat_feat,
        marginal="violin", opacity=0.6,
        category_orders={cat_feat: list(cat_order)},
        histnorm=stat if stat != "count" else None,
    )
    for trace in hist_fig.data:
        fig.add_trace(trace, row=2, col=1)

    fig.update_layout(showlegend=False)
    return fig


# ====================================================================
# Public API (dispatchers)
# ====================================================================

def crosstab(
        df,
        x_col,
        y_col,
        values=None,
        aggfunc=None,
        title=None,
        color_title=None,
        is_abs=True,
        is_norm=True,
        figsize=None,
        method="auto",
        alpha=0.05,
        *,
        engine=None,
        **kwargs,
):
    """
    Plot crosstab with improved settings and statistical analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data for crosstab analysis.
    x_col : str
        The name of the column to use as the x-axis.
    y_col : str
        The name of the column to use as the y-axis.
    values : str or None, optional, default=None
        The column name to aggregate. If None, the crosstab will count
        occurrences.
    aggfunc : callable or None, optional, default=None
        The aggregation function to use if `values` is specified.
    title : str or None, optional, default=None
        The title of the plot. If None, an automatic title with
        statistics is generated.
    color_title : str or None, optional, default=None
        The color of the title text (matplotlib only).
    is_abs : bool, optional, default=True
        If True, plot the crosstab with absolute values.
    is_norm : bool, optional, default=True
        If True, plot the crosstab normalized by row indices.
    figsize : tuple or None, optional, default=None
        The size of the figure (matplotlib only).
    method : {'auto', 'fisher', 'chi2'}, optional, default='auto'
        The statistical test to use.
    alpha : float, optional, default=0.05
        The threshold for statistical significance (p-value).
    engine : {"matplotlib", "plotly"} or None, default=None
        The rendering backend.
    **kwargs : dict
        Additional keyword arguments passed to ``sns.heatmap`` or
        ``go.Heatmap``.

    Returns
    -------
    plotly.graph_objects.Figure or None

    Examples
    --------
    >>> import pandas as pd
    >>> from pltstat.twofeats import crosstab
    >>> data = pd.DataFrame({
    ...     "Gender": ["Male", "Female", "Male", "Female", "Male"],
    ...     "Preference": ["A", "B", "A", "A", "B"]
    ... })
    >>> crosstab(data, x_col="Gender", y_col="Preference")
    """
    engine = _resolve_engine(engine)

    if engine == "plotly":
        return _crosstab_plotly(
            df, x_col, y_col, values, aggfunc, title, color_title,
            is_abs, is_norm, figsize, method, alpha, **kwargs,
        )

    _crosstab_mpl(
        df, x_col, y_col, values, aggfunc, title, color_title,
        is_abs, is_norm, figsize, method, alpha, **kwargs,
    )
    return None


def corr(
        df,
        col_x,
        col_y,
        ax=None,
        show_means=True,
        show_regression=True,
        *,
        engine=None,
        **kwargs,
):
    """
    Plot correlation with enhanced settings.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data for analysis.
    col_x : str
        The column name for the x-axis.
    col_y : str
        The column name for the y-axis.
    ax : matplotlib.axes.Axes or None, optional, default=None
        The axes on which to draw the plot (matplotlib only).
    show_means : bool, optional, default=True
        Whether to show the mean lines for both x and y axes.
    show_regression : bool, optional, default=True
        Whether to display the regression line with the correlation
        coefficient.
    engine : {"matplotlib", "plotly"} or None, default=None
        The rendering backend.
    **kwargs : dict, optional
        Additional keyword arguments passed to ``ax.scatter`` or
        ``px.scatter``.

    Returns
    -------
    plotly.graph_objects.Figure or None

    Examples
    --------
    >>> import pandas as pd
    >>> from pltstat.twofeats import corr
    >>> data = pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 4, 6, 8]})
    >>> corr(data, "A", "B")
    """
    engine = _resolve_engine(engine)

    if engine == "plotly":
        return _corr_plotly(df, col_x, col_y, show_means,
                            show_regression, **kwargs)

    _corr_mpl(df, col_x, col_y, ax, show_means, show_regression, **kwargs)
    return None


def boxplot(
        df,
        cat_feat,
        num_feat,
        size="compact",
        cat_order=None,
        fig_return=False,
        alpha=0.05,
        ax=None,
        palette="pastel",
        *,
        engine=None,
        **kwargs,
):
    """
    Plot a boxplot for a numeric feature grouped by a categorical
    feature with statistical testing.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    cat_feat : str
        The name of the categorical feature to group by.
    num_feat : str
        The name of the numeric feature to plot.
    size : {'tiny', 'compact', 'normal', 'huge'}, optional, default 'compact'
        The size of the figure (matplotlib only).
    cat_order : list or array-like, optional, default None
        The desired order of categories in the plot.
    fig_return : bool, optional, default False
        If True, returns the figure object (matplotlib only).
    alpha : float, optional, default 0.05
        The significance level for the statistical test.
    ax : matplotlib.axes.Axes, optional, default None
        The axes on which to plot the boxplot (matplotlib only).
    palette : str or list, optional, default 'pastel'
        The color palette to use for the plot.
    engine : {"matplotlib", "plotly"} or None, default=None
        The rendering backend.
    **kwargs : additional keyword arguments, optional
        Additional arguments passed to ``sns.boxplot`` or ``px.box``.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure or None

    Notes
    -----
    The function computes a statistical test based on the number of
    unique values in the categorical feature: if there are exactly 2
    categories, the Mann-Whitney U-test is applied; if there are
    between 3 and 16 categories, the Kruskal-Wallis H-test is applied.
    The p-value from the test is displayed in the plot title.

    Raises
    ------
    ValueError
        If the `size` parameter is not one of 'tiny', 'compact',
        'normal', or 'huge', or if the categorical feature has more
        than 20 unique values.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.twofeats import boxplot
    >>>
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     'category': np.random.choice(['A', 'B', 'C'], size=100),
    ...     'value': np.random.randn(100)
    ... })
    >>> boxplot(df, cat_feat='category', num_feat='value', size='normal')
    """
    engine = _resolve_engine(engine)
    spec = _boxplot_spec(df, cat_feat, num_feat, size, cat_order, alpha)

    if spec is None:
        print(f"Number of dataframe rows for columns "
              f"{cat_feat} and {num_feat} is zero")
        return None

    if engine == "plotly":
        return _boxplot_plotly(spec, palette=palette, **kwargs)

    return _boxplot_mpl(spec, ax=ax, fig_return=fig_return,
                        palette=palette, **kwargs)


def dis_box_plot(
        df,
        cat_feat,
        num_feat,
        cat_order=None,
        stat="count",
        figsize=(20, 3.5),
        palette="pastel",
        alpha=0.05,
        ax_return=False,
        *,
        engine=None,
):
    """
    Plot a boxplot and a distribution plot for a numeric feature
    grouped by a categorical feature.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be visualized.
    cat_feat : str
        The name of the categorical feature by which the data will be
        grouped.
    num_feat : str
        The name of the numeric feature to plot.
    cat_order : list or array-like, optional, default None
        The desired order of categories for the categorical feature.
    stat : {'count', 'probability', 'density', 'frequency'}, optional,
        default='count'
        The statistic to plot in the distribution plot.
    figsize : tuple of int, optional, default (20, 3.5)
        The size of the figure (matplotlib only).
    palette : str or list, optional, default 'pastel'
        The color palette to use for the plot.
    alpha : float, optional, default 0.05
        The significance level for the statistical test.
    ax_return : bool, optional, default False
        If True, returns the axes object(s) (matplotlib only).
    engine : {"matplotlib", "plotly"} or None, default=None
        The rendering backend.

    Returns
    -------
    plotly.graph_objects.Figure or None

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.twofeats import dis_box_plot
    >>>
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     'target': np.random.choice(['A', 'B'], size=100),
    ...     'value': np.random.randn(100)
    ... })
    >>> dis_box_plot(df, cat_feat='target', num_feat='value')
    """
    engine = _resolve_engine(engine)

    if engine == "plotly":
        return _dis_box_plot_plotly(
            df, cat_feat, num_feat, cat_order, stat, figsize,
            palette, alpha,
        )

    return _dis_box_plot_mpl(
        df, cat_feat, num_feat, cat_order, stat, figsize,
        palette, alpha, ax_return,
    )