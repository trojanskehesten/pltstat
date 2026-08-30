"""
Provides tools for analyzing interactions between two features.
Includes functions for creating crosstabs, computing correlations,
and visualizing results using violin plots, boxplots,
and distribution box plots. These functions also display p-values
and other statistical metrics to summarize relationships between
the two features.
"""

from dataclasses import dataclass
from types import ModuleType

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import seaborn as sns

import numpy as np
import pandas as pd

from scipy import stats

from sklearn.linear_model import LinearRegression

from . import cm
from .cm import format_matrix as _format_matrix
from .config import _figsize_to_px, _import_plotly, _resolve_engine, _warn_ignored_mpl_params
from .stat_methods import cramer_v_by_obs, chi2_fisher_by_cat, matthews
from .stat_methods import kde_curve, kruskal_by_cat, mannwhitneyu_by_cat


# --- Specs: plot-ready data shared by both engines ---


@dataclass(frozen=True)
class _CrosstabSpec:
    """
    Plot-ready data of a crosstab.

    Attributes
    ----------
    crosstab_abs : pd.DataFrame or None
        Crosstab of the absolute values, or None when the subset is empty.
    crosstab_norm : pd.DataFrame or None
        Crosstab normalized by index, or None when the subset is empty.
    p_value : float or None
        p-value of the test of independence of the two features.
    correlation : float or None
        Association between the two features.
    test_type_mpl : str or None
        Name of the test, with the mathtext markup understood by matplotlib.
    test_type_plotly : str or None
        Name of the test, as plain text for plotly.
    corr_type : str or None
        Name of the association measure, "Matthews" or "Cramer V".
    is_empty : bool
        True when the two features have no common non missing observation.
    """

    crosstab_abs: object
    crosstab_norm: object
    p_value: object
    correlation: object
    test_type_mpl: object
    test_type_plotly: object
    corr_type: object
    is_empty: bool


@dataclass(frozen=True)
class _CorrSpec:
    """
    Plot-ready data of a correlation scatter plot.

    Attributes
    ----------
    x : np.ndarray
        Observations of the feature drawn on the x axis.
    y : np.ndarray
        Observations of the feature drawn on the y axis.
    col_x : str
        Name of the feature drawn on the x axis.
    col_y : str
        Name of the feature drawn on the y axis.
    x_mean : float
        Mean of ``x``.
    y_mean : float
        Mean of ``y``.
    reg_x : np.ndarray or None
        Ends of the regression line on the x axis, or None when it is hidden.
    reg_y : np.ndarray or None
        Ends of the regression line on the y axis, or None when it is hidden.
    reg_label : str or None
        Label of the regression line, or None when it is hidden.
    title : str or None
        Title with the correlation and its p-value, or None when the
        regression is hidden.
    """

    x: object
    y: object
    col_x: str
    col_y: str
    x_mean: float
    y_mean: float
    reg_x: object
    reg_y: object
    reg_label: object
    title: object


def _crosstab_spec(df, x_col, y_col, values=None, aggfunc=None, method="auto"):
    """
    Compute the plot-ready data of a crosstab.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the two categorical features.
    x_col : str
        Name of the feature drawn on the x axis.
    y_col : str
        Name of the feature drawn on the y axis.
    values : str or None, default: None
        Name of the column to aggregate, passed to :func:`pandas.crosstab`.
    aggfunc : callable or None, default: None
        Aggregation applied to ``values``.
    method : {"auto", "fisher", "chi2"}, default: "auto"
        Test of independence used to compute the p-value.

    Returns
    -------
    spec : _CrosstabSpec
        Crosstabs, p-value and association of the two features.

    Notes
    -----
    The statistics are computed once here, so that both engines annotate the
    heatmap with the same numbers.

    Examples
    --------
    >>> import pandas as pd
    >>> from pltstat.twofeats import _crosstab_spec
    >>> data = pd.DataFrame({"A": ["x", "x", "y", "y"], "B": ["u", "v", "u", "v"]})
    >>> _crosstab_spec(data, "A", "B").corr_type
    'Matthews'
    """
    df_subset = df[[x_col, y_col]].dropna()

    if df_subset.shape[0] == 0:
        return _CrosstabSpec(
            crosstab_abs=None,
            crosstab_norm=None,
            p_value=None,
            correlation=None,
            test_type_mpl=None,
            test_type_plotly=None,
            corr_type=None,
            is_empty=True,
        )

    crosstab_abs = pd.crosstab(df_subset[x_col], df_subset[y_col], values=values, aggfunc=aggfunc)
    crosstab_norm = pd.crosstab(
        df_subset[x_col],
        df_subset[y_col],
        normalize="index",
        values=values,
        aggfunc=aggfunc,
    )

    _, p_value, method = chi2_fisher_by_cat(df_subset, x_col, y_col, method=method)
    if method == "fisher":
        test_type_mpl = "Exact Fisher"
        test_type_plotly = "Exact Fisher"
    else:
        test_type_mpl = "$chi^2$"
        # Plotly does not render the mathtext markup of matplotlib
        test_type_plotly = "chi2"

    if crosstab_abs.shape == (2, 2):
        corr_type = "Matthews"
        correlation = matthews(df_subset[x_col], df_subset[y_col])
    else:
        corr_type = "Cramer V"
        correlation = cramer_v_by_obs(crosstab_abs)

    return _CrosstabSpec(
        crosstab_abs=crosstab_abs,
        crosstab_norm=crosstab_norm,
        p_value=p_value,
        correlation=correlation,
        test_type_mpl=test_type_mpl,
        test_type_plotly=test_type_plotly,
        corr_type=corr_type,
        is_empty=False,
    )


def _corr_spec(df, col_x, col_y, show_regression=True):
    """
    Compute the plot-ready data of a correlation scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the two numerical features.
    col_x : str
        Name of the feature drawn on the x axis.
    col_y : str
        Name of the feature drawn on the y axis.
    show_regression : bool, default: True
        If True, the regression line and the correlation are computed.

    Returns
    -------
    spec : _CorrSpec
        Observations, means and regression line of the two features.

    Examples
    --------
    >>> import pandas as pd
    >>> from pltstat.twofeats import _corr_spec
    >>> data = pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 4, 6, 8]})
    >>> _corr_spec(data, "A", "B").title
    'r = 1.000, p_value = 0.000'
    """
    data = df[[col_x, col_y]].dropna()
    x = data[col_x].values
    y = data[col_y].values

    reg_x = None
    reg_y = None
    reg_label = None
    title = None

    if show_regression:
        lr = LinearRegression().fit(x.reshape((-1, 1)), y)
        r, p_value = stats.pearsonr(x, y)
        reg_x = np.array([x.min(), x.max()])
        reg_y = lr.predict(reg_x.reshape((-1, 1)))
        reg_label = f"{lr.intercept_:.3f} + {lr.coef_[0]:.3f}x"
        title = f"r = {r:.3f}, p_value = {p_value:.3f}"

    return _CorrSpec(
        x=x,
        y=y,
        col_x=col_x,
        col_y=col_y,
        x_mean=x.mean(),
        y_mean=y.mean(),
        reg_x=reg_x,
        reg_y=reg_y,
        reg_label=reg_label,
        title=title,
    )


# --- Matplotlib renderers ---


def _crosstab_mpl(spec, title=None, color_title=None, is_abs=True, is_norm=True,
                  figsize=None, alpha=0.05, **kwargs):
    """
    Draw a crosstab with matplotlib.

    Parameters
    ----------
    spec : _CrosstabSpec
        Plot-ready data built by :func:`_crosstab_spec`.
    title : str or None, default: None
        Title of the heatmap of absolute values.
    color_title : str or None, default: None
        Colour of the titles. If None, the title of the absolute values is
        green when the p-value is significant and red otherwise.
    is_abs : bool, default: True
        If True, the heatmap of absolute values is drawn.
    is_norm : bool, default: True
        If True, the heatmap normalized by index is drawn.
    figsize : tuple or None, default: None
        Size of the figure in inches.
    alpha : float, default: 0.05
        Significance level used to colour the title.
    **kwargs : keyword arguments, optional
        Additional arguments passed to ``sns.heatmap()``.

    Returns
    -------
    None
        The function modifies the plot in place and does not return any value.
    """
    def plot_crosstab_abs(ax_count):
        """Plot Heatmap with absolute values crosstab and statistics"""
        if spec.is_empty:
            return

        if title is None:
            ax_count.set_title(
                "Crosstab. Absolute values\n%s p_value = %.3f; %s corr = %.3f"
                % (spec.test_type_mpl, spec.p_value, spec.corr_type, spec.correlation),
                color=color_title or ("g" if spec.p_value <= alpha else "r"),
            )
        else:
            ax_count.set_title(title, color=color_title or "k")
        sns.heatmap(
            spec.crosstab_abs, annot=True, fmt=".0f", linewidths=1,
            cmap="coolwarm", ax=ax_count, **kwargs,
        )

    def plot_crosstab_norm(ax_norm):
        """Plot Heatmap with normalized by row indices crosstab"""
        if spec.is_empty:
            return

        ax_norm.set_title("Crosstab. Normalized by index", color=color_title or "k")
        sns.heatmap(
            spec.crosstab_norm,
            annot=True,
            fmt=".2f",
            vmin=0,
            vmax=1,
            linewidths=1,
            cmap="coolwarm",
            ax=ax_norm,
            **kwargs,
        )

    # Plot Heatmaps and calculate statistics:
    if is_abs and is_norm:
        figsize = figsize or (10, 3)
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        plot_crosstab_abs(ax_count=ax[0])
        plot_crosstab_norm(ax_norm=ax[1])
    else:
        figsize = figsize or (5, 3)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if is_abs is None:
            plot_crosstab_abs(ax_count=ax)
        elif is_norm is None:
            plot_crosstab_norm(ax_norm=ax)
        else:
            raise ValueError("Not less that one of is_abs or is_norm must be True")


def _corr_mpl(spec, ax=None, show_means=True, show_regression=True, **kwargs):
    """
    Draw a correlation scatter plot with matplotlib.

    Parameters
    ----------
    spec : _CorrSpec
        Plot-ready data built by :func:`_corr_spec`.
    ax : matplotlib.axes.Axes or None, default: None
        Axes to draw on. If None, a new figure and Axes are created.
    show_means : bool, default: True
        If True, the mean of each feature is drawn as a dashed line.
    show_regression : bool, default: True
        If True, the regression line is drawn.
    **kwargs : keyword arguments, optional
        Additional arguments passed to ``ax.scatter()``.

    Returns
    -------
    None
        The function modifies the plot in place and does not return any value.
    """
    x = spec.x
    y = spec.y

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(x, y, c="green", s=2, label="Data Points", **kwargs)

    # Show mean lines
    if show_means:
        ax.plot(
            [spec.x_mean] * 2,
            [y.min(), y.max()],
            "--r",
            label=r"$\overline{%s}$" % spec.col_x.replace("_", r"\_"),
        )
        ax.plot(
            [x.min(), x.max()],
            [spec.y_mean] * 2,
            "--b",
            label=r"$\overline{%s}$" % spec.col_y.replace("_", r"\_"),
        )

    # Show regression line and correlation
    if show_regression:
        ax.plot(spec.reg_x, spec.reg_y, label=spec.reg_label)
        ax.set_title(spec.title)

    # Axis labels
    ax.set_xlabel(spec.col_x)
    ax.set_ylabel(spec.col_y)

    # Legend
    ax.legend()


# --- Plotly renderers ---


def _to_plotly_color(color):
    """
    Convert a matplotlib colour into a colour understood by plotly.

    Parameters
    ----------
    color : str or None
        Colour as accepted by matplotlib, including the single letter codes
        such as "g" or "r". None is returned unchanged.

    Returns
    -------
    color : str or None
        Hexadecimal colour, or None when `color` is None.

    Notes
    -----
    Plotly rejects the single letter colour codes of matplotlib, so every
    colour is normalized to its hexadecimal form.

    Examples
    --------
    >>> from pltstat.twofeats import _to_plotly_color
    >>> _to_plotly_color("g")
    '#008000'
    >>> _to_plotly_color(None) is None
    True
    """
    if color is None:
        return None
    return to_hex(color)


def _crosstab_plotly(spec, title=None, color_title=None, is_abs=True, is_norm=True,
                     figsize=None, alpha=0.05, **kwargs):
    """
    Draw a crosstab with plotly.

    Parameters
    ----------
    spec : _CrosstabSpec
        Plot-ready data built by :func:`_crosstab_spec`.
    title : str or None, default: None
        Title of the heatmap of absolute values.
    color_title : str or None, default: None
        Colour of the titles. If None, the title of the absolute values is
        green when the p-value is significant and red otherwise.
    is_abs : bool, default: True
        If True, the heatmap of absolute values is drawn.
    is_norm : bool, default: True
        If True, the heatmap normalized by index is drawn.
    figsize : tuple or None, default: None
        Size of the figure in inches, converted to pixels.
    alpha : float, default: 0.05
        Significance level used to colour the title.
    **kwargs : keyword arguments, optional
        Additional arguments passed to ``plotly.graph_objects.Heatmap``.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The crosstab.
    """
    go, make_subplots = _import_plotly()

    if is_abs and is_norm:
        figsize = figsize or (10, 3)
    else:
        figsize = figsize or (5, 3)
    width, height = _figsize_to_px(figsize)

    if spec.is_empty:
        return go.Figure(layout={"width": width, "height": height})

    if title is None:
        title_abs = (
            "Crosstab. Absolute values<br>%s p_value = %.3f; %s corr = %.3f"
            % (spec.test_type_plotly, spec.p_value, spec.corr_type, spec.correlation)
        )
        color_abs = _to_plotly_color(color_title) or ("green" if spec.p_value <= alpha else "red")
    else:
        title_abs = title
        color_abs = _to_plotly_color(color_title) or "black"

    title_norm = "Crosstab. Normalized by index"
    color_norm = _to_plotly_color(color_title) or "black"

    if is_abs and is_norm:
        titles = [title_abs, title_norm]
        colors = [color_abs, color_norm]
        panels = [(spec.crosstab_abs, ".0f", None, None, False),
                  (spec.crosstab_norm, ".2f", 0, 1, True)]
    elif is_abs is None:
        titles, colors = [title_abs], [color_abs]
        panels = [(spec.crosstab_abs, ".0f", None, None, True)]
    elif is_norm is None:
        titles, colors = [title_norm], [color_norm]
        panels = [(spec.crosstab_norm, ".2f", 0, 1, True)]
    else:
        raise ValueError("Not less that one of is_abs or is_norm must be True")

    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=titles)

    for position, (data, fmt, zmin, zmax, showscale) in enumerate(panels, start=1):
        fig.add_trace(
            go.Heatmap(
                z=data.values,
                x=[str(column) for column in data.columns],
                y=[str(index) for index in data.index],
                zmin=zmin,
                zmax=zmax,
                colorscale="RdBu_r",
                text=_format_matrix(data.values, fmt),
                texttemplate="%{text}",
                # Reproduce the white grid drawn by seaborn
                xgap=1,
                ygap=1,
                showscale=showscale,
                hovertemplate="%{y} / %{x}<br>%{z}<extra></extra>",
                **kwargs,
            ),
            row=1,
            col=position,
        )
        fig.update_xaxes(title_text=str(data.columns.name), row=1, col=position)
        fig.update_yaxes(
            title_text=str(data.index.name),
            # Seaborn draws the first row at the top, plotly at the bottom
            autorange="reversed",
            row=1,
            col=position,
        )

    for annotation, color in zip(fig.layout.annotations, colors):
        annotation.font.color = color

    fig.update_layout(width=width, height=height)

    return fig


def _corr_plotly(spec, show_means=True, show_regression=True, figsize=(8, 6), **kwargs):
    """
    Draw a correlation scatter plot with plotly.

    Parameters
    ----------
    spec : _CorrSpec
        Plot-ready data built by :func:`_corr_spec`.
    show_means : bool, default: True
        If True, the mean of each feature is drawn as a dashed line.
    show_regression : bool, default: True
        If True, the regression line is drawn.
    figsize : tuple, default: (8, 6)
        Size of the figure in inches, converted to pixels.
    **kwargs : keyword arguments, optional
        Additional arguments passed to ``plotly.graph_objects.Scattergl``.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The correlation scatter plot.
    """
    go, _ = _import_plotly()
    width, height = _figsize_to_px(figsize)

    fig = go.Figure(
        go.Scattergl(
            x=spec.x,
            y=spec.y,
            mode="markers",
            marker={"color": "green", "size": 3},
            name="Data Points",
            **kwargs,
        )
    )

    if show_means:
        # Plotly has no mathtext, so the means are named in plain text
        fig.add_trace(
            go.Scatter(
                x=[spec.x_mean] * 2,
                y=[spec.y.min(), spec.y.max()],
                mode="lines",
                line={"color": "red", "dash": "dash"},
                name=f"mean({spec.col_x})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[spec.x.min(), spec.x.max()],
                y=[spec.y_mean] * 2,
                mode="lines",
                line={"color": "blue", "dash": "dash"},
                name=f"mean({spec.col_y})",
            )
        )

    if show_regression:
        fig.add_trace(
            go.Scatter(
                x=spec.reg_x,
                y=spec.reg_y,
                mode="lines",
                name=spec.reg_label,
            )
        )

    fig.update_layout(
        title=spec.title,
        width=width,
        height=height,
        xaxis_title=spec.col_x,
        yaxis_title=spec.col_y,
    )

    return fig


# --- Public plotting functions ---


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
        engine=None,
        **kwargs,
):
    """
    Plot crosstab with improved settings and detailed statistical analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data for crosstab analysis.
    x_col : str
        The name of the column to use as the x-axis.
    y_col : str
        The name of the column to use as the y-axis.
    values : str or None, optional, default=None
        The column name to aggregate. If None, the crosstab will count occurrences.
    aggfunc : callable or None, optional, default=None
        The aggregation function to use if `values` is specified.
    title : str or None, optional, default=None
        The title of the plot. If None, an automatic title with statistics is generated.
    color_title : str or None, optional, default=None
        The color of the title text. If None, the color is green when the differences are significant and red in other cases.
    is_abs : bool, optional, default=True
        If True, plot the crosstab with absolute values.
    is_norm : bool, optional, default=True
        If True, plot the crosstab normalized by row indices.
    figsize : tuple or None, optional, default=None
        The size of the figure. If None, default sizes are used.
    method : {'auto', 'fisher', 'chi2'}, optional, default='auto'
        The statistical test to use. If 'auto', Fisher's exact test is used
        when any cell count in the crosstab is less than 5; otherwise the
        chi-squared test is used.
    alpha : float, optional, default=0.05
        The threshold for statistical significance (p-value).
    engine : {"matplotlib", "plotly"} or None, optional, default=None
        The rendering engine. If None, the engine set by
        :func:`pltstat.set_backend` is used.
    **kwargs : dict
        Additional keyword arguments to further customize the heatmaps. They are
        passed to `sns.heatmap` with matplotlib and to
        ``plotly.graph_objects.Heatmap`` with plotly, so they are engine specific.

    Returns
    -------
    fig : plotly.graph_objects.Figure or None
        The figure when ``engine="plotly"``. With matplotlib the function
        creates and displays the plots and returns None.

    Notes
    -----
    - If both `is_abs` and `is_norm` are True, two plots are displayed: absolute values and normalized values.
    - The name of the chi-squared test uses the mathtext markup of matplotlib,
      which plotly does not render, so the plotly title shows "chi2".
    - The function can automatically detect and apply the appropriate statistical test (Chi-square or Fisher's exact test).
    - For binary 2x2 tables, Matthews correlation is calculated; otherwise, Cramér's V is used.

    Example
    --------
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> from pltstat.twofeats import crosstab
    >>> data = pd.DataFrame({
    >>>     "Gender": ["Male", "Female", "Male", "Female", "Male"],
    >>>     "Preference": ["A", "B", "A", "A", "B"]
    >>> })
    >>> crosstab(data, x_col="Gender", y_col="Preference")
    """
    engine = _resolve_engine(engine)
    spec = _crosstab_spec(df, x_col, y_col, values=values, aggfunc=aggfunc, method=method)

    if spec.is_empty:
        print(f"Number of dataframe rows for columns {x_col} and {y_col} is zero")

    if engine == "matplotlib":
        return _crosstab_mpl(
            spec,
            title=title,
            color_title=color_title,
            is_abs=is_abs,
            is_norm=is_norm,
            figsize=figsize,
            alpha=alpha,
            **kwargs,
        )

    return _crosstab_plotly(
        spec,
        title=title,
        color_title=color_title,
        is_abs=is_abs,
        is_norm=is_norm,
        figsize=figsize,
        alpha=alpha,
        **kwargs,
    )


def corr(
    df,
    col_x,
    col_y,
    ax=None,
    show_means=True,
    show_regression=True,
    figsize=(8, 6),
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
        The axes on which to draw the plot. If None, a new figure and axes are created.
        Ignored when ``engine="plotly"``.
    show_means : bool, optional, default=True
        Whether to show the mean lines for both x and y axes.
    show_regression : bool, optional, default=True
        Whether to display the regression line with the correlation coefficient.
    figsize : tuple of (float, float), optional, default=(8, 6)
        The size of the figure in inches. Ignored if `ax` is not None.
        With ``engine="plotly"`` it is converted to pixels at 100 dpi.
    engine : {"matplotlib", "plotly"} or None, optional, default=None
        The rendering engine. If None, the engine set by
        :func:`pltstat.set_backend` is used.
    **kwargs : dict, optional
        Additional keyword arguments to further customize the scatter plot.
        They are passed to `ax.scatter()` with matplotlib and to
        ``plotly.graph_objects.Scattergl`` with plotly, so they are engine
        specific.

    Returns
    -------
    fig : plotly.graph_objects.Figure or None
        The figure when ``engine="plotly"``. With matplotlib the function
        displays the plot and returns None.

    Notes
    -----
    The mean lines are labelled with the mathtext markup of matplotlib, which
    plotly does not render, so the plotly legend shows "mean(column)".

    Example
    --------
    >>> import pandas as pd
    >>> from pltstat.twofeats import corr
    >>> data = pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 4, 6, 8]})
    >>> corr(data, "A", "B")
    """

    engine = _resolve_engine(engine)
    spec = _corr_spec(df, col_x, col_y, show_regression=show_regression)

    if engine == "matplotlib":
        return _corr_mpl(
            spec,
            ax=ax,
            show_means=show_means,
            show_regression=show_regression,
            **kwargs,
        )

    _warn_ignored_mpl_params(engine, ax=ax)
    return _corr_plotly(
        spec,
        show_means=show_means,
        show_regression=show_regression,
        figsize=figsize,
        **kwargs,
    )


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
    **kwargs,
):
    """
    Plot a boxplot for a numeric feature grouped by a categorical feature with statistical testing.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    cat_feat : str
        The name of the categorical feature to group by.
    num_feat : str
        The name of the numeric feature to plot.
    size : {'tiny', 'compact', 'normal', 'huge'}, optional, default 'compact'
        The size of the figure. 'tiny' results in a small figure, 'compact' is the default,
        'normal' is a medium size, and 'huge' produces a larger figure.
    cat_order : list or array-like, optional, default None
        The desired order of categories in the plot. If None, categories are ordered by their appearance in the data.
    fig_return : bool, optional, default False
        If True, the function returns the figure object. If False, the figure is not returned.
    alpha : float, optional, default 0.05
        The significance level for the statistical test. Determines the threshold for p-value coloring.
    ax : matplotlib.axes.Axes, optional, default None
        The axes on which to plot the boxplot. If None, a new axes object is created.
    palette : str or list, optional, default 'pastel'
        The color palette to use for the plot. It can be a predefined palette name or a list of colors.
    **kwargs : additional keyword arguments, optional
        Additional arguments passed to `sns.boxplot()` for further customization of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        If `fig_return` is True, the function returns the matplotlib figure object. Otherwise, it returns None.

    Notes
    -----
    The function computes a statistical test based on the number of unique values in the categorical feature:
    - If there are exactly 2 categories, the Mann–Whitney U-test is applied.
    - If there are between 3 and 16 categories, the Kruskal-Wallis H-test is applied.
    The p-value from the test is displayed in the plot title. The color of the title will be green for a p-value
    less than `alpha` and red otherwise.
    Counts for each category are displayed on the plot, with categories having fewer than 30 data points highlighted in red.

    Raises
    ------
    ValueError
        If the `size` parameter is not one of 'tiny', 'compact', 'normal', or 'huge'.
        If the categorical feature has more than 20 unique values, which is not supported for the test.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.twofeats import boxplot
    >>>
    >>> # Example DataFrame
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    >>>     'category': np.random.choice(['A', 'B', 'C'], size=100),
    >>>     'value': np.random.randn(100)
    >>> })
    >>>
    >>> # Create the boxplot
    >>> boxplot(df, cat_feat='category', num_feat='value', size='normal', alpha=0.05)
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
            f"Value of size must be 'tiny', 'compact', 'normal', or 'huge' but given: {size}"
        )

    df = df[[cat_feat, num_feat]].dropna()
    if df.shape[0] == 0:
        print(f"Number of dataframe rows for columns {cat_feat} and {num_feat} is zero")
        return

    df[cat_feat] = df[cat_feat].astype("str")
    if cat_order is not None:
        cat_order = np.array(cat_order).astype("str")
        cat_order = cat_order[np.isin(cat_order, df[cat_feat].unique())]
    else:
        cat_order = df[cat_feat].astype("str").unique()
        cat_order = np.sort(cat_order)

    n_cat_feat = len(df[cat_feat].unique())
    n_cat_feat_tr1 = 2
    n_cat_feat_tr2 = 16

    if n_cat_feat == n_cat_feat_tr1:
        p = mannwhitneyu_by_cat(df, cat_feat=cat_feat, num_feat=num_feat)[1]
    elif (n_cat_feat > n_cat_feat_tr1) and (n_cat_feat <= n_cat_feat_tr2):
        p = kruskal_by_cat(df, cat_feat=cat_feat, num_feat=num_feat)[1]
    elif n_cat_feat > n_cat_feat_tr2:
        raise ValueError(
            f"Too many unique values of categorical feature '{cat_feat}': {n_cat_feat}"
        )
    else:
        raise ValueError(
            f"The categorical feature '{cat_feat}' has {n_cat_feat} unique value(s), which seems unusual for this function."
        )

    if ax is None:
        h_size = n_cat_feat / coef  # _np.ceil(n_cat_feat / coef)
        w_size = 18
        fig, ax = plt.subplots(1, 1, figsize=(w_size, h_size))

    fig = sns.boxplot(
        data=df,
        x=num_feat,
        y=cat_feat,
        hue=cat_feat,
        legend=False,
        orient="h",
        fliersize=1,
        showmeans=True,
        order=cat_order,
        hue_order=cat_order,
        ax=ax,
        palette=palette,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },  # "markersize": "10"
        **kwargs,
    )

    if cat_order is None:
        cat_order = fig.axes.get_yticklabels()
        cat_order = [plt_text.get_text() for plt_text in cat_order]
        dtype = df[cat_feat].dtype
        cat_order = np.array(cat_order).astype(dtype)

    # if n_cat_feat == 2:
    if p <= alpha:
        color = "g"
    else:
        color = "r"
    if n_cat_feat == 2:
        test_type = "Mann–Whitney U-test"
    else:
        test_type = "Kruskal-Wallis H-test"
    ax.set_title(f"{test_type} p-value={p:.3f}", color=color)

    counts = df.groupby(cat_feat, observed=False)[num_feat].count().loc[cat_order]
    xmin, xmax, ymin, ymax = ax.axis()
    x = (xmin + xmax) / 2
    for y, val in enumerate(counts):
        if val < 30:
            color = "r"
        else:
            color = "k"
        ax.text(x, y, "count=" + str(val), color=color, fontweight="bold")

    if fig_return:
        return fig


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
):
    """
    Plot a boxplot and a displot for a numeric feature (`num_feat`) of a DataFrame,
    grouped by a binary or nominal categorical feature (`cat_feat`).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be visualized.
    cat_feat : str
        The name of the categorical target feature (binary or nominal) by which
        the data will be grouped.
    num_feat : str
        The name of the numeric feature to plot.
    cat_order : list or array-like, optional, default None
        The desired order of categories for the `target` feature. If None, categories
        will be ordered by their appearance in the data.
   stat : {'count', 'probability', 'density', 'frequency'}, optional, default 'count'
    The statistic to plot in the displot. The available options are:
    - 'count': shows the number of occurrences.
    - 'probability': shows the relative frequencies of each bin.
    - 'density': shows the kernel density estimate.
    - 'frequency': shows the raw count in each bin.
    figsize : tuple of int, optional, default (20, 3.5)
        The size of the figure to be created, in inches (width, height).
    palette : str or list, optional, default 'pastel'
        The color palette to use for the plot. Can be a predefined palette name or a list of colors.
    alpha : float, optional, default 0.05
        The significance level for the statistical test. Determines the threshold for p-value coloring.
    ax_return : bool, optional, default False
        If True, the function will return the axes object(s) for further customization.

    Returns
    -------
    fig : matplotlib.axes.Axes or None
        If `ax_return` is True, the function returns the axes object(s). Otherwise, it returns None.

    Notes
    -----
    This function generates two plots:
    1. A boxplot (using the `boxplot` function) that shows the distribution of `col` grouped by `target`.
    2. A displot (using `sns.histplot`) that shows the distribution of `col` with a
       Kernel Density Estimate (KDE), separated by the levels of `target`.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.twofeats import dis_box_plot
    >>>
    >>> # Create a sample DataFrame with a binary target and a numeric column
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    >>>     'target': np.random.choice(['A', 'B'], size=100),
    >>>     'value': np.random.randn(100)
    >>> })
    >>>
    >>> # Call the dis_box_plot function
    >>> dis_box_plot(df, cat_feat='target', num_feat='value')
    """

    df_subset = df[[cat_feat, num_feat]].dropna()
    if df_subset.shape[0] == 0:
        print(f"Number of dataframe rows for columns {cat_feat} and {num_feat} is zero")
        return

    df_subset[cat_feat] = df_subset[cat_feat].astype("str")

    if cat_order is not None:
        cat_order = np.array(cat_order).astype("str")
        cat_order = cat_order[np.isin(cat_order, df_subset[cat_feat].unique())]
    else:
        cat_order = df_subset[cat_feat].astype("str").unique()
        cat_order = np.sort(cat_order)

    _, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 2]})
    fig = boxplot(
        df_subset,
        cat_feat,
        num_feat,
        fig_return=True,
        cat_order=cat_order,
        ax=ax[0],
        palette=palette,
        alpha=alpha,
    )

    title = fig.axes.get_title()
    ax[0].set_title(f"{num_feat}\n{title}")

    sns.histplot(
        data=df_subset,
        x=num_feat,
        hue=cat_feat,
        kde=True,
        hue_order=cat_order,
        stat=stat,
        common_norm=False,
        ax=ax[1],
        palette=palette,
    )
    if ax_return is True:
        return ax
