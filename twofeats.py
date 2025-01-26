from types import ModuleType

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

from sklearn.linear_model import LinearRegression
from sklearn.metrics import matthews_corrcoef

from corr_methods import cramer_v_by_obs

# Fisher exact test from R:
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

numpy2ri.activate()
_stats_r = importr("stats")


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
    exact="auto",
    alpha=0.05,
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
    exact : {'auto', True, False}, optional, default='auto'
        Whether to use Fisher's exact test. If 'auto', Fisher's test is used when any cell count is less than 5.
    alpha : float, optional, default=0.05
        The threshold for statistical significance (p-value).
    **kwargs : dict
        Additional keyword arguments passed to `sns.heatmap`.

    Returns
    -------
    None
        The function creates and displays the plots.

    Notes
    -----
    - If both `is_abs` and `is_norm` are True, two plots are displayed: absolute values and normalized values.
    - The function can automatically detect and apply the appropriate statistical test (Chi-square or Fisher's exact test).
    - For binary 2x2 tables, Matthews correlation is calculated; otherwise, Cramér's V is used.

    Examples
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
    df_subset = df[[x_col, y_col]].dropna()

    def plot_crosstab_abs(exact):
        """Plot Heatmap with absolute values crosstab and statistics"""
        if df_subset.shape[0] == 0:
            print(f"Number of dataframe rows for columns {x_col} and {y_col} is zero")
            return
        crosstab_df = pd.crosstab(df_subset[x_col], df_subset[y_col], values=values, aggfunc=aggfunc)

        if exact == "auto":
            exact = crosstab_df.min(axis=None) < 5
            # Convert numpy.bool to bool:
            exact = bool(exact)
        if type(exact) is not bool:
            err_msg = f"param 'exact' must be False, True, or 'auto', but the value is {exact}"
            raise ValueError(err_msg)

        if exact is True:
            test_type = "Exact Fisher"
            # Python Scipy realization (only 2x2 table):
            # g, p_value = _fisher_exact(crosstab_df)
            # R language Stats realization (MxN table):
            p_value = _stats_r.fisher_test(crosstab_df.values, conf_int=True, conf_level=alpha)[0][0]
        else:
            test_type = "$chi^2$"
            p_value = chi2_contingency(crosstab_df)[1]

        color = "g" if p_value <= alpha else "r"

        if crosstab_df.shape == (2, 2):
            corr_type = "Matthews"
            x_col_un = df_subset[x_col].unique()
            y_col_un = df_subset[y_col].unique()
            correlation = matthews_corrcoef(
                df_subset[x_col].map({x_col_un[0]: 0, x_col_un[1]: 1}),
                df_subset[y_col].map({y_col_un[0]: 0, y_col_un[1]: 1}),
            )
        else:
            corr_type = "Cramer V"
            correlation = cramer_v_by_obs(crosstab_df)

        if title is None:
            plt.title(
                "Crosstab. Absolute values\n%s p_value = %.3f; %s corr = %.3f"
                % (test_type, p_value, corr_type, correlation),
                color=color,
            )
        else:
            plt.title(title, color=color_title)
        sns.heatmap(crosstab_df, annot=True, fmt=".0f", linewidths=1, cmap="coolwarm", **kwargs)

        return exact

    def plot_crosstab_norm():
        """Plot Heatmap with normalized by row indices crosstab"""
        crosstab_df = pd.crosstab(
            df_subset[x_col],
            df_subset[y_col],
            normalize="index",
            values=values,
            aggfunc=aggfunc,
        )

        plt.title("Crosstab. Normalized by index")
        sns.heatmap(
            crosstab_df,
            annot=True,
            fmt=".2f",
            vmin=0,
            vmax=1,
            linewidths=1,
            cmap="coolwarm",
            **kwargs,
        )

    # Plot Heatmaps and calculate statistics:
    if is_abs and is_norm:
        figsize = figsize or (10, 6)
        plt.figure(figsize=figsize)
        plt.subplot(221)
        plot_crosstab_abs(exact)
        plt.subplot(222)
        plot_crosstab_norm()
    else:
        figsize = figsize or (5, 3)
        plt.figure(figsize=figsize)
        if is_abs is None:
            plot_crosstab_abs(exact)
        elif is_norm is None:
            plot_crosstab_norm()
        else:
            raise ValueError("Not less that one of is_abs or is_norm must be True")



# def corr(df, col_x, col_y, ax=None, show_means=True, show_regression=True, point_size=2, c="green"):
def corr(
    df,
    col_x,
    col_y,
    ax=None,
    show_means=True,
    show_regression=True,
    # point_color="green",
    # line_color="blue",
    # mean_colors=("red", "blue"),
    # point_size=5,
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
    show_means : bool, optional, default=True
        Whether to show the mean lines for both x and y axes.
    show_regression : bool, optional, default=True
        Whether to display the regression line with the correlation coefficient.
    **kwargs : dict, optional
        Additional keyword arguments passed to `ax.scatter()`

    Returns
    -------
    None
        The function displays the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> from pltstat.twofeats import corr
    >>> data = pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 4, 6, 8]})
    >>> corr(data, "A", "B")
    """
    # point_color : str, optional, default="green"
    #         The color of the scatter plot points.
    #     line_color : str, optional, default="blue"
    #         The color of the regression line.
    #     mean_colors : tuple of str, optional, default=("red", "blue")
    #         The colors for the x and y mean lines, respectively.
    #     point_size : int, optional, default=5
    #         The size of the points in the scatter plot.

    # Prepare data
    data = df[[col_x, col_y]].dropna()
    x = data[col_x].values
    y = data[col_y].values

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(x, y, c="green", s=2, label="Data Points", **kwargs)

    # Show mean lines
    if show_means:
        # ax.axvline(x.mean(), color=mean_colors[0], linestyle="--", label=f"Mean {col_x}")
        # ax.axhline(y.mean(), color=mean_colors[1], linestyle="--", label=f"Mean {col_y}")
        ax.plot(
            [x.mean()] * 2,
            [y.min(), y.max()],
            "--r",
            label=r"$\overline{%s}$" % col_x.replace("_", r"\_"),
        )
        ax.plot(
            [x.min(), x.max()],
            [y.mean()] * 2,
            "--b",
            label=r"$\overline{%s}$" % col_y.replace("_", r"\_"),
        )

    # Show regression line and correlation
    if show_regression:
        lr = LinearRegression().fit(x.reshape((-1, 1)), y)
        r, p_value = stats.pearsonr(x, y)
        x_th = np.array([x.min(), x.max()])
        y_th = lr.predict(x_th.reshape((-1, 1)))
        ax.plot(x_th, y_th, label=f"{lr.intercept_:.3f} + {lr.coef_[0]:.3f}x")
        ax.set_title(f"r = {r:.3f}, p_value = {p_value:.3f}")

    # Axis labels
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)

    # Legend
    ax.legend()


def violin(
    df,
    cat_feat,
    num_feat,
    size="compact",
    fig_return=False,
    alpha=0.05,
    split2=False,
    inner="box",
    **kwargs,
):
    """
    Plot a violin plot with enhanced customization and statistical analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data for the plot.
    cat_feat : str
        The name of the categorical feature to group the data by.
    num_feat : str
        The name of the numeric feature to plot as the distribution in the violin plot.
    size : str, optional, default='compact'
        The size of the figure. Options are:
        - 'compact': Small-sized plot.
        - 'normal': Medium-sized plot.
        - 'huge': Large-sized plot.
    fig_return : bool, optional, default=False
        If True, returns the matplotlib figure object.
    alpha : float, optional, default=0.05
        The significance threshold for the Mann–Whitney U test (applies when the categorical feature has two unique values).
    split2 : bool, optional, default=False
        If the categorical feature has exactly two unique values:
        - True: Splits the violins to overlay the distributions.
        - False: Plots two separate violins.
    inner : str, optional, default='box'
        Determines what is displayed inside the violin:
        - 'box': A small box plot inside the violin.
        - 'quartile': Quartiles displayed inside the violin.
        - 'point': Individual data points inside the violin.
        - 'stick': Vertical lines within the violin.
    **kwargs : dict
        Additional keyword arguments passed to `sns.violinplot` for further customization.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object if `fig_return=True`, otherwise None.

    Notes
    -----
    - If the number of unique categories in `cat_feat` is greater than 16, the function will print a warning and exit.
    - For two-category comparisons, the Mann–Whitney U test is performed, and the p-value is displayed in the plot title.
    - Categories with fewer than 30 observations are annotated in red on the plot.

    Examples
    --------
    Example 1: Basic violin plot for a categorical and numeric feature.

    >>> import seaborn as sns
    >>> from matplotlib import pyplot as plt
    >>> from pltstat.twofeats import violin
    >>> tips = sns.load_dataset("tips")
    >>> violin(tips, cat_feat="day", num_feat="total_bill")
    """


    """
    Plot violinplot with good settings

    :param df: pd.DataFrame
    :param cat_feat: name of categorical feature
    :param num_feat: name of numeric feature
    :param size: size of figure - 'compact', 'normal', 'huge'
    :param fig_return: if True - return figure
    :param alpha: Threshold for p_value
    :param split2: If cat_feat has 2 unique value - how to print it?
      True - split; False - plot two violins
    :param inner: param inner of the sns.violinplot(). What to print inside a violin?
    :return: figure if fig_return is True, else - None
    """
    if size == "compact":
        coef = 1
    elif size == "normal":
        coef = 1.5
    elif size == "huge":
        coef = 2
    else:
        raise ValueError(
            f"Value of size must be 'compact', 'normal' or 'huge' but given: {size}"
        )

    df = df[[cat_feat, num_feat]].dropna()
    n_cat_feat = df[cat_feat].nunique()

    if n_cat_feat == 0:
        print("Number of unique categorical features is 0")
        return
    elif n_cat_feat > 16:
        print("There a lot of unique categorical features (%d)" % n_cat_feat)
        return

    w_size = 18
    if (n_cat_feat == 2) and (split2 is True):
        h_size = coef
    else:
        h_size = coef * n_cat_feat
    figsize = (w_size, h_size)
    fig = plt.figure(figsize=figsize)

    if n_cat_feat == 1:
        sns.violinplot(
            x=df[num_feat], palette="muted", orient="h", linewidth=1, inner=inner
        )
    elif n_cat_feat == 2:
        x = df[df[cat_feat] == df[cat_feat].unique()[0]][num_feat]
        y = df[df[cat_feat] == df[cat_feat].unique()[1]][num_feat]
        _, p = mannwhitneyu(x, y)
        if p <= alpha:
            color = "g"
        else:
            color = "r"
        plt.title("Mann–Whitney U-test p-value=%.3f" % p, color=color)
        if split2 is True:
            y = df.shape[0] * [""]
            sns.violinplot(
                x=df[num_feat],
                y=y,
                hue=df[cat_feat],
                palette="muted",
                split=True,
                orient="h",
                linewidth=1,
                inner=inner,
            )
        else:
            sns.violinplot(
                x=df[num_feat],
                y=df[cat_feat],
                palette="muted",
                split=True,
                orient="h",
                linewidth=1,
                inner=inner,
            )
    else:  # n_cat_feat > 2
        sns.violinplot(
            x=df[num_feat],
            y=df[cat_feat],
            palette="muted",
            split=True,
            orient="h",
            linewidth=1,
            inner=inner,
            **kwargs,
        )

    counts = df.groupby(cat_feat)[num_feat].count()
    xmin, xmax, ymin, ymax = plt.axis()
    x = (xmin + xmax) / 2

    def _val_color(val):
        return "r" if val < 30 else "k"

    if (n_cat_feat == 2) and (split2 is True):
        for y, val in zip([-0.1, 0.2], counts):
            plt.text(
                x, y, "count=" + str(val), color=_val_color(val), fontweight="bold"
            )
    else:
        for y, val in enumerate(counts):
            plt.text(
                x,
                y - 0.1,
                "count=" + str(val),
                color=_val_color(val),
                fontweight="bold",
            )

    if fig_return is True:
        return fig


def boxplot(
    df,
    cat_feat,
    num_feat,
    size="compact",
    cat_order=None,
    fig_return=False,
    alpha=0.05,
    ax=None,
    palette="tab10",
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
    palette : str or list, optional, default 'tab10'
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
    if size == "tiny":
        coef = 2.5
    elif size == "compact":
        coef = 2
    elif size == "normal":
        coef = 1.5
    elif size == "huge":
        coef = 1
    else:
        raise ValueError(
            f"Value of size must be 'compact', 'normal' or 'huge' but given: {size}"
        )

    df = df[[cat_feat, num_feat]].dropna()
    if df.shape[0] == 0:
        print(
            "Number of dataframe rows for columns %s and %s is zero"
            % (cat_feat, num_feat)
        )
        return

    df[cat_feat] = df[cat_feat].astype("str")
    if cat_order is not None:
        cat_order = np.array(cat_order).astype("str")
        cat_order = cat_order[np.isin(cat_order, df[cat_feat].unique())]

    n_cat_feat = len(df[cat_feat].unique())
    if n_cat_feat == 2:
        x = df[df[cat_feat] == df[cat_feat].unique()[0]][num_feat]
        y = df[df[cat_feat] == df[cat_feat].unique()[1]][num_feat]
        p = mannwhitneyu(x, y)[1]
    elif (n_cat_feat > 2) and (n_cat_feat <= 16):
        x = df.groupby(cat_feat)[num_feat].agg(list).to_numpy()
        p = kruskal(*x)[1]
    elif n_cat_feat > 20:
        raise ValueError(
            f"Too many unique values of categorical feature '{cat_feat}': {n_cat_feat}"
        )
    else:
        raise ValueError(
            f"Strange number of categorical feature '{cat_feat}': {n_cat_feat}"
        )

    if ax is None:
        h_size = n_cat_feat / coef  # _np.ceil(n_cat_feat / coef)
        w_size = 18
        fig, ax = plt.subplots(1, 1, figsize=(w_size, h_size))

    fig = sns.boxplot(
        data=df,
        x=num_feat,
        y=cat_feat,
        orient="h",
        fliersize=1,
        showmeans=True,
        order=cat_order,
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

    counts = df.groupby(cat_feat)[num_feat].count().loc[cat_order]
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
    palette="tab10",
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
    palette : str or list, optional, default 'tab10'
        The color palette to use for the plot. Can be a predefined palette name or a list of colors.
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
