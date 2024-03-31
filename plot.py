from types import ModuleType

import os

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns

import umap.umap_ as umap

import numpy as np
import pandas as pd

# Fisher exact test from R:
# import rpy2.robjects.numpy2ri
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

from scipy import stats
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr

from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import matthews_corrcoef

numpy2ri.activate()
_stats_r = importr("stats")


def pie(df_column, ax=None):
    """
    Plot pie with good settings

    :param df_column: pd.Series
    :return: None
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
    )


def countplot(df_column, is_count_order=True, x_rotation=90):
    """
    Plot countplot with good settings

    :param df_column: pd.Series
    :return: None
    """
    plt.figure(figsize=(18, 6))
    if is_count_order:
        ax = sns.countplot(df_column, order=df_column.value_counts().index)
    else:
        ax = sns.countplot(df_column)
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


def distplot(df_column, is_limits=False, bins=None):  # , n_modes=0):
    """
    Plot distplot with good settings

    :param df_column: pd.Series
    :param is_limits: limits for kde plot
    :param bins: number of bins
    :return: None
    """

    if is_limits:
        min_val = df_column.min()
        max_val = df_column.max()
        ax = sns.histplot(
            df_column, kde=True, kde_kws={"clip": (min_val, max_val)}, bins=bins
        )
    else:
        ax = sns.histplot(df_column, kde=True, bins=bins)

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
):
    """
    Plot crosstab with good settings

    :param df: pd.DataFrame
    :param x_col, y_col: name of columns
    :param values: if None - value counts, or string - value the column
    :param aggfunc: function for aggregation
    :param is_abs: print absolute values
    :param is_norm: print normalized by indexes
    :param figsize: figsize of plt
    :param exact: use exact Fisher test (use it for small data), if 'auto', Fisher test
    is used when one of the contingency table cells is less than 5
    :param alpha: threshold for p_value
    :return: None
    """

    def plot_crosstab_abs(exact):
        df_subset = df[[x_col, y_col]].dropna()
        if df_subset.shape[0] == 0:
            print(
                "Number of dataframe rows for columns %s and %s is zero"
                % (x_col, y_col)
            )
            return
        crosstab_df = pd.crosstab(
            df_subset[x_col], df_subset[y_col], values=values, aggfunc=aggfunc
        )

        if exact == "auto":
            exact = crosstab_df.min(axis=None) < 5
            # Convert numpy.bool to bool:
            exact = bool(exact)
        if type(exact) is not bool:
            err_msg = f"param 'exact' must be False, True, or 'auto', but the value is {exact}"  # noqa: E501
            raise ValueError(err_msg)

        if exact is True:
            test_type = "Exact Fisher"
            # Python Scipy realization (only 2x2 table):
            # g, p_value = _fisher_exact(crosstab_df)
            # R language Stats realization (MxN table):
            p_value = _stats_r.fisher_test(crosstab_df.values)[0][0]
        else:
            test_type = "$chi^2$"
            g, p_value = chi2_contingency(crosstab_df)[:2]

        if p_value <= alpha:
            color = "g"
        else:
            color = "r"

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
            correlation = _cramer_v_by_obs(crosstab_df)

        if title is None:
            plt.title(
                "Crosstab. Absolute values\n%s p_value = %.3f; %s corr = %.3f"
                % (test_type, p_value, corr_type, correlation),
                color=color,
            )
        else:
            plt.title(title, color=color_title)
        sns.heatmap(crosstab_df, annot=True, fmt=".0f", linewidths=1, cmap="coolwarm")

        return exact

    def plot_crosstab_norm():
        df_subset = df[[x_col, y_col]].dropna()
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
        )

    if is_abs and is_norm:
        if figsize is None:
            figsize = (10, 6)
        plt.figure(figsize=figsize)
        plt.subplot(221)
        plot_crosstab_abs(exact)
        plt.subplot(222)
        plot_crosstab_norm()
    elif is_abs:
        if figsize is None:
            figsize = (5, 3)
        plt.figure(figsize=figsize)
        plot_crosstab_abs(exact)
    elif is_norm:
        if figsize is None:
            figsize = (5, 3)
        plt.figure(figsize=figsize)
        plot_crosstab_norm()
    else:
        raise ValueError("Not less that one of is_abs or is_norm must be True")


def corr(df, col_x, col_y, plot=plt, show_means=True, show_regression=True):
    """
    Plot correlation with good settings

    :param df: pd.DataFrame
    :param col_x: x_col, y_col: name of columns
    :param plot: ax or plt
    :return: None
    """
    data = df.copy()
    data = data[[col_x, col_y]].dropna()
    x = data[col_x].values
    y = data[col_y].values
    plot.scatter(x, y, c="green", s=2)
    if show_means:
        plot.plot(
            [x.mean()] * 2,
            [y.min(), y.max()],
            "--r",
            label=r"$\overline{%s}$" % col_x.replace("_", r"\_"),
        )
        plot.plot(
            [x.min(), x.max()],
            [y.mean()] * 2,
            "--b",
            label=r"$\overline{%s}$" % col_y.replace("_", r"\_"),
        )
    if show_regression:
        lr = LinearRegression().fit(x.reshape((-1, 1)), y)
        r, p_value = stats.pearsonr(x, y)
        x_th = np.array([x.min(), x.max()])
        y_th = lr.predict(x_th.reshape((-1, 1)))
        plot.plot(x_th, y_th, label=f"{lr.intercept_:.3f} + {lr.coef_[0]:.3f}x")
        if isinstance(plot, ModuleType):
            plot.xlabel(col_x)
            plot.ylabel(col_y)
            plot.title(f"r = {r:.3f}, p_value = {p_value:.3f}")
        else:
            plot.set_xlabel(col_x)
            plot.set_ylabel(col_y)
            plot.set_title(f"r = {r:.3f}, p_value = {p_value:.3f}")
    plot.legend()


def violin(
    df,
    cat_feat,
    num_feat,
    size="compact",
    fig_return=False,
    alpha=0.05,
    split2=False,
    inner="box",
):
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
):
    """
    Plot boxplot with good settings

    :param df: pd.DataFrame
    :param cat_feat: name of categorical feature
    :param num_feat: name of numeric feature
    :param size: size of figure - 'compact', 'normal', 'huge'
    :param cat_order: list/array of categories' order for plotting
    :param ax: ax of matplotlib, None is by default
    :return: None
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


def auto_naive_plot(  # noqa: C901
    df_column,
    is_ordinal=False,
    is_limits=None,
    is_show_average=True,
    is_count_order=True,
):
    """
    Plot info about column of df. Type of column depends on count of unique values.
    A naive analyse column, you need to check obtained information.
    The function can recognize only categorical and continuous features.
    :param df_column: series for analyzing
    :param is_limits: True/False/None. If None - auto choosing
    :param is_ordinal: True/False. Function can't recognise ordinal feature.
    :param is_show_average: True/False. If it's continuous feature -
       print (or not print) mean and median
    :param is_count_order: True/False. If it's True that it's order by count for
      countplotting
    :return: None
    """
    threshold_cont = 20  # If categories more than 20 and numbers - continuous
    threshold_big_cat = (
        5  # If categories more than 5 - too many categories, don't plot pie plot
    )

    n_nan = np.sum(df_column.isnull())
    n_unique = len(df_column.dropna().unique())
    top5 = df_column.head().to_string()
    if n_nan > 0:
        df_column = df_column.dropna()

    dtype = "Categorical"  # initial value

    try:
        # If everything is ok - it's continuous
        # check number of unique values:
        assert n_unique > threshold_cont
        # do and check that it's possible to change format type:
        df_column = df_column.astype("float64")
        dtype = "Continuous"
    except Exception:
        # Else: categorical
        # I can't use np.array_equal, because df_column can has float features:
        if n_unique == 2:  # and _np.allclose(_np.sort(df_column.unique()), [0, 1]):
            dtype += ":Boolean"
        elif is_ordinal:
            dtype += ":Ordinal"
        else:
            dtype += ":Nominal"

    if dtype == "Continuous":
        if np.allclose(df_column, df_column.astype("int64")):
            dtype += ":Integer"
        else:
            if (
                (df_column.max() <= 1)
                and (df_column.max() > 0)  # noqa: W503
                and (df_column.min() >= -1)  # noqa: W503
            ):
                dtype += ":Proportion"
            else:
                dtype += ":Float"

    if is_limits is None:
        if dtype.startswith("Continuous"):
            if (
                dtype.endswith("Proportion")
                or (df_column.min() == 0)  # noqa: W503
                or (df_column.min() == 1)  # noqa: W503
            ):  # prop or counter
                is_limits = True
            else:
                is_limits = False
    print("Name of feature: '%s'" % df_column.name)
    print("Feature -", dtype)
    # if dtype.startswith('Categorical'):
    print("Number of unique values:", n_unique)
    if n_nan > 0:
        print("Number of nan values:", n_nan)
    else:
        print("Nan values aren't found")
    print()
    print("First 5 values:")
    print(top5)
    print()
    if dtype.startswith("Continuous"):
        print(
            "Min / max values: %s / %s"
            % tuple(np.around([df_column.min(), df_column.max()], decimals=3))
        )
        if is_show_average:
            print(
                "Mean / median values: %s / %s"
                % tuple(np.around([df_column.mean(), df_column.median()], decimals=3))
            )
        print()

    try:
        if n_unique < 1:
            print("Error. Number of unique values:", n_unique)
        elif n_unique == 1:
            print("It contains only one value:", df_column[0])
        elif n_unique < threshold_big_cat:
            pie(df_column)
        elif n_unique < threshold_cont:
            countplot(df_column, is_count_order)
        else:
            distplot(df_column, is_limits=is_limits)
    except Exception:
        print("It isn't possible to understand format and print it")
    # print('un', n_unique)
    # print('# nan:', n_nan)


def nulls(
    df,
    figsize=(20, 10),
    index=None,
    n_ticks=None,
    print_str_index=False,
    print_all=True,
):
    """
    Plot graph of nulls (black color) for features of DataFrame

    :param df: pd.DataFrame
    :param figsize: tuple
    :param index: name of column, str for ylabel
    :param n_ticks: number of y ticks
    :param print_str_index: if index is string, print it instead of range numbers?
    :param print_all: print all string indexes or only n (n_ticks) of its?
    :return: None
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        df.isnull().apply(np.invert), yticklabels=False, cbar=False, vmin=0, vmax=1
    )
    plt.title("Null values (black)")
    if n_ticks is None:
        n_ticks = 11  # 10+1
    y_ticks = np.linspace(0, len(df) - 1, n_ticks).astype("int64")

    if index is None:
        index = "index"
        if df.index.dtype != "O":
            y_labels = np.percentile(df.index, np.linspace(0, 100, n_ticks)).astype(
                "int64"
            )
        else:
            if print_str_index:
                y_labels = df.index
                if print_all:
                    n_ticks = len(df)
                    y_ticks = np.arange(n_ticks)
                else:
                    y_labels = df.index[y_ticks]
            else:
                y_labels = y_ticks
    else:
        if df[index].dtype != "O":
            y_labels = np.percentile(df[index], np.linspace(0, 100, n_ticks)).astype(
                "int64"
            )
        else:
            if print_str_index:
                y_labels = df[index]
                if print_all:
                    n_ticks = len(df)
                    y_ticks = np.arange(n_ticks)
                else:
                    y_labels = df[index][y_ticks]
            else:
                y_labels = y_ticks

    plt.ylabel(index)
    plt.yticks(y_ticks, y_labels)


def dist_qq_plot(df, figsize):
    """
    Plot histogram and 4-plot for each feature of df

    :param df: pd.DataFrame
    :param figsize: tuple
    :return: None
    """
    n_cols_df = df.shape[1]

    if n_cols_df > 9:
        n_cols = 8
    elif n_cols_df > 4:
        n_cols = 6
    elif n_cols_df == 4:
        n_cols = 4
    elif n_cols_df == 3:
        n_cols = 6
    elif n_cols_df == 2:
        n_cols = 4
    elif n_cols_df == 1:
        n_cols = 2
    elif n_cols_df == 0:
        print("DF is empty, no columns")
        return
    else:
        print("I can't calculate number of columns")
        return
    n_rows = int(np.ceil(2 * n_cols_df / n_cols))

    #     if n_cols == 8:  ## TODO: calc width and height of figsize
    #         width = 20
    #     elif n_cols == 6:
    #         height = int((n_rows / n_cols) * .95 * width)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    i = 0
    shapiros = []
    if n_cols_df > 3:
        for col in df:
            sns.distplot(df[col], ax=axs[i // n_cols, i % n_cols])
            median = df[col].median()
            axs[i // n_cols, i % n_cols].set_title(col + "\nMedian=%.2f" % median)
            i += 1
            stats.probplot(
                df[col], dist="norm", plot=axs[i // n_cols, i % n_cols], rvalue=True
            )
            pval = stats.shapiro(df[col]).pvalue
            axs[i // n_cols, i % n_cols].set_title(col + "\nShapiro pval=%.2f" % pval)
            i += 1
            shapiros.append(pval)
    else:
        for col in df:
            sns.distplot(df[col], ax=axs[i])
            median = df[col].median()
            axs[i].set_title(col + "\nMedian=%.2f" % median)
            i += 1
            stats.probplot(df[col], dist="norm", plot=axs[i], rvalue=True)
            pval = stats.shapiro(df[col]).pvalue
            axs[i].set_title(col + "\nShapiro pval=%.2f" % pval)
            i += 1
            shapiros.append(pval)

    return np.array(shapiros)


def save_plt(fig, filepath, dpi=None):
    """
    Save a figure to a file if file is not exist. If file exists - resave it

    :param fig: figure of plt
    :param filepath: string
    :param dpi: integer
    :return: None
    """
    if os.path.isfile(filepath):
        os.remove(filepath)
    if dpi is not None:
        fig.savefig(filepath, dpi=dpi)
    else:
        fig.savefig(filepath)


def _cramer_v_by_obs(obs):
    chi2 = stats.chi2_contingency(obs, correction=False)[0]
    n = np.sum(obs).sum()
    min_dim = min(obs.shape) - 1
    corr_cramer_v = np.sqrt((chi2 / n) / min_dim)
    return corr_cramer_v


def _cramer_v(data1, data2):
    obs = pd.crosstab(data1, data2)
    corr_cramer_v = _cramer_v_by_obs(obs)
    return corr_cramer_v


def heatmap_corr(
    df,
    corr_type="pearson",
    figsize=(30, 20),
    fmt=".2f",
    index=None,
    cols=None,
    threshold=None,
    linecolor="white",
):
    """
    Create df.corr and print heatmap of it with tuned parameters

    :param df: pd.DataFrame
    :param type: str, method of df.corr(). Also 'cramer_v' is accepted
    :param figsize: tuple
    :param fmt: string
    :param index: list/array of features which is necessary to be in rows
    :param cols: list/array of features which is necessary to be in columns
    :return: None
    """
    plt.figure(figsize=figsize)

    if corr_type == "cramer_v":
        corr_type = _cramer_v

    corr = df.corr(corr_type)

    if (index is not None) or (cols is not None):
        if (index is not None) and (cols is not None):
            corr = corr.loc[index, cols]
        elif index is not None:
            corr = corr.loc[index, :]
        elif cols is not None:
            corr = corr.loc[:, cols]

    if (threshold is not None) and (corr_type == _cramer_v):
        interval = (1 - threshold) / 2
        cmap = [
            (0, "White"),
            (0 + threshold, "White"),
            # (0 + threshold + interval, 'IndianRed'),
            (1, "Red"),
        ]
        cmap = LinearSegmentedColormap.from_list("custom", cmap)
        vmin = 0
    elif threshold is not None:
        interval = (1 - threshold) / 2
        threshold /= 2
        interval /= 2

        cmap = [
            (0, "Blue"),
            # (0.5 - threshold - interval, 'LightBlue'),
            (0.5 - threshold, "White"),
            (0.5 + threshold, "White"),
            # (0.5 + threshold + interval, 'IndianRed'),
            (1, "Red"),
        ]
        cmap = LinearSegmentedColormap.from_list("custom", cmap)
        vmin = -1
    else:
        cmap = "coolwarm"
        vmin = -1

    sns.heatmap(
        corr,
        annot=True,
        fmt=fmt,
        vmin=vmin,
        vmax=1,
        linewidths=0.5,
        linecolor=linecolor,
        cmap=cmap,
    )


def time_hist(
    a, n_bins, bottom=0.1, title=None, figsize=None, ax=None, return_ax=False
):
    """
    Histogram for time data

    :param a: numpy array of hours
    :param n_bins: count of bins
    :param bottom: proportion
    :param title: title of histogram, is None - no title
    :param figsize: tuple
    :param ax: ax from matplotlib which is used for plottng. If None - function
      will create new ax
    :param return_ax: if True return ax, else return None
    :return: ax if return_ax is True, else - return None
    """

    a = a / 12 * np.pi  # h to rad

    def radian_function(x, y):
        rad_x = x / np.pi
        return f"{str(round(rad_x * 12, 1))} h"

    theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    bins = np.linspace(0, 2 * np.pi, n_bins + 1, endpoint=True)

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
        # ax = plt.subplot(111, polar=True)

    max_bin = np.histogram(a, bins=bins)[0].max()
    bottom = max_bin * bottom

    # get yticks:
    fig, ax2 = plt.subplots(subplot_kw={"projection": "polar"})
    ax2.hist(a, bins=bins, edgecolor="black")
    yticks = ax2.axes.yaxis.get_ticklocs()
    plt.close(fig)

    ax.hist(a, bins=bins, edgecolor="black", bottom=bottom)

    # arrange graph
    ax.set(
        theta_offset=np.pi / 2,
        theta_direction=-1,
        xticks=theta,
        yticks=yticks + bottom,
        yticklabels=yticks,
    )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(radian_function))
    ax.set_title(title)
    if return_ax:
        return ax

    # Bars and colors:
    # Use custom colors and opacity
    # bottom = 0
    # max_height = 10
    # radii = max_height*np.random.rand(N)
    # width = (2*np.pi) / N
    # bars = ax.bar(theta, radii, width=width) #, bottom=bottom)
    # for r, bar in zip(radii, bars):
    #     bar.set_facecolor(plt.cm.jet(r / 10.))
    #     bar.set_alpha(0.8)


def dis_box_plot(
    df,
    target,
    col,
    cat_order=None,
    stat="count",
    figsize=(20, 3.5),
    palette="tab10",
    ax_return=False,
):
    """Plot boxplot and displot by col of df divided by target (binary/nominal)
    feature"""

    df_subset = df[[col, target]].dropna()
    if df_subset.shape[0] == 0:
        print(f"Number of dataframe rows for columns {target} and {col} is zero")
        return

    df_subset[target] = df_subset[target].astype("str")

    if cat_order is not None:
        cat_order = np.array(cat_order).astype("str")
        cat_order = cat_order[np.isin(cat_order, df_subset[target].unique())]
    else:
        cat_order = df_subset[target].astype("str").unique()
        cat_order = np.sort(cat_order)

    _, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 2]})
    fig = boxplot(
        df_subset,
        target,
        col,
        fig_return=True,
        cat_order=cat_order,
        ax=ax[0],
        palette=palette,
    )

    title = fig.axes.get_title()
    ax[0].set_title(f"{col}\n{title}")

    sns.histplot(
        data=df_subset,
        x=col,
        hue=target,
        kde=True,
        hue_order=cat_order,
        stat=stat,
        common_norm=False,
        ax=ax[1],
        palette=palette,
    )
    if ax_return is True:
        return ax


def embeddings_creation(X, random_state=0):
    """
    Create 2D representation of data.
    Return umap and tsne data.
    """
    reducer = umap.UMAP(random_state=random_state)
    X_umap = reducer.fit_transform(X)

    reducer = TSNE(random_state=random_state)
    X_tsne = reducer.fit_transform(X)

    print("Shape of umap is", X_umap.shape, "; Shape of tsne is", X_tsne.shape)
    return X_umap, X_tsne


def plot_umap_tsne(
    X_umap, X_tsne, labels=None, clust_name="None", unnoisy_idx=None, figsize=(16, 6)
):
    """
    Plotting clusterization on umap and tsne representation without noisy points
    (if data has noisy points).

    :param X_umap: numpy.array
        UMAP 2D data
    :param X_tsne: numpy.array
        t-SNE 2D data
    :param labels: numpy.array
        array of points labels
    :param clust_name: str
        Name of clusterization method
    :param unnoisy_idx: numpy.array
        Indexes of unnoisy points
    :return: None
    """
    if isinstance(labels, str):
        labels = labels.astype("str")  # TODO str for categorical palette
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    ax[0].set_title("%s clusterization and UMAP projection" % clust_name)
    ax[1].set_title("%s clusterization and TSNE projection" % clust_name)

    if unnoisy_idx is None:
        sns.scatterplot(
            x=X_umap[:, 0], y=X_umap[:, 1], hue=labels, legend="full", ax=ax[0]
        )
        sns.scatterplot(
            x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, legend="full", ax=ax[1]
        )
    else:
        sns.scatterplot(
            x=X_umap[unnoisy_idx, 0],
            y=X_umap[unnoisy_idx, 1],
            hue=labels[unnoisy_idx],
            legend="full",
            ax=ax[0],
        )
        sns.scatterplot(
            x=X_tsne[unnoisy_idx, 0],
            y=X_tsne[unnoisy_idx, 1],
            hue=labels[unnoisy_idx],
            legend="full",
            ax=ax[1],
        )


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


def r_pval(
    df,
    dm_corr_cols,
    stat_cols,
    figsize,
    cmap_thr=0.8,
    is_T=False,
    annot=True,
    show_pvals=True,
    annot_rot=0,
):
    """
    Create Heatmaps with correlations and p-values by Spearman's statistic
    :param df: pd.DataFrame for analysis
    :param dm_corr_cols: Demographic numeric columns
    :param stat_cols: Statistical numeric columns
    :param figsize: tuple, figure size
    :param cmap_thr: cmap threshold of correlations significancy
    :param is_T: pivoting table, bool
    :param annot: add annotation to cells, bool
    :param show_pvals: plot heatmap with p-values, bool
    :param annot_rot: rotation angle of annotation, int
    :return: pandas.DataFrames with p-values and with correlation coefficients
    """
    df_pvals = pd.DataFrame(index=dm_corr_cols, columns=stat_cols, dtype="float64")
    df_corrs = pd.DataFrame(index=dm_corr_cols, columns=stat_cols, dtype="float64")
    for dm_corr_col in dm_corr_cols:
        for stat_col in stat_cols:
            df_subset = df[[dm_corr_col, stat_col]].dropna()
            x = df_subset.loc[:, dm_corr_col]
            y = df_subset.loc[:, stat_col]
            corr, p = spearmanr(x, y)
            df_pvals.loc[dm_corr_col, stat_col] = p
            df_corrs.loc[dm_corr_col, stat_col] = corr

    if show_pvals:
        nrows = 2
    else:
        nrows = 1

    fig, ax = plt.subplots(
        nrows=nrows, ncols=1, figsize=figsize, constrained_layout=True
    )
    if not show_pvals:
        ax = [ax]
    if is_T:
        df_pvals = df_pvals.T

    cmap_corrs = get_corr_thr_cmap(threshold=cmap_thr)
    sns.heatmap(
        df_corrs,
        vmin=-1,
        vmax=1,
        cmap=cmap_corrs,
        annot=annot,
        fmt=".1f",
        linewidths=1,
        ax=ax[0],
        annot_kws={"rotation": annot_rot},
    )
    ax[0].set_title("Spearman correlations, correlation coefficient")
    xticks = df_corrs.columns
    ax[0].set_xticks(np.arange(len(xticks)) + 0.5, xticks)
    if show_pvals:
        cmap, cbar_kws = get_pval_legend_thr_cmap()
        sns.heatmap(
            df_pvals,
            vmin=0,
            vmax=1,
            cmap=cmap,
            annot=annot,
            fmt=".2f",
            linewidths=1,
            cbar_kws=cbar_kws,
            ax=ax[1],
        )
        ax[1].set_title("Spearman correlations, p-value")
        xticks = df_pvals.columns
        ax[1].set_xticks(np.arange(len(xticks)) + 0.5, xticks)

    return df_corrs, df_pvals


def mw_pval(
    df,
    cat_cols,
    num_cols,
    figsize,
    is_T=False,
    annot=True,
    annot_rot=0,
    annot_size=None,
):
    """
    Create Heatmaps with Mann-Whitney p-values between numerical data divided by
    categorical columns into two datasets
    :param df: pd.DataFrame for analysis
    :param cat_cols: Categorical columns with two unique values
    :param num_cols: Numerical columns
    :param figsize: tuple, figure size
    :param is_T: pivoting table, bool
    :param annot: add annotation to cells, bool
    :param annot_rot: rotation angle of annotation, int
    :param annot_size: fontsize of annotation, int
    :return: pandas.DataFrames with Mann-Whitney p-values
    """
    df_pvals = pd.DataFrame(index=cat_cols, columns=num_cols, dtype="float64")
    for cat_col in cat_cols:
        for num_col in num_cols:
            df_subset = df[[cat_col, num_col]].dropna()
            if df_subset.loc[:, cat_col].nunique() < 2:
                # Less than 2 different values of categorical feature in the subset
                p = np.nan
            else:
                x = df_subset.loc[
                    df_subset.loc[:, cat_col] == df_subset.loc[:, cat_col].unique()[0],
                    num_col,
                ]
                y = df_subset.loc[
                    df_subset.loc[:, cat_col] == df_subset.loc[:, cat_col].unique()[1],
                    num_col,
                ]
                p = mannwhitneyu(x, y)[1]
            df_pvals.loc[cat_col, num_col] = p

    plt.figure(figsize=figsize)
    if is_T:
        df_pvals = df_pvals.T
    cmap, cbar_kws = get_pval_legend_thr_cmap()
    sns.heatmap(
        df_pvals,
        vmin=0,
        vmax=1,
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        linewidths=1,
        cbar_kws=cbar_kws,
        annot_kws={"rotation": annot_rot, "fontsize": annot_size},
    )
    xticks = df_pvals.columns

    plt.xticks(np.arange(len(xticks)) + 0.5, xticks)
    plt.title("Mann-Whitney p-values")
    return df_pvals


def phik_corrs(
    df,
    x=None,
    y=None,
    threshold=0.8,
    annot=True,
    fmt=".2f",
    figsize=None,
    annot_rot=0,
    annot_size=None,
    **kwargs,
):
    """P,ot Heatmap with Phik correlations between specific x and y lists of columns"""
    cmap = get_corr_thr_cmap(threshold=threshold, vmin=0)
    if (x is not None) and (y is not None):
        xy = np.concatenate((x, y))
        xy = np.unique(xy)
        df_phik = df[xy].phik_matrix()
        df_phik = df_phik.loc[y, x]
    else:
        df_phik = df.phik_matrix()
    plt.figure(figsize=figsize)
    sns.heatmap(
        df_phik,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=annot,
        fmt=fmt,
        annot_kws={"rotation": annot_rot, "fontsize": annot_size},
        **kwargs,
    )
