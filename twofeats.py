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
            err_msg = f"param 'exact' must be False, True, or 'auto', but the value is {exact}"
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
            correlation = cramer_v_by_obs(crosstab_df)

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
