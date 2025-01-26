import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns

import umap.umap_ as umap

import numpy as np
import pandas as pd


from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr

from sklearn.manifold import TSNE

from cm import get_corr_thr_cmap, get_pval_legend_thr_cmap

from corr_methods import cramer_v

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
        corr_type = cramer_v

    corr = df.corr(corr_type)

    if (index is not None) or (cols is not None):
        if (index is not None) and (cols is not None):
            corr = corr.loc[index, cols]
        elif index is not None:
            corr = corr.loc[index, :]
        elif cols is not None:
            corr = corr.loc[:, cols]

    if (threshold is not None) and (corr_type == cramer_v):
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
