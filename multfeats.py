import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns

import umap.umap_ as umap

import numpy as np
import pandas as pd


from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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
    Plot a heatmap to visualize null values in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    figsize : tuple, optional, default=(20, 10)
        The size of the figure (width, height) in inches.
    index : str, optional, default=None
        The name of the column to use as the y-axis label. If None, the index is used.
    n_ticks : int, optional, default=None
        The number of y-axis ticks to display. If None, 10+1 ticks will be displayed.
    print_str_index : bool, optional, default=False
        If True and the index is a string type, print the index values as labels.
    print_all : bool, optional, default=True
        If True, display all index values; if False, display only the `n_ticks` specified.

    Returns
    -------
    None
        The function creates a heatmap plot of null values and does not return any value.

    Notes
    -----
    - The plot shows the null values in the DataFrame as black color.
    - The function dynamically adjusts the y-axis labels depending on the input DataFrame index or the specified `index` column.

    Example
    --------
    Basic usage with default settings.

    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.multfeats import nulls
    >>>
    >>> df = pd.DataFrame({
    >>>     'A': [1, 2, np.nan, 4],
    >>>     'B': [np.nan, 2, 3, 4],
    >>>     'C': [1, np.nan, np.nan, 4]
    >>> })
    >>>
    >>> nulls(df)
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


def dist_qq_plot(df, figsize, **kwargs):
    """
    Plot histograms and Q-Q plots for each feature of the DataFrame, along with the Shapiro-Wilk test p-values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the features to be plotted. Each feature will have its own histogram and Q-Q plot.
    figsize : tuple
        The size of the figure (width, height) in inches.
    **kwargs : keyword arguments
        Additional arguments passed to the `sns.histplot()` function for customizing the histogram plots.

    Returns
    -------
    shapiros : np.ndarray
        An array containing the Shapiro-Wilk test p-values for each feature in the DataFrame.

    Notes
    -----
    - The function dynamically determines the number of rows and columns needed to plot the histograms and Q-Q plots
      based on the number of features in the DataFrame.
    - Each feature is plotted with its histogram and a Q-Q plot to assess its distribution.
    - The median of each feature is displayed in the plot title.
    - The Shapiro-Wilk p-value is shown in the plot title to indicate if the data is normally distributed.
    - The number of rows and columns in the plot grid is adjusted based on the number of features in the DataFrame.

    Example
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.multfeats import dist_qq_plot
    >>> np.random.seed(42)
    >>>
    >>> df = pd.DataFrame({
    >>>     'A': np.random.normal(0, 1, 100),
    >>>     'B': np.random.normal(5, 2, 100)
    >>> })
    >>> dist_qq_plot(df, figsize=(12, 8), kde=True)
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
            sns.histplot(df[col], ax=axs[i // n_cols, i % n_cols], **kwargs)
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
            sns.histplot(df[col], ax=axs[i], **kwargs)
            median = df[col].median()
            axs[i].set_title(col + "\nMedian=%.2f" % median)
            i += 1
            stats.probplot(df[col], dist="norm", plot=axs[i], rvalue=True)
            pval = stats.shapiro(df[col]).pvalue
            axs[i].set_title(col + "\nShapiro pval=%.2f" % pval)
            i += 1
            shapiros.append(pval)

    shapiros = np.array(shapiros)

    return shapiros


def embeddings_creation(X, n_components=2, standardize=True, random_state=0, umap_kwargs=None, tsne_kwargs=None):
    """
    Create 2D representation of data using UMAP and t-SNE.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data for dimensionality reduction.
    n_components : int, optional, default=2
        Number of dimensions for the reduced representation.
    standardize : bool, optional, default=True
        Whether to standardize the input data before applying dimensionality reduction.
    random_state : int, optional, default=0
        The seed used by the random number generator. Used to ensure reproducibility of the results.
    umap_kwargs : dict, optional
        Additional keyword arguments to pass to UMAP.
    tsne_kwargs : dict, optional
        Additional keyword arguments to pass to t-SNE.

    Returns
    -------
    X_umap : array, shape (n_samples, n_components)
        The UMAP embeddings of the input data.
    X_tsne : array, shape (n_samples, n_components)
        The t-SNE embeddings of the input data.

    Notes
    -----
    The function uses both UMAP and t-SNE algorithms to reduce the input data `X` to 2D representations.
    The `random_state` ensures that the results are reproducible across different runs of the function.

    Example
    --------
    >>> from sklearn.datasets import load_iris
    >>> import pandas as pd
    >>> from pltstat.multfeats import embeddings_creation
    >>>
    >>> iris = load_iris()
    >>> X = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> X_umap, X_tsne = embeddings_creation(X, standardize=True, random_state=42)
    >>> X_umap.shape
    (150, 2)
    >>> X_tsne.shape
    (150, 2)
    """

    if standardize is True:
        X = StandardScaler().fit_transform(X)

    umap_kwargs = umap_kwargs or {}
    tsne_kwargs = tsne_kwargs or {}

    reducer = umap.UMAP(n_components=n_components, random_state=random_state, **umap_kwargs)
    X_umap = reducer.fit_transform(X)

    reducer = TSNE(n_components=n_components, random_state=random_state, **tsne_kwargs)
    X_tsne = reducer.fit_transform(X)

    print("Shape of umap is", X_umap.shape, "; Shape of tsne is", X_tsne.shape)
    return X_umap, X_tsne


def plot_umap_tsne(X_umap, X_tsne, labels=None, title_pref="", unnoisy_idx=None, figsize=(16, 6)):
    """
    Plot UMAP and t-SNE projections of data with cluster labels, optionally excluding noisy points.

    Parameters
    ----------
    X_umap : array, shape (n_samples, 2)
        2D UMAP embeddings of the data.
    X_tsne : array, shape (n_samples, 2)
        2D t-SNE embeddings of the data.
    labels : array, shape (n_samples,), optional
        Cluster labels for each sample. If not provided, no coloring will be applied.
    title_pref : str, optional, default=""
        A preferred title prefix.
            If not provided (empty string), the titles will be "UMAP projection" and "TSNE projection".
            If specified, the prefix will be used at the beginning of each title, followed by the projection name.
    unnoisy_idx : array, shape (n_samples,), optional
        Indices of non-noisy points. If provided, only those points will be plotted.
    figsize : tuple, optional, default=(16, 6)
        The size of the figure.

    Returns
    -------
    None
        The function creates a plot in place and does not return any value.

    Notes
    -----
    The function generates two side-by-side scatter plots showing the results of
    dimensionality reduction using UMAP and t-SNE, with the points colored according to
    the given cluster labels. If `unnoisy_idx` is provided, only the non-noisy points
    will be plotted.

    Example
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from pltstat.multfeats import plot_umap_tsne
    >>>
    >>> iris = load_iris()
    >>> X = iris.data
    >>> labels = np.array([0, 1, 2] * 50)  # Example labels for 3 clusters
    >>> plot_umap_tsne(X_umap, X_tsne, labels=labels, clust_name="KMeans")
    """
    if isinstance(labels, str):
        labels = labels.astype("str")  # TODO str for categorical palette
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    if title_pref is not "":
        title_pref += " and "
    ax[0].set_title(f"{title_pref}UMAP projection")
    ax[1].set_title(f"{title_pref}TSNE projection")

    palette = "tab10"
    legend = "full"

    if unnoisy_idx is None:
        sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels, legend=legend, palette=palette, ax=ax[0])
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, legend=legend, palette=palette, ax=ax[1])
    else:
        sns.scatterplot(
            x=X_umap[unnoisy_idx, 0],
            y=X_umap[unnoisy_idx, 1],
            hue=labels[unnoisy_idx],
            legend=legend,
            palette=palette,
            ax=ax[0],
        )
        sns.scatterplot(
            x=X_tsne[unnoisy_idx, 0],
            y=X_tsne[unnoisy_idx, 1],
            hue=labels[unnoisy_idx],
            legend=legend,
            palette=palette,
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
    **kwargs,
):
    """
    Compute correlation matrix and visualize it using a heatmap.

    Parameters
    ----------
    df : DataFrame
       Input DataFrame containing the data to compute the correlation matrix.
    corr_type : str, optional, default="pearson"
       The method to compute correlation. Supported options include "pearson",
       "kendall", "spearman", and "cramer_v" for categorical data.
    figsize : tuple, optional, default=(30, 20)
       The size of the figure (width, height) in inches.
    fmt : str, optional, default=".2f"
       The format string for the annotation of correlation values in the heatmap.
    index : list or array, optional
       List or array of features to be used as rows in the correlation matrix. If None, all features will be used.
    cols : list or array, optional
       List or array of features to be used as columns in the correlation matrix. If None, all features will be used.
    threshold : float, optional
       Threshold value for customizing the heatmap colors. If specified, only correlations
       above this threshold will be displayed.
    linecolor : str, optional, default="white"
       The color of the lines separating the cells in the heatmap.
    **kwargs : keyword arguments
       Additional parameters passed to `sns.heatmap`.

    Returns
    -------
    None
       The function creates a heatmap in place and does not return any value.

    Notes
    -----
    This function computes the correlation matrix of the DataFrame using the specified
    method (`corr_type`). The resulting matrix is visualized as a heatmap, where the
    color intensity represents the strength of the correlation between the variables.
    If `threshold` is provided, correlations below the threshold are ignored.

    Example
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.multfeats import heatmap_corr
    >>>
    >>> np.random.seed(42)
    >>> df = pd.DataFrame(np.random.rand(10, 5), columns=list("ABCDE"))
    >>> heatmap_corr(df, corr_type="pearson")
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
        **kwargs,
    )


def r_pval(
    df,
    corr_type="pearson",
    figsize=(30, 20),
    index=None,
    cols=None,
    cmap_thr=0.8,
    is_T=False,
    annot=True,
    show_pvals=True,
    annot_rot=0,
    **kwargs,
):
    """
    Create Heatmaps with correlations and p-values by Spearman's statistic.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame for analysis.
    dm_corr_cols : list
        Demographic numeric columns.
    stat_cols : list
        Statistical numeric columns.
    figsize : tuple
        Size of the figure.
    cmap_thr : float, optional
        Threshold for cmap significance. Default is 0.8.
    is_T : bool, optional
        If True, pivot the table. Default is False.
    annot : bool, optional
        If True, annotate cells. Default is True.
    show_pvals : bool, optional
        If True, display p-values in the second heatmap. Default is True.
    annot_rot : int, optional
        Rotation angle for annotation. Default is 0.
    **kwargs : keyword arguments
        Additional arguments passed to `sns.heatmap`.

    Returns
    -------
    df_corrs : pandas.DataFrame
        DataFrame of Spearman correlation coefficients.
    df_pvals : pandas.DataFrame
        DataFrame of Spearman p-values.

    Examples
    --------
    Create a DataFrame with random data and compute correlation and p-values:

    >>> import numpy as np
    >>> import pandas as pd
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    >>>     'age': np.random.randint(18, 70, size=100),
    >>>     'income': np.random.randint(20000, 100000, size=100),
    >>>     'education_years': np.random.randint(10, 20, size=100)
    >>> })
    >>> dm_corr_cols = ['age', 'education_years']
    >>> stat_cols = ['income']
    >>> figsize = (10, 6)
    >>> df_corrs, df_pvals = r_pval(df, dm_corr_cols, stat_cols, figsize)
    >>> print(df_corrs)
    >>> print(df_pvals)
    """

    """Create Heatmaps with correlations and p-values by Spearman's statistic
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
        **kwargs,
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
    """
    Plot Heatmap with Phik correlations between specific x and y lists of columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data for correlation calculation.
    x : list, optional
        List of columns to use for the x-axis in the correlation matrix. If None,
        correlations will be calculated for all columns in df.
    y : list, optional
        List of columns to use for the y-axis in the correlation matrix. If None,
        correlations will be calculated for all columns in df.
    threshold : float, optional
        The threshold value for displaying Phik correlation values on the heatmap.
        Only correlations equal to or greater than this threshold will be shown. Default is 0.8.
    annot : bool, optional
        If True, annotate the cells in the heatmap with the correlation values. Default is True.
    fmt : str, optional
        Format for displaying correlation values in the heatmap. Default is ".2f".
    figsize : tuple, optional
        Figure size for the plot. Default is None, which uses the default size.
    annot_rot : int, optional
        Rotation angle for the annotations. Default is 0.
    annot_size : int, optional
        Font size for the annotations. Default is None.
    **kwargs : keyword arguments
        Additional arguments passed to `sns.heatmap`.

    Returns
    -------
    None
        The function generates a heatmap and does not return any values.

    Examples
    --------
    Create a DataFrame and plot the Phik correlation heatmap between specific columns:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pltstat.multfeats import phik_corrs
    >>>
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    >>>     'age': np.random.randint(18, 70, size=100),
    >>>     'income': np.random.randint(20000, 100000, size=100),
    >>>     'education_years': np.random.randint(10, 20, size=100)
    >>> })
    >>> x = ['age', 'income']
    >>> y = ['education_years']
    >>> phik_corrs(df, x=x, y=y, figsize=(8, 6))
    """
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
