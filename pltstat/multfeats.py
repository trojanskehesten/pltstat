"""
Provides tools for analyzing relationships between multiple features.
Includes visualization functions for analyzing missing data, comparing distributions,
and visualizing dimensionality reductions.
Additionally, it provides methods for creating heatmaps that display correlations and p-values,
including Spearman's correlation, Mann-Whitney p-values, and Phik correlations.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns

import umap.umap_ as umap

import numpy as np
import pandas as pd
from phik import phik_matrix

from scipy import stats
from scipy.stats import spearmanr, pearsonr, chi2_contingency, fisher_exact

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef

from . import cm

from .stat_methods import cramer_v
from .stat_methods import kruskal_by_cat, mannwhitneyu_by_cat

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

    if title_pref != "":
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
    x=None,
    y=None,
    corr_type="pearson",
    threshold=None,
    annot=True,
    fmt=".2f",
    figsize=(30, 20),
    linecolor="white",
    ax=None,
    **kwargs,
):
    """
    Compute correlation matrix and visualize it using a heatmap.

    Parameters
    ----------
    df : DataFrame
       Input DataFrame containing the data to compute the correlation matrix.
    x : list or array, optional
       List or array of features to be used as columns in the correlation matrix. If None, all features will be used.
    y : list or array, optional
       List or array of features to be used as rows in the correlation matrix. If None, all features will be used.
    corr_type : str, optional, default="pearson"
           The method to compute correlation. Supported options include "pearson",
           "kendall", "spearman", "cramer_v" for categorical data, "matthews" for binary data, and "phik".
           If "phik" is chosen, the `phik_corrs` function is called.
    threshold : float, optional
       Threshold value for customizing the heatmap colors. If specified, only correlations
       above this threshold will be displayed.
    annot : bool, optional
        If True, annotate the cells in the heatmap with the correlation values. Default is True.
    fmt : str, optional, default=".2f"
       The format string for the annotation of correlation values in the heatmap.
    figsize : tuple, optional, default=(30, 20)
       The size of the figure (width, height) in inches.
       If `ax` is provided, `figsize` will be ignored.
    linecolor : str, optional, default="white"
       The color of the lines separating the cells in the heatmap.
    ax : matplotlib.axes.Axes or None, optional, default=None
        The axes on which to draw the plot. If None, a new figure and axes are created.
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
    try:
        corr_method_title = {
            "pearson": "Pearson's r Correlation",
            "kendall": "Kendall's τ Correlation",
            "spearman": "Spearman's ρ Correlation",
            "cramer_v": "Cramér's V Statistic",
            "matthews": "Matthews Correlation Coefficient",
            "phik": "Phi Coefficient (phik)",
        }[corr_type]
    except:
        raise ValueError(f"Invalid `corr_type`. Choose 'pearson', 'spearman', 'kendall', 'cramer_v', 'matthews', or 'phik'. But [{corr_type}] is given")
    if corr_type in ["matthews", "cramer_v"]:

        # df.corr works only with numerical data, correct it:
        if (y is not None) and (x is not None):
            cols = np.concatenate((x, y))
        else:
            cols = df.columns

        if corr_type == "cramer_v":
            corr_type = cramer_v
            for col in cols:
                df.loc[:, col] = pd.Categorical(df.loc[:, col]).codes
        else:
            corr_type = matthews_corrcoef
            for col in cols:
                col_un_vals = df.loc[:, col].unique()
                df.loc[:, col] = df.loc[:, col].map({col_un_vals[0]: 0, col_un_vals[1]: 1})

    if corr_type == "phik":
        phik_corrs(
            df=df,
            x=x,
            y=y,
            threshold=threshold,
            annot=annot,
            fmt=fmt,
            figsize=figsize,
            ax=ax,
            **kwargs,
        )
        return

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    corr = df.corr(corr_type)

    if (y is not None) or (x is not None):
        if (y is not None) and (x is not None):
            corr = corr.loc[y, x]
        elif y is not None:
            corr = corr.loc[y, :]
        elif x is not None:
            corr = corr.loc[:, x]

    if (threshold is not None) and (corr_type == cramer_v):
        vmin = 0
        cmap = cm.get_corr_thr_cmap(threshold=threshold, vmin=vmin)
    elif threshold is not None:
        threshold /= 2
        vmin = -1
        cmap = cm.get_corr_thr_cmap(threshold=threshold, vmin=vmin)

    else:
        cmap = "coolwarm"
        vmin = -1

    sns.heatmap(
        corr,
        annot=annot,
        fmt=fmt,
        vmin=vmin,
        vmax=1,
        linewidths=0.5,
        linecolor=linecolor,
        cmap=cmap,
        ax=ax,
        **kwargs,
    )

    ax.set_title(corr_method_title)


def pvals_num(
        df,
        num_cols1=None,
        num_cols2=None,
        figsize=None,
        method='pearson',
        fmt=".2f",
        annot=True,
        alpha=0.05,
        annot_rot=0,
        annot_size=None,
        ax=None,
        **kwargs,
):
    """
    Compute and plot p-values for pairwise correlations between numeric columns.

    Parameters
    ----------
    df : DataFrame
       Input DataFrame containing numeric columns.
    num_cols1, num_cols2 : list, optional
       List of column names to be used as the first and the second set of variables.
       If None, all columns in `df` are used.
    figsize : tuple, optional
       Figure size for the heatmap. Ignored if `ax` is not None.
    method : {'pearson', 'spearman'}, default='pearson'
       Correlation method to use.
    fmt : str, default=".2f"
       String formatting for displayed p-values.
    annot : bool, default=True
       If True, display p-values on the heatmap.
    alpha : float, default=0.05
       Significance level for highlighting values.
    annot_rot : int, default=0
       Rotation angle for annotations.
    annot_size : int, optional
       Font size for annotations.
    ax : matplotlib.axes.Axes, optional
       Axis object to plot the heatmap.
    **kwargs
       Additional keyword arguments passed to seaborn heatmap.

    Returns
    -------
    DataFrame
       A DataFrame containing p-values for pairwise correlations.

    Examples
    --------
    >>> import pandas as pd
    >>> np.random.seed(42)
    >>> df = pd.DataFrame(np.random.rand(10, 3), columns=['A', 'B', 'C'])
    >>> pvals_num(df, method='pearson')
    """
    if method == 'pearson':
        stat_func = pearsonr
        stat_method = "Pearson's Correlation"
    elif method == 'spearman':
        stat_func = spearmanr
        stat_method = "Spearman's Rank Correlation"
    else:
        raise ValueError(f"Invalid `method`. Choose 'pearson' or 'spearman'. But [{method}] is given")

    cols = df.columns
    num_cols1 = num_cols1 or cols
    num_cols2 = num_cols2 or cols

    df_pvals = df.corr(method=lambda a, b: stat_func(a, b)[1])
    np.fill_diagonal(df_pvals.values, 0)

    df_pvals = df_pvals.loc[num_cols1, num_cols2]

    # Plot dataframe:
    _plot_pvals(df_pvals, stat_method, figsize=figsize, fmt=fmt, annot=annot, ax=ax, alpha=alpha, annot_rot=annot_rot,
                annot_size=annot_size, **kwargs)

    return df_pvals


def pvals_cat(
        df,
        cat_cols1=None,
        cat_cols2=None,
        figsize=None,
        method='auto',
        fmt=".2f",
        annot=True,
        alpha=0.05,
        annot_rot=0,
        annot_size=None,
        ax=None,
        **kwargs,
    ):
    """
    Compute and plot p-values for pairwise correlations between categorical columns.

    Parameters
    ----------
    df : DataFrame
       Input DataFrame containing categorical columns.
    cat_cols1, cat_cols2 : list, optional
       List of column names to be used as the first and the second set of variables.
       If None, all columns in `df` are used.
    figsize : tuple, optional
       Figure size for the heatmap. Ignored if `ax` is not None.
    method : {'auto', 'fisher', 'chi2'}, default='auto'
       Correlation method to use. If `auto`, the method will be chosen by the data.
    fmt : str, default=".2f"
       String formatting for displayed p-values.
    annot : bool, default=True
       If True, display p-values on the heatmap.
    alpha : float, default=0.05
       Significance level for highlighting values.
    annot_rot : int, default=0
       Rotation angle for annotations.
    annot_size : int, optional
       Font size for annotations.
    ax : matplotlib.axes.Axes, optional
       Axis object to plot the heatmap.
    **kwargs
       Additional keyword arguments passed to seaborn heatmap.

    Returns
    -------
    DataFrame
       A DataFrame containing p-values for pairwise correlations.

    Examples
    --------
    >>> import pandas as pd
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    >>>     'A': np.random.choice(['Red', 'Blue', 'Green'], size=10),
    >>>     'B': np.random.choice(['Yes', 'No'], size=10),
    >>>     'C': np.random.choice(['Low', 'Medium', 'High'], size=10)
    >>> })
    >>> pvals_cat(df)
    """
    if method == 'auto':
        stat_func = lambda crosstab_df: fisher_exact(crosstab_df) if crosstab_df.min(axis=None) < 5 else chi2_contingency(crosstab_df)
        stat_method = "Fisher's Exact or Chi-squared Test"
    elif method == 'fisher':
        stat_func = lambda crosstab_df: fisher_exact(crosstab_df)
        stat_method = "Fisher's Exact Test"
    elif method == 'chi2':
        stat_func = lambda ct_df: chi2_contingency(ct_df)
        stat_method = 'Chi-squared Test'
    else:
        raise ValueError("Invalid `method`. Choose 'auto', 'fisher', or 'chi2'. But [{corr_type}] is given")

    cols = df.columns
    cat_cols1 = cat_cols1 or cols
    cat_cols2 = cat_cols2 or cols

    df_pvals = pd.DataFrame(index=cat_cols1, columns=cat_cols2, dtype="float64")
    for cat_col1 in cat_cols1:
        for cat_col2 in cat_cols2:
            if cat_col1 == cat_col2:
                p_value = 0.
            else:
                df_subset = df[[cat_col1, cat_col2]].dropna()
                crosstab_df = pd.crosstab(df_subset[cat_col1], df_subset[cat_col2])

                p_value = stat_func(crosstab_df)[1]

            df_pvals.loc[cat_col1, cat_col2] = p_value

    # Plot dataframe:
    _plot_pvals(df_pvals, stat_method, figsize=figsize, fmt=fmt, annot=annot, ax=ax, alpha=alpha, annot_rot=annot_rot,
                annot_size=annot_size, **kwargs)

    return df_pvals


def pvals_num_cat(
    df,
    cat_cols,
    num_cols,
    alpha=0.05,
    figsize=None,
    fmt=".2f",
    method='auto',
    is_T=False,
    annot=True,
    annot_rot=0,
    annot_size=None,
    ax=None,
    **kwargs,
):
    """
    Compute Mann-Whitney or Kruskal-Wallis p-values between numerical columns grouped by categorical columns.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing both numerical and categorical columns.
    cat_cols : list
        List of categorical column names. Each column must have at least two unique values.
    num_cols : list
        List of numerical column names.
    alpha : float, default=0.05
        Significance level for highlighting values in the heatmap.
    figsize : tuple, optional
        Figure size for the heatmap. Ignored if `ax` is not None.
    fmt : str, default=".2f"
        String formatting for displayed p-values.
    method : {'auto', 'mw', 'kruskal'}, default='auto'
        Statistical test to use:
        - 'mw' for Mann-Whitney U test (for two groups only)
        - 'kruskal' for Kruskal-Wallis test (for multiple groups)
        - 'auto' selects Mann-Whitney if there are exactly two groups, otherwise Kruskal-Wallis.
    is_T : bool, default=False
        If True, transpose the result DataFrame.
    annot : bool, default=True
        If True, display p-values on the heatmap.
    annot_rot : int, default=0
        Rotation angle for annotations.
    annot_size : int, optional
        Font size for annotations.
    ax : matplotlib.axes.Axes, optional
        Axis object to plot the heatmap. If provided, `figsize` is ignored.
    **kwargs
        Additional keyword arguments passed to seaborn heatmap.

    Returns
    -------
    DataFrame
        A DataFrame containing p-values for each numerical-categorical combination.

    Examples
    --------
    >>> import pandas as pd
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     'Category': np.random.choice(['A', 'B'], size=10),
    ...     'Value1': np.random.rand(10),
    ...     'Value2': np.random.rand(10)
    ... })
    >>> pvals_num_cat(df, cat_cols=['Category'], num_cols=['Value1', 'Value2'], method='mw')
    """
    if method == 'auto':
        stat_func = lambda n: mannwhitneyu_by_cat if n == 2 else kruskal_by_cat
    elif method == 'mw':
        stat_func = lambda n: mannwhitneyu_by_cat
    elif method == 'kruskal':
        stat_func = lambda n: kruskal_by_cat

    df_pvals = pd.DataFrame(index=cat_cols, columns=num_cols, dtype="float64")
    for cat_col in cat_cols:
        for num_col in num_cols:
            df_subset = df[[cat_col, num_col]].dropna()
            n_cats = df_subset.loc[:, cat_col].nunique()

            if n_cats < 2:
                # Less than 2 different values of categorical feature in the subset
                p = np.nan
            else:
                p = stat_func(n_cats)(df_subset, cat_col, num_col)[1]

            df_pvals.loc[cat_col, num_col] = p

    if is_T:
        df_pvals = df_pvals.T

    # Plot dataframe:
    stat_method = {
        "mw": "Mann-Whitney U Test",
        "kruskal": "Kruskal-Wallis Test",
        "auto": "Auto Mann-Whitney U or Kruskal-Wallis Test"
    }[method]
    _plot_pvals(df_pvals, stat_method, figsize=figsize, fmt=fmt, annot=annot, ax=ax, alpha=alpha, annot_rot=annot_rot,
                annot_size=annot_size, **kwargs)

    return df_pvals


def _plot_pvals(df_pvals, stat_method, figsize=None, fmt=".2f", annot=True, ax=None, alpha=0.5, annot_rot=0, annot_size=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    cmap, cbar_kws = cm.get_pval_legend_thr_cmap(alpha=alpha)
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
        ax=ax,
        **kwargs,
    )
    xticks = df_pvals.columns

    # ax.set_xticks(np.arange(len(xticks)) + 0.5, xticks)

    ax.set_title(f"{stat_method} p-value")


def phik_corrs(
    df,
    interval_cols=None,
    x=None,
    y=None,
    threshold=0.8,
    annot=True,
    fmt=".2f",
    figsize=None,
    annot_rot=0,
    annot_size=None,
    ax=None,
    bins=10,
    njobs=-1,
    heatmap_kwargs=None,
    phik_kwargs=None,
):
    """
    Plot Heatmap with Phik correlations between specific x and y lists of columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data for correlation calculation.
    interval_cols : list of str, optional
        List of columns to treat as interval variables for Phik correlation calculation.
        If None, columns will be automatically determined as interval (continuous) or categorical by the program.
        This parameter is useful when you want to specify certain columns to be treated as interval (continuous) variables.
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
        If `ax` is provided, `figsize` will be ignored.
    annot_rot : int, optional
        Rotation angle for the annotations. Default is 0.
    annot_size : int, optional
        Font size for the annotations. Default is None.
    ax : matplotlib.axes.Axes or None, optional, default=None
        The axes on which to draw the plot. If None, a new figure and axes are created.
    bins : int, optional
        Number of bins to use for discretizing continuous variables. Default is 10.
    njobs : int, optional
        Number of parallel jobs to use for the Phik calculation.
        Default is -1, which uses all available processors.
    heatmap_kwargs : dict, optional
        Additional keyword arguments to pass to `sns.heatmap` for customization of the heatmap.
    phik_kwargs : dict, optional
        Additional keyword arguments to pass to the `phik_matrix` function for Phik calculation.

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

    phik_kwargs = phik_kwargs or {}
    heatmap_kwargs = heatmap_kwargs or {}

    cmap = cm.get_corr_thr_cmap(threshold=threshold, vmin=0)
    if (x is not None) and (y is not None):
        xy = np.concatenate((x, y))
        xy = np.unique(xy)
        df_phik = df[xy].phik_matrix(interval_cols=interval_cols, bins=bins, njobs=njobs, **phik_kwargs)
        df_phik = df_phik.loc[y, x]
    else:
        df_phik = df.phik_matrix(interval_cols=interval_cols, bins=bins, njobs=njobs, **phik_kwargs)

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df_phik,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=annot,
        fmt=fmt,
        annot_kws={"rotation": annot_rot, "fontsize": annot_size},
        ax=ax,
        **heatmap_kwargs,
    )


# def r_pval(
#     df,
#     x=None,
#     y=None,
#     corr_type="pearson",
#     threshold=None,
#     annot=True,
#     fmt_corrs=".2f",
#     fmt_pvals=".2f",
#     figsize=(30, 20),
#     linecolor="white",
#     **kwargs,
# ):
#     """"""
#     fig, ax = plt.subplots(1, 2, figsize=figsize)
#
#     # Create corrs:
#     heatmap_corr(
#         df,
#         x=x,
#         y=y,
#         corr_type=corr_type,
#         threshold=threshold,
#         annot=annot,
#         fmt=fmt_corrs,
#         linecolor=linecolor,
#         ax=ax[0],
#         **kwargs,
#     )
#
#     # n n
#     CORR SPE PEAR            [PVALS]
#
#     # n c
#     def pvals_num(df, cat_cols, num_cols)
#     def pvals_cat(df, cat_cols1=None, cat_cols2=None)
#     def pvals_num_cat(df, num_cols1=None, num_cols2=None)
#     [PHIK?]                  MW KRUSKAL
#
#     # c c
#     [CRAMER V]               CHI2 FISHER
#
#     # Create p-values:
#     cmap, cbar_kws = cm.get_pval_legend_thr_cmap()
#     sns.heatmap(
#         df_pvals,
#         vmin=0,
#         vmax=1,
#         cmap=cmap,
#         annot=annot,
#         fmt=fmt_pvals,
#         linewidths=1,
#         cbar_kws=cbar_kws,
#         ax=ax[1],
#     )
#     ax[1].set_title("Spearman correlations, p-value")
#     xticks = df_pvals.columns
#     ax[1].set_xticks(np.arange(len(xticks)) + 0.5, xticks)
#
#
# def r_pval2(
#     df,
#     corr_type="pearson",
#     figsize=(30, 20),
#     index=None,
#     cols=None,
#     cmap_thr=0.8,
#     is_T=False,
#     annot=True,
#     show_pvals=True,
#     annot_rot=0,
#     **kwargs,
# ):
#     """
#     Create Heatmaps with correlations and p-values by Spearman's statistic.
#
#     Parameters
#     ----------
#     df : pandas.DataFrame
#         DataFrame for analysis.
#     dm_corr_cols : list
#         Demographic numeric columns.
#     stat_cols : list
#         Statistical numeric columns.
#     figsize : tuple
#         Size of the figure.
#     cmap_thr : float, optional
#         Threshold for cmap significance. Default is 0.8.
#     is_T : bool, optional
#         If True, pivot the table. Default is False.
#     annot : bool, optional
#         If True, annotate cells. Default is True.
#     show_pvals : bool, optional
#         If True, display p-values in the second heatmap. Default is True.
#     annot_rot : int, optional
#         Rotation angle for annotation. Default is 0.
#     **kwargs : keyword arguments
#         Additional arguments passed to `sns.heatmap`.
#
#     Returns
#     -------
#     df_corrs : pandas.DataFrame
#         DataFrame of Spearman correlation coefficients.
#     df_pvals : pandas.DataFrame
#         DataFrame of Spearman p-values.
#
#     Examples
#     --------
#     Create a DataFrame with random data and compute correlation and p-values:
#
#     >>> import numpy as np
#     >>> import pandas as pd
#     >>> np.random.seed(42)
#     >>> df = pd.DataFrame({
#     >>>     'age': np.random.randint(18, 70, size=100),
#     >>>     'income': np.random.randint(20000, 100000, size=100),
#     >>>     'education_years': np.random.randint(10, 20, size=100)
#     >>> })
#     >>> dm_corr_cols = ['age', 'education_years']
#     >>> stat_cols = ['income']
#     >>> figsize = (10, 6)
#     >>> df_corrs, df_pvals = r_pval(df, dm_corr_cols, stat_cols, figsize)
#     >>> print(df_corrs)
#     >>> print(df_pvals)
#     """
#
#     """Create Heatmaps with correlations and p-values by Spearman's statistic
#     :param df: pd.DataFrame for analysis
#     :param dm_corr_cols: Demographic numeric columns
#     :param stat_cols: Statistical numeric columns
#     :param figsize: tuple, figure size
#     :param cmap_thr: cmap threshold of correlations significancy
#     :param is_T: pivoting table, bool
#     :param annot: add annotation to cells, bool
#     :param show_pvals: plot heatmap with p-values, bool
#     :param annot_rot: rotation angle of annotation, int
#     :return: pandas.DataFrames with p-values and with correlation coefficients
#     """
#     df_pvals = pd.DataFrame(index=dm_corr_cols, columns=stat_cols, dtype="float64")
#     df_corrs = pd.DataFrame(index=dm_corr_cols, columns=stat_cols, dtype="float64")
#     for dm_corr_col in dm_corr_cols:
#         for stat_col in stat_cols:
#             df_subset = df[[dm_corr_col, stat_col]].dropna()
#             x = df_subset.loc[:, dm_corr_col]
#             y = df_subset.loc[:, stat_col]
#             corr, p = spearmanr(x, y)
#             df_pvals.loc[dm_corr_col, stat_col] = p
#             df_corrs.loc[dm_corr_col, stat_col] = corr
#
#     if show_pvals:
#         nrows = 2
#     else:
#         nrows = 1
#
#     fig, ax = plt.subplots(
#         nrows=nrows, ncols=1, figsize=figsize, constrained_layout=True
#     )
#     if not show_pvals:
#         ax = [ax]
#     if is_T is True:
#         df_pvals = df_pvals.T
#
#     cmap_corrs = cm.get_corr_thr_cmap(threshold=cmap_thr)
#     sns.heatmap(
#         df_corrs,
#         vmin=-1,
#         vmax=1,
#         cmap=cmap_corrs,
#         annot=annot,
#         fmt=".1f",
#         linewidths=1,
#         ax=ax[0],
#         annot_kws={"rotation": annot_rot},
#         **kwargs,
#     )
#     ax[0].set_title("Spearman correlations, correlation coefficient")
#     xticks = df_corrs.columns
#     ax[0].set_xticks(np.arange(len(xticks)) + 0.5, xticks)
#     if show_pvals:
#         cmap, cbar_kws = cm.get_pval_legend_thr_cmap()
#         sns.heatmap(
#             df_pvals,
#             vmin=0,
#             vmax=1,
#             cmap=cmap,
#             annot=annot,
#             fmt=".2f",
#             linewidths=1,
#             cbar_kws=cbar_kws,
#             ax=ax[1],
#         )
#         ax[1].set_title("Spearman correlations, p-value")
#         xticks = df_pvals.columns
#         ax[1].set_xticks(np.arange(len(xticks)) + 0.5, xticks)
#
#     return df_corrs, df_pvals
