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
from scipy.stats import spearmanr, pearsonr

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef

from . import cm

from .stat_methods import cramer_v
from .stat_methods import chi2_fisher_by_cat, kruskal_by_cat, mannwhitneyu_by_cat
from .config import _resolve_engine, _ensure_plotly


# ====================================================================
# Shared plotly heatmap helper
# ====================================================================

def _heatmap_plotly(data, colorscale, zmin, zmax, title, annot=True,
                    fmt=".2f", **kwargs):
    """Build a plotly Heatmap figure from a 2D array or DataFrame."""
    _ensure_plotly()
    import plotly.graph_objects as go

    if isinstance(data, pd.DataFrame):
        z = data.values
        x = [str(c) for c in data.columns]
        y = [str(i) for i in data.index]
    else:
        z = np.asarray(data)
        x = None
        y = None

    text = None
    if annot:
        text = np.array([[fmt.format(v) if isinstance(v, float)
                          else str(v) for v in row] for row in z])

    heatmap = go.Heatmap(
        z=z, x=x, y=y, colorscale=colorscale,
        zmin=zmin, zmax=zmax, text=text,
        texttemplate="%{text}" if annot else None,
        **kwargs,
    )
    fig = go.Figure(data=heatmap)
    fig.update_layout(title=title)
    return fig


# ====================================================================
# nulls
# ====================================================================

def nulls(
    df,
    figsize=(20, 10),
    index=None,
    n_ticks=None,
    print_str_index=False,
    print_all=True,
    *,
    engine=None,
):
    """
    Plot a heatmap to visualize null values in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    figsize : tuple, optional, default=(20, 10)
        The size of the figure (matplotlib only).
    index : str, optional, default=None
        The name of the column to use as the y-axis label.
    n_ticks : int, optional, default=None
        The number of y-axis ticks to display.
    print_str_index : bool, optional, default=False
        If True and the index is a string type, print the index values
        as labels.
    print_all : bool, optional, default=True
        If True, display all index values (matplotlib only).
    engine : {"matplotlib", "plotly"} or None, default=None
        The rendering backend.

    Returns
    -------
    plotly.graph_objects.Figure or None

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.multfeats import nulls
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, np.nan, 4],
    ...     'B': [np.nan, 2, 3, 4],
    ...     'C': [1, np.nan, np.nan, 4]
    ... })
    >>> nulls(df)
    """
    engine = _resolve_engine(engine)

    if engine == "plotly":
        _ensure_plotly()
        import plotly.express as px

        null_matrix = df.isnull()
        fig = px.imshow(
            null_matrix,
            color_continuous_scale=[[0, "white"], [1, "black"]],
            title="Null values (black)",
            labels=dict(x="columns", y="rows"),
        )
        fig.update_layout(coloraxis_showscale=False)
        return fig

    # matplotlib
    plt.figure(figsize=figsize)
    sns.heatmap(
        df.isnull().apply(np.invert), yticklabels=False,
        cbar=False, vmin=0, vmax=1,
    )
    plt.title("Null values (black)")
    if n_ticks is None:
        n_ticks = 11
    y_ticks = np.linspace(0, len(df) - 1, n_ticks).astype("int64")

    if index is None:
        index = "index"
        if df.index.dtype != "O":
            y_labels = np.percentile(
                df.index, np.linspace(0, 100, n_ticks)
            ).astype("int64")
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
            y_labels = np.percentile(
                df[index], np.linspace(0, 100, n_ticks)
            ).astype("int64")
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
    return None


# ====================================================================
# dist_qq_plot
# ====================================================================

def dist_qq_plot(df, figsize, *, engine=None, fig_return=False, **kwargs):
    """
    Plot histograms and Q-Q plots for each feature, along with
    Shapiro-Wilk test p-values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the features to be plotted.
    figsize : tuple
        The size of the figure (matplotlib only).
    engine : {"matplotlib", "plotly"} or None, default=None
        The rendering backend.
    fig_return : bool, optional, default=False
        If True and engine is plotly, returns ``(shapiros, fig)``.
    **kwargs : keyword arguments
        Additional arguments for histogram customization.

    Returns
    -------
    shapiros : np.ndarray
        Shapiro-Wilk test p-values for each feature.  When
        ``engine="plotly"`` and ``fig_return=True``, returns
        ``(shapiros, fig)``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.multfeats import dist_qq_plot
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     'A': np.random.normal(0, 1, 100),
    ...     'B': np.random.normal(5, 2, 100)
    ... })
    >>> dist_qq_plot(df, figsize=(12, 8))
    """
    engine = _resolve_engine(engine)

    if engine == "plotly":
        return _dist_qq_plot_plotly(df, fig_return=fig_return, **kwargs)

    return _dist_qq_plot_mpl(df, figsize, **kwargs)


def _dist_qq_plot_mpl(df, figsize, **kwargs):
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
        return np.array([])
    else:
        print("I can't calculate number of columns")
        return np.array([])
    n_rows = int(np.ceil(2 * n_cols_df / n_cols))

    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=figsize, constrained_layout=True
    )
    i = 0
    shapiros = []
    if n_cols_df > 3:
        for col in df:
            sns.histplot(df[col], ax=axs[i // n_cols, i % n_cols],
                          **kwargs)
            median = df[col].median()
            axs[i // n_cols, i % n_cols].set_title(
                col + "\nMedian=%.2f" % median
            )
            i += 1
            stats.probplot(
                df[col], dist="norm",
                plot=axs[i // n_cols, i % n_cols], rvalue=True
            )
            pval = stats.shapiro(df[col]).pvalue
            axs[i // n_cols, i % n_cols].set_title(
                col + "\nShapiro pval=%.2f" % pval
            )
            i += 1
            shapiros.append(pval)
    else:
        for col in df:
            sns.histplot(df[col], ax=axs[i], **kwargs)
            median = df[col].median()
            axs[i].set_title(col + "\nMedian=%.2f" % median)
            i += 1
            stats.probplot(df[col], dist="norm", plot=axs[i],
                           rvalue=True)
            pval = stats.shapiro(df[col]).pvalue
            axs[i].set_title(col + "\nShapiro pval=%.2f" % pval)
            i += 1
            shapiros.append(pval)

    return np.array(shapiros)


def _dist_qq_plot_plotly(df, fig_return=False, **kwargs):
    _ensure_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_cols_df = df.shape[1]
    if n_cols_df == 0:
        print("DF is empty, no columns")
        return np.array([])

    n_rows = n_cols_df
    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=[],
    )

    shapiros = []
    for row_idx, col in enumerate(df.columns, start=1):
        data = df[col].dropna()

        # Histogram
        hist = go.Histogram(x=data.values, name=f"{col} histogram")
        fig.add_trace(hist, row=row_idx, col=1)

        median = data.median()
        pval = stats.shapiro(data).pvalue
        shapiros.append(pval)

        fig.update_annotations(
            dict(
                text=f"{col}<br>Median={median:.2f}, "
                     f"Shapiro pval={pval:.2f}",
                x=0.25, xref=f"x{row_idx * 2 - 1} domain" if row_idx > 1
                    else "x domain",
            ) if row_idx == 1 else
            dict(text=f"{col}<br>Median={median:.2f}"),
        )

        # Q-Q plot (manual)
        osm, osr = stats.probplot(data.values, dist="norm")[:2]
        osm = np.asarray(osm)
        osr = np.asarray(osr)
        qq = go.Scatter(
            x=osm, y=osr, mode="markers",
            name=f"{col} Q-Q",
        )
        fig.add_trace(qq, row=row_idx, col=2)

        # Reference line y = x (for standardized data)
        ref = go.Scatter(
            x=[float(osm.min()), float(osm.max())],
            y=[float(osr.min()), float(osr.max())] if osr.std() > 0
              else [0, 0],
            mode="lines", line=dict(dash="dash", color="red"),
            showlegend=False,
        )
        fig.add_trace(ref, row=row_idx, col=2)

    fig.update_layout(title="Histograms and Q-Q plots")

    if fig_return:
        return np.array(shapiros), fig
    return np.array(shapiros)


# ====================================================================
# embeddings_creation (pure compute, no engine)
# ====================================================================

def embeddings_creation(X, n_components=2, standardize=True,
                         random_state=0, umap_kwargs=None,
                         tsne_kwargs=None):
    """
    Create 2D representation of data using UMAP and t-SNE.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data for dimensionality reduction.
    n_components : int, optional, default=2
        Number of dimensions for the reduced representation.
    standardize : bool, optional, default=True
        Whether to standardize the input data before applying
        dimensionality reduction.
    random_state : int, optional, default=0
        The seed used by the random number generator.
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

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> import pandas as pd
    >>> from pltstat.multfeats import embeddings_creation
    >>> iris = load_iris()
    >>> X = pd.DataFrame(iris.data, columns=iris.feature_names)
    >>> X_umap, X_tsne = embeddings_creation(X, random_state=42)
    >>> X_umap.shape
    (150, 2)
    """
    if standardize is True:
        X = StandardScaler().fit_transform(X)

    umap_kwargs = umap_kwargs or {}
    tsne_kwargs = tsne_kwargs or {}

    reducer = umap.UMAP(
        n_components=n_components, random_state=random_state,
        **umap_kwargs,
    )
    X_umap = reducer.fit_transform(X)

    reducer = TSNE(
        n_components=n_components, random_state=random_state,
        **tsne_kwargs,
    )
    X_tsne = reducer.fit_transform(X)

    print("Shape of umap is", X_umap.shape,
          "; Shape of tsne is", X_tsne.shape)
    return X_umap, X_tsne


# ====================================================================
# plot_umap_tsne
# ====================================================================

def plot_umap_tsne(X_umap, X_tsne, labels=None, title_pref="",
                   unnoisy_idx=None, figsize=(16, 6), *, engine=None):
    """
    Plot UMAP and t-SNE projections with cluster labels.

    Parameters
    ----------
    X_umap : array, shape (n_samples, 2)
        2D UMAP embeddings of the data.
    X_tsne : array, shape (n_samples, 2)
        2D t-SNE embeddings of the data.
    labels : array, optional
        Cluster labels for each sample.
    title_pref : str, optional, default=""
        A preferred title prefix.
    unnoisy_idx : array, optional
        Indices of non-noisy points.
    figsize : tuple, optional, default=(16, 6)
        The size of the figure (matplotlib only).
    engine : {"matplotlib", "plotly"} or None, default=None
        The rendering backend.

    Returns
    -------
    plotly.graph_objects.Figure or None

    Examples
    --------
    >>> import numpy as np
    >>> from pltstat.multfeats import plot_umap_tsne
    >>> # X_umap, X_tsne = embeddings_creation(X)
    >>> # plot_umap_tsne(X_umap, X_tsne, labels=labels)
    """
    engine = _resolve_engine(engine)

    if engine == "plotly":
        return _plot_umap_tsne_plotly(
            X_umap, X_tsne, labels, title_pref, unnoisy_idx
        )

    return _plot_umap_tsne_mpl(
        X_umap, X_tsne, labels, title_pref, unnoisy_idx, figsize
    )


def _plot_umap_tsne_mpl(X_umap, X_tsne, labels, title_pref,
                        unnoisy_idx, figsize):
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    if title_pref != "":
        title_pref += " and "
    ax[0].set_title(f"{title_pref}UMAP projection")
    ax[1].set_title(f"{title_pref}TSNE projection")

    palette = "tab10"
    legend = "full"

    if unnoisy_idx is None:
        sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels,
                        legend=legend, palette=palette, ax=ax[0])
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels,
                        legend=legend, palette=palette, ax=ax[1])
    else:
        sns.scatterplot(
            x=X_umap[unnoisy_idx, 0], y=X_umap[unnoisy_idx, 1],
            hue=labels[unnoisy_idx], legend=legend, palette=palette,
            ax=ax[0],
        )
        sns.scatterplot(
            x=X_tsne[unnoisy_idx, 0], y=X_tsne[unnoisy_idx, 1],
            hue=labels[unnoisy_idx], legend=legend, palette=palette,
            ax=ax[1],
        )
    return None


def _plot_umap_tsne_plotly(X_umap, X_tsne, labels, title_pref,
                           unnoisy_idx):
    _ensure_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if unnoisy_idx is not None:
        X_umap = X_umap[unnoisy_idx]
        X_tsne = X_tsne[unnoisy_idx]
        if labels is not None:
            labels = np.asarray(labels)[unnoisy_idx]

    if labels is None:
        labels = np.zeros(len(X_umap))

    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"{title_pref}UMAP projection",
                         f"{title_pref}TSNE projection"),
    )

    for proj_idx, (X_proj, name) in enumerate(
        [(X_umap, "UMAP"), (X_tsne, "TSNE")], start=1
    ):
        for lab in unique_labels:
            mask = labels == lab
            fig.add_trace(
                go.Scatter(
                    x=X_proj[mask, 0], y=X_proj[mask, 1],
                    mode="markers", name=str(lab),
                    showlegend=(proj_idx == 1),
                ),
                row=1, col=proj_idx,
            )

    return fig


# ====================================================================
# heatmap_corr
# ====================================================================

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
    *,
    engine=None,
    **kwargs,
):
    """
    Compute correlation matrix and visualize it using a heatmap.

    Parameters
    ----------
    df : DataFrame
       Input DataFrame containing the data.
    x : list or array, optional
       Features to use as columns in the correlation matrix.
    y : list or array, optional
       Features to use as rows in the correlation matrix.
    corr_type : str, optional, default="pearson"
       The method to compute correlation. Supported: "pearson",
       "kendall", "spearman", "cramer_v", "matthews", "phik".
    threshold : float, optional
       Threshold for the heatmap colors.
    annot : bool, optional, default=True
       If True, annotate the cells.
    fmt : str, optional, default=".2f"
       Format string for annotations.
    figsize : tuple, optional, default=(30, 20)
       Figure size (matplotlib only).
    linecolor : str, optional, default="white"
       Cell separator color (matplotlib only).
    ax : matplotlib.axes.Axes or None, optional, default=None
       The axes to draw on (matplotlib only).
    engine : {"matplotlib", "plotly"} or None, default=None
       The rendering backend.
    **kwargs : keyword arguments
       Additional parameters passed to ``sns.heatmap`` or
       ``go.Heatmap``.

    Returns
    -------
    plotly.graph_objects.Figure or None

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.multfeats import heatmap_corr
    >>> np.random.seed(42)
    >>> df = pd.DataFrame(np.random.rand(10, 5), columns=list("ABCDE"))
    >>> heatmap_corr(df, corr_type="pearson")
    """
    engine = _resolve_engine(engine)

    try:
        corr_method_title = {
            "pearson": "Pearson's r Correlation",
            "kendall": "Kendall's Correlation",
            "spearman": "Spearman's Correlation",
            "cramer_v": "Cramer's V Statistic",
            "matthews": "Matthews Correlation Coefficient",
            "phik": "Phi Coefficient (phik)",
        }[corr_type]
    except KeyError:
        raise ValueError(
            f"Invalid `corr_type`. Choose 'pearson', 'spearman', "
            f"'kendall', 'cramer_v', 'matthews', or 'phik'. "
            f"But [{corr_type}] is given"
        )

    if corr_type in ["matthews", "cramer_v"]:
        if (y is not None) and (x is not None):
            cols = np.concatenate((x, y))
        else:
            cols = df.columns

        if corr_type == "cramer_v":
            corr_type_fn = cramer_v
            for col in cols:
                df.loc[:, col] = pd.Categorical(df.loc[:, col]).codes
        else:
            corr_type_fn = matthews_corrcoef
            for col in cols:
                col_un_vals = df.loc[:, col].unique()
                df.loc[:, col] = df.loc[:, col].map(
                    {col_un_vals[0]: 0, col_un_vals[1]: 1}
                )
    else:
        corr_type_fn = corr_type

    if corr_type_fn == "phik":
        return phik_corrs(
            df=df, x=x, y=y, threshold=threshold, annot=annot,
            fmt=fmt, figsize=figsize, ax=ax, engine=engine, **kwargs,
        )

    corr = df.corr(corr_type_fn)

    if (y is not None) or (x is not None):
        if (y is not None) and (x is not None):
            corr = corr.loc[y, x]
        elif y is not None:
            corr = corr.loc[y, :]
        elif x is not None:
            corr = corr.loc[:, x]

    if engine == "plotly":
        _ensure_plotly()
        if (threshold is not None) and (corr_type_fn == cramer_v):
            colorscale, vmin, vmax = cm.get_corr_colorscale(
                threshold=threshold, vmin=0
            )
        elif threshold is not None:
            colorscale, vmin, vmax = cm.get_corr_colorscale(
                threshold=threshold / 2, vmin=-1
            )
        else:
            colorscale = "RdBu"
            vmin = -1
            vmax = 1
        return _heatmap_plotly(
            corr, colorscale, vmin, vmax, corr_method_title,
            annot=annot, fmt=fmt, **kwargs,
        )

    # matplotlib
    if (threshold is not None) and (corr_type_fn == cramer_v):
        vmin = 0
        cmap = cm.get_corr_thr_cmap(threshold=threshold, vmin=vmin)
    elif threshold is not None:
        threshold /= 2
        vmin = -1
        cmap = cm.get_corr_thr_cmap(threshold=threshold, vmin=vmin)
    else:
        cmap = "coolwarm"
        vmin = -1

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr, annot=annot, fmt=fmt, vmin=vmin, vmax=1,
        linewidths=0.5, linecolor=linecolor, cmap=cmap, ax=ax,
        **kwargs,
    )
    ax.set_title(corr_method_title)
    return None


# ====================================================================
# pvals_num, pvals_cat, pvals_num_cat
# ====================================================================

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
        *,
        engine=None,
        fig_return=False,
        **kwargs,
):
    """
    Compute and plot p-values for pairwise correlations between numeric
    columns.

    Parameters
    ----------
    df : DataFrame
       Input DataFrame containing numeric columns.
    num_cols1, num_cols2 : list, optional
       Column names for the first and second set of variables.
    figsize : tuple, optional
       Figure size (matplotlib only).
    method : {'pearson', 'spearman'}, default='pearson'
       Correlation method to use.
    fmt : str, default=".2f"
       String formatting for displayed p-values.
    annot : bool, default=True
       If True, display p-values on the heatmap.
    alpha : float, default=0.05
       Significance level for highlighting values.
    annot_rot : int, default=0
       Rotation angle for annotations (matplotlib only).
    annot_size : int, optional
       Font size for annotations (matplotlib only).
    ax : matplotlib.axes.Axes, optional
       Axis object to plot the heatmap (matplotlib only).
    engine : {"matplotlib", "plotly"} or None, default=None
       The rendering backend.
    fig_return : bool, optional, default=False
       If True and engine is plotly, returns ``(df_pvals, fig)``.
    **kwargs
       Additional keyword arguments passed to the heatmap.

    Returns
    -------
    DataFrame
       A DataFrame containing p-values.  When ``engine="plotly"`` and
       ``fig_return=True``, returns ``(df_pvals, fig)``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.multfeats import pvals_num
    >>> np.random.seed(42)
    >>> df = pd.DataFrame(np.random.rand(10, 3), columns=['A','B','C'])
    >>> pvals_num(df, method='pearson')
    """
    engine = _resolve_engine(engine)

    if method == 'pearson':
        stat_func = pearsonr
        stat_method = "Pearson's Correlation"
    elif method == 'spearman':
        stat_func = spearmanr
        stat_method = "Spearman's Rank Correlation"
    else:
        raise ValueError(
            f"Invalid `method`. Choose 'pearson' or 'spearman'. "
            f"But [{method}] is given"
        )

    cols = df.columns
    num_cols1 = num_cols1 or cols
    num_cols2 = num_cols2 or cols

    df_pvals = df.corr(method=lambda a, b: stat_func(a, b)[1])
    np.fill_diagonal(df_pvals.values, 0)
    df_pvals = df_pvals.loc[num_cols1, num_cols2]

    if engine == "plotly":
        result = _plot_pvals_plotly(
            df_pvals, stat_method, alpha=alpha, annot=annot, fmt=fmt,
            **kwargs,
        )
        if fig_return:
            return df_pvals, result
        return df_pvals

    _plot_pvals(
        df_pvals, stat_method, figsize=figsize, fmt=fmt, annot=annot,
        ax=ax, alpha=alpha, annot_rot=annot_rot, annot_size=annot_size,
        **kwargs,
    )
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
        *,
        engine=None,
        fig_return=False,
        **kwargs,
):
    """
    Compute and plot p-values for pairwise correlations between
    categorical columns.

    Parameters
    ----------
    df : DataFrame
       Input DataFrame containing categorical columns.
    cat_cols1, cat_cols2 : list, optional
       Column names for the first and second set of variables.
    figsize : tuple, optional
       Figure size (matplotlib only).
    method : {'auto', 'fisher', 'chi2'}, default='auto'
       Statistical method to use.
    fmt : str, default=".2f"
       String formatting for displayed p-values.
    annot : bool, default=True
       If True, display p-values on the heatmap.
    alpha : float, default=0.05
       Significance level for highlighting values.
    annot_rot : int, default=0
       Rotation angle for annotations (matplotlib only).
    annot_size : int, optional
       Font size for annotations (matplotlib only).
    ax : matplotlib.axes.Axes, optional
       Axis object to plot the heatmap (matplotlib only).
    engine : {"matplotlib", "plotly"} or None, default=None
       The rendering backend.
    fig_return : bool, optional, default=False
       If True and engine is plotly, returns ``(df_pvals, fig)``.
    **kwargs
       Additional keyword arguments passed to the heatmap.

    Returns
    -------
    DataFrame
       A DataFrame containing p-values.  When ``engine="plotly"`` and
       ``fig_return=True``, returns ``(df_pvals, fig)``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.multfeats import pvals_cat
    >>> df = pd.DataFrame({
    ...     'A': np.random.choice(['Red','Blue','Green'], size=10),
    ...     'B': np.random.choice(['Yes','No'], size=10),
    ... })
    >>> pvals_cat(df)
    """
    engine = _resolve_engine(engine)

    if method == 'auto':
        stat_method = "Fisher's Exact or Chi-squared Test"
    elif method == 'fisher':
        stat_method = "Fisher's Exact Test"
    elif method == 'chi2':
        stat_method = 'Chi-squared Test'
    else:
        raise ValueError(
            "Invalid `method`. Choose 'auto', 'fisher', or 'chi2'."
        )

    cols = df.columns
    cat_cols1 = cat_cols1 or cols
    cat_cols2 = cat_cols2 or cols

    df_pvals = pd.DataFrame(
        index=cat_cols1, columns=cat_cols2, dtype="float64"
    )
    for cat_col1 in cat_cols1:
        for cat_col2 in cat_cols2:
            if cat_col1 == cat_col2:
                p_value = 0.
            else:
                _, p_value, _ = chi2_fisher_by_cat(
                    df, cat_col1, cat_col2, method=method
                )
            df_pvals.loc[cat_col1, cat_col2] = p_value

    if engine == "plotly":
        result = _plot_pvals_plotly(
            df_pvals, stat_method, alpha=alpha, annot=annot, fmt=fmt,
            **kwargs,
        )
        if fig_return:
            return df_pvals, result
        return df_pvals

    _plot_pvals(
        df_pvals, stat_method, figsize=figsize, fmt=fmt, annot=annot,
        ax=ax, alpha=alpha, annot_rot=annot_rot, annot_size=annot_size,
        **kwargs,
    )
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
    *,
    engine=None,
    fig_return=False,
    **kwargs,
):
    """
    Compute Mann-Whitney or Kruskal-Wallis p-values between numerical
    columns grouped by categorical columns.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing both numerical and categorical
        columns.
    cat_cols : list
        List of categorical column names.
    num_cols : list
        List of numerical column names.
    alpha : float, default=0.05
        Significance level for highlighting values.
    figsize : tuple, optional
        Figure size (matplotlib only).
    fmt : str, default=".2f"
        String formatting for displayed p-values.
    method : {'auto', 'mw', 'kruskal'}, default='auto'
        Statistical test to use.
    is_T : bool, default=False
        If True, transpose the result DataFrame.
    annot : bool, default=True
        If True, display p-values on the heatmap.
    annot_rot : int, default=0
        Rotation angle for annotations (matplotlib only).
    annot_size : int, optional
        Font size for annotations (matplotlib only).
    ax : matplotlib.axes.Axes, optional
        Axis object to plot the heatmap (matplotlib only).
    engine : {"matplotlib", "plotly"} or None, default=None
        The rendering backend.
    fig_return : bool, optional, default=False
       If True and engine is plotly, returns ``(df_pvals, fig)``.
    **kwargs
        Additional keyword arguments passed to the heatmap.

    Returns
    -------
    DataFrame
       A DataFrame containing p-values.  When ``engine="plotly"`` and
       ``fig_return=True``, returns ``(df_pvals, fig)``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pltstat.multfeats import pvals_num_cat
    >>> df = pd.DataFrame({
    ...     'Category': np.random.choice(['A','B'], size=10),
    ...     'Value1': np.random.rand(10),
    ...     'Value2': np.random.rand(10)
    ... })
    >>> pvals_num_cat(df, cat_cols=['Category'],
    ...               num_cols=['Value1','Value2'], method='mw')
    """
    engine = _resolve_engine(engine)

    if method == 'auto':
        stat_func = lambda n: (mannwhitneyu_by_cat if n == 2
                               else kruskal_by_cat)
    elif method == 'mw':
        stat_func = lambda n: mannwhitneyu_by_cat
    elif method == 'kruskal':
        stat_func = lambda n: kruskal_by_cat

    df_pvals = pd.DataFrame(
        index=cat_cols, columns=num_cols, dtype="float64"
    )
    for cat_col in cat_cols:
        for num_col in num_cols:
            df_subset = df[[cat_col, num_col]].dropna()
            n_cats = df_subset.loc[:, cat_col].nunique()

            if n_cats < 2:
                p = np.nan
            else:
                p = stat_func(n_cats)(df_subset, cat_col, num_col)[1]

            df_pvals.loc[cat_col, num_col] = p

    if is_T:
        df_pvals = df_pvals.T

    stat_method = {
        "mw": "Mann-Whitney U Test",
        "kruskal": "Kruskal-Wallis Test",
        "auto": "Auto Mann-Whitney U or Kruskal-Wallis Test"
    }[method]

    if engine == "plotly":
        result = _plot_pvals_plotly(
            df_pvals, stat_method, alpha=alpha, annot=annot, fmt=fmt,
            **kwargs,
        )
        if fig_return:
            return df_pvals, result
        return df_pvals

    _plot_pvals(
        df_pvals, stat_method, figsize=figsize, fmt=fmt, annot=annot,
        ax=ax, alpha=alpha, annot_rot=annot_rot, annot_size=annot_size,
        **kwargs,
    )
    return df_pvals


def _plot_pvals(df_pvals, stat_method, figsize=None, fmt=".2f",
                annot=True, ax=None, alpha=0.5, annot_rot=0,
                annot_size=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    cmap, cbar_kws = cm.get_pval_legend_thr_cmap(alpha=alpha)
    sns.heatmap(
        df_pvals, vmin=0, vmax=1, cmap=cmap, annot=annot, fmt=".2f",
        linewidths=1, cbar_kws=cbar_kws,
        annot_kws={"rotation": annot_rot, "fontsize": annot_size},
        ax=ax, **kwargs,
    )
    ax.set_title(f"{stat_method} p-value")


def _plot_pvals_plotly(df_pvals, stat_method, alpha=0.05, annot=True,
                        fmt=".2f", **kwargs):
    _ensure_plotly()
    colorscale, vmin, vmax, cbar_ticks = cm.get_pval_colorscale(
        alpha=alpha
    )
    return _heatmap_plotly(
        df_pvals, colorscale, vmin, vmax,
        f"{stat_method} p-value", annot=annot, fmt=fmt, **kwargs,
    )


# ====================================================================
# phik_corrs
# ====================================================================

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
    *,
    engine=None,
    heatmap_kwargs=None,
    phik_kwargs=None,
):
    """
    Plot Heatmap with Phik correlations between specific x and y
    lists of columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data for correlation calculation.
    interval_cols : list of str, optional
        Columns to treat as interval variables for Phik calculation.
    x : list, optional
        Columns for the x-axis in the correlation matrix.
    y : list, optional
        Columns for the y-axis in the correlation matrix.
    threshold : float, optional, default=0.8
        The threshold value for displaying Phik correlation values.
    annot : bool, optional, default=True
        If True, annotate the cells.
    fmt : str, optional, default=".2f"
        Format for displaying correlation values.
    figsize : tuple, optional
        Figure size (matplotlib only).
    annot_rot : int, optional, default=0
        Rotation angle for annotations (matplotlib only).
    annot_size : int, optional
        Font size for annotations (matplotlib only).
    ax : matplotlib.axes.Axes or None, optional, default=None
        The axes to draw on (matplotlib only).
    bins : int, optional, default=10
        Number of bins for discretizing continuous variables.
    njobs : int, optional, default=-1
        Number of parallel jobs for the Phik calculation.
    engine : {"matplotlib", "plotly"} or None, default=None
       The rendering backend.
    heatmap_kwargs : dict, optional
        Additional keyword arguments for ``sns.heatmap`` or
        ``go.Heatmap``.
    phik_kwargs : dict, optional
        Additional keyword arguments for ``phik_matrix``.

    Returns
    -------
    plotly.graph_objects.Figure or None

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pltstat.multfeats import phik_corrs
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     'age': np.random.randint(18, 70, size=100),
    ...     'income': np.random.randint(20000, 100000, size=100),
    ... })
    >>> phik_corrs(df, figsize=(8, 6))
    """
    engine = _resolve_engine(engine)
    phik_kwargs = phik_kwargs or {}
    heatmap_kwargs = heatmap_kwargs or {}

    if (x is not None) and (y is not None):
        xy = np.concatenate((x, y))
        xy = np.unique(xy)
        df_phik = df[xy].phik_matrix(
            interval_cols=interval_cols, bins=bins,
            njobs=njobs, **phik_kwargs,
        )
        df_phik = df_phik.loc[y, x]
    else:
        df_phik = df.phik_matrix(
            interval_cols=interval_cols, bins=bins,
            njobs=njobs, **phik_kwargs,
        )

    if engine == "plotly":
        colorscale, vmin, vmax = cm.get_corr_colorscale(
            threshold=threshold, vmin=0
        )
        return _heatmap_plotly(
            df_phik, colorscale, vmin, vmax, "Phi Coefficient (phik)",
            annot=annot, fmt=fmt, **heatmap_kwargs,
        )

    # matplotlib
    cmap = cm.get_corr_thr_cmap(threshold=threshold, vmin=0)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df_phik, cmap=cmap, vmin=0, vmax=1, annot=annot, fmt=fmt,
        annot_kws={"rotation": annot_rot, "fontsize": annot_size},
        ax=ax, **heatmap_kwargs,
    )
    return None