"""
Includes methods for calculating correlation matrices and related statistical relationships.
"""

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu

from sklearn.metrics import matthews_corrcoef

import warnings


def matthews(x, y):
    """
    Calculate the Matthews correlation coefficient (MCC) for binary categorical variables.

    The Matthews correlation coefficient is a measure of the strength of association
    between two binary categorical variables.

    Parameters
    ----------
    x : array-like
       First binary categorical feature.
    y : array-like
       Second binary categorical feature.

    Returns
    -------
    corr_matthews : float
       The Matthews correlation coefficient, ranging from -1 (perfect negative correlation)
       to +1 (perfect positive correlation). A value of 0 indicates no correlation.

    Raises
    ------
    ValueError
       If either `x` or `y` has more than two unique values or contains fewer than two unique values.

    Notes
    -----
    The function removes missing values before computing the correlation.

    Examples
    --------
    >>> import numpy as np
    >>> from pltstat.stat_methods import matthews
    >>> x = np.array(["yes", "no", "yes", "yes", "no", "no"])
    >>> y = np.array(["no", "no", "yes", "yes", "no", "no"])
    >>> matthews(x, y)
    # np.float64(0.7071067811865476)
    """
    df = pd.DataFrame({"x": x, "y": y})
    df = df.dropna()

    x_unique = np.sort(df["x"].unique())
    y_unique = np.sort(df["y"].unique())

    if (len(x_unique) != 2) or (len(y_unique) != 2):
        raise ValueError("Matthews correlation coefficient can only be calculated for binary categorical variables.")

    x_mapped = df["x"].map({x_unique[0]: 0, x_unique[1]: 1})
    y_mapped = df["y"].map({y_unique[0]: 0, y_unique[1]: 1})
    corr_matthews = matthews_corrcoef(x_mapped, y_mapped)

    return corr_matthews


def cramer_v_by_obs(obs):
    """
    Compute the Cramér's V correlation coefficient from a contingency table.

    Cramér's V measures the strength of association between two categorical variables,
    based on the chi-squared statistic. It is suitable for contingency tables
    of any size and ranges between 0 (no association) and 1 (perfect association).

    Parameters
    ----------
    obs : array-like of shape (n_rows, n_columns)
        A contingency table (crosstab) of counts. It can be created using
        `pandas.crosstab` or other similar methods.
        For example:
        >>> obs = pd.crosstab(df["A"], df["B"])

    Returns
    -------
    corr_cramer_v : float
        The Cramér's V correlation coefficient, ranging between 0 and 1.

    Notes
    -----
    - The chi-squared statistic is computed using `scipy.stats.chi2_contingency`
      without Yates' correction.
    - Cramér's V is normalized by the minimum of the number of rows and columns
      in the contingency table minus 1 (`min_dim = min(obs.shape) - 1`).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import chi2_contingency
    >>> obs = pd.DataFrame([[10, 20], [20, 40]])
    >>> cramer_v_by_obs(obs)
    0.0

    >>> obs = pd.DataFrame([[50, 0], [0, 50]])
    >>> cramer_v_by_obs(obs)
    1.0
    """
    chi2 = stats.chi2_contingency(obs, correction=False)[0]
    n = obs.sum(axis=0).sum(axis=0)
    min_dim = min(obs.shape) - 1
    if min_dim == 0:
        return 0.0
    corr_cramer_v = np.sqrt((chi2 / n) / min_dim)
    corr_cramer_v = float(corr_cramer_v)
    return corr_cramer_v


def cramer_v(data1, data2):
    """
     Compute the Cramér's V correlation coefficient between two categorical variables.

     Cramér's V measures the association between two categorical variables.
     It is based on the chi-squared statistic and ranges between 0 (no association)
     and 1 (perfect association).

     Parameters
     ----------
     data1 : array-like
         The first categorical variable. Can be a list, NumPy array, pandas Series, or similar.
     data2 : array-like
         The second categorical variable. Must have the same length as `data1`.

     Returns
     -------
     corr_cramer_v : float
         The Cramér's V correlation coefficient, ranging between 0 and 1.

     Notes
     -----
     - The function internally uses a contingency table created with `pandas.crosstab`
       to calculate the chi-squared statistic.
     - This implementation relies on an auxiliary function `cramer_v_by_obs`,
       which computes the Cramér's V given a contingency table.

     Examples
     --------
     >>> import pandas as pd
     >>> data1 = ['A', 'A', 'B', 'B', 'C', 'C']
     >>> data2 = ['X', 'Y', 'X', 'Y', 'X', 'Y']
     >>> cramer_v(data1, data2)
     0.0

     >>> data1 = ['A', 'A', 'A', 'B', 'B', 'C']
     >>> data2 = ['X', 'X', 'Y', 'X', 'Y', 'Y']
     >>> cramer_v(data1, data2)
     0.4714045207910317
     """
    obs = pd.crosstab(data1, data2)
    corr_cramer_v = cramer_v_by_obs(obs)
    return corr_cramer_v


def mannwhitneyu_by_cat(df, cat_feat, num_feat):
    """
    Perform the Mann-Whitney U test for a numeric variable across two categorical groups.

    This function tests whether the distributions of a numeric feature differ
    between two groups defined by a categorical feature.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the categorical and numeric features.
    cat_feat : str
        The name of the categorical feature with exactly two unique categories.
    num_feat : str
        The name of the numeric feature to compare across the two categories.

    Returns
    -------
    statistic : float
        The Mann-Whitney U statistic.
    p_value : float
        The p-value of the Mann-Whitney U test, indicating the likelihood
        that the two distributions are from the same population.
        Returns NaN if `cat_feat` does not contain exactly two unique categories.

    Raises
    ------
    UserWarning
        If `cat_feat` does not have exactly two unique categories.

    Notes
    -----
    - The test is non-parametric and does not assume normality of the numeric variable.
    - Missing values are removed before performing the test.

    Examples
    --------
    >>> import pandas as pd
    >>> from stat_methods import mannwhitneyu_by_cat
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     "group": np.random.choice(["A", "B"], size=10),
    ...     "value": np.random.randn(10)
    ... })
    >>> mannwhitneyu_by_cat(df, "group", "value")
    # np.float64(0.8333333333333333)
    """
    df_subset = df[[cat_feat, num_feat]].dropna()
    if df_subset[cat_feat].nunique() != 2:
        warnings.warn(f"Feature `{cat_feat}` does not have exactly two unique categories. Returning NaN.", UserWarning)
        return np.nan, np.nan

    x = df.groupby(cat_feat)[num_feat].agg(list).to_numpy()
    statistic, p_value = mannwhitneyu(*x)

    # x = df[df[cat_feat] == df[cat_feat].unique()[0]][num_feat]
    # y = df[df[cat_feat] == df[cat_feat].unique()[1]][num_feat]
    # p_value = mannwhitneyu(x, y)[1]

    return statistic, p_value


def kruskal_by_cat(df, cat_feat, num_feat):
    """
    Perform the Kruskal-Wallis H test to compare distributions of a numerical feature across multiple categories.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the categorical and numerical features.
    cat_feat : str
        Name of the categorical feature with at least two unique categories.
    num_feat : str
        Name of the numerical feature to compare.

    Returns
    -------
    statistic : float
        The Kruskal-Wallis H statistic, corrected for ties.
    p_value : float
        p-value from the Kruskal-Wallis H test. Returns NaN if the categorical feature has fewer than two unique categories.

    Raises
    ------
    UserWarning
        If the categorical feature has fewer than two unique categories.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from stat_methods import kruskal_by_cat
    >>> data = pd.DataFrame({
    ...     "group": ["A", "A", "B", "B", "B", "C", "C"],
    ...     "value": [3.2, 3.8, 2.1, 2.5, 2.8, 4.0, 4.2]
    ... })
    >>> kruskal_by_cat(data, "group", "value")
    # np.float64(0.06866117151308508)
    """
    df_subset = df[[cat_feat, num_feat]].dropna()
    if df_subset[cat_feat].nunique() < 2:
        warnings.warn(f"Feature `{cat_feat}` has less than two unique categories. Returning NaN.", UserWarning)
        return np.nan, np.nan

    x = df.groupby(cat_feat)[num_feat].agg(list).to_numpy()
    statistic, p_value = kruskal(*x)

    return statistic, p_value


def chi2_fisher_by_cat(df, cat_feat1, cat_feat2, method="auto"):
    """
    Perform Fisher's exact test or chi-squared test for two categorical features.

    Tests the independence of two categorical variables using either Fisher's
    exact test (suitable for small samples) or Pearson's chi-squared test
    (suitable for larger samples).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the categorical features.
    cat_feat1 : str
        The name of the first categorical feature.
    cat_feat2 : str
        The name of the second categorical feature.
    method : {'auto', 'fisher', 'chi2'}, default='auto'
        The statistical test to use. If 'auto', Fisher's exact test is used
        when any cell count in the contingency table is less than 5;
        otherwise, the chi-squared test is used.

    Returns
    -------
    statistic : float
        The test statistic: odds ratio for Fisher's exact test, chi-squared
        statistic for the chi-squared test.
    p_value : float
        The p-value of the test.
    method : {'fisher', 'chi2'}
        The test actually performed. When ``method='auto'`` is passed in, this
        reports which test was selected based on the cell counts; otherwise it
        echoes the requested ``method``. Useful for callers that need to label
        the test (e.g. in a plot title).

    Notes
    -----
    - ``scipy.stats.fisher_exact`` supports tables of any size (R×C) since
      scipy 1.11; no external package is required for exact inference.
    - Missing values are removed before constructing the contingency table.

    Examples
    --------
    >>> import pandas as pd
    >>> from pltstat.stat_methods import chi2_fisher_by_cat
    >>> data = pd.DataFrame({
    ...     "A": ["x", "x", "y", "y"],
    ...     "B": ["a", "b", "a", "b"],
    ... })
    >>> chi2_fisher_by_cat(data, "A", "B", method="auto")
    # (1.0, 1.0, 'fisher')
    """
    df_subset = df[[cat_feat1, cat_feat2]].dropna()
    crosstab_df = pd.crosstab(df_subset[cat_feat1], df_subset[cat_feat2])

    if method == "auto":
        # Use Fisher's exact test when any cell count is small (< 5);
        # otherwise fall back to the chi-squared test.
        method = "fisher" if crosstab_df.min(axis=None) < 5 else "chi2"

    if method == "fisher":
        statistic, p_value = fisher_exact(crosstab_df)
    elif method == "chi2":
        statistic, p_value = chi2_contingency(crosstab_df)[:2]
    else:
        raise ValueError(f"`method` must be 'auto', 'fisher', or 'chi2', but got {method}.")

    return statistic, p_value, method


def kde_curve(values, clip=None, n_points=200, scale=1.0):
    """
    Evaluate a gaussian kernel density estimate on a regular grid.

    The function returns the curve as arrays instead of drawing it, so that it
    can be rendered by any plotting engine.

    Parameters
    ----------
    values : array-like
        Sample to estimate the density of. Missing values are removed.
    clip : tuple[float, float] or None, default: None
        Lower and upper bound of the grid. None uses the range of `values`.
    n_points : int, default: 200
        Number of points of the grid.
    scale : float, default: 1.0
        Factor applied to the density. Use the bin width times the number of
        observations to overlay the curve on a histogram of counts.

    Returns
    -------
    grid : np.ndarray
        Points at which the density is evaluated.
    density : np.ndarray
        Estimated density multiplied by `scale`.

    Raises
    ------
    ValueError
        If `values` has fewer than two distinct observations, because the
        kernel bandwidth is then undefined.

    Notes
    -----
    The estimate uses Scott's rule for the bandwidth, which is the default of
    :class:`scipy.stats.gaussian_kde` and of the seaborn density plots.

    Examples
    --------
    >>> import numpy as np
    >>> from pltstat.stat_methods import kde_curve
    >>> grid, density = kde_curve(np.array([1.0, 2.0, 2.0, 3.0]), n_points=5)
    >>> grid
    array([1. , 1.5, 2. , 2.5, 3. ])
    >>> bool(density.argmax() == 2)
    True
    """
    values = pd.Series(values).dropna().to_numpy(dtype=float)

    if np.unique(values).size < 2:
        raise ValueError(
            "`values` must contain at least two distinct observations "
            "to estimate a density."
        )

    if clip is None:
        low, high = values.min(), values.max()
    else:
        low, high = clip

    grid = np.linspace(low, high, n_points)
    density = stats.gaussian_kde(values)(grid) * scale

    return grid, density
