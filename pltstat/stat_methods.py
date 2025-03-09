"""
Includes methods for calculating correlation matrices and related statistical relationships.
"""

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import kruskal, mannwhitneyu

from sklearn.metrics import matthews_corrcoef

import warnings

# Fisher exact test from R:
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

numpy2ri.activate()
_stats_r = importr("stats")


def fisher_test(obs, alpha=0.05):
    """
        Perform Fisher's Exact Test on a 2x2 contingency table.

        Parameters
        ----------
        obs : pd.DataFrame or np.ndarray
            2x2 contingency table with observed frequencies.
        alpha : float, optional
            Significance level for the confidence interval (default is 0.05).

        Returns
        -------
        float
            p-value from Fisher's Exact Test.

        Notes
        --------
        This implementation uses the `fisher_test` function from R's 'stats' package,
        which works for any MxN table but here is specifically applied to 2x2 tables.

        Examples
        --------
        # Example usage:
        >>> import pandas as pd
        >>> data = {'Category A': [30, 10],
        >>>         'Category B': [20, 40]}
        >>> obs = pd.DataFrame(data)
        >>> fisher_test(obs)
        # np.float64(8.308658706239191e-05)

        Notes
        -----
        The Fisher's Exact Test is used for categorical data to determine
        if there are nonrandom associations between two categorical variables.
        """
    # Python Scipy realization (only 2x2 table):
    # g, p_value = _fisher_exact(crosstab_df)
    # R language Stats realization (MxN table):
    obs = obs.values
    p_value = _stats_r.fisher_test(obs, conf_int=True, conf_level=alpha)[0][0]

    return None, p_value


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
     1.0

     >>> data1 = ['A', 'A', 'A', 'B', 'B', 'C']
     >>> data2 = ['X', 'X', 'Y', 'X', 'Y', 'Y']
     >>> cramer_v(data1, data2)
     0.6454972243679028
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
    float
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
        return np.nan

    x = df.groupby(cat_feat)[num_feat].agg(list).to_numpy()
    p_value = mannwhitneyu(*x)[1]

    # x = df[df[cat_feat] == df[cat_feat].unique()[0]][num_feat]
    # y = df[df[cat_feat] == df[cat_feat].unique()[1]][num_feat]
    # p_value = mannwhitneyu(x, y)[1]

    return p_value


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
    float
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
        return np.nan

    x = df.groupby(cat_feat)[num_feat].agg(list).to_numpy()
    p_value = kruskal(*x)[1]

    return p_value

# TODO: Fisher?
# p_value = _stats_r.fisher_test(crosstab_df.values)[0][0]
# g, p_value = chi2_contingency(crosstab_df)[:2]
# # Perform Fisher exact test with confidence interval = 0.05
# result = stats_r.fisher_test(crosstab, conf_int=True, conf_level=0.95)
#
# # Extract the p-value and confidence interval from the result
# p_value = result.rx2("p.value")[0]
# conf_int = result.rx2("conf.int")
#
# # Calculate Pearson correlation coefficient
# # Convert crosstab to a flattened list for correlation computation
# # You may need to format the data differently based on your specific case
# flattened_data = crosstab.flatten()
# x = [0, 0, 1, 1]  # Corresponds to rows
# y = [0, 1, 0, 1]  # Corresponds to columns
# weights = flattened_data
#
# # Weighted Pearson correlation
# correlation_coefficient, _ = pearsonr(
#     np.repeat(x, weights), np.repeat(y, weights)
# )
#
# print("P-value (Fisher Test):", p_value)
# print("Confidence Interval (Fisher Test):", list(conf_int))
# print("Correlation Coefficient (Pearson):", correlation_coefficient)
