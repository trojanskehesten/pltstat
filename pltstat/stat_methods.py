"""
Includes methods for calculating correlation matrices and related statistical relationships.
"""

import numpy as np
import pandas as pd

from scipy import stats


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
