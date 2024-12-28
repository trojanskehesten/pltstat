# Correlation calculation functions

import numpy as np
import pandas as pd

from scipy import stats


def cramer_v_by_obs(obs):
    """
    Compute the Cramer V correlation coefficient by crosstab between two features.

    Parameters
    ----------
    obs : array_like
        Counts crosstab of two features. It can be created by:
        >>> crosstab_df = pd.crosstab(df["A"], df["B"])

    Returns
    -------
    corr_cramer_v : float
        Cramer V Correlation coefficient
    """
    chi2 = stats.chi2_contingency(obs, correction=False)[0]
    n = np.sum(obs).sum()
    min_dim = min(obs.shape) - 1
    corr_cramer_v = np.sqrt((chi2 / n) / min_dim)
    return corr_cramer_v


def cramer_v(data1, data2):
    """
    Compute the Cramer V correlation coefficient by data of two features.

    Parameters
    ----------
    data1, data2 : array_like
        Values arrays of two features

    Returns
    -------
    corr_cramer_v : float
        Cramer V Correlation coefficient
    """
    obs = pd.crosstab(data1, data2)
    corr_cramer_v = cramer_v_by_obs(obs)
    return corr_cramer_v

# TODO: Fisher?
