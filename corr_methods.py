import numpy as np
import pandas as pd

from scipy import stats


def cramer_v_by_obs(obs):
    chi2 = stats.chi2_contingency(obs, correction=False)[0]
    n = np.sum(obs).sum()
    min_dim = min(obs.shape) - 1
    corr_cramer_v = np.sqrt((chi2 / n) / min_dim)
    return corr_cramer_v


def cramer_v(data1, data2):
    obs = pd.crosstab(data1, data2)
    corr_cramer_v = cramer_v_by_obs(obs)
    return corr_cramer_v

# TODO: Fisher?
