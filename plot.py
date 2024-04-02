from types import ModuleType

import os

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns

import umap.umap_ as umap

import numpy as np
import pandas as pd

# Fisher exact test from R:
# import rpy2.robjects.numpy2ri
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

from scipy import stats
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr

from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import matthews_corrcoef

numpy2ri.activate()
_stats_r = importr("stats")