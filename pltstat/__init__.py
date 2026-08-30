# Author: Dmitrii Beregovoi
# License: BSD 3-Clause License, see LICENSE file

# Copyright: (c) 2025 Dmitrii Beregovoi

from . import config
from .config import backend, get_backend, set_backend

__version__ = "0.10.1"
__author__ = "Dmitrii Beregovoi"
__email__ = "dimaforth@gmail.com"
__all__ = [
    "__version__",
    "config",
    "backend",
    "get_backend",
    "set_backend",
    "stat_methods",
    "cm",
    "circle",
    "in_out",
    "singlefeat",
    "twofeats",
    "multfeats",
]