# Author: Dmitrii Beregovoi
# License: BSD 3-Clause License, see LICENSE file

# Copyright: (c) 2025 Dmitrii Beregovoi

from .config import config, set_backend, get_backend

__version__ = "0.11.0"
__author__ = "Dmitrii Beregovoi"
__email__ = "dimaforth@gmail.com"
__all__ = [
    "__version__",
    "config",
    "set_backend",
    "get_backend",
    "stat_methods",
    "cm",
    "circle",
    "in_out",
    "singlefeat",
    "twofeats",
    "multfeats",
]