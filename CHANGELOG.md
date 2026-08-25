# Changelog

All notable changes to this project will be documented in this file.

## [0.10.1] - 2026-08-25
### Fixed
- p-value cmap now uses a discrete BoundaryNorm instead of a continuous
  LinearSegmentedColormap, which removes the blended color transition at the
  alpha threshold.
- Cramér's V returns 0.0 when the crosstab has a single column or row,
  avoiding a division by zero.

### Improved
- Added color_signif and color_non_signif options to get_pval_legend_thr_cmap
  and to the pvals_num, pvals_cat and pvals_num_cat heatmaps.

## [0.10.0] - 2025-03-16
### Improved
- Fisher's exact test no longer requires the installation of the R language for functionality.
- Refactored the project structure for improved maintainability.

## [0.9.7] - 2025-03-03
### First stable release
- Initial fully functional version with correct statistical computations.
