# Changelog

All notable changes to this project will be documented in this file.

## [0.11.0] - 2026-08-30
### Added
- Dual rendering engine: every plotting function can now draw with matplotlib
  (the default) or with plotly.
- `pltstat.set_backend` and `pltstat.get_backend` to choose the engine used by
  default, and the `pltstat.backend` context manager to change it for a block.
- An `engine` parameter on every plotting function, which overrides the
  default engine for a single call.
- `cm.get_pval_thr_colorscale` and `cm.get_corr_thr_colorscale`, the plotly
  counterparts of the matplotlib colormaps, built from the same stops so both
  engines colour a heatmap alike. `cm.get_palette_hex` and `cm.format_matrix`
  support them.
- `stat_methods.kde_curve` to estimate a density as arrays, which lets both
  engines draw the same curve.
- `in_out.save` to save a figure of either engine. `in_out.save_plt` is
  unchanged.
- A `fig_return` parameter on `pvals_num`, `pvals_cat`, `pvals_num_cat` and
  `dist_qq_plot`, which returns the figure next to the data.
- `plotly` as a dependency.

### Fixed
- `cm.get_corr_thr_cmap` raised a `NameError` because
  `LinearSegmentedColormap` was not imported, which broke `phik_corrs` and
  `heatmap_corr` with a threshold.
- The `fmt` parameter of `pvals_num`, `pvals_cat` and `pvals_num_cat` was
  ignored and the annotations were always formatted with two decimals.
- A pie slice could be labelled with the count of another category when two
  categories had the same number of observations.
- `countplot` renamed the Series of the caller when it had no name.
- The name of the Mann-Whitney U-test used a non ASCII dash.

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
