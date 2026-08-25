# Changelog

All notable changes to this project will be documented in this file.

## [0.11.0] - 2025-08-25
### Added
- Dual rendering backend: every visualization function can now draw through
  either plotly (default) or matplotlib.
- Global `pltstat.set_backend("plotly"|"matplotlib")` and
  `pltstat.get_backend()` for switching the default engine.
- Per-call `engine=` keyword-only parameter on all plotting functions to
  override the global backend for a single call.
- `pltstat.config` singleton (`config.engine`) as the single source of truth
  for the active backend.
- `cm.get_pval_colorscale` and `cm.get_corr_colorscale` adapters that mirror
  the matplotlib colormaps as plotly-compatible colorscales.
- `in_out.save` polymorphic saver supporting both matplotlib and plotly
  figures (HTML and image formats).
- `plotly` added as a runtime dependency.

### Changed
- Default backend is now plotly; `set_backend("matplotlib")` restores the
  previous behavior.  The matplotlib backend preserves the exact return
  behavior of v0.10.0.
- `__version__` bumped to 0.11.0.

## [0.10.0] - 2025-03-16
### Improved
- Fisher's exact test no longer requires the installation of the R language for functionality.
- Refactored the project structure for improved maintainability.

## [0.9.7] - 2025-03-03
### First stable release
- Initial fully functional version with correct statistical computations.
