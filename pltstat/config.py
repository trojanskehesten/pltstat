"""
Global configuration for the pltstat package.

Provides a singleton ``config`` object that controls the active rendering
backend (plotly or matplotlib), along with convenient ``set_backend`` /
``get_backend`` helpers exposed at the package level.

Notes
-----
The ``engine`` attribute defaults to ``"matplotlib"`` so that every
plotting function draws with matplotlib out of the box.  Users who prefer
plotly can set ``pltstat.config.engine = "plotly"`` or call
``pltstat.set_backend("plotly")`` once before any plotting calls.
Individual functions also accept a keyword-only ``engine`` parameter that
overrides the global setting for that single call.
"""

from __future__ import annotations

from typing import Literal

Backend = Literal["matplotlib", "plotly"]


class Config:
    """Global configuration singleton for pltstat.

    Attributes
    ----------
    engine : Backend
        The default rendering engine.  Must be ``"matplotlib"`` or
        ``"plotly"``.  Defaults to ``"matplotlib"``.
    """

    _engine: Backend

    def __init__(self, engine: Backend = "matplotlib") -> None:
        self._engine = engine

    @property
    def engine(self) -> Backend:
        """The default rendering backend."""
        return self._engine

    @engine.setter
    def engine(self, value: Backend) -> None:
        if value not in ("matplotlib", "plotly"):
            raise ValueError(
                f"engine must be 'matplotlib' or 'plotly', got {value!r}"
            )
        self._engine = value


#: Module-level singleton - the single source of truth for the active backend.
config = Config()


def set_backend(engine: Backend) -> None:
    """Set the default rendering backend for all subsequent plotting calls.

    Parameters
    ----------
    engine : {"matplotlib", "plotly"}
        The backend to use by default.

    Raises
    ------
    ValueError
        If *engine* is not one of the allowed values.

    Examples
    --------
    >>> from pltstat import set_backend, get_backend
    >>> set_backend("matplotlib")
    >>> get_backend()
    'matplotlib'
    >>> set_backend("plotly")
    """
    config.engine = engine


def get_backend() -> Backend:
    """Return the current default rendering backend.

    Returns
    -------
    Backend
        Either ``"matplotlib"`` or ``"plotly"``.

    Examples
    --------
    >>> from pltstat import get_backend
    >>> get_backend()
    'matplotlib'
    """
    return config.engine


def _resolve_engine(engine: Backend | None) -> Backend:
    """Resolve the effective engine from an optional per-call override.

    Parameters
    ----------
    engine : Backend or None
        The per-call ``engine`` keyword argument.  When ``None`` the global
        ``config.engine`` is used.

    Returns
    -------
    Backend
        The resolved backend string.
    """
    return config.engine if engine is None else engine


# --- Lazy plotly-import helpers used throughout the viz modules ----------

_PLOTLY_AVAILABLE: bool | None = None  # tri-state: None = unchecked


def _ensure_plotly() -> None:
    """Lazily import plotly to keep startup fast.

    Raises
    ------
    ImportError
        If plotly is not installed, with a hint to install it.
    """
    global _PLOTLY_AVAILABLE
    if _PLOTLY_AVAILABLE is None:
        try:
            import plotly  # noqa: F401
            _PLOTLY_AVAILABLE = True
        except ImportError:
            _PLOTLY_AVAILABLE = False
            raise ImportError(
                "plotly is required for the plotly backend. "
                "Install it with: pip install plotly"
            )