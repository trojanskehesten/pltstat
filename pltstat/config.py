"""
Provides the global configuration of the rendering engine used by every
plotting function of the package.

The engine is selected in two ways: globally with :func:`set_backend` or the
:func:`backend` context manager, and per call with the ``engine`` keyword
argument accepted by every plotting function.
"""

import warnings
from contextlib import contextmanager

ENGINES = ("matplotlib", "plotly")

_DEFAULT_ENGINE = "matplotlib"
_DEFAULT_AUTO_SHOW = True

_state = {
    "engine": _DEFAULT_ENGINE,
    "auto_show": _DEFAULT_AUTO_SHOW,
}

# Number of pixels per inch used to convert a matplotlib ``figsize`` into the
# ``width`` and ``height`` of a plotly figure.
_PIXELS_PER_INCH = 100


def _validate_engine(engine):
    """
    Validate a rendering engine name and return it in lower case.

    Parameters
    ----------
    engine : str
        Name of the engine. Must be "matplotlib" or "plotly".

    Returns
    -------
    engine : str
        The validated engine name in lower case.

    Raises
    ------
    ValueError
        If `engine` is not one of the supported engines.

    Examples
    --------
    >>> from pltstat.config import _validate_engine
    >>> _validate_engine("Plotly")
    'plotly'
    """
    if isinstance(engine, str) and engine.lower() in ENGINES:
        return engine.lower()

    engines_str = " or ".join(f"'{name}'" for name in ENGINES)
    raise ValueError(f"Invalid `engine`. Choose {engines_str}. But [{engine}] is given")


def set_backend(engine):
    """
    Set the rendering engine used by default by every plotting function.

    Parameters
    ----------
    engine : {"matplotlib", "plotly"}
        Name of the engine to use by default.

    Returns
    -------
    None
        The function updates the global configuration and returns nothing.

    Raises
    ------
    ValueError
        If `engine` is not one of the supported engines.
    ImportError
        If `engine` is "plotly" and the plotly package is not installed.

    Notes
    -----
    The ``engine`` keyword argument of a plotting function overrides this
    setting for a single call.

    Examples
    --------
    >>> import pltstat
    >>> pltstat.set_backend("matplotlib")
    >>> pltstat.get_backend()
    'matplotlib'
    """
    engine = _validate_engine(engine)
    if engine == "plotly":
        # Fail at configuration time rather than at the first plotting call
        _import_plotly()
    _state["engine"] = engine


def get_backend():
    """
    Get the rendering engine currently used by default.

    Returns
    -------
    engine : str
        Name of the engine, "matplotlib" or "plotly".

    Examples
    --------
    >>> import pltstat
    >>> pltstat.set_backend("matplotlib")
    >>> pltstat.get_backend()
    'matplotlib'
    """
    return _state["engine"]


@contextmanager
def backend(engine):
    """
    Temporarily use `engine` as the default rendering engine.

    Parameters
    ----------
    engine : {"matplotlib", "plotly"}
        Name of the engine to use inside the ``with`` block.

    Yields
    ------
    engine : str
        The validated engine name in lower case.

    Raises
    ------
    ValueError
        If `engine` is not one of the supported engines.
    ImportError
        If `engine` is "plotly" and the plotly package is not installed.

    Notes
    -----
    The previous engine is restored on exit, including when the block raises.
    The configuration is global, so this context manager is not thread safe.

    Examples
    --------
    >>> import pltstat
    >>> from pltstat import singlefeat as sf
    >>> with pltstat.backend("plotly"):
    ...     fig = sf.pie(df["group"])  # doctest: +SKIP
    """
    previous = _state["engine"]
    set_backend(engine)
    try:
        yield _state["engine"]
    finally:
        _state["engine"] = previous


def set_auto_show(value):
    """
    Set whether plotly figures are displayed when they are not returned.

    Parameters
    ----------
    value : bool
        If True, functions which return data instead of a figure call
        ``fig.show()`` on the plotly figure they build.

    Returns
    -------
    None
        The function updates the global configuration and returns nothing.

    Examples
    --------
    >>> import pltstat
    >>> pltstat.config.set_auto_show(False)
    >>> pltstat.config.get_auto_show()
    False
    >>> pltstat.config.set_auto_show(True)
    """
    _state["auto_show"] = bool(value)


def get_auto_show():
    """
    Get whether plotly figures are displayed when they are not returned.

    Returns
    -------
    auto_show : bool
        True when figures are displayed automatically.

    Examples
    --------
    >>> import pltstat
    >>> pltstat.config.get_auto_show()
    True
    """
    return _state["auto_show"]


def _resolve_engine(engine):
    """
    Get the engine of a single call, falling back to the global setting.

    Parameters
    ----------
    engine : str or None
        Value of the ``engine`` keyword argument of a plotting function.
        None means that the global setting is used.

    Returns
    -------
    engine : str
        Name of the engine to render with, in lower case.

    Raises
    ------
    ValueError
        If `engine` is neither None nor one of the supported engines.

    Examples
    --------
    >>> from pltstat.config import _resolve_engine
    >>> _resolve_engine("plotly")
    'plotly'
    >>> _resolve_engine(None)
    'matplotlib'
    """
    if engine is None:
        return _state["engine"]
    return _validate_engine(engine)


def _import_plotly():
    """
    Import plotly lazily and raise a helpful error when it is missing.

    Returns
    -------
    go : module
        The ``plotly.graph_objects`` module.
    make_subplots : callable
        The ``plotly.subplots.make_subplots`` function.

    Raises
    ------
    ImportError
        If the plotly package is not installed.

    Notes
    -----
    Importing plotly lazily keeps ``import pltstat`` cheap for users who only
    render with matplotlib.

    Examples
    --------
    >>> from pltstat.config import _import_plotly
    >>> go, make_subplots = _import_plotly()
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise ImportError(
            "The plotly engine requires the plotly package. "
            "Install it with 'pip install plotly'."
        ) from error

    return go, make_subplots


def _figsize_to_px(figsize):
    """
    Convert a matplotlib figure size in inches into plotly pixel sizes.

    Parameters
    ----------
    figsize : tuple of float or None
        Figure size as ``(width, height)`` in inches. None means that the
        plotly defaults are used.

    Returns
    -------
    width : int or None
        Width in pixels, or None when `figsize` is None.
    height : int or None
        Height in pixels, or None when `figsize` is None.

    Examples
    --------
    >>> from pltstat.config import _figsize_to_px
    >>> _figsize_to_px((18, 6))
    (1800, 600)
    >>> _figsize_to_px(None)
    (None, None)
    """
    if figsize is None:
        return None, None

    width, height = figsize
    return round(width * _PIXELS_PER_INCH), round(height * _PIXELS_PER_INCH)


def _warn_ignored_mpl_params(engine, **params):
    """
    Warn about the parameters which the active engine cannot honor.

    Parameters
    ----------
    engine : str
        Name of the engine in use, quoted in the warning message.
    **params : keyword arguments
        Mapping of a parameter name to the value it was given. A warning is
        emitted for every parameter whose value is neither None nor False.

    Returns
    -------
    None
        The function emits a warning and returns nothing.

    Notes
    -----
    A single warning lists every ignored parameter, so that one call never
    emits more than one warning.

    Examples
    --------
    >>> import warnings
    >>> from pltstat.config import _warn_ignored_mpl_params
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     _warn_ignored_mpl_params("plotly", ax=None, return_ax=False)
    >>> len(caught)
    0
    """
    ignored = sorted(
        name for name, value in params.items() if value is not None and value is not False
    )
    if not ignored:
        return

    names = ", ".join(f"`{name}`" for name in ignored)
    warnings.warn(
        f"{names} is ignored when `engine` is '{engine}'"
        if len(ignored) == 1
        else f"{names} are ignored when `engine` is '{engine}'",
        UserWarning,
        stacklevel=3,
    )
