"""Provides utilities for reading, writing, and preprocessing input and output data files."""

import os


def save_plt(fig, filepath, dpi='figure'):
    """
    Save a Matplotlib figure to a file, overwriting the file if it already exists.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object to save.
    filepath : str or path-like or binary file-like
        The path or file-like object where the figure should be saved. If the file
        already exists, it will be overwritten.
    dpi: int, float or 'figure', default: rcParams["savefig.dpi"] (default: 'figure')
        The resolution in dots per inch. If 'figure', use the figure's dpi value.
    """
    if os.path.isfile(filepath):
        os.remove(filepath)

    fig.savefig(filepath, dpi=dpi)


def save(fig, filepath, dpi='figure', **kwargs):
    """
    Save a matplotlib or plotly figure to a file.

    The backend is inferred from the figure type.  Matplotlib figures are
    saved via ``fig.savefig``; plotly figures use ``write_html`` for ``.html``
    files and ``write_image`` (requires ``kaleido``) for image formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The figure object to save.
    filepath : str or path-like
        Path where the figure should be saved.  If the file already exists
        it will be overwritten.
    dpi : int, float or 'figure', default: 'figure'
        Resolution for matplotlib figures only.  Ignored for plotly.
    **kwargs
        Passed through to ``fig.write_html`` or ``fig.write_image`` for
        plotly figures.

    Raises
    ------
    TypeError
        If *fig* is neither a matplotlib nor a plotly figure.
    ImportError
        If *fig* is a plotly figure and ``kaleido`` is needed but not
        installed (for non-HTML formats).

    Examples
    --------
    >>> from pltstat.in_out import save
    >>> # Matplotlib figure
    >>> import matplotlib.pyplot as plt
    >>> fig, _ = plt.subplots()
    >>> save(fig, "plot.png", dpi=150)
    >>>
    >>> # Plotly figure
    >>> import plotly.express as px
    >>> fig = px.scatter(x=[1, 2], y=[3, 4])
    >>> save(fig, "chart.html")
    """
    if os.path.isfile(filepath):
        os.remove(filepath)

    # matplotlib path (no lazy import needed, always available)
    try:
        from matplotlib.figure import Figure
        if isinstance(fig, Figure):
            fig.savefig(filepath, dpi=dpi)
            return
    except ImportError:
        pass

    # --- plotly path ----------------------------------------------------
    try:
        import plotly.graph_objects as go
        if isinstance(fig, go.Figure):
            ext = os.path.splitext(str(filepath))[1].lower()
            if ext in (".html", ".htm"):
                fig.write_html(str(filepath), **kwargs)
            else:
                fig.write_image(str(filepath), **kwargs)
            return
    except ImportError:
        pass

    raise TypeError(
        f"Unsupported figure type: {type(fig).__name__}. "
        "Expected matplotlib.figure.Figure or plotly.graph_objects.Figure."
    )
