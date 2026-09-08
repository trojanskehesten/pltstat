"""Provides utilities for reading, writing, and preprocessing input and output data files."""

import os

from matplotlib.figure import Figure


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

    Notes
    -----
    This function only saves Matplotlib figures. Use :func:`save` to save a
    figure built by either engine.
    """
    if os.path.isfile(filepath):
        os.remove(filepath)

    fig.savefig(filepath, dpi=dpi)


def save(fig, filepath, dpi="figure", **kwargs):
    """
    Save a matplotlib or a plotly figure, overwriting the file if it exists.

    The engine is chosen from the type of ``fig``. Matplotlib figures are
    written with ``Figure.savefig``. Plotly figures are written with
    ``Figure.write_html`` when ``filepath`` ends with ".html" or ".htm", and
    with ``Figure.write_image`` otherwise.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The figure to save.
    filepath : str or path-like
        The path where the figure should be saved. If the file already
        exists, it will be overwritten.
    dpi : int, float or 'figure', default: 'figure'
        The resolution in dots per inch of a Matplotlib figure. For a plotly
        image it is turned into a scale factor of ``dpi / 100``; the value
        'figure' leaves the plotly default untouched.
    **kwargs
        Additional keyword arguments passed to ``write_html`` or to
        ``write_image`` for a plotly figure.

    Returns
    -------
    None
        The function writes the file and does not return any value.

    Raises
    ------
    TypeError
        If ``fig`` is neither a matplotlib nor a plotly figure.
    ImportError
        If a static image of a plotly figure is asked for and the kaleido
        package is not installed.

    Notes
    -----
    Plotly is never imported here, so this module can be used when only
    matplotlib is installed.

    Examples
    --------
    >>> from pltstat.in_out import save
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> save(fig, "plot.png", dpi=150)  # doctest: +SKIP
    """
    if os.path.isfile(filepath):
        os.remove(filepath)

    if isinstance(fig, Figure):
        fig.savefig(filepath, dpi=dpi)
        return

    # Duck typing keeps plotly out of the imports of this module
    if type(fig).__module__.startswith("plotly") and hasattr(fig, "write_html"):
        extension = os.path.splitext(str(filepath))[1].lower()
        if extension in (".html", ".htm"):
            fig.write_html(filepath, **kwargs)
            return

        if isinstance(dpi, (int, float)) and ("scale" not in kwargs):
            kwargs["scale"] = dpi / 100
        try:
            fig.write_image(filepath, **kwargs)
        except Exception as error:
            # Plotly reports a missing kaleido as a ValueError naming it
            if "kaleido" not in str(error).lower():
                raise
            raise ImportError(
                "Saving a plotly figure as an image requires the kaleido "
                "package. Install it with 'pip install kaleido', or save the "
                "figure as an '.html' file."
            ) from error
        return

    raise TypeError(
        "`fig` must be a matplotlib or a plotly figure, "
        f"but {type(fig).__name__} is given"
    )
