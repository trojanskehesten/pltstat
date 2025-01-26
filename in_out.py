"""Functions for save and load files"""
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
