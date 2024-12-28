"""Functions for save and load files"""
import os


def save_plt(fig, filepath, dpi=None):
    """
    Save a figure to a file if file is not exist. If file exists - resave it

    Parameters
    ----------
    fig: plt.figure
        Matplotlib figure to save into a file
    filepath: str or path-like or binary file-like
        Path to save the figure
    dpi: int or float or :class:`plt.figure` or None, default: :rc:`savefig.dpi`
        The resolution in dots per inch. If 'plt.figure', use the figure's dpi value.
        If None, ignore `dpi`.
    """
    if os.path.isfile(filepath):
        os.remove(filepath)
    if dpi is not None:
        fig.savefig(filepath, dpi=dpi)
    else:
        fig.savefig(filepath)
