import os


def save_plt(fig, filepath, dpi=None):
    """
    Save a figure to a file if file is not exist. If file exists - resave it

    :param fig: figure of plt
    :param filepath: string
    :param dpi: integer
    :return: None
    """
    if os.path.isfile(filepath):
        os.remove(filepath)
    if dpi is not None:
        fig.savefig(filepath, dpi=dpi)
    else:
        fig.savefig(filepath)
