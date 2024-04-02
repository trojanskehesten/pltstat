import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np


def time_hist(
    a, n_bins, bottom=0.1, title=None, figsize=None, ax=None, return_ax=False
):
    """
    Histogram for time data

    :param a: numpy array of hours
    :param n_bins: count of bins
    :param bottom: proportion
    :param title: title of histogram, is None - no title
    :param figsize: tuple
    :param ax: ax from matplotlib which is used for plottng. If None - function
      will create new ax
    :param return_ax: if True return ax, else return None
    :return: ax if return_ax is True, else - return None
    """

    a = a / 12 * np.pi  # h to rad

    def radian_function(x, y):
        rad_x = x / np.pi
        return f"{str(round(rad_x * 12, 1))} h"

    theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    bins = np.linspace(0, 2 * np.pi, n_bins + 1, endpoint=True)

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
        # ax = plt.subplot(111, polar=True)

    max_bin = np.histogram(a, bins=bins)[0].max()
    bottom = max_bin * bottom

    # get yticks:
    fig, ax2 = plt.subplots(subplot_kw={"projection": "polar"})
    ax2.hist(a, bins=bins, edgecolor="black")
    yticks = ax2.axes.yaxis.get_ticklocs()
    plt.close(fig)

    ax.hist(a, bins=bins, edgecolor="black", bottom=bottom)

    # arrange graph
    ax.set(
        theta_offset=np.pi / 2,
        theta_direction=-1,
        xticks=theta,
        yticks=yticks + bottom,
        yticklabels=yticks,
    )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(radian_function))
    ax.set_title(title)
    if return_ax:
        return ax
    #
    # Bars and colors:
    # Use custom colors and opacity
    # bottom = 0
    # max_height = 10
    # radii = max_height*np.random.rand(N)
    # width = (2*np.pi) / N
    # bars = ax.bar(theta, radii, width=width) #, bottom=bottom)
    # for r, bar in zip(radii, bars):
    #     bar.set_facecolor(plt.cm.jet(r / 10.))
    #     bar.set_alpha(0.8)


def mean_circle(a, high=24):
    """
    Get mean of numpy array with directional data
    """
    # to rad
    a = np.array(a) / high * 2 * np.pi

    # cos, sin, radius and mean
    c = np.mean(np.cos(a))
    s = np.mean(np.sin(a))
    r = np.sqrt(c ** 2 + s ** 2)

    # if radius is 0 than it is not possible to calculate mean value
    # 1e-10 for 7 signs after dot accuracy
    if np.isclose(r, 0, atol=1e-10):
        return np.nan

    tau = np.arctan2(s, c)

    # return from rad
    tau = tau / np.pi / 2 * high
    if tau < 0:
        tau += high
    elif tau >= high:
        tau -= high
    return tau


def std_circle(a, high=24):
    """
    Get std of numpy array with directional data
    """
    # to rad
    a = np.array(a) / high * 2 * np.pi

    # cos, sin, radius and mean
    c = np.mean(np.cos(a))
    s = np.mean(np.sin(a))
    r = np.sqrt(c ** 2 + s ** 2)

    # if radius is 0 than it is not possible to calculate mean value
    # 1e-10 for 7 signs after dot accuracy
    # if np.isclose(r, 0, atol=1e-10):
    #     return np.nan

    v = np.sqrt(-2 * np.log(r))

    # return from rad
    v = v / np.pi / 2 * high
    return v
