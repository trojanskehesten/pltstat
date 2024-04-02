# TODO: Description
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

_HIGH=24


"""Return a list of the marginal sums of the array `a`.

    Parameters
    ----------
    a : ndarray
        The array for which to compute the marginal sums.

    Returns
    -------
    margsums : list of ndarrays
        A list of length `a.ndim`.  `margsums[k]` is the result
        of summing `a` over all axes except `k`; it has the same
        number of dimensions as `a`, but the length of each axis
        except axis `k` will be 1.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.contingency import margins

    >>> a = np.arange(12).reshape(2, 6)
    >>> a
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11]])
    >>> m0, m1 = margins(a)
    >>> m0
    array([[15],
           [51]])
    >>> m1
    array([[ 6,  8, 10, 12, 14, 16]])

"""





def rad2val(a, high=_HIGH):
    """Convert values `a` from radians to measures with the highest value `high`

    Parameters
    ----------
    a : array_like
        Input array in radians
    high : float or int
        High boundary for the sample range

    Returns
    -------
    rad2val : ndarray
        The corresponding values with `high` boundary. This is a scalar if `a` is a scalar.

    Example
    --------
    >>> import numpy as np
    >>> from pltstat.circle import rad2val

    >>> a = [0, np.pi, 4, 8]
    >>> rad2val(a, high=24)
    array([ 0., 12., 15.27887454, 30.55774907])
    """
    a = np.array(a)
    a = a / np.pi / 2 * high
    return a

def val2rad(a, high=_HIGH):
    """Convert values `a` from measures with the highest value `high` to radians

        Parameters
        ----------
        a : array_like
            Input array in measures with the `high` boundary fot the sample range
        high : float or int
            High boundary for the sample range

        Returns
        -------
        rad2val : ndarray
            The corresponding radian values. This is a scalar if `a` is a scalar.

        Example
        --------
        >>> import numpy as np
        >>> from pltstat.circle import val2rad

        >>> a = [0, 3, 19, 25]
        >>> val2rad(a, high=24)
        array([0., 0.78539816, 4.97418837, 6.54498469])

        >>> val2rad(a, high=24) / np.pi
        array([0., 0.25, 1.58333333, 2.08333333])
    """
    a = np.array(a)
    a = a / high * 2 * np.pi
    return a

def hist(
    a, n_bins, high=_HIGH, bottom=0.1, title=None, figsize=None, ax=None, return_ax=False, **kwargs,
):
    """
    Histogram for time data

    :param a: numpy array of hours
    :param n_bins: count of bins
    :param high: the highest possible value
    :param bottom: proportion
    :param title: title of histogram, is None - no title
    :param figsize: tuple
    :param ax: ax from matplotlib which is used for plottng. If None - function
      will create new ax
    :param return_ax: if True return ax, else return None
    :return: ax if return_ax is True, else - return None
    """

    a = a / high * 2 * np.pi  # h to rad

    def radian_function(x, y):
        rad_x = x / np.pi
        return f"{str(round(rad_x * 12, 1))} h"

    theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    bins = np.linspace(0, 2 * np.pi, n_bins + 1, endpoint=True)

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
        # ax = plt.subplot(111, polar=True)

    max_bin = np.histogram(a, bins=bins, **kwargs)[0].max()
    bottom = max_bin * bottom

    # get yticks:
    fig, ax2 = plt.subplots(subplot_kw={"projection": "polar"})
    ax2.hist(a, bins=bins, edgecolor="black")
    yticks = ax2.axes.yaxis.get_ticklocs()
    plt.close(fig)

    ax.hist(a, bins=bins, edgecolor="black", bottom=bottom, **kwargs)

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


def mean(a, high=_HIGH, atol=1e-10):
    """
    Get mean of numpy array with directional data
    :param a: list of values,
    :param high: the highest possible value,
    :param atol: tolerance of radius-vector calculation.
    If the radius is less than tolerance, it is impossible to calculate mean value.
    1e-10 is enough for getting the mean value with 1e-07 accuracy
    :return: circle mean value
    """
    # to rad
    a = np.array(a) / high * 2 * np.pi

    # cos, sin, radius and mean
    c = np.mean(np.cos(a))
    s = np.mean(np.sin(a))
    r = np.sqrt(c ** 2 + s ** 2)

    # if radius is 0 than it is not possible to calculate mean value
    # 1e-10 for 7 signs after dot accuracy
    if np.isclose(r, 0, atol=atol):
        return np.nan

    tau = np.arctan2(s, c)

    # return from rad
    tau = tau / np.pi / 2 * high
    if tau < 0:
        tau += high
    elif tau >= high:
        tau -= high
    return tau


def std(a, high=_HIGH):
    """
    Get std of numpy array with directional data
    :param a: list of values,
    :param high: the highest possible value,
    :return: circle std value
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
