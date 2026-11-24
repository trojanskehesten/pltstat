"""
Contains functions and methods related to circular statistical visualizations,
such as radar charts or circular histograms.
"""

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import seaborn as sns

from .config import _figsize_to_px, _import_plotly, _resolve_engine, _warn_ignored_mpl_params


_HIGH = 24
_N_BINS = 24
_N_TICKS = 24


def rad2val(a, high=_HIGH):
    """
    Convert values ``a`` from radians to measures with the highest value ``high``

    Parameters
    ----------
    a : array_like
        Input array in radians
    high : float or int
        High boundary for the sample range

    Returns
    -------
    rad2val : ndarray
        The corresponding values with `high` boundary. This is a scalar if ``a`` is a scalar.

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
    """
    Convert values `a` from measures with the highest value `high` to radians

    Parameters
    ----------
    a : array_like
        Input array in measures with the ``high`` boundary fot the sample range
    high : float or int
        High boundary for the sample range

    Returns
    -------
    rad2val : ndarray
        The corresponding radian values. This is a scalar if ``a`` is a scalar.

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


def _hist_yticks(a, bins):
    """
    Get the radial ticks matplotlib chooses for a circular histogram.

    Parameters
    ----------
    a : array_like
        Input array in radians.
    bins : array_like
        Edges of the bins, in radians.

    Returns
    -------
    yticks : np.ndarray
        Positions of the radial ticks, before any offset is added.

    Notes
    -----
    The ticks are read from a throwaway polar figure, which is closed right
    away. Matplotlib pads the radial axis past the tallest bin in a way that
    a locator alone does not reproduce, so drawing is the reliable way to get
    the very same ticks with either engine.
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.hist(a, bins=bins, edgecolor="black")
    yticks = ax.axes.yaxis.get_ticklocs()
    plt.close(fig)

    return yticks


def _hist_plotly(counts, bins, theta, yticks, bottom, high, title, figsize=None):
    """
    Draw a circular histogram with plotly.

    Parameters
    ----------
    counts : np.ndarray
        Number of observations in every bin.
    bins : np.ndarray
        Edges of the bins, in radians.
    theta : np.ndarray
        Positions of the ticks of the angular axis, in radians.
    yticks : np.ndarray
        Positions of the radial ticks, before the offset is added.
    bottom : float
        Radial offset of the bins, which hollows the centre of the circle.
    high : float or int
        High boundary of the sample range, used to label the angular axis.
    title : str or None
        Title of the chart.
    figsize : tuple or None, default: None
        Size of the figure in inches, converted to pixels.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The circular histogram.

    Notes
    -----
    Plotly measures the angles in degrees while matplotlib uses radians, so
    the angles are converted here. Setting the angular axis to start at the
    top and to run clockwise matches ``theta_offset`` and ``theta_direction``
    of matplotlib.
    """
    go, _ = _import_plotly()
    width, height = _figsize_to_px(figsize)

    centers = (bins[:-1] + bins[1:]) / 2

    fig = go.Figure(
        go.Barpolar(
            r=counts,
            theta=np.degrees(centers),
            width=np.degrees(np.diff(bins)),
            base=bottom,
            marker={"line": {"color": "black", "width": 1}},
            hovertemplate="count = %{r}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        polar={
            "angularaxis": {
                "rotation": 90,
                "direction": "clockwise",
                "tickmode": "array",
                "tickvals": np.degrees(theta),
                "ticktext": [f"{value / np.pi / 2 * high:.3g}" for value in theta],
            },
            "radialaxis": {
                "tickmode": "array",
                "tickvals": yticks + bottom,
                "ticktext": [f"{value:g}" for value in yticks],
            },
        },
    )

    return fig


def _scatter_plotly(rads, y, theta, high, title, y_range, figsize=None):
    """
    Draw a scatter plot of a circular dataset with plotly.

    Parameters
    ----------
    rads : np.ndarray
        Angle of every point, in radians.
    y : array_like
        Radius of every point.
    theta : np.ndarray
        Positions of the ticks of the angular axis, in radians.
    high : float or int
        High boundary of the sample range, used to label the angular axis.
    title : str or None
        Title of the chart.
    y_range : tuple[float, float]
        Lowest and highest radius drawn, which pads the points.
    figsize : tuple or None, default: None
        Size of the figure in inches, converted to pixels.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The scatter plot.

    Notes
    -----
    Plotly measures the angles in degrees while matplotlib uses radians, so
    the angles are converted here.
    """
    go, _ = _import_plotly()
    width, height = _figsize_to_px(figsize)

    fig = go.Figure(
        go.Scatterpolar(
            r=y,
            theta=np.degrees(rads),
            mode="markers",
            hovertemplate="%{r}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        showlegend=False,
        polar={
            "angularaxis": {
                "rotation": 90,
                "direction": "clockwise",
                "tickmode": "array",
                "tickvals": np.degrees(theta),
                "ticktext": [f"{value / np.pi / 2 * high:.3g}" for value in theta],
            },
            "radialaxis": {"range": list(y_range)},
        },
    )

    return fig


def hist(
    a, n_bins=_N_BINS, high=_HIGH, bottom=0.1, title=None, figsize=None, ax=None, return_ax=False,
    engine=None, **kwargs,
):
    if n_bins < 2:
        raise ValueError("Received an invalid number of bins. Number of bins must be at least 2, and must be an int.")

    a = a / high * 2 * np.pi  # h to rad

    def radian_function(x, y):
        """Helper function for labeling the x-axis"""
        rad_x = x / np.pi / 2
        return f"{(rad_x * high):.3g}"

    engine = _resolve_engine(engine)

    theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    bins = np.linspace(0, 2 * np.pi, n_bins + 1, endpoint=True)

    counts = np.histogram(a, bins=bins, **kwargs)[0]
    max_bin = counts.max()
    bottom = max_bin * bottom

    # get yticks:
    yticks = _hist_yticks(a, bins)

    if engine == "plotly":
        _warn_ignored_mpl_params(engine, ax=ax, return_ax=return_ax)
        return _hist_plotly(
            counts, bins, theta, yticks, bottom, high, title, figsize=figsize
        )

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
        # ax = plt.subplot(111, polar=True)

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

    return None


hist.__doc__ = """\
Plot histogram to show distributions of circular datasets.

Parameters
----------
a : array_like
    Input array in measures with the ``high`` boundary for the sample range.
n_bins : int, default: {n_bins}
    The number of bins to produce. Raises ValueError if ``n_bins < 2``.
high : float or int, default: {high}
    High boundary for the sample range.
bottom : float, default: 0.1
    Proportion of location of the bottom of each bin, where 0 is center of the circle, and 1 is the edge.
    Bins are drawn from ``bottom * max(bins)`` to ``bottom * max(bins) + hist(x, bins)``.
    Valid range is [0, 1].
title : str or None, default: None
    Text to use for the title.
figsize : (float, float) or None, default: None
    Width, height in inches.
ax : :class:`matplotlib.axes.Axes` or array of Axes or None, default: None
   Axes object to draw the plot onto, otherwise uses the current Axes. If ``ax`` is None, create nex ``ax``.
return_ax : bool, default: False
    Show, it is necessary to return ``ax``.
    Ignored when ``engine="plotly"``, which always returns the figure.
engine : {{"matplotlib", "plotly"}} or None, default: None
    The rendering engine. If None, the engine set by
    :func:`pltstat.set_backend` is used.
kwargs : key, value mappings
    Other keywords arguments are passed down to :meth:`maptplotlib.axes.Axes.hist`.

Returns
-------
ax : :class:`matplotlib.axes.Axes` or array of Axes or :class:`plotly.graph_objects.Figure`
    Returns the Axes object with the plot drawn onto it if ``return_ax argument`` is ``True``.
    Returns the plotly Figure when ``engine="plotly"``.

Example
--------
>>> import numpy as np
>>> from pltstat.circle import hist
>>> np.random.seed(0)
>>> a = np.random.randint(0, 24, 30)
>>> a
array([ 6, 23, 11, 14, 18,  0, 14,  3, 21, 12, 10, 20, 11,  4,  6,  4, 15,
       20,  3, 12,  4, 20,  8, 14, 15, 20,  3, 23, 15, 13])
>>> hist(a, 12)""".format(n_bins=_N_BINS, high=_HIGH)


def mean(a, high=_HIGH, atol=1e-10):
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
    # instead of elif, because-1.614809932057922e-15 + 360 => 360, for example a=[10, 350], high=360:
    if tau >= high:
        tau -= high
    return tau


mean.__doc__ = """\
    Compute the circular mean for samples in a range. 
    The function shows more accurate result than :meth:`scipy.stats.circmean`

    Parameters
    ----------
    a : array_like
        Input array in measures with the ``high`` boundary for the sample range.
    high : float or int, default: {high}
        High boundary for the sample range.
    atol : float, default: 1e-10
        The threshold for radius calculation. If the radius is less than ``atol``, it will be set to 0 and the mean value
        will be NaN. When the radius is 0, it is not possible to calculate the mean value. If the radius is close to 0,
        the mean value can be extremly inaccurate.
        An ``atol`` equal to ``1e-10`` ensures a mean value accurate to approximately 7 decimal places. 

    Returns
    -------
    mean : float
        Circular mean

    Examples
    --------
    >>> import numpy as np
    >>> from pltstat.circle import mean
    >>> from scipy.stats import circmean
    >>> a = [10, 350]
    >>> mean(a, 360)
    0.0
    >>> circmean(a, 360)
    359.99999999999994""".format(high=_HIGH)


def std(a, high=_HIGH):
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


std.__doc__ = """\
    Compute the circular standard deviation for samples in a range.

    Parameters
    ----------
    a : array_like
        Input array in measures with the ``high`` boundary for the sample range.
    high : float or int, default: {high}
        High boundary for the sample range. 

    Returns
    -------
    std : float
        Circular standard deviation

    Examples
    --------
    >>> import numpy as np
    >>> from pltstat.circle import std
    >>> from scipy.stats import circstd
    >>> a = [10, 350]
    >>> std(a, 360)
    10.0255602484647
    >>> circstd(a, 360)
    10.025560248464737""".format(high=_HIGH)


def scatter(
    deg, y, high=_HIGH, n_ticks=_N_TICKS, title=None, figsize=None, ax=None, return_ax=False,
    engine=None, **kwargs
):
    BOTTOM_EDGE = 0.2
    UPPER_EDGE = 0.1

    rads = deg / high * 2 * np.pi  # h to rad
    theta = np.linspace(0, 2 * np.pi, n_ticks, endpoint=False)

    def radian_function(x, y):
        """Helper function for labeling the x-axis"""
        rad_x = x / np.pi / 2
        return f"{(rad_x * high):.3g}"

    engine = _resolve_engine(engine)

    y_max = max(y)
    y_min = min(y)
    y_diff = y_max - y_min
    y_range = (y_min - y_diff * BOTTOM_EDGE, y_max + y_diff * UPPER_EDGE)

    if engine == "plotly":
        _warn_ignored_mpl_params(engine, ax=ax, return_ax=return_ax)
        return _scatter_plotly(rads, y, theta, high, title, y_range, figsize=figsize)

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)

    sns.scatterplot(x=rads, y=y, ax=ax, **kwargs)
    ax.set_ylim(y_range)

    # arrange graph
    ax.set(
        theta_offset=np.pi / 2,
        theta_direction=-1,
        xticks=theta,
        # yticks=yticks,
        # yticklabels=yticks,
    )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(radian_function))
    ax.set_title(title)
    if return_ax:
        return ax

    return None


scatter.__doc__ = """\
Draw a scatter plot of circular datasets.

Parameters
----------
deg : array_like
    Variables that specify positions on the angle axis.
y : array_like
    Variables that specify positions on the y axis.
high : float or int, default: {high}
    High boundary for the sample range.
n_ticks : int, default: {n_ticks}
    The number of angle ticks to produce.
ax : :class:`matplotlib.axes.Axes` or array of Axes or None, default: None
   Axes object to draw the plot onto, otherwise uses the current Axes. If ``ax`` is None, create nex ``ax``.
return_ax : bool, default: False
    Show, it is necessary to return ``ax``.
    Ignored when ``engine="plotly"``, which always returns the figure.
engine : {{"matplotlib", "plotly"}} or None, default: None
    The rendering engine. If None, the engine set by
    :func:`pltstat.set_backend` is used.
kwargs : key, value mappings
    Other keywords arguments are passed down to :meth:`seaborn.scatterplot`.

Returns
-------
ax : :class:`matplotlib.axes.Axes` or array of Axes
    Returns the Axes object with the plot drawn onto it if ``return_ax argument`` is ``True``.


    :return: ax if return_ax is True, else - return None

Example
--------
>>> import numpy as np
>>> from pltstat.circle import scatter
>>> np.random.seed(0)
>>> deg = np.linspace(0, 24, 20, endpoint=False)
>>> deg
array([ 0. ,  1.2,  2.4,  3.6,  4.8,  6. ,  7.2,  8.4,  9.6, 10.8, 12. ,
       13.2, 14.4, 15.6, 16.8, 18. , 19.2, 20.4, 21.6, 22.8])
>>> temp = np.concatenate((np.repeat([36.6], 10), np.linspace(36.6, 40, 10, endpoint=False)))
>>> temp
array([36.6 , 36.6 , 36.6 , 36.6 , 36.6 , 36.6 , 36.6 , 36.6 , 36.6 ,
       36.6 , 36.6 , 36.94, 37.28, 37.62, 37.96, 38.3 , 38.64, 38.98,
       39.32, 39.66])
>>> scatter(deg, temp, s=20, marker='o')""".format(n_ticks=_N_TICKS, high=_HIGH)
