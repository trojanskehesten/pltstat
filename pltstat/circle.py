"""
Contains functions and methods related to circular statistical visualizations,
such as radar charts or circular histograms.
"""

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import seaborn as sns

from .config import _resolve_engine, _ensure_plotly


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
    """
    a = np.array(a)
    a = a / high * 2 * np.pi
    return a


def hist(
    a, n_bins=_N_BINS, high=_HIGH, bottom=0.1, title=None, figsize=None,
    ax=None, return_ax=False, *, engine=None, **kwargs,
):
    """\

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
        Proportion of location of the bottom of each bin, where 0 is center
        of the circle, and 1 is the edge. Bins are drawn from
        ``bottom * max(bins)`` to ``bottom * max(bins) + hist(x, bins)``.
        Valid range is [0, 1].
    title : str or None, default: None
        Text to use for the title.
    figsize : (float, float) or None, default: None
        Width, height in inches (matplotlib only).
    ax : :class:`matplotlib.axes.Axes` or None, default: None
       Axes object to draw the plot onto (matplotlib only).
    return_ax : bool, default: False
        If True, return the ``ax`` (matplotlib only).
    engine : {{"matplotlib", "plotly"}} or None, default: None
        The rendering backend.
    **kwargs : key, value mappings
        Other keyword arguments are passed to ``ax.hist`` (matplotlib).

    Returns
    -------
    ax or plotly.graph_objects.Figure or None
        For matplotlib: the Axes if ``return_ax`` is True, else None.
        For plotly: a ``go.Figure``.

    Raises
    ------
    ValueError
        If ``n_bins < 2``.

    Example
    --------
    >>> import numpy as np
    >>> from pltstat.circle import hist
    >>> np.random.seed(0)
    >>> a = np.random.randint(0, 24, 30)
    >>> hist(a, 12)
    """.format(n_bins=_N_BINS, high=_HIGH)

    if n_bins < 2:
        raise ValueError(
            "Received an invalid number of bins. Number of bins must be "
            "at least 2, and must be an int."
        )

    engine = _resolve_engine(engine)
    a = np.array(a)
    a_rad = a / high * 2 * np.pi

    theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    bins = np.linspace(0, 2 * np.pi, n_bins + 1, endpoint=True)

    counts, _ = np.histogram(a_rad, bins=bins)
    max_bin = counts.max()
    bottom_val = max_bin * bottom

    if engine == "plotly":
        return _hist_plotly(
            theta, counts, bins, bottom_val, title, high, n_bins,
        )

    return _hist_mpl(
        a_rad, theta, bins, bottom_val, title, figsize, ax, return_ax,
        **kwargs,
    )


def _hist_mpl(a_rad, theta, bins, bottom_val, title, figsize, ax,
              return_ax, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": "polar"}, figsize=figsize
        )

    def radian_function(x, y):
        rad_x = x / np.pi / 2
        return f"{(rad_x * _HIGH):.3g}"

    # get yticks from a temporary figure
    fig, ax2 = plt.subplots(subplot_kw={"projection": "polar"})
    ax2.hist(a_rad, bins=bins, edgecolor="black")
    yticks = ax2.axes.yaxis.get_ticklocs()
    plt.close(fig)

    ax.hist(a_rad, bins=bins, edgecolor="black", bottom=bottom_val,
            **kwargs)

    ax.set(
        theta_offset=np.pi / 2,
        theta_direction=-1,
        xticks=theta,
        yticks=yticks + bottom_val,
        yticklabels=yticks,
    )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(radian_function))
    ax.set_title(title)
    if return_ax:
        return ax
    return None


def _hist_plotly(theta, counts, bins, bottom_val, title, high, n_bins):
    _ensure_plotly()
    import plotly.graph_objects as go

    # Barpolar uses degrees by default; convert theta to degrees
    theta_deg = np.degrees(theta)
    # Width of each bar in degrees
    bar_width = 360.0 / n_bins

    bar = go.Barpolar(
        r=counts,
        theta=theta_deg,
        width=bar_width,
        offset=0,
        base=bottom_val,
    )
    fig = go.Figure(data=bar)
    fig.update_layout(
        title=title,
        polar=dict(
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                period=high,
            ),
            radialaxis=dict(showline=True),
        ),
    )
    return fig


hist.__doc__ = hist.__doc__


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
    # instead of elif, because -1.614809932057922e-15 + 360 => 360,
    # for example a=[10, 350], high=360:
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
    deg, y, high=_HIGH, n_ticks=_N_TICKS, title=None, figsize=None,
    ax=None, return_ax=False, *, engine=None, **kwargs
):
    """\

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
    figsize : (float, float) or None, default: None
        Width, height in inches (matplotlib only).
    ax : :class:`matplotlib.axes.Axes` or None, default: None
       Axes object to draw the plot onto (matplotlib only).
    return_ax : bool, default: False
        If True, return the ``ax`` (matplotlib only).
    engine : {{"matplotlib", "plotly"}} or None, default: None
        The rendering backend.
    **kwargs : key, value mappings
        Other keyword arguments are passed to ``sns.scatterplot``
        (matplotlib) or ``go.Scatterpolar`` (plotly).

    Returns
    -------
    ax or plotly.graph_objects.Figure or None
        For matplotlib: the Axes if ``return_ax`` is True, else None.
        For plotly: a ``go.Figure``.

    Example
    --------
    >>> import numpy as np
    >>> from pltstat.circle import scatter
    >>> np.random.seed(0)
    >>> deg = np.linspace(0, 24, 20, endpoint=False)
    >>> temp = np.concatenate((
    ...     np.repeat([36.6], 10),
    ...     np.linspace(36.6, 40, 10, endpoint=False),
    ... ))
    >>> scatter(deg, temp, s=20, marker='o')
    """.format(n_ticks=_N_TICKS, high=_HIGH)

    engine = _resolve_engine(engine)

    if engine == "plotly":
        return _scatter_plotly(deg, y, high, n_ticks, title)

    return _scatter_mpl(deg, y, high, n_ticks, title, figsize, ax,
                       return_ax, **kwargs)


def _scatter_mpl(deg, y, high, n_ticks, title, figsize, ax, return_ax,
                 **kwargs):
    rads = np.array(deg) / high * 2 * np.pi
    theta = np.linspace(0, 2 * np.pi, n_ticks, endpoint=False)

    def radian_function(x, y):
        rad_x = x / np.pi / 2
        return f"{(rad_x * high):.3g}"

    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": "polar"}, figsize=figsize
        )

    y = np.array(y)
    y_max = max(y)
    y_min = min(y)
    y_diff = y_max - y_min
    BOTTOM_EDGE = 0.2
    UPPER_EDGE = 0.1

    sns.scatterplot(x=rads, y=y, ax=ax, **kwargs)
    ax.set_ylim(
        (y_min - y_diff * BOTTOM_EDGE, y_max + y_diff * UPPER_EDGE)
    )

    ax.set(
        theta_offset=np.pi / 2,
        theta_direction=-1,
        xticks=theta,
    )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(radian_function))
    ax.set_title(title)
    if return_ax:
        return ax
    return None


def _scatter_plotly(deg, y, high, n_ticks, title):
    _ensure_plotly()
    import plotly.graph_objects as go

    deg = np.array(deg)
    y = np.array(y)
    theta_deg = deg / high * 360.0

    scatter = go.Scatterpolar(
        r=y,
        theta=theta_deg,
        mode="markers",
    )
    fig = go.Figure(data=scatter)
    fig.update_layout(
        title=title,
        polar=dict(
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                period=high,
            ),
        ),
    )
    return fig