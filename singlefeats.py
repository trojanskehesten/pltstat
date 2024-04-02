import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np


def pie(df_column, ax=None):
    """
    Plot pie with good settings

    :param df_column: pd.Series
    :return: None
    """
    value_counts = df_column.value_counts()
    value_counts_norm = df_column.value_counts(normalize=True)
    cat_number = value_counts.shape[0]

    def func(pct, allvals_norm, allvals):
        TOL = 0.01
        absolute = allvals[
            np.where(np.abs(round(pct, 2) - 100 * np.round(allvals_norm, 4)) < TOL)[
                0
            ][0]
        ]

        # absolute = int(pct*np.sum(allvals)/100)
        return f"{pct:.1f}% ({absolute:d} )"

    if ax is None:
        plt.title(df_column.name)
        ax = plt
    else:
        ax.set_title(df_column.name)

    ax.pie(
        value_counts.values,
        labels=value_counts.index,
        autopct=lambda pct: func(pct, value_counts_norm.values, value_counts.values),
        # autopct='%.1f %%',
        # startangle= 120,
        explode=[0.02] * cat_number,
    )


def countplot(df_column, is_count_order=True, x_rotation=90):
    """
    Plot countplot with good settings

    :param df_column: pd.Series
    :return: None
    """
    plt.figure(figsize=(18, 6))
    if is_count_order:
        ax = sns.countplot(df_column, order=df_column.value_counts().index)
    else:
        ax = sns.countplot(df_column)
    total = len(df_column)
    for p in ax.patches:
        text = f"{100 * p.get_height() / total:.1f}% \n ({p.get_height()})"
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.text(
            x,
            y,
            text,
            fontsize=12,
            horizontalalignment="center",
            verticalalignment="center",
        )
        # ax.annotate(text, (x, y), ha='center', va='center', xytext=(x, y))
    plt.xticks(rotation=x_rotation)



def histplot(df_column, is_limits=False, bins=None):  # , n_modes=0):
    """
    Plot histplot with good settings

    :param df_column: pd.Series
    :param is_limits: limits for kde plot by min and max values of the data
    :param bins: number of bins
    :return: None
    """

    if is_limits:
        min_val = df_column.min()
        max_val = df_column.max()
        ax = sns.histplot(
            df_column, kde=True, kde_kws={"clip": (min_val, max_val)}, bins=bins
        )
    else:
        ax = sns.histplot(df_column, kde=True, bins=bins)

    top_values = df_column.value_counts().index.to_numpy()
    top_counts = df_column.value_counts().values

    mode = top_values[0]
    bins_most_height = np.array([p.get_height() for p in ax.patches]).max()
    # coef = bins_most_height / top_counts[0]

    plt.vlines(mode, 0, bins_most_height, colors="r", label="mode")
    plt.text(
        mode,
        bins_most_height,
        "mode=%.2f" % mode,  # '%.2f (mode)' % mode,
        fontsize=12,
        horizontalalignment="right",
        verticalalignment="top",
        rotation="vertical",
    )
    plt.text(
        mode,
        bins_most_height,
        "count=%d" % top_counts[0],
        fontsize=12,
        horizontalalignment="left",
        verticalalignment="top",
        rotation="vertical",
    )

    plt.legend()


def auto_naive_plot(
    df_column,
    is_ordinal=False,
    is_limits=None,
    is_show_average=True,
    is_count_order=True,
):
    """
    Plot info about column of df. Type of column depends on count of unique values.
    A naive analyse column, you need to check obtained information.
    The function can recognize only categorical and continuous features.
    :param df_column: series for analyzing
    :param is_limits: True/False/None. If None - auto choosing
    :param is_ordinal: True/False. Function can't recognise ordinal feature.
    :param is_show_average: True/False. If it's continuous feature -
       print (or not print) mean and median
    :param is_count_order: True/False. If it's True that it's order by count for
      countplotting
    :return: None
    """
    threshold_cont = 20  # If categories more than 20 and numbers - continuous
    threshold_big_cat = (
        5  # If categories more than 5 - too many categories, don't plot pie plot
    )

    n_nan = np.sum(df_column.isnull())
    n_unique = len(df_column.dropna().unique())
    top5 = df_column.head().to_string()
    if n_nan > 0:
        df_column = df_column.dropna()

    dtype = "Categorical"  # initial value

    try:
        # If everything is ok - it's continuous
        # check number of unique values:
        assert n_unique > threshold_cont
        # do and check that it's possible to change format type:
        df_column = df_column.astype("float64")
        dtype = "Continuous"
    except Exception:
        # Else: categorical
        # I can't use np.array_equal, because df_column can has float features:
        if n_unique == 2:  # and _np.allclose(_np.sort(df_column.unique()), [0, 1]):
            dtype += ":Boolean"
        elif is_ordinal:
            dtype += ":Ordinal"
        else:
            dtype += ":Nominal"

    if dtype == "Continuous":
        if np.allclose(df_column, df_column.astype("int64")):
            dtype += ":Integer"
        else:
            if (
                (df_column.max() <= 1)
                and (df_column.max() > 0)
                and (df_column.min() >= -1)
            ):
                dtype += ":Proportion"
            else:
                dtype += ":Float"

    if is_limits is None:
        if dtype.startswith("Continuous"):
            if (
                dtype.endswith("Proportion")
                or (df_column.min() == 0)
                or (df_column.min() == 1)
            ):  # prop or counter
                is_limits = True
            else:
                is_limits = False
    print("Name of feature: '%s'" % df_column.name)
    print("Feature -", dtype)
    # if dtype.startswith('Categorical'):
    print("Number of unique values:", n_unique)
    if n_nan > 0:
        print("Number of nan values:", n_nan)
    else:
        print("Nan values aren't found")
    print()
    print("First 5 values:")
    print(top5)
    print()
    if dtype.startswith("Continuous"):
        print(
            "Min / max values: %s / %s"
            % tuple(np.around([df_column.min(), df_column.max()], decimals=3))
        )
        if is_show_average:
            print(
                "Mean / median values: %s / %s"
                % tuple(np.around([df_column.mean(), df_column.median()], decimals=3))
            )
        print()

    try:
        if n_unique < 1:
            print("Error. Number of unique values:", n_unique)
        elif n_unique == 1:
            print("It contains only one value:", df_column[0])
        elif n_unique < threshold_big_cat:
            pie(df_column)
        elif n_unique < threshold_cont:
            countplot(df_column, is_count_order)
        else:
            histplot(df_column, is_limits=is_limits)
    except Exception:
        print("It isn't possible to understand format and print it")
    # print('un', n_unique)
    # print('# nan:', n_nan)


# TODO:Not circle but heatmap with HDE and values of bins!
def plot_hist_man_ing(hours, title, y_step, h_step=2, height=6, aspect=2.5):
    """
    Plot histogram of ingestion time
    """
    bins = np.arange(0, 25, h_step)
    bin_values = np.histogram(hours, bins=bins)
    max_y = bin_values[0].max()
    max_y = max_y + y_step - max_y % y_step

    sns.displot(hours, kde=True, bins=bins, height=height, aspect=aspect)
    if (h_step - int(h_step)) != 0:
        plt.xticks(bins, bins, rotation=45);
    else:
        plt.xticks(bins, bins)
    plt.yticks(range(0, max_y+1, y_step), range(0, max_y+1, y_step))
    plt.ylabel('Count of ingestions')
    plt.xlim(0, 24)
    plt.ylim(0, max_y)
    for x, y in zip(bin_values[1], bin_values[0]):
        plt.text(x+h_step/2, y, y, horizontalalignment='center', verticalalignment='bottom')
    plt.grid()
    plt.title(title);
