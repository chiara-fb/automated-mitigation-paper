import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO: set default plt.rcParams


def quantiles(
    quant_df: pd.DataFrame, outlier_ix: np.array = None, ax: plt.Axes = None, **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """Plot quantiles as a time series line plot.
    Quantiles is a matrix of shape (n_observations, n_quantiles)."""

    if ax is None:
        fig, ax = plt.subplots()

    else:
        fig = ax.get_figure()

    outlier_ix = outlier_ix if outlier_ix is not None else []
    for ix, row in tqdm(
        quant_df.iterrows(), total=len(quant_df), desc="Plotting quantiles"
    ):

        if ix in outlier_ix:
            ax.plot(row, c="red", alpha=0.1, marker="o")
        else:
            ax.plot(row, c="black", alpha=0.1, marker="o")
    ax.set(**kwargs)

    return fig, ax


def outliers(
    outlier_score: np.array,  # outlier scores for each observation
    outlier_ix: np.array,  # indices of outlier observations
    vlines: np.array = None,  # optional list of vertical lines to plot
    xlines: np.array = None,  # optional list of horizontal lines to plot
    other_lines: list = None,  # optional list of other lines to plot on secondary y-axis
    ax: plt.Axes = None,
    **kwargs,  # args for main axis
) -> tuple[plt.Figure, plt.Axes]:
    """Plot outlier scores as a time series line plot
    with outliers highlighted in red.
    NO TIME INDEX ASSUMED."""

    if ax is None:
        fig, ax = plt.subplots()

    else:
        fig = ax.get_figure()

    ax.plot(outlier_score)

    ax.plot(
        outlier_ix, outlier_score[outlier_ix], c="red", marker="o", linestyle="None"
    )

    if vlines is not None:
        ymin, ymax = ax.get_ylim()

        ax.vlines(vlines, ymin=ymin, ymax=ymax, color="green", linestyle="--")

    if xlines is not None:
        xmin, xmax = ax.get_xlim()
        ax.hlines(xlines, xmin=xmin, xmax=xmax, color="green", linestyle="--")

    if other_lines:
        ax2 = ax.twinx()

        for line in other_lines:
            ax2.plot(line, color="black")

    ax.set(**kwargs)

    return fig, ax
