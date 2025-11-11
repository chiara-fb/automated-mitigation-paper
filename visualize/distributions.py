import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import re


def correlation(
    df: pd.DataFrame, ax: plt.Axes = None, **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """Wrapper to plot the correlation of the input matrix with seaborn."""
    corr = df.corr()
    ax = sns.heatmap(corr, ax=ax, **kwargs)
    fig = ax.get_figure()
    return fig, ax


def univariate_hist(
    df: pd.DataFrame, cols: list, hue: str, **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """Plot univariate histograms for the input columns."""

    L = len(cols)
    fig, axes = plt.subplots(L // 2 + L % 2, 2, figsize=(5 * L, 5))

    for i in tqdm(range(L), desc="Plotting univariate histograms:"):
        legend = True if i == 0 else False
        ax = axes.flatten()[i]
        sns.histplot(
            df,
            x=cols[i],
            hue=hue,
            ax=ax,
            multiple="dodge",
            common_norm=False,
            stat="probability",
            legend=legend,
            bins=50,
        )
        ax.set(**kwargs)

    return fig, axes


def bivariate_kde(
    df, x: str, ylist: list, scatter_ix: pd.Series = None, **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """Plot bivariate KDEs for the input columns."""

    L = len(ylist)
    fig, axes = plt.subplots(L // 2 + L % 2, 2, figsize=(5 * L, 5))

    for i in tqdm(range(L), desc="Plotting bivariate KDEs:"):
        ax = axes.flatten()[i]
        y = ylist[i]
        sns.kdeplot(
            df,
            x=x,
            y=y,
            ax=ax,
            fill=True,
        )
        sns.scatterplot(
            df.loc[scatter_ix] if scatter_ix is not None else df,
            x=x,
            y=y,
            ax=ax,
            s=5,
        )
        ax.set(**kwargs)

    return fig, axes
