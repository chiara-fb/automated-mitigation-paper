
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml


plt.style.use('ggplot')

with open("visualize/matplotlib_config.yaml", "r") as f:
    config = yaml.safe_load(f)

mpl.rcParams.update(config)



def bids_violinplot(isone_data: pd.DataFrame, nyiso_data: pd.DataFrame, year=2019) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot violin plots of maximum and average incremental bids in ISO-NE and NYISO for a given year.
    Args:
        data (pd.DataFrame): Bids data for ISO-NE and NYISO.
        year (int): Year to filter the bids data.
    Returns:
        tuple: A tuple containing the figure and axes of the violin plots.
    """
    
    fig, (ax0, ax1) = plt.subplots(1,2, tight_layout=True, sharey=True)    
    isone_data['Treatment (RSI ≤ 1)'] = isone_data['rsi'] <= 1
    nyiso_data['Treatment (congestion ≥ 0.04)'] = nyiso_data['avg_cong_1h_lag'] >= 0.04

    sns.violinplot(data=isone_data, x='Treatment (RSI ≤ 1)', y='max_bid', hue='Treatment (RSI ≤ 1)', ax=ax0, legend=False)
    sns.violinplot(data=nyiso_data, x='Treatment (congestion ≥ 0.04)', y='max_bid', hue='Treatment (congestion ≥ 0.04)', ax=ax1, legend=False)
    #fig.suptitle(f'Maximum bid prices in ISO-NE and NYISO ({year})')
    #ax0.set_title('Maximum bid')
    ax0.set_title('ISO-NE', fontsize=24)
    ax1.set_title('NYISO', fontsize=24)
    ax0.set_xlabel('Treatment (RSI ≤ 1)', fontsize=16)
    ax1.set_xlabel('Treatment (congestion ≥ 0.04)', fontsize=16)
    ax0.set_ylabel('Maximum bid price ($/MWh)', fontsize=16)
    ax0.tick_params(axis='both', labelsize=16)
    ax1.tick_params(axis='both', labelsize=16)

    #ax1.set_title('Average bid')

    return fig, (ax0, ax1)



def max_boxplot(corr_df: pd.DataFrame,
                year=2019) -> tuple[plt.Figure, plt.Axes]:
    """Boxplot unit-level correlation with of maximum incremental bids in ISO-NE and NYISO for a given year.
    Args:
        corr_df (pd.DataFrame): contains unit-level correlations and unit market.
        year (int): Year to filter the bids data.
    Returns:
        tuple: A tuple containing the figure and axes of the boxplot.
    """
    
    fig, ax = plt.subplots(figsize=(12, 10), tight_layout=True)
    corr_df_long = pd.melt(corr_df, id_vars=['Masked Asset ID', 'Market'])
    sns.boxplot(data=corr_df_long, x='variable', y='value', hue='Market', gap=.1)
    ax.set_title(f'Correlation between average bid and variables ({year})', fontsize=24)
    ax.set_xlabel('')
    ax.set_ylabel('Correlation')

    return fig, ax



def plot_ref_level(df, name): 
        """
        Plots reference level and maximum bid for one unit. 
            Returns: tuple (fig, ax).
        """
        fig, ax = plt.subplots(tight_layout=True)
        dfc = df.copy().droplevel([1])
        dfc['ref_level'].plot(ax=ax, label='Reference level')
        dfc['max_bid'].plot(ax=ax, label='Maximum bid')
        ax.set_ylabel('$/MWh')
        ax.set_xlabel('')
        ax.set_title(f"Generation unit ID: {name}")
        ax.legend(loc="lower right")

        return fig, ax



if __name__ == "__main__":
    # Load data
    isone_bids = pd.read_parquet("data/2025-08-12_iso-ne_dataset.parquet")
    isone_bids = isone_bids[isone_bids.index.get_level_values('DateTime').year == 2019]
    nyiso_bids = pd.read_parquet("data/2025-08-12_nyiso_dataset.parquet")
    nyiso_bids = nyiso_bids[nyiso_bids.index.get_level_values('DateTime').year == 2019]
    fig, axes = bids_violinplot(isone_bids, nyiso_bids, year=2019)
    fig.savefig("bids_violinplot.pdf", bbox_inches='tight')

    df = isone_bids.xs(44623, level='Masked Asset ID')
    ref_fig, ref_ax = plot_ref_level(df, "44623")
    ref_fig.savefig("ref_level.pdf", bbox_inches='tight')