
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




path = Path(r"C:\Users\c.fusarbassini\OneDrive - Hertie School\25 ML-Strom\2 Literatur & Research ideas\AP 3")

# isone_bids = pd.read_parquet(path / "data" / "ISO-NE" / "rt_bids_2018-2019.parquet")
# isone_bids['Market'] = 'ISO-NE'
# nyiso_bids = pd.read_parquet(path / "data" / "NYISO" / "rt_bids_2018-2019.parquet")
# nyiso_bids['Market'] = 'NYISO'
# bids = pd.concat([isone_bids, nyiso_bids], axis=0)

# isone_load_fcst = pd.read_parquet(path / "data" / "ISO-NE" / "load_forecast_2018-2019.parquet").sum(axis=1)
# isone_reserves = pd.read_parquet(path / "data" / "ISO-NE" / "reserves_2018-2019.parquet").sum(axis=1)

# nyiso_load_fcst = pd.read_parquet(path / "data" / "NYISO" / "load_forecast_2018-2019.parquet")
# nyiso_rt_cong = pd.read_parquet(path / "data" / "NYISO" / "rt_shadow_prices_2018-2019.parquet")

# bids = []
# corr_df = []

# variables = {'Load forecast': 'load_forecast', 
#              'Gas price': 'gas',
#              'Average temperature': 'temperature'}

# YEAR = 2019
# market_df = {}
# for market in ['ISO-NE', 'NYISO']:

#     market_bids = pd.read_parquet(path / "data" / market / "rt_bids_2018-2019.parquet")
#     market_bids = market_bids[market_bids.index.get_level_values('DateTime').year == YEAR]
#     market_bids['Max bid'] = market_bids.filter(regex=r'Segment \d+ Price').max(axis=1)
#     if market == 'ISO-NE':
#         rsi = residual_supplier_index(market_bids, isone_load_fcst, isone_reserves).rename('rsi')
#         market_bids = market_bids.join(rsi, on=['DateTime', 'Masked Lead Participant ID'], how='left') 
    
#     else:
#         avg_cong_1h_lag = make_congestion_treatment(nyiso_rt_cong, nyiso_load_fcst)['avg_cong_1h_lag']
#         market_bids = market_bids.join(avg_cong_1h_lag, on='DateTime', how='left')
        
#     p, q = market_bids.filter(regex=r'Segment \d+ Price'), market_bids.filter(regex=r'Segment \d+ MW')
#     market_bids['Average bid'] = np.nansum(p.values * q.values, axis=1) / np.nansum(q.values, axis=1)
#     market_bids['Market'] = market
#     market_df[market] = market_bids

    # market_corr = []

    # for name, var in variables.items():       
    #     if var == 'gas':
    #         var_df = pd.read_parquet(path / "data" / f"{var}_2018-2019.parquet")
    #     else:
    #         var_df = pd.read_parquet(path / "data" / market / f"{var}_2018-2019.parquet")
        
    #     if var == 'load_forecast':
    #         var_df = var_df.sum(axis=1)
    #     elif var == 'temperature':
    #         var_df = var_df['AverageTemperature']
    #     else:
    #         var_df = var_df['Price']
    #     var_corr = market_bids.reset_index([1,2]).groupby('Masked Asset ID').apply(lambda x: x['Max bid'].corr(var_df))      
    #     var_corr = var_corr.rename(name).to_frame() 
    #     market_corr.append(var_corr)
        

    # market_corr  = pd.concat(market_corr, axis=1)
    # market_corr['Market'] = market
    # corr_df.append(market_corr)


# corr_df = pd.concat(corr_df, axis=0)
# corr_df = corr_df.reset_index()


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
    isone_bids = pd.read_parquet(path / "data" / "2025-08-12_iso-ne_dataset.parquet")
    isone_bids = isone_bids[isone_bids.index.get_level_values('DateTime').year == 2019]
    nyiso_bids = pd.read_parquet(path / "data" / "2025-08-12_nyiso_dataset.parquet")
    nyiso_bids = nyiso_bids[nyiso_bids.index.get_level_values('DateTime').year == 2019]
    fig, axes = bids_violinplot(isone_bids, nyiso_bids, year=2019)
    fig.savefig(path / "pictures" / "bids_violinplot.pdf", bbox_inches='tight')

    df = isone_bids.xs(44623, level='Masked Asset ID')
    ref_fig, ref_ax = plot_ref_level(df, "44623")
    ref_fig.savefig(path / "pictures" / "ref_level.pdf", bbox_inches='tight')