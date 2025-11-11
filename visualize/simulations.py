import yaml
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

with open("visualize/matplotlib_config.yaml", "r") as f:
    config = yaml.safe_load(f)

mpl.rcParams.update(config)
plt.style.use('ggplot')



### TODO: make this nicer to a function ###

def plot_simulations(all_runs: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the results of the simulations.
    
    Parameters:
    all_runs (pd.DataFrame): DataFrame containing simulation results with columns 'real-time', 'a', 'b', 'c', 'd', 'e'.
    """
    
    all_runs = all_runs.loc[start:end] # Filter the DataFrame for the specified date range
    fig, ax = plt.subplots(tight_layout=True)
    ix = all_runs.index
    ax.plot(ix, all_runs['real-time'].values, 
            label='Real-time LMP', color='black', linestyle=':', linewidth=.5, zorder=0)
    ax.plot(ix, all_runs['a'].values, 
            label='Baseline', color='black', zorder=2)

    markers = ['o', 'd', '^', 'P']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = {'a': 'Baseline', 'b': 'Low conduct', 'c': 'Low impact', 'd': 'Low conduct & impact', 'e': 'No pivotal test'}

    for i, col in enumerate(['b', 'c', 'd', 'e']):
        mitigated = all_runs.loc[all_runs[col] != all_runs['a']]
        ax.plot(mitigated.index, mitigated[col].values, 
                label=labels[col], linestyle='', 
                marker=markers[i], markersize=8, 
                markeredgecolor='white', markerfacecolor=colors[i], zorder=2)
        ax.fill_between(all_runs.index, all_runs[col], all_runs['a'], 
                        interpolate=False, where=all_runs[col] <= all_runs['a'], 
                        color=colors[i], zorder=1, alpha=0.5)
    ax.set_ylabel('$/MWh')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(facecolor='white', loc='upper left')
    ax.set_title("Mitigated versus baseline real-time LMP")
    return fig, ax


if __name__ == "__main__":
    # Example usage
    all_runs = pd.read_parquet("output/all_runs.parquet")
    start_date = pd.Timestamp('2019-11-01')
    end_date = pd.Timestamp('2019-12-31')
    fig, ax = plot_simulations(all_runs, start_date, end_date)
    fig.savefig("simulation_plot.svg", bbox_inches='tight', dpi=300)
