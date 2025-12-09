import yaml
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('ggplot')

with open("visualize/matplotlib_config.yaml", "r") as f:
    config = yaml.safe_load(f)

mpl.rcParams.update(config)


def plot_simulations(all_runs: pd.DataFrame, starts: tuple[pd.Timestamp], ends: tuple[pd.Timestamp]) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the results of the simulations.
    
    Parameters:
    all_runs (pd.DataFrame): DataFrame containing simulation results with columns 'real-time', 'a', 'b', 'c', 'd', 'e'.
    """
    
    fig, axes = plt.subplots(1, len(starts), sharey=True, tight_layout=True)

    markers = ['o', 'd', '^', 'P']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = {'a': 'Baseline', 'b': 'Low conduct', 'c': 'Low impact', 'd': 'Low conduct & impact', 'e': 'No pivotal test'}
    
    for ax, start, end in zip(axes, starts, ends):
        run = all_runs.loc[start:end] # Filter the DataFrame for the specified date range
        ix = run.index
        ax.plot(ix, run['real-time'].values, 
                label='Real-time LMP', color='black', linestyle=':', linewidth=.5, zorder=0)
        ax.plot(ix, run['a'].values, 
                label='Baseline', color='black', zorder=2)

        for i, col in enumerate(['b', 'c', 'd', 'e']):
            mitigated = run.loc[run[col] != run['a']]
            ax.plot(mitigated.index, mitigated[col].values, 
                    label=labels[col], linestyle='', 
                    marker=markers[i], markersize=8, 
                    markeredgecolor='white', markerfacecolor=colors[i], zorder=2)
            ax.fill_between(ix, run[col], run['a'], 
                            interpolate=False, where=run[col] <= run['a'], 
                            color=colors[i], zorder=1, alpha=0.5)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title(start.strftime('%b %Y'), fontsize=12)
    
    axes[0].set_ylabel('$/MWh')
    axes[-1].legend(fontsize='small', loc='upper left', ncol=2)
    fig.suptitle("Mitigated versus baseline real-time LMP")
    
    return fig, axes


if __name__ == "__main__":
    # Example usage
    all_runs = pd.read_parquet("data/all_runs.parquet")
    starts = (pd.Timestamp('2019-11-01'),pd.Timestamp('2019-12-09'))
    ends = (pd.Timestamp('2019-11-19'),pd.Timestamp('2019-12-25'))
    fig, axes = plot_simulations(all_runs, starts, ends)
    
    fig.savefig("simulation_plot.pdf", bbox_inches='tight', dpi=300)
