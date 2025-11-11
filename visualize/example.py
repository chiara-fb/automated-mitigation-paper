
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib as mpl
from pathlib import Path
import yaml

with open("visualize/matplotlib_config.yaml", "r") as f:
    config = yaml.safe_load(f)

mpl.rcParams.update(config)
plt.style.use('ggplot')

cmap = mcolors.LinearSegmentedColormap.from_list('white_gray', ['white', 'gray'])

def smooth_pulse(x, t0, t1, smoothness=1):
    """
    Smooth pulse function:
    - rises from 0 to 1 around step t0
    - falls back to 0 around step t1
    - stays at 0 afterwards for t2 steps
    
    Parameters:
    x : input (can be array or scalar)
    t0 : step where rise happens
    t1 : step where fall happens
    smoothness : controls steepness of transitions
    
    Returns:
    float or array
    """
    # sigmoid function for smooth transitions
    sigmoid = lambda t: 1 / (1 + np.exp(-t / smoothness))

    rise = sigmoid(x - t0)      # goes from 0 → 1 around t0
    fall = sigmoid(-(x - t1))   # goes from 1 → 0 around t1
    return rise * fall


def plot_example() -> tuple[plt.Figure, plt.Axes]:    
    """Plot an example of strategic bidding to avoid conduct-and-impact test."""

    t0, t1 = 50, 150
    x = np.linspace(0, 200, 200)
    rsi = smooth_pulse(x, t0, t1)
    noise = np.random.normal(size=len(x))
    reference = np.ones(200) * 20
    #low_thres = reference * 2
    #high_thres = reference * 3
    low_bid = reference + noise + rsi * (noise + 5) + (5 * np.isclose(rsi, 1))
    high_bid = low_bid + (15 * np.isclose(rsi, 1))


    fig, (ax0, ax1) = plt.subplots(1,2, tight_layout=True, sharey=True, sharex=True)
    

    for i, ax in enumerate((ax0, ax1)):
        

        if i == 0: 
            ax.plot(x, low_bid, label='Max bid (observed)', color='black', zorder=1)
            ax.plot(x, high_bid, label='Max bid (counterfactual)', color='black', linestyle=':',zorder=1)
            ax.plot(x, reference * 2, label='Conduct threshold', color='red', linestyle='-.', zorder=2)
            #ax.plot(x, low_thres, label='Threshold (2x reference)', color='red', linestyle='-.', zorder=2)
            ax.set_title('Low conduct threshold (2x reference)') 
            ax.set_ylabel('Bid price ($/MWh)')
        
        else:
            ax.plot(x, high_bid,  color='black', zorder=1)
            ax.plot(x, reference * 3, label=None, color='red', linestyle='-.', zorder=2)
            #ax.plot(x, high_thres, label='Threshold (3x reference)', color='red', linestyle='--', zorder=2)
            ax.set_title('High conduct threshold (3x reference)')

        ax.set_ylim(10, 70)
        ax.set_xlim(0, 200)
        ax.plot(x, reference, label='Reference level' if i == 1 else None, color='green', zorder=2)
        
        im = ax.imshow(np.repeat(rsi[None,:], 200,axis=0), cmap=cmap, alpha=0.3, zorder=0, aspect='auto')
        ax.fill_between(x, 70, where=np.isclose(rsi, 1), alpha=0.3, color='blue', zorder=0)

        
        ax.set_xlabel('Time (hours)')
        ax.set_xticklabels("")
        ax.set_yticklabels("")
        ax.grid(False)
        
  
    fig.legend(ncols=4, loc='upper center', bbox_to_anchor=(0.5, 0.01), 
            facecolor='white', fontsize=16, framealpha=1, edgecolor='black')

    fig.suptitle('Strategic bidding to avoid conduct-and-impact test')
    return fig, (ax0, ax1)


if __name__ == "__main__":
    fig, ax = plot_example()
    path = Path(r"C:\Users\c.fusarbassini\OneDrive - Hertie School\25 ML-Strom\2 Literatur & Research ideas\AP 3")
    fig.savefig(path / "pictures" / "example4.svg", bbox_inches='tight')
