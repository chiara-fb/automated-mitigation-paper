import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the range for x values
x = np.linspace(-.5, .5, 1000)
s = np.linspace(-1, 1, 100)

# Define different standard deviations (variances are their squares)
std_devs = [0.01, .05, 0.1]
colors = ['red', 'green', 'blue']
cutoff = 0
# Create subplots: 1 row, 2 columns

def plot_fuzzy_cdf(s, std_devs, colors, cutoff, ax):

# Plot CDFs on the left
    for std, color in zip(std_devs, colors):
        cdf = norm.cdf((s - cutoff) / std)
        #cdf = norm.cdf((s - cutoff) / std + norm.ppf(0.8))
        ax.plot(s, cdf, label=f'$\sigma={std}$', color=color)
    ax.set_title('Expected treatment assignment $D_i$')
    ax.axvline(x=cutoff, color='black', linestyle='--', label='Cutoff')
    ax.set_ylabel('$P(D_i = 1)$')
    ax.set_xlabel('$\hat{S}_i$')
    ax.legend()
    ax.grid(True)
    return ax



def plot_fuzzy_pdf(x, std_devs, colors, ax):
    # Plot PDFs on the right
    for std, color in zip(std_devs, colors):
        pdf = norm.pdf(x, loc=0, scale=std)
        ax.plot(x, pdf, label=f'$\sigma={std}$', color=color)
    ax.set_title(r'Distribution of noise variable $\varepsilon$')
    ax.set_xlabel(r'$\varepsilon$')
    ax.legend()
    ax.grid(True)
    
    return ax


if __name__ == "__main__":
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
    ax0 = plot_fuzzy_cdf(s, std_devs, colors, cutoff, ax0)
    ax1 = plot_fuzzy_pdf(x, std_devs, colors, ax1)
    fig.savefig('fuzzy_treatment.svg')