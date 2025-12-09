import pandas as pd
import numpy as np
import doubleml as dml
from doubleml.rdd import RDFlex
from amp_tests.utils import fuzzy_treatment_assignment
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib as mpl
import yaml
import matplotlib.pyplot as plt

plt.style.use('ggplot')
with open("visualize/matplotlib_config.yaml", "r") as f:
    config = yaml.safe_load(f)

mpl.rcParams.update(config)

# Data import (change the path accordingly)
path = Path(r"C:\\Users\\c.fusarbassini\\OneDrive - Hertie School\\25 ML-Strom\\2 Literatur & Research ideas\\AP 3\\data")

if __name__ == "__main__":
    iso_ne = pd.read_parquet(path / "2025-08-12_iso-ne_dataset.parquet")
    iso_ne = iso_ne[iso_ne.index.get_level_values('DateTime').year == 2019]  # filter for the year 2019

    nyiso = pd.read_parquet(path / "2025-08-12_nyiso_dataset.parquet")
    nyiso = nyiso[nyiso.index.get_level_values('DateTime').year == 2019]  # filter for the year 2019


    fig, (ax0, ax1) = plt.subplots(1,2, sharey=True, tight_layout=True)
    sns.histplot(iso_ne["rsi"], bins=np.arange(iso_ne["rsi"].min(), iso_ne["rsi"].max() +.2, .2), stat="probability", ax=ax0)
    sns.histplot(nyiso["avg_cong_1h_lag"], bins=np.arange(nyiso["avg_cong_1h_lag"].min(), nyiso["avg_cong_1h_lag"].max() +10, 10), stat="probability", label="Variable distribution")
    ax0.set_xlabel("Residual Supply Index")
    ax0.axvline(1, color="blue", ls="--", lw=3)
    ax1.set_xlabel("Avg. lagged congestion ($/MWh)")
    ax1.set_xlim(-100,50)
    ax1.axvline(0.04, color="blue", ls="--", lw=3, label="Cutoff value")
    ax1.legend()
    fig.suptitle("Distribution of the score variable")
    ax0.set_title("ISO-NE")
    ax1.set_title("NYISO")

    fig.savefig("score_variables.pdf", bbox_inches='tight', dpi=300)