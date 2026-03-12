import sys
from pathlib import Path
# Add the parent directory to sys.path to import modules from there
sys.path.append(
    str(Path(__file__).parent.parent)
)  # add the path to the parent directory to sys.path
from amp_tests.structural_test import residual_supplier_index, congested_area_test
from amp_tests.conduct_test import ref_level, mitigate_bids
from datetime import datetime as dt, timedelta as td
from amp_tests.utils import get_incremental_bids
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import ThreadPool



def compute_metrics(x, ref_price, load):
    mit_hours = lambda x: (x != ref_price).sum()
    price_impact = lambda x: ((x - ref_price)[x !=ref_price]).mean()
    welfare_impact = lambda x: ((ref_price - x) * load).sum()
    welfare_per_hour = lambda x: ((ref_price - x)[x !=ref_price] * load).mean()
    
    metrics_dict = {"mitigated_hours": mit_hours(x), 
                    "average_price": x.mean(),
                    "average_price_impact": price_impact(x), 
                    "tot_welfare_impact": welfare_impact(x), 
                    "welfare_impact_per_hour": welfare_per_hour(x)}
    
    return pd.Series(metrics_dict)

    
if __name__ == "__main__":
    # parse arguments
    # args = parser.parse_args()
    # TODO: remove this before submission
    all_runs_file = "data/all_runs.parquet"
    load_file = "data/isone_rawdata/load_2018-2019.parquet"

    prices = pd.read_parquet(all_runs_file)
    load = pd.read_parquet(load_file)["Load"]
    ref_col = "no_amp"
    ref_price = prices[ref_col]
    res  = {col:
        compute_metrics(prices[col], ref_price, load)
        for col in prices.columns if col != ref_col
    }
    res_df = pd.DataFrame().from_dict(res, orient="columns")
    res_df.to_csv("statistics.csv")