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

FOLDER = "data/isone_rawdata"


def read_source(
    path: Path,
    start: dt = None,
    end: dt = None,
    sum_ax1: bool = False,
    multiindex: bool = False,
) -> pd.DataFrame:
    """Reads a .parquet source file and returns a DataFrame with the data between start and end dates."""
    source = pd.read_parquet(path)

    if sum_ax1:
        source = source.sum(axis=1)

    if not (start is None or end is None):
        if multiindex:
            source = source[
                (source.index.get_level_values("DateTime").date >= start.date())
                & (source.index.get_level_values("DateTime").date <= end.date())
            ]
        else:
            source = source[
                (source.index.date >= start.date()) & (source.index.date <= end.date())
            ]

    return source


def mitigate_impact(price: pd.Series, mit_price: pd.Series, rel_impact_threshold: int = 2, abs_impact_threshold: int = 100) -> pd.Series:
    """
    Accepts only bid mitigation if they have a significant impact, otherwise transforms the mitigated price back to the original 
    price. Returns the mitigated price series.
    """
    rel_impact = lambda s0, s1: (s0 / s1) > rel_impact_threshold
    abs_impact = lambda s0, s1: (s0 - s1) > abs_impact_threshold
    impact = lambda s0, s1: abs_impact(s0, s1) | rel_impact(s0, s1)
    no_impact = ~impact(price, mit_price)
    mit_price[no_impact] = price[no_impact]

    return mit_price
    


def moc_equilibrium(bids:pd.DataFrame, demand:float=None) -> float:
    """
    Computes the clearing price for a given set of incremental bids and a demand.
    The function sorts the bids by price, cumulatively sums the MW, and finds the price based
    on the merit order curve (MOC)."""
    
    inc_bids = get_incremental_bids(bids, p_floor=-151, p_ceil=1001)
    inc_bids = inc_bids.reset_index(drop=True)
    inc_bids = inc_bids.sort_values(by='Price')
    inc_bids['Tot_MW'] = inc_bids['MW'].cumsum()
    ix = (inc_bids['Tot_MW'] >= demand).idxmax()
    lmp = inc_bids.loc[ix, 'Price']
    return lmp


def run_simulation(
    input_folder: str,
    start_str: str = "2019-01-01",
    end_str: str = "2019-12-01",
    mitigate_conduct: bool = True,
    structural_threshold: int = 1,  # threshold for structural test
    rel_conduct_threshold: int = 3, # relative threshold for conduct mitigation
    abs_conduct_threshold: int = 100, # absolute threshold for conduct mitigation
    verbose: bool = True,
) -> None:

    # TODO: include reserves and interchange
    FILEPATH = Path(input_folder)

    date_range = pd.date_range(
        start=start_str, end=end_str, freq="h", inclusive="left")

    bids = read_source(
        FILEPATH / "rt_bids_2018-2019.parquet", multiindex=True
    )
    rt_prices = read_source(FILEPATH / "rt_prices_2018-2019.parquet")
   
    load_fcst = read_source(FILEPATH / "load_forecast_2018-2019.parquet", sum_ax1=True)
    reserves = read_source(FILEPATH / "reserves_2018-2019.parquet", sum_ax1=True)
    flag_hour = read_source(
        FILEPATH / "mitigated_hours_2018-2019.parquet",
    )
    flag_hour = flag_hour["Real-Time mitigated?"]

    rsi = residual_supplier_index(
        bids, load_fcst, reserves=reserves
    )
    
    pst = (rsi < structural_threshold)
    ref_levels = ref_level(
        bids, min_bid=0, max_bid=800, days=90
    ).rename('ref_level')  
    
    const_hour = congested_area_test(rt_prices)
    print("Reference levels and pivotal supplier test computed.\n") if verbose else None
    (
        print(
            f"% hours with at least one pivotal supplier: {(pst.groupby("DateTime").sum() >= 1).mean():.2%}"
        )
        if verbose
        else None
    )

    ix = []
    bids_lmp = []
    
    for t in tqdm(date_range):

        print(f"PROCESSING {t}.\n") if verbose else None
  
        if t not in load_fcst.index or t not in rt_prices.index:
            print(f"Skipping {t} because it is not in the load or price.\n")

        elif const_hour[t]:
            print(f"Skipping {t} because it is congested.\n")
                 
        elif flag_hour[t]:
            print(f"Skipping {t} because it is mitigated.\n")
            
        else:

            bids_t = bids[bids.index.get_level_values("DateTime") == t]
            
            if mitigate_conduct:
                ref_t = ref_levels[
                    ref_levels.index.get_level_values("DateTime") == t
                ]
                pst_t = pst[pst.index.get_level_values("DateTime") == t]
                bids_t = mitigate_bids(bids_t, pst_t, ref_t, rel_ref=rel_conduct_threshold, abs_ref=abs_conduct_threshold, verbose=False)

            bids_lmp.append((bids_t, load_fcst[t]))
            ix.append(t)

        print(f"PROCESSED {t}.\n") if verbose else None

    with ThreadPool(10) as pool:
        prices = pool.starmap(moc_equilibrium, bids_lmp)
    
    # Create a DataFrame to store the results
    res = pd.Series(index=ix, data=prices, name="price")
    res.index.name = "DateTime"

    return res

    
if __name__ == "__main__":
    # parse arguments
    # args = parser.parse_args()
    #example usage 
    res = run_simulation(
        input_folder=FOLDER,
        start_str="2019-01-01",  # Start date for the simulation
        end_str="2020-01-01",  # End date for the simulation (not inclusive)
        structural_threshold=np.inf,  # Threshold for structural test (change to make test stricter)
        mitigate_conduct=True,  # Whether to mitigate bids
        rel_conduct_threshold=3,  # Relative threshold for mitigation (change to make mitigation stricter)
        abs_conduct_threshold=100, # Absolute threshold for mitigation (change to make mitigation stricter)
    )

    res.to_frame().to_parquet("output/e_no_impact.parquet")
    
    a = pd.read_parquet("output/a.parquet")['price']
    res = mitigate_impact(a, res, rel_impact_threshold=2, abs_impact_threshold=100)
    res.to_frame().to_parquet("output/e.parquet")


    #TODO: in main, add a parameter to remove the pivotality test