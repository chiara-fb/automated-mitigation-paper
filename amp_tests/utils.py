
import pandas as pd, numpy as np
from scipy.stats import norm, bernoulli


def fuzzy_prob(centered_x:list|np.ndarray, 
               std=0.1, # Uncertainty around the cutoff
               ) -> np.ndarray: # Expected value of treatment at the cutoff
    """Define treatment probability for fuzzy design"""
    
    centered_x = np.array(centered_x)
    #const = norm.ppf(cutoff_prob) # constant to adjust the probability at the cutoff
    #prob = norm.cdf(centered_x / std + const)
    prob = norm.cdf(centered_x / std)
    return prob


def fuzzy_treatment_assignment(centered_x:list|np.ndarray, 
                               std=0.1, # Uncertainty around the cutoff
                               seed=None) -> np.ndarray: # seed for reproducibility
    """
    Assigns treatment based on a fuzzy design around a cutoff.
    The treatment is always assigned to the right of the cutoff.
    """

    prob = fuzzy_prob(centered_x, std=std)
    rng = np.random.default_rng(seed=int(seed)) # set seed
    treat_assigned = bernoulli.rvs(p=prob, random_state=rng)
    #treat_assigned = pd.Series(treat_assigned, index=centered_x.index)
    
    return treat_assigned


def sharp_treatment_assignment(centered_x: list|np.ndarray) -> np.ndarray: # Cutoff value for treatment assignment

    """
    Assigns treatment based on a sharp design around a cutoff.
    The treatment is always assigned to the right of the cutoff.
    """    
    centered_x = np.array(centered_x)
    treat_assigned = (centered_x >= 0).astype(int)
    
    return treat_assigned


def get_incremental_bids(
    bids: pd.DataFrame,
    p_floor: int = -150.0,
    p_ceil: int = 1000.0,
    must_run: bool = True,
) -> pd.DataFrame:
    """Returns all pairs of incremental bids (price, quantity)
    for each hour, capping economic bids to be within the
    range (p_floor, p_ceil). The range is NOT inclusive.
    Energy Offer Floor and Cap are defined by ISO-NE. Sources:
    https://www.iso-ne.com/participate/support/faq/emarket"""

    if must_run: 
        econ_bids = bids[bids["Unit Status"] != "UNAVAILABLE"]
    else:
        econ_bids = bids[bids["Unit Status"] == "ECONOMIC"]
        
    inc_bids = []

    for i in range(1, 11):
        segm_bids = econ_bids[[f"Segment {i} Price", f"Segment {i} MW"]]
        segm_bids = segm_bids.rename(
            columns={f"Segment {i} Price": "Price", f"Segment {i} MW": "MW"}
        )
        segm_bids = segm_bids.dropna(how="all", axis=0)
        inc_bids.append(segm_bids)
    inc_bids = pd.concat(inc_bids, axis=0)
    inc_bids = inc_bids[(inc_bids["Price"] > p_floor) & (inc_bids["Price"] < p_ceil)]
    #inc_bids["Price"] = inc_bids["Price"].clip(lower=p_floor, upper=p_ceil)
    
    return inc_bids