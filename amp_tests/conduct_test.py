import pandas as pd
import numpy as np
import re



def ref_level(bids:pd.DataFrame, min_bid=0, max_bid=800, days=90) -> pd.Series:
    
    price = bids.filter(regex='Segment [0-9]+ Price')
    mw = bids.filter(regex='Segment [0-9]+ MW')
    price = price.rename(columns={c: i for i, c in enumerate(price.columns)})
    mw = mw.rename(columns={c: i for i, c in enumerate(mw.columns)})
    
    for col in mw.columns: # considers only bids within bounds
        mw[col] = mw[col].where(price[col] > min_bid, other=0)
        mw[col] = mw[col].where(price[col] < max_bid, other=0)
    
    avg_bid = (price * mw).sum(axis=1) / mw.sum(axis=1)   
    avg_group = avg_bid.groupby("Masked Asset ID", group_keys=False)
    ref = lambda x, days: x.shift(24).rolling(days * 24, min_periods=1).mean()
    ref_level = avg_group.apply(lambda x: ref(x, days))

    return ref_level



def reference_levels(
    bids:pd.DataFrame,
    hub_price:pd.Series,
    accepted_only:bool = False,
    fill_nans:bool = True,
    must_run:bool = False,
    lower_bound: float = -1000,
    upper_bound: float = 1000,
    days: int = 90,
) -> pd.DataFrame:
    """ "Computes offer-based reference levels for a series of bids indexed by
    DateTime, Masked Lead Participant ID and Masked Asset ID. First,
    computes daily average of bids then computes a rolling average of the last
    days (current day not included). The reference levels are then resampled to
    hourly values, i.e. each day has 24 identical reference levels.

    Args:
        bids (pd.DataFrame): DataFrame with bids indexed by DateTime, Masked Lead Participant ID and Masked Asset ID.
        hub_price (pd.Series): Series with hub price indexed by DateTime.
        accepted_only (bool): If True, only considers accepted bids. Defaults to True.
        fill_nans (bool): If True, fills NaNs with forward and backward filling. Defaults to True.
        must_run (bool): If True, includes must run bids. Defaults to False.
        lower_bound (float): Lower bound for the bids. Bids <= are set to 0 MW. Defaults to -1000.
        upper_bound (float): Upper bound for the bids. Bids >= are set to 0 MW. Defaults to 1000.
        days (int): Number of days to consider for the rolling average. Defaults to 90.

    Returns:
        ref_levels (pd.Series): Series with reference levels indexed by DateTime, Masked Lead Participant ID and Masked Asset ID.
        NOTE: Some assets might have incomplete ref_levels due to lack of accepted bids.
    """

    copy_bids = bids.copy()
    
    if must_run: # if must run, keep both must run and economic units
        copy_bids = copy_bids[copy_bids["Unit Status"] != "UNAVAILABLE"]
    else: 
        copy_bids = copy_bids[copy_bids["Unit Status"] == "ECONOMIC"]

    
    copy_bids, hub_price = copy_bids.align(hub_price, axis=0, join='inner')

    revenue = pd.Series(0, index=copy_bids.index, dtype=float)
    quantity = pd.Series(0, index=copy_bids.index, dtype=float)

    for s in range(1, 11):
        price_s = copy_bids[f"Segment {s} Price"]
        quantity_s = copy_bids[f"Segment {s} MW"]

        # if accepted only, the upper bound should be at least as tight as the upper bound, lower if hub price is lower
        if accepted_only: 
            upper_bound = hub_price.where(hub_price < upper_bound, other=upper_bound)
        in_range = (price_s > lower_bound) & (price_s < upper_bound)
        price_s = price_s.where(cond=in_range, other=0)
        quantity_s = quantity_s.where(cond=in_range, other=0)
        
        revenue += price_s * quantity_s
        quantity += quantity_s
      
    avg = revenue / quantity
    group_avg = avg.groupby(["Masked Lead Participant ID", "Masked Asset ID"], group_keys=False)

    # shift back by one day (24 hours) and compute rolling average
    ref = group_avg.apply(lambda x: x.shift(24).rolling(days*24).mean())

    if fill_nans:
        ref = ref.groupby(["Masked Lead Participant ID", "Masked Asset ID"]).ffill().bfill()
    
    ref = ref.dropna(how="all")
    ref.name = "Reference Level"

    return ref
    



def mitigate_bids(
    bids: pd.DataFrame,
    pst: pd.Series,
    ref_levels: pd.Series,
    rel_ref: int = 3,
    abs_ref: int = 100,
    default_ref: float = 0,
    verbose: bool = True,
) -> pd.DataFrame:


    ### fill missing ref levels so that no unit is removed
    ref_fill = lambda x: x.ffill().bfill().fillna(default_ref)
    ref_levels = ref_levels.groupby('Masked Asset ID').transform(ref_fill)
    df = bids.join(pst).join(ref_levels)
    bids, pst, ref_levels = df.iloc[:, :-2], df.iloc[:, -2], df.iloc[:, -1]
    threshold = ref_levels.map(lambda x: min(x + abs_ref, x * rel_ref))

    print("Structural test (# bids):", pst.sum()) if verbose else None
    
    for col in bids.columns:
        # check whether unit is PST (structure) and bid is above ref level (conduct)
        if re.match("Segment [0-9]+ Price", col):
            cond = (bids[col] > threshold) & pst
            bids.loc[cond, col] = ref_levels[cond]

        else:
            continue
        print(col, "mitigated bids:", cond.sum()) if verbose else None

    bids = bids.dropna(how="all")
    return bids
