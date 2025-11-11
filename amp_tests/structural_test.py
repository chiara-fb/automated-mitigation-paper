import pandas as pd


def residual_supplier_index(
    bids: pd.DataFrame,
    load: pd.Series,
    reserves: pd.Series = 0,
    interchange: pd.Series = 0,
    group_by: str|list = ["DateTime", "Masked Lead Participant ID"],
    substract_must_run: bool = True,
    remove_unavailable: bool = True
) -> pd.Series:
    """Computes the residual supplier index for each supplier in the market.
    The residual supplier index is the ratio of the available capacity of a
    supplier to the total available capacity in the market minus
    the demand (load forecast, reserves, interchange)."""

    if remove_unavailable:
        avail_bids = bids[bids["Unit Status"] != "UNAVAILABLE"]
    else:
        avail_bids = bids.copy()
    
    avail_mw = avail_bids["Economic Maximum"] 
    
    if substract_must_run:
        # Remove must run bids from available capacity
        avail_mw -= avail_bids["Must Take Energy"]
    
    tot_mw = avail_mw.groupby("DateTime").sum()
    supplier_mw = avail_mw.groupby(group_by).sum()
    
    if type(interchange) == pd.Series:
        interchange = interchange.bfill()
    if type(reserves) == pd.Series:
        reserves = reserves.bfill()
    
    demand_mw = load.bfill() + interchange + reserves   
    rsi = (tot_mw - supplier_mw) / demand_mw
    rsi.name = "Residual Supplier Index"

    return rsi



def pivotal_supplier_test(
    bids: pd.DataFrame,
    load: pd.Series,
    reserves: pd.Series = 0,
    interchange: pd.Series = 0,
    group_by: str|list = ["DateTime", "Masked Lead Participant ID"],
    substract_must_run: bool = True,
    remove_unavailable: bool = True
    ) -> pd.Series:
    """Computes the pivotal supplier test for each supplier in the market.
    The pivotal supplier test checks if market has enough supply capacity
    w/o supplier.
    #TODO: check numbers are different than IS0-NE market monitor (30%)
    """

    rsi = residual_supplier_index(bids, load, reserves, interchange, group_by, substract_must_run, remove_unavailable)
    pst = rsi < 1
    pst.name = "Pivotal Supplier Test"

    return pst


def congested_area_test(prices: pd.DataFrame) -> pd.Series:
    """Computes a series of boolean depending on whether an aread is congested (difference to
    Hub LMP >= 25 $/MWh) for any zonal node."""

    zonal_prices = prices.filter(like=".Z")
    hub_prices = prices[".H.Internal_Hub"]
    congested = (zonal_prices.sub(hub_prices, axis=0)) > 25
    return congested.any(axis=1)
