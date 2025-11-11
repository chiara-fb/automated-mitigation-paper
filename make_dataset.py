import pandas as pd
from amp_tests.structural_test import residual_supplier_index
from pathlib import Path
from datetime import datetime


def offer_based_ref(x, days):
    ref = x.shift(24).rolling(days * 24, min_periods=1).mean() #adding min_periods=1 to avoid NaN
    return ref


def make_outcome(bids:pd.DataFrame) -> pd.DataFrame:
    """
    Computes max bid and reference levels.
    Returns pd.DataFrame with index [DateTime, Masked Asset ID, Masked Lead Participant ID].
    """
    bids = bids.copy()
    bids = bids.sort_index(level="DateTime")
    if 'Unit Status' in bids.columns: # for NYISO, unit status are unclear
        bids = bids[bids["Unit Status"] != "UNAVAILABLE"]

    price = bids.filter(regex='Segment [0-9]+ Price')
    mw = bids.filter(regex='Segment [0-9]+ MW')
    price = price.rename(columns={c: i for i, c in enumerate(price.columns)})
    mw = mw.rename(columns={c: i for i, c in enumerate(mw.columns)})
    max_bid = price.max(axis=1)
    
    for col in mw.columns: # considers only bids within bounds
        mw[col] = mw[col].where(price[col] > 0, other=0)
        mw[col] = mw[col].where(price[col] < 800, other=0)
    
    avg_bid = (price * mw).sum(axis=1) / mw.sum(axis=1)   
    avg_group = avg_bid.groupby("Masked Asset ID", group_keys=False)
    ref_level = avg_group.apply(lambda x: offer_based_ref(x, 90))
    dep_vars = pd.concat([max_bid, ref_level], axis=1)
    dep_vars.columns = ['max_bid', 'ref_level']
    
    return dep_vars



def make_pivotality_treatment(bids:pd.DataFrame, 
                   load_fcst: pd.DataFrame, 
                   reserves: pd.DataFrame|int=0
                   ) -> pd.DataFrame:
    """
    Computes treatment variables for pivotality. 
    Returns: pd.DataFrame with index [DateTime, Masked Asset ID, Masked Lead Participant ID].
    """

    rsi = residual_supplier_index(bids, load_fcst, 
                                  reserves, 
                                  substract_must_run=True)
    _, rsi = bids.align(rsi, axis=0, join='left')
    pst = (rsi > 1).astype(int)
    treat_vars = pd.concat([rsi, pst], axis=1)
    treat_vars.columns = ['rsi', 'is_not_pivotal']
    treat_vars = treat_vars.dropna(how='any', axis=0)

    return treat_vars



def make_congestion_treatment(rt_cong:pd.DataFrame,
                              load_fcst:pd.Series) -> pd.DataFrame:
    """
    Computes treatment variables for congestion for day-ahead and real-time markets.
    Returns: pd.DataFrame with index [DateTime].
    """
    #ignore nyc because it is always considered a constrained area
    load_zones = ['capitl', 'centrl', 'dunwod', 'genese', 'hudvl', 'longil', 'mhkvl', 'millwd', 'north', 'west']
    load_fcst, rt_cong = load_fcst.copy()[load_zones], rt_cong.copy()[load_zones]
    rel_load = load_fcst[load_zones].div(load_fcst[load_zones].sum(axis=1), axis=0)
    cong_dict = {}
  
    cong_zones = rt_cong * rel_load
    cong_dict = {'avg_cong': cong_zones.sum(axis=1),
                 'max_cong': rt_cong.max(axis=1),
                 'avg_cong_1h_lag': cong_zones.sum(axis=1).shift(1),
                 'max_cong_1h_lag': rt_cong.max(axis=1).shift(1),
                 'avg_cong_3h_lag': cong_zones.sum(axis=1).shift(3),
                 'max_cong_3h_lag': rt_cong.max(axis=1).shift(3)}
    
    treat_vars = []
    for name, score in cong_dict.items():
        score = score.rename(name)
        treatment = (score > 0.04).astype(int)
        treatment.name = f'is_{name}'
        treat_vars.extend([score, treatment])
    
    treat_vars = pd.concat(treat_vars, axis=1)
    
    return treat_vars



def make_covariates(bids:pd.DataFrame, 
                    load_fcst:pd.Series, 
                    gas_prices:pd.DataFrame, 
                    wind_fcst:pd.Series,
                    net_imports:pd.Series,
                    da_must_take:pd.Series,
                    temperature:pd.Series,
                    ) -> pd.DataFrame:
    """
    Add covariates for the regression and the cluster analysis:
        - load forecast
        - res load fcst
        - day-ahead must take 
        - week-before gas price
        - time dummies
        - economic maximum (asset_mw)
        - economic maximum (company_mw)
    
    Returns: pd.DataFrame with index [DateTime, Masked Asset ID, Masked Lead Participant ID].
    """
    gas = gas_prices['Price'].rename('gas_prices').shift(7) #use gas price from the previous week
    gas = gas.resample('1h').ffill()
    gas.index.name = 'DateTime'
    
    res_load = (load_fcst - wind_fcst - net_imports).rename('res_load')
    covs = pd.concat([load_fcst, res_load, da_must_take, gas, temperature], axis=1)
    covs.columns = ['load_fcst', 'res_load', 'da_must_take', 'gas_prices', 'temperature']
    _, covs = bids.align(covs, axis=0, join='left')

    asset_mw = bids['Economic Maximum'].rename('asset_mw')
    company_mw = bids.groupby(['DateTime','Masked Lead Participant ID'])['Economic Maximum'].sum()
    _,company_mw = bids.align(company_mw.rename('company_mw'), axis=0, join='left')
    covs = pd.concat([covs, asset_mw, company_mw], axis=1)
    
    index = bids.index.to_frame(index=True)
    
    for freq in ['hour', 'quarter']: 
        time = getattr(index['DateTime'].dt, freq)
        dummies = pd.get_dummies(time, prefix=freq, 
                                drop_first=False)
        dummies = dummies.set_index(covs.index).astype(int)
        covs = pd.concat([covs, dummies], axis=1)   
    
    covs = covs.dropna(how='any', axis=0)

    return covs



if __name__ == "__main__":
    MARKET = 'ISO-NE' # 'ISO-NE' or 'NYISO'
    PATH = Path(r'C:\Users\c.fusarbassini\OneDrive - Hertie School\25 ML-Strom\2 Literatur & Research ideas\AP 3\data')
    
    rt_bids = pd.read_parquet(PATH / MARKET / 'rt_bids_2018-2019.parquet')
    da_bids = pd.read_parquet(PATH / MARKET / 'da_bids_2018-2019.parquet')
    da_must_take = da_bids['Must Take Energy'].groupby('DateTime').sum()
    gas_prices = pd.read_parquet(PATH / 'gas_2018-2019.parquet')
    load_fcst_zones = pd.read_parquet(PATH / MARKET / 'load_forecast_2018-2019.parquet')
    load_fcst = load_fcst_zones.sum(axis=1).rename('load_forecast')
    temperature = pd.read_parquet(PATH / MARKET / 'temperature_2018-2019.parquet')['AverageTemperature']


    outcome = make_outcome(rt_bids) # substitute with da_bids if you want to compute day-ahead
    print(f'Outcome variables computed.')

    if MARKET == 'ISO-NE':
        wind_fcst = pd.read_parquet(PATH / MARKET / 'wind_forecast_2018-2019.parquet')['Wind'] # missing from nyiso
        reserves = pd.read_parquet(PATH / MARKET / 'reserves_2018-2019.parquet') # missing from nyiso
        reserves = reserves.sum(axis=1).rename('reserves')
        net_imports = pd.read_parquet(PATH / MARKET / 'interchange_2018-2019.parquet') # missing from nyiso
        net_imports = net_imports.sum(axis=1).rename('net_imports')
        treat = make_pivotality_treatment(rt_bids, load_fcst, reserves)
        print(f'Treatment variables computed.')
    
    elif MARKET == 'NYISO':
        wind_fcst, net_imports = 0, 0
        da_congestion = pd.read_parquet(PATH / MARKET / 'da_shadow_prices_2018-2019.parquet')
        rt_congestion = pd.read_parquet(PATH / MARKET / 'rt_shadow_prices_2018-2019.parquet')
        treat = make_congestion_treatment(rt_congestion, load_fcst_zones)
        outcome, treat = outcome.align(treat, axis=0, join='left')
        print(f'Treatment variables computed.')
    

    covariates = make_covariates(rt_bids, load_fcst, 
                                 gas_prices, 
                                 wind_fcst=wind_fcst, 
                                 da_must_take=da_must_take, 
                                 net_imports=net_imports, 
                                 temperature=temperature)
    print(f'Covariates computed.')
    
    dataset = pd.concat([outcome, treat, covariates], axis=1)
    dataset = dataset.dropna(how='any', axis=0)
    dataset.to_parquet(f'{datetime.now().strftime("%Y-%m-%d")}_{MARKET.lower()}_dataset.parquet')

    
