
from tqdm import tqdm 
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import doubleml as dml
from doubleml.rdd import RDFlex
from utils import sharp_treatment_assignment, fuzzy_treatment_assignment
import yaml
import warnings
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')


class RegressionDiscontinuity:
    
    def __init__(self, cfg):
        self.cfg = cfg

    ### DOES ###
    def fit_regression(self, df):
        """
        Fit RDD regression.
            Returns: doubleml.rdd.RDFlex.
        """
        
        np.random.seed(self.cfg['seed'])
        
        if cfg['unit_dummy']:
            unit_dummy = pd.get_dummies(df.index.get_level_values('Masked Asset ID'), dtype=float)
            unit_dummy = unit_dummy.set_index(df.index)
            df = pd.concat([df, unit_dummy], axis=1)
            units = unit_dummy.columns.tolist()
        else:
            units = []

        dml_data = dml.DoubleMLData(df,     
                                y_col='max_bid', 
                                x_cols=self.cfg['x_cols'] + units, 
                                d_cols='d_cols', 
                                s_col='s_col', 
                                )
    
        ml_g = RandomForestRegressor(n_estimators=self.cfg['n_estimators'], 
                            max_depth=self.cfg['max_depth'],
                            random_state=self.cfg['seed']) 
        
        ml_m = RandomForestClassifier(n_estimators=self.cfg['n_estimators'], 
                             max_depth=self.cfg['max_depth'],
                             random_state=self.cfg['seed'])
  
        dml_obj = RDFlex(dml_data, 
                         ml_g=ml_g, ml_m=ml_m, 
                         cutoff=0, 
                         h = self.cfg['bandwidth'],
                         fuzzy=self.cfg['fuzzy'], 
                        )   

        dml_obj.fit()
        
        return dml_obj

    
    
    def evaluate_regression(self, dml_obj):
        """
        Evaluate RDD regression.
            Returns: dict.
        """
        
        res = {
            'bandwidth': dml_obj.h[0],
            't_stat': dml_obj.t_stat[0],
            'p_val': dml_obj.pval[0],
            'coef': dml_obj.coef[0],
            'ci_lower': dml_obj.confint(level=0.95).iloc[0,0],
            'ci_upper': dml_obj.confint(level=0.95).iloc[0,1],
        }
        return res
    

    
    def iterate_assets(self, dataset, test=False): 
        """Iterate over all assets. 
            Returns: dict."""

        results = {}
        i = 0

        for (name, df) in tqdm(dataset.groupby(self.cfg["groupby"])):
            if test:
                i+=1
                if i > test:
                    break
            
            try:
                dml_obj = self.fit_regression(df) 
                res = self.evaluate_regression(dml_obj)
                results[name] = res

            except Exception as e:
                print(f'Exception for {self.cfg["groupby"]} {name}: {e}')

        return results
    


if __name__ == "__main__":

    
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    dataset = pd.read_parquet(cfg["filepath"])
    dataset = dataset[dataset.index.get_level_values('DateTime').year == 2019]
    dataset = dataset.sort_index()

    if cfg['market'] == 'iso-ne':
        dataset['s_col'] = cfg['cutoff'] - dataset[cfg['s_col']]
    else:
        dataset['s_col'] = dataset[cfg['s_col']] - cfg['cutoff']

    if cfg['fuzzy']:
        dataset['d_cols'] = fuzzy_treatment_assignment(dataset['s_col'], std=cfg['fuzzy_std'], seed=cfg['seed'])
    else:
        dataset['d_cols'] = sharp_treatment_assignment(dataset['s_col'])

    rdd = RegressionDiscontinuity(cfg)
    res = rdd.iterate_assets(dataset)
    res_df = pd.DataFrame.from_dict(res, orient='index')

    folder = Path(datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(exist_ok=True)
    res_df.to_excel(folder / 'rdd_res.xlsx')
    with open(folder / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f)
    print(f'Results saved for {len(res_df)} {cfg["groupby"]}s.')