
"""

Here we compute the bidder-level statistics for NYISO and ISO-NE data.

"""

import pandas as pd
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')



    
def compute_statistics(dataset):

    stats = ["num_units", 
             "median_volume",
             "avg_volume", 
             "median_bid",
             "avg_bid", 
             "std_bid"]
    stats_df = pd.DataFrame(columns=stats)

    for (b, bidder_df) in dataset.groupby("bidder"):
        groupby_df = bidder_df.groupby("datetime")
        # max number of bidding units per datetime
        stats_df.loc[b, "num_units"] = groupby_df["unit"].nunique().max()
        # median / avg total bid volume by datetime
        stats_df.loc[b, "median_volume"] = groupby_df["asset_mw"].sum().median()
        stats_df.loc[b, "avg_volume"] = groupby_df["asset_mw"].sum().mean()
        # median / avg / std of bids in the dataset
        stats_df.loc[b, "median_bid"] = bidder_df["max_bid"].median()
        stats_df.loc[b, "avg_bid"] = bidder_df["max_bid"].mean()
        stats_df.loc[b, "std_bid"] = bidder_df["max_bid"].std()
    
    return stats_df


        


if __name__ == "__main__":

    path = Path("data")

    for market in ["iso-ne", "nyiso"]:

        df = pd.read_parquet(path / f"2025-08-12_{market}_dataset.parquet")
        df = df[df.index.get_level_values('DateTime').year == 2019]
        df = df.reset_index()
        df = df.rename(columns={"DateTime": "datetime", 
                                "Masked Lead Participant ID": "bidder", 
                                "Masked Asset ID": "unit"})

        stats = compute_statistics(df)
        stats.to_excel(path / f"{market}_bidder_stats.xlsx")
