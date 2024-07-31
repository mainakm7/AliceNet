from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
import os
from ..utils.data_dir_path import data_dir_path

def mi_regression_all(sf_exp_df, sf_events_df):
    """
    Perform mutual information regression for all combinations of genes and splicing events,
    save results to a CSV file.
    """
    
    if sf_events_df.empty or sf_exp_df.empty:
        raise ValueError("Input DataFrames are empty.")
    
    # Extract column and index names
    cols = sf_events_df.index
    ind = sf_exp_df.index

    def compute_mutual_info(i, j):
        """
        Compute mutual information between a gene and a splicing event.
        """
        
        event = sf_events_df.iloc[j, :].values
        gene = sf_exp_df.iloc[i, :].values

        # Filter out NaNs
        mask = ~np.isnan(event)
        y = event[mask]
        X = gene[mask].reshape(-1, 1)

        # Compute mutual information regression
        mi_reg_val = mutual_info_regression(X, y)
        return mi_reg_val[0]

    # Parallel computation
    results = Parallel(n_jobs=-1)(
        delayed(compute_mutual_info)(i, j)
        for i in range(len(sf_exp_df))
        for j in range(len(sf_events_df))
    )

    # Reshape results into array
    mi_reg_parallel = np.array(results).reshape(len(sf_exp_df), len(sf_events_df))

    # Convert to DataFrame
    mi_reg_df = pd.DataFrame(mi_reg_parallel, index=ind, columns=cols)

    data_path_whole = data_dir_path(subdir="MI")
    save_path = os.path.join(data_path_whole, "mutualinfo_reg_one_to_one_MI_all.csv")

    # Save to CSV
    mi_reg_df.to_csv(save_path)

    return mi_reg_df