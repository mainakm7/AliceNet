from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from datetime import datetime
import os
from ..utils.data_matrices import events_mat, genes_mat, sf_events_upd, sf_exp_upd
from ..utils.data_dir_path import data_dir_path


def mi_regression_all():
    cols = sf_events_upd.index
    ind = sf_exp_upd.index

    def compute_mutual_info(i, j):
        event = events_mat[j, :]
        gene = genes_mat[i, :]

        # Filter out NaNs
        mask = ~np.isnan(event)
        y = event[mask]
        X = gene[mask].reshape(-1, 1)

        # Compute mutual information regression
        mi_reg_val = mutual_info_regression(X, y)
        return mi_reg_val[0]

    # Parallel computation
    results1 = Parallel(n_jobs=-1)(
        delayed(compute_mutual_info)(i, j)
        for i in range(len(sf_exp_upd))
        for j in range(len(sf_events_upd))
    )

    # Create results array
    mi_reg_parallel = np.zeros((len(sf_exp_upd), len(sf_events_upd)))
    k = 0
    for i in range(len(sf_exp_upd)):
        for j in range(len(sf_events_upd)):
            mi_reg_parallel[i, j] = results1[k]
            k += 1

    # Convert to DataFrame
    mi_reg_df = pd.DataFrame(mi_reg_parallel, index=ind, columns=cols)
    
    # Define the paths

    data_path_whole = data_dir_path()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    save_path = os.path.join(data_path_whole, f"mi_reg_all_{timestamp}.csv")

    # Save to CSV
    mi_reg_df.to_csv(save_path)

    return mi_reg_df

