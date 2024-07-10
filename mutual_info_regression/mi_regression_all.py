from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from datetime import datetime
import os
from ..utils.data_matrices import sf_events_upd, sf_exp_upd
from ..utils.data_dir_path import data_dir_path


def mi_regression_all():
    """
    Perform mutual information regression for all combinations of genes and splicing events,
    save results to a CSV file.

    Returns:
        pd.DataFrame: DataFrame containing mutual information regression values.
    """
    # Extract column and index names
    cols = sf_events_upd.index
    ind = sf_exp_upd.index

    def compute_mutual_info(i, j):
        """
        Compute mutual information between a gene and a splicing event.

        Args:
            i (int): Index of the gene.
            j (int): Index of the splicing event.

        Returns:
            float: Mutual information regression value.
        """
        event = sf_events_upd.iloc[j, :].values
        gene = sf_exp_upd.iloc[i, :].values

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
        for i in range(len(sf_exp_upd))
        for j in range(len(sf_events_upd))
    )

    # Reshape results into array
    mi_reg_parallel = np.array(results).reshape(len(sf_exp_upd), len(sf_events_upd))

    # Convert to DataFrame
    mi_reg_df = pd.DataFrame(mi_reg_parallel, index=ind, columns=cols)

    # Define the save path using current timestamp
    data_path_whole = data_dir_path()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    save_path = os.path.join(data_path_whole, f"mi_reg_all_{timestamp}.csv")

    # Save to CSV
    mi_reg_df.to_csv(save_path)

    return mi_reg_df
