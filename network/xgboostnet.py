import pandas as pd
import numpy as np
import optuna
from .custom_model import custom_model
from .hyperparameter_tuning import hyperparameter_tuning
from sklearn.metrics import mean_squared_error
from ..utils.data_dir_path import data_dir_path
from .data_preparation import data_preparation
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
import shap
from typing import Tuple, Optional
import json
import os
import requests
from fastapi import HTTPException, status

def xgboostnet(event_name: str) -> Tuple[dict, float, pd.DataFrame]:
    """
    Train an XGBoost model with PCA preprocessing and hyperparameter optimization using Optuna.

    Args:
        event_name (str): The name of the event to save in the results.

    Returns:
        tuple: Best hyperparameters found by Optuna, fit_rmse, Adjacency matrix for features after PC clustering.
    """
    try:
        # Perform data preparation
        response_data = requests.get("http://localhost:8000/network/data_prepare")
        response_data.raise_for_status()
        data = response_data.json()

        train_X = pd.DataFrame.from_dict(data["train_X"])
        train_y = pd.Series(data["train_y"])
        test_X = pd.DataFrame.from_dict(data["test_X"])
        test_y = pd.Series(data["test_y"])

        # Perform hyperparameter tuning
        response_study = requests.get("http://localhost:8000/network/hptuning")
        response_study.raise_for_status()
        study_custom = response_study.json()

        # Get best hyperparameters
        best_params_custom = study_custom["best_params"]

        # Train final model with best hyperparameters
        final_model_custom = custom_model()
        final_model_custom.pca_fit(train_X, min(10, train_X.shape[1]))  # Assuming train_X is the correct dataframe
        final_model_custom.fit(
            train_X.values, train_y.values,
            params=best_params_custom, random_state=42, apply_pca=True, verbose=False, n_splits=5
        )

        # Evaluate final model
        final_preds_custom = final_model_custom.predict(test_X.values)
        final_rmse_custom = mean_squared_error(test_y.values, final_preds_custom) ** 0.5

        # Calculating SHAP values for the model
        train_X_transformed = final_model_custom.pca_transform(train_X)

        explainer_custom = shap.TreeExplainer(final_model_custom.xgb_model, train_X_transformed)
        shap_values_custom = explainer_custom.shap_values(train_X_transformed)
        shap_values_custom_abs = np.abs(shap_values_custom)
        loadings = final_model_custom.pca_model.components_
        num_principal_components = loadings.shape[0]
        pc_name_list = ["PC" + str(i + 1) for i in range(num_principal_components)]
        shap_df_custom_abs2 = pd.DataFrame(shap_values_custom_abs, columns=pc_name_list)
        
        # Clustering the features for individual PCs and forming an adjacency matrix
        adj_matrix_whole_df = pd.DataFrame(np.zeros((train_X.shape[1], train_X.shape[1])), index=train_X.columns, columns=train_X.columns)

        for i, pci in enumerate(pc_name_list):
            pcshap = shap_df_custom_abs2[pci].copy()
            pcshap  = pcshap.values.reshape(-1, 1)
            pcloading = loadings[i, :].copy()
            pcloading = pcloading.reshape(1, -1)
            PC_sf_shap = np.dot(pcshap, pcloading)
            PC_sf_shap_df = pd.DataFrame(PC_sf_shap, columns=train_X.columns)
            Z = linkage(PC_sf_shap.T, method='ward')
            dendrogram(Z, labels=PC_sf_shap_df.columns.tolist())
            maxxluster_threshold = 11 
            cluster_labels = fcluster(Z, maxxluster_threshold, criterion='maxclust')
            clustered_data = PC_sf_shap_df.copy()
            clustered_data.loc['Cluster'] = cluster_labels
            clustered_data_pc = clustered_data.loc['Cluster']
            cluster_list = [clustered_data_pc.index[clustered_data_pc == clusterid].tolist() for clusterid in clustered_data_pc.unique()]
            for clist in cluster_list:
                for sf1 in clist:
                    for sf2 in clist:
                        if sf1 != sf2:
                            adj_matrix_whole_df.loc[sf1][sf2] += 1
        
        # Save results to a JSON file
        fit = {
            "model_best_fit_param": best_params_custom,
            "fit_RMSE": final_rmse_custom,
            "fit_adj_matrix": adj_matrix_whole_df.to_dict(orient="split")
        }
        
        event_data = {event_name: fit}

        data_path = data_dir_path(subdir="network")
        file_name = f"{event_name}_model_fit_results.json"
        whole_save_path = os.path.join(data_path, file_name)
        with open(whole_save_path, "w") as f:
            json.dump(event_data, f)

        return best_params_custom, final_rmse_custom, adj_matrix_whole_df

    except requests.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while making a request: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {e}"
        )
