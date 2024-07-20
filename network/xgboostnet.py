import pandas as pd
import numpy as np
import optuna
from .custom_model import custom_model
from .hyperparameter_tuning import hyperparameter_tuning
from sklearn.metrics import mean_squared_error
from ..utils.data_dir_path import data_dir_path
from .data_preparation import data_preparation
from typing import Optional, Tuple
import json
import os
import requests
from fastapi import HTTPException, status



def xgboostnet() -> Tuple[dict, float]:
    """
    Train an XGBoost model with PCA preprocessing and hyperparameter optimization using Optuna.

    Returns:
        tuple: Best hyperparameters found by Optuna, fit_rmse.
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
        final_model_custom.fit(train_X.values, train_y.values, params=best_params_custom, random_state=42, apply_pca=True, verbose=False, n_splits=5)

        # Evaluate final model
        final_preds_custom = final_model_custom.predict(test_X.values)
        final_rmse_custom = mean_squared_error(test_y.values, final_preds_custom) ** 0.5

        # Save results to a JSON file
        fit = {
            "model_best_fit_param": best_params_custom,
            "fit_RMSE": final_rmse_custom
        }
        data_path = data_dir_path(subdir="network")
        file_name = "model_fit_results.json"
        whole_save_path = os.path.join(data_path, file_name)
        with open(whole_save_path, "w") as f:
            json.dump(fit, f)

        return best_params_custom, final_rmse_custom

    except requests.RequestException as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred while making a request: {e}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {e}")
