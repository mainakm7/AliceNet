import pandas as pd
import numpy as np
from .custom_model import custom_model
from sklearn.metrics import mean_squared_error
from ..utils.data_dir_path import data_dir_path
from typing import Tuple, Dict
import json
import os
import requests
from fastapi import HTTPException, status

final_model_fit: custom_model
train_data: pd.DataFrame

def xgboostnet(hparams: Dict, dataparams: Dict) -> Tuple[Dict, float, custom_model, pd.DataFrame]:
    """
    Train an XGBoost model with PCA preprocessing and hyperparameter optimization using Optuna.

    Args:
        hparams (Dict): Hyperparameters for tuning.
        dataparams (Dict): Data parameters including specific_gene, event, and test_size.

    Returns:
        Tuple[Dict, float, custom_model, pd.DataFrame]: A tuple containing the best hyperparameters found by Optuna,
            the root mean squared error of the model, the fitted custom model, and the training data used.
    """
    specific_gene = dataparams.get("specific_gene")
    event = dataparams.get("event")
    test_size = dataparams.get("test_size")
    
    try:
        # Perform data preparation
        response_data = requests.post(
            "http://localhost:8000/network/data_prepare",
            json={"specific_gene": specific_gene, "event": event, "test_size": test_size}
        )
        response_data.raise_for_status()
        data = response_data.json()

        # Convert JSON data to DataFrames
        train_X = pd.DataFrame.from_dict(data["train_X"])
        train_y = pd.Series(data["train_y"])
        test_X = pd.DataFrame.from_dict(data["test_X"])
        test_y = pd.Series(data["test_y"])

        # Perform hyperparameter tuning
        response_study = requests.post(
            "http://localhost:8000/network/hptuning",
            json={**hparams, **dataparams}
        )
        response_study.raise_for_status()
        study_custom = response_study.json()

        # Get best hyperparameters
        best_params_custom = study_custom["best_params"]

        # Initialize and train the final model with best hyperparameters
        final_model_custom = custom_model()
        final_model_custom.pca_fit(train_X, min(10, train_X.shape[1]))  # Perform PCA
        final_model_custom.fit(
            train_X.values, train_y.values,
            params=best_params_custom, random_state=42, apply_pca=True, verbose=False, n_splits=5
        )

        # Evaluate the final model
        final_preds_custom = final_model_custom.predict(test_X.values)
        final_rmse_custom = mean_squared_error(test_y.values, final_preds_custom) ** 0.5

        global final_model_fit, train_data
        
        final_model_fit = final_model_custom
        train_data = train_X
        
        # Save results to a JSON file
        fit = {
            "model_best_fit_param": best_params_custom,
            "fit_RMSE": final_rmse_custom
        }
        
        event_data = {event: fit}

        data_path = data_dir_path(subdir="network")
        file_name = f"{event}_model_fit_results.json"
        whole_save_path = os.path.join(data_path, file_name)
        
        # Save JSON data
        with open(whole_save_path, "w") as f:
            json.dump(event_data, f, indent=4)

        return best_params_custom, final_rmse_custom, final_model_custom, train_X

    except requests.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while making a request: {e}"
        )
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected data format: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {e}"
        )
