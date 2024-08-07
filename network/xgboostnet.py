import pandas as pd
import numpy as np
from .custom_model import custom_model
from sklearn.metrics import mean_squared_error
from ..utils.data_dir_path import data_dir_path
from typing import Tuple, Dict, Any
import json
import os
from fastapi import HTTPException, status


def xgboostnet(data_dict: Dict[str, pd.DataFrame], best_fit: Dict[str, Any], Dataparam: Dict[str, Any], *args, **kwargs) -> Tuple[Dict[str, Any], float, custom_model, pd.DataFrame]:
    """
    Train an XGBoost model with PCA preprocessing and hyperparameter optimization using Optuna.

    Args:
        data_dict (Dict[str, pd.DataFrame]): Data including train and test datasets.
        best_fit (Dict[str, Any]): Best hyperparameters for the model.
        Dataparam (Dict[str, Any]): Additional parameters such as specific_gene, event, and test_size.

    Returns:
        Tuple[Dict[str, Any], float, custom_model, pd.DataFrame]: Best hyperparameters, RMSE, trained model, and training data.
    """
    try:
        train_X, train_y, test_X, test_y = data_dict["train_X"], data_dict["train_y"], data_dict["test_X"], data_dict["test_y"]
        event = Dataparam.get("eventname")
        
        # Initialize and train the final model with best hyperparameters
        final_model_custom = custom_model()
        final_model_custom.pca_fit(train_X, min(10, train_X.shape[1]))  # Perform PCA
        final_model_custom.fit(
            train_X.values, train_y.values,
            params=best_fit, random_state=42, apply_pca=True, verbose=False, n_splits=5
        )

        # Evaluate the final model
        final_preds_custom = final_model_custom.predict(test_X.values)
        final_rmse_custom = mean_squared_error(test_y.values, final_preds_custom) ** 0.5

        # Save results to a JSON file
        fit = {
            "model_best_fit_param": best_fit,
            "fit_RMSE": final_rmse_custom
        }
        event_data = {event: fit}

        data_path = data_dir_path(subdir="network")
        file_name = f"{event}_model_fit_results.json"
        whole_save_path = os.path.join(data_path, file_name)
        
        # Save JSON data
        with open(whole_save_path, "w") as f:
            json.dump(event_data, f, indent=4)

        return best_fit, final_rmse_custom, final_model_custom, train_X

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {str(e)}")