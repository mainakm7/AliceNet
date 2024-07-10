import pandas as pd
import numpy as np
import optuna
from .custom_model import custom_model
from sklearn.metrics import mean_squared_error
from ..utils.data_dir_path import data_dir_path
from .data_preparation import data_preparation
from typing import Optional, Tuple
import json
import os

def xgboostnet(event_index: int = 1, specific_gene: Optional[str] = None) -> Tuple[dict, float]:
    """
    Train an XGBoost model with PCA preprocessing and hyperparameter optimization using Optuna.

    Args:
        event_index (int): Index of the splicing event to use.
        specific_gene (Optional[str]): Gene to filter the splicing events. If None, uses all genes.

    Returns:
        tuple: Best hyperparameters found by Optuna, fit_rmse.
    """
    
    # Perform data preparation
    train_X, train_y, test_X, test_y = data_preparation(event_index, specific_gene)

    def objective_custom(trial):
        """
        Objective function for Optuna to optimize hyperparameters.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.

        Returns:
            float: Root Mean Squared Error (RMSE) of the model.
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1e1, log=True),
            'gamma': trial.suggest_float('gamma', 1e-3, 1e1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1e1, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1e1, log=True)
        }
        
        model = custom_model()
        model.pca_fit(train_X, min(10, train_X.shape[1]))  # Assuming train_X is the correct dataframe
        model.fit(train_X.values, train_y.values, params=params, early_stopping_rounds=10, random_state=42, apply_pca=True, verbose=False, n_splits=5)
        
        preds = model.predict(test_X.values)
        rmse = mean_squared_error(test_y.values, preds) ** 0.5
        return rmse

    # Perform hyperparameter optimization
    study_custom = optuna.create_study(direction='minimize')
    study_custom.optimize(objective_custom, n_trials=100, n_jobs=-1)

    # Get best hyperparameters
    best_params_custom = study_custom.best_params

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
    sf_events_df_individual = train_X.columns  # Assuming this gives the correct splicing events
    file_name = f"{sf_events_df_individual.name}.json"
    whole_save_path = os.path.join(data_path, file_name)
    with open(whole_save_path, "w") as f:
        json.dump(fit, f)

    print("\n Best Custom XGBoostReg parameters:", best_params_custom)
    print("\n Final RMSE for custom model: ", final_rmse_custom)
    
    return best_params_custom, final_rmse_custom
