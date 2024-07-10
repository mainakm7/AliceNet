import pandas as pd
import numpy as np
import optuna
from .custom_model import custom_model
from sklearn.metrics import mean_squared_error


def hyperparameter_tuning(train_X: pd.DataFrame, train_y: pd.DataFrame, test_X: pd.DataFrame, test_y: pd.DataFrame):
    """
    Perform hyperparameter tuning using Optuna for a custom model.

    Args:
        train_X (pd.DataFrame): Training features.
        train_y (pd.Series): Training labels.
        test_X (pd.DataFrame): Testing features.
        test_y (pd.Series): Testing labels.

    Returns:
        function: Objective function for Optuna hyperparameter optimization.
    """
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

    return objective_custom
