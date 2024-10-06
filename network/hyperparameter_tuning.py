import pandas as pd
import numpy as np
import optuna
from .custom_model import custom_model
from sklearn.metrics import mean_squared_error
import logging



def hyperparameter_tuning(train_X: pd.DataFrame, train_y: pd.Series, test_X: pd.DataFrame, test_y: pd.Series, *args, **kwargs):
    """
    Perform hyperparameter tuning using Optuna for a custom model.
    """
    

    def objective_custom(trial):
        """
        Objective function for Optuna to optimize hyperparameters.
        """
        try:
            # Extract hyperparameters from kwargs with default values
            n_estimators = trial.suggest_int('n_estimators', *kwargs.get('n_estimators', (50, 200)))
            max_depth = trial.suggest_int('max_depth', *kwargs.get('max_depth', (3, 9)))
            learning_rate = trial.suggest_float('learning_rate', *kwargs.get('learning_rate', (0.01, 0.3)), log=True)
            min_child_weight = trial.suggest_float('min_child_weight', *kwargs.get('min_child_weight', (1e-3, 1e1)), log=True)
            gamma = trial.suggest_float('gamma', *kwargs.get('gamma', (1e-3, 1e1)), log=True)
            subsample = trial.suggest_float('subsample', *kwargs.get('subsample', (0.5, 1.0)))
            colsample_bytree = trial.suggest_float('colsample_bytree', *kwargs.get('colsample_bytree', (0.5, 1.0)))
            reg_alpha = trial.suggest_float('reg_alpha', *kwargs.get('reg_alpha', (1e-3, 1e1)), log=True)
            reg_lambda = trial.suggest_float('reg_lambda', *kwargs.get('reg_lambda', (1e-3, 1e1)), log=True)

            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'min_child_weight': min_child_weight,
                'gamma': gamma,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda
            }

            

            # Initialize and train the model
            model = custom_model()

            model.pca_fit(train_X, min(10, train_X.shape[1]))
            model.fit(train_X.values, train_y.values, params=params, early_stopping_rounds=10, random_state=42, apply_pca=True, verbose=False, n_splits=5)
            
            # Make predictions and evaluate
            preds = model.predict(test_X.values)
            rmse = mean_squared_error(test_y.values, preds) ** 0.5
            return rmse
        except Exception as e:
            return float('inf')  # Return a large number to indicate failure

    # Create and optimize the Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_custom, n_trials=100, n_jobs=-1)

    # Return the best parameters and best value
    return study.best_params, study.best_value
