import pandas as pd
import numpy as np
import optuna
from .custom_model import custom_model
from sklearn.metrics import mean_squared_error


def hyperparameter_tuning(train_X: pd.DataFrame, train_y: pd.DataFrame, test_X: pd.DataFrame, test_y: pd.DataFrame, *args, **kwargs):
    """
    Perform hyperparameter tuning using Optuna for a custom model.

    Args:
        train_X (pd.DataFrame): Training features.
        train_y (pd.Series): Training labels.
        test_X (pd.DataFrame): Testing features.
        test_y (pd.Series): Testing labels.
        **kwargs: Additional keyword arguments for dynamic parameter tuning.

        List of keywords:
        - n_estimators (int): Number of trees in the forest.
        - max_depth (int): Maximum depth of the tree.
        - learning_rate (float): Boosting learning rate.
        - min_child_weight (float): Minimum sum of instance weight (hessian) needed in a child.
        - gamma (float): Minimum loss reduction required to make a further partition on a leaf node.
        - subsample (float): Subsample ratio of the training instance.
        - colsample_bytree (float): Subsample ratio of columns when constructing each tree.
        - reg_alpha (float): L1 regularization term on weights.
        - reg_lambda (float): L2 regularization term on weights.

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
        # Example of using kwargs to modify parameters
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
        
        model = custom_model()
        model.pca_fit(train_X, min(10, train_X.shape[1]))
        model.fit(train_X.values, train_y.values, params=params, early_stopping_rounds=10, random_state=42, apply_pca=True, verbose=False, n_splits=5)
        
        preds = model.predict(test_X.values)
        rmse = mean_squared_error(test_y.values, preds) ** 0.5
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_custom, n_trials=100)
    return study.best_params, study.best_value
