import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from ..data_matrices import sf_events_upd, sf_exp_upd
from ..load_data import load_mi_data
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
    # Load mutual information data
    mi_df = load_mi_data()

    adj_df_mi = mi_df.groupby("Splicing events").apply(lambda x: dict(zip(x["Splicing factors"], x["MI-value"]))).reset_index(name="adj_list")
    adj_df_mi.set_index("Splicing events", inplace=True)

    sf_events_df = sf_events_upd.copy()
    sf_exp_df = sf_exp_upd.copy()
    sf_exp_df_t = sf_exp_df.T
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])

    na_list = [series.dropna().shape[0] for _, series in sf_events_df.iterrows()]

    if specific_gene:
        sf_events_df_gene = sf_events_df[sf_events_df["gene"] == specific_gene]
        if sf_events_df_gene.empty:
            raise ValueError(f"No events found for the gene: {specific_gene}")
        if event_index >= len(sf_events_df_gene):
            raise IndexError(f"Event index {event_index} is out of bounds for gene {specific_gene}")
        sf_events_df_individual = sf_events_df_gene.iloc[event_index, :-1]
    else:
        if event_index >= len(sf_events_df):
            raise IndexError(f"Event index {event_index} is out of bounds")
        sf_events_df_individual = sf_events_df.iloc[event_index, :-1]

    def individual_dataset(X_df, y_df):
        """
        Prepare individual dataset for a specific splicing event.

        Args:
            X_df (pd.DataFrame): DataFrame of splicing factors.
            y_df (pd.Series): Series of individual splicing event.

        Returns:
            tuple: Processed X and y DataFrames, and list of samples used.
        """
        sf_dict = adj_df_mi.loc[y_df.name]['adj_list']
        keys_list, values_list = list(sf_dict.keys()), list(sf_dict.values())
        X_df = X_df[keys_list] * values_list
        y_df = y_df.dropna()
        pat_list = list(np.intersect1d(X_df.index, y_df.index))
        X_df, y_df = X_df.loc[pat_list], y_df.loc[pat_list]
        samples = X_df.shape[0]
        ix = np.random.choice(samples, size=(min(na_list),), replace=False)
        return X_df.iloc[ix], y_df.iloc[ix], pat_list

    Xdata, ydata, pat_list = individual_dataset(sf_exp_df_t, sf_events_df_individual)

    def data_splitting(X, y):
        """
        Split data into training and testing sets.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target variable.

        Returns:
            tuple: Training and testing sets for X and y.
        """
        return train_test_split(X, y, test_size=0.3, random_state=42)

    train_X, train_y, test_X, test_y = data_splitting(Xdata, ydata)

    class custom_model:
        """
        Custom model class for handling PCA and XGBoost regression.
        """
        def __init__(self):
            """
            Initialize the custom model with XGBRegressor and PCA.
            """
            self.xgb_model = XGBRegressor()
            self.pca_model = None
        
        def pca_fit(self, whole_X, n_components=10):
            """
            Fit PCA on the given data.

            Args:
                whole_X (array-like): Data to fit PCA on.
                n_components (int): Number of components to keep.

            Returns:
                PCA: Fitted PCA object.
            """
            self.pca_model = PCA(n_components=n_components).fit(whole_X)
            return self.pca_model
        
        def pca_transform(self, partial_X):
            """
            Transform the given data using the fitted PCA.

            Args:
                partial_X (array-like): Data to transform.

            Returns:
                array-like: Transformed data.

            Raises:
                ValueError: If PCA model is not fitted.
            """
            if self.pca_model is None:
                raise ValueError("PCA model has not been fitted. Please call pca_fit first.")
            return self.pca_model.transform(partial_X)
        
        def fit(self, train_X, train_y, dev_X=None, dev_y=None, params=None, early_stopping_rounds=10, random_state=42, apply_pca=True, verbose=False, n_splits=5):
            """
            Fit the XGBRegressor model using k-fold cross-validation.

            Args:
                train_X (array-like): Training features.
                train_y (array-like): Training target.
                dev_X (array-like, optional): Validation features.
                dev_y (array-like, optional): Validation target.
                params (dict, optional): XGBRegressor parameters.
                early_stopping_rounds (int): Early stopping rounds.
                random_state (int): Random state for reproducibility.
                apply_pca (bool): Whether to apply PCA.
                verbose (bool): Whether to print verbose output.
                n_splits (int): Number of folds for cross-validation.

            Returns:
                custom_model: Fitted model.
            """
            if apply_pca:
                train_X = self.pca_transform(train_X)
                if dev_X is not None:
                    dev_X = self.pca_transform(dev_X)
            
            params = params or {}
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            eval_results = []

            for train_index, val_index in kf.split(train_X):
                X_train, X_val = train_X[train_index], train_X[val_index]
                y_train, y_val = train_y[train_index], train_y[val_index]
                
                model = XGBRegressor(**params, random_state=random_state)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
                
                eval_results.append(model.best_score)
            
            self.eval_results = eval_results
            self.xgb_model = model  # Keep the model from the last fold

            return self
        
        def predict(self, test_X, apply_pca=True):
            """
            Predict using the fitted XGBRegressor model.

            Args:
                test_X (array-like): Test features.
                apply_pca (bool): Whether to apply PCA.

            Returns:
                array-like: Predictions.
            """
            if apply_pca and self.pca_model is not None:
                test_X = self.pca_transform(test_X)
            return self.xgb_model.predict(test_X)

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
        model.pca_fit(Xdata, min(10, train_X.shape[1]))
        model.fit(train_X.values, train_y.values, params=params, early_stopping_rounds=10, random_state=42, apply_pca=True, verbose=False, n_splits=5)
        
        preds = model.predict(test_X.values)
        rmse = mean_squared_error(test_y.values, preds) ** 0.5
        return rmse

    study_custom = optuna.create_study(direction='minimize')
    study_custom.optimize(objective_custom, n_trials=100, n_jobs=-1)

    best_params_custom = study_custom.best_params

    final_model_custom = custom_model()
    final_model_custom.pca_fit(Xdata, min(10, train_X.shape[1]))
    final_model_custom.fit(train_X.values, train_y.values, params=best_params_custom, random_state=42, apply_pca=True, verbose=False, n_splits=5)

    final_preds_custom = final_model_custom.predict(test_X.values)
    final_rmse_custom = mean_squared_error(test_y.values, final_preds_custom) ** 0.5
    
    fit = {
        "model_best_fit_param": best_params_custom,
        "fit_RMSE": final_rmse_custom
    }

    file_name = f"{sf_events_df_individual.name}.json"
    with open(file_name, "w") as f:
        json.dump(fit, f)

    print("\n Best Custom XGBoostReg parameters:", best_params_custom)
    print("\n Final RMSE for custom model: ", final_rmse_custom)
    
    return best_params_custom, final_rmse_custom
