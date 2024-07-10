from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

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
