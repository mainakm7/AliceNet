import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

def data_preparation(
    event: str, 
    test_size: float, 
    mi_melted_df: pd.DataFrame, 
    sf_exp_upd: pd.DataFrame, 
    sf_events_upd: pd.DataFrame,
    *args,
    **kwargs
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepare machine learning data for predicting splicing events associated with splicing factors.

    Args:
        event (str): specific splicing event to consider.
        test_size (float): Proportion of the dataset to include in the test split.
        mi_df (pd.DataFrame): Mutual information dataframe.
        sf_exp_upd (pd.DataFrame): Splicing factor expression dataframe.
        sf_events_upd (pd.DataFrame): Splicing factor events dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: A tuple containing:
            - train_X (pd.DataFrame): Training features for machine learning.
            - train_y (pd.Series): Training labels representing specific splicing events.
            - test_X (pd.DataFrame): Testing features for validation.
            - test_y (pd.Series): Testing labels representing specific splicing events.

    Raises:
        ValueError: If no events are found for the specified gene or if the event index is out of bounds.
    """
    
    def prepare_adjacency_list(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate mutual information into adjacency list per splicing event."""
        return df.groupby("Splicing events").apply(
            lambda x: dict(zip(x["Splicing factors"], x["MI-value"]))
        ).reset_index(name="adj_list").set_index("Splicing events")
    
    def prepare_individual_dataset(X_df,y_df):
        
        sf_dict = adj_df_mi.loc[y_df.name]['adj_list']

            
        keys_list = list(sf_dict.keys())
        values_list = list(sf_dict.values())

        X_df = X_df[keys_list] * values_list
        y_df = y_df.dropna()
        pat_list = list(np.intersect1d(X_df.index,y_df.index))
        X_df = X_df.loc[pat_list]
        y_df = y_df.loc[pat_list]
        samples = X_df.shape[0]
        ix = np.random.choice(samples, size= (min(na_list),), replace=False)
        X_df = X_df.iloc[ix]
        y_df = y_df.iloc[ix]
        return X_df, y_df, pat_list
    
    # Aggregate mutual information into adjacency list per splicing event
    adj_df_mi = prepare_adjacency_list(mi_melted_df)

    # Copy and prepare dataframes
    sf_events_df = sf_events_upd
    sf_exp_df = sf_exp_upd.T

    sf_events_df_individual = sf_events_df.loc[event]
    
    na_list = []
    for row in sf_events_df.index:
        series = sf_events_df.loc[row]
        series = series.dropna()
        na_list.append(series.shape[0])

    # Prepare individual dataset
    Xdata, ydata, pat_list = prepare_individual_dataset(sf_exp_df, sf_events_df_individual)

    # Split data into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(Xdata, ydata, test_size=test_size, random_state=42)

    return train_X, train_y, test_X, test_y
