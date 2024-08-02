import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

def data_preparation(
    specific_gene: str, 
    event: int, 
    test_size: float, 
    mi_df: pd.DataFrame, 
    sf_exp_upd: pd.DataFrame, 
    sf_events_upd: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepare machine learning data for predicting splicing events associated with splicing factors.

    Args:
        specific_gene (str): Specific gene to filter events for.
        event (int): Index of the splicing event to consider.
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
    
    def prepare_individual_dataset(X_df: pd.DataFrame, y_df: pd.Series, adj_df_mi: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
        """Prepare individual dataset for a specific splicing event."""
        sf_dict = adj_df_mi.loc[y_df.name]['adj_list']
        keys_list, values_list = list(sf_dict.keys()), list(sf_dict.values())
        X_df_filtered = X_df[keys_list] * values_list
        y_df_filtered = y_df.dropna()
        pat_list = list(np.intersect1d(X_df_filtered.index, y_df_filtered.index))
        if len(pat_list) == 0:
            raise ValueError("No overlapping samples found between features and labels.")
        ix = np.random.choice(len(pat_list), size=min(len(pat_list), 100), replace=False)  # Ensure `ix` does not exceed `pat_list` length
        return X_df_filtered.loc[pat_list[ix]], y_df_filtered.loc[pat_list[ix]], pat_list
    
    # Aggregate mutual information into adjacency list per splicing event
    adj_df_mi = prepare_adjacency_list(mi_df)

    # Copy and prepare dataframes
    sf_events_df = sf_events_upd.copy()
    sf_exp_df = sf_exp_upd.copy().T
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])

    # Filter for specific gene and event
    sf_events_df_gene = sf_events_df[sf_events_df["gene"] == specific_gene]
    if sf_events_df_gene.empty:
        raise ValueError(f"No events found for the gene: {specific_gene}")
    
    if event >= len(sf_events_df_gene) or event < 0:
        raise ValueError(f"Event index {event} is out of bounds.")

    sf_events_df_individual = sf_events_df_gene.iloc[event, :-1]  # Exclude the last column which is "gene"

    # Prepare individual dataset
    Xdata, ydata, pat_list = prepare_individual_dataset(sf_exp_df, sf_events_df_individual, adj_df_mi)

    # Split data into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(Xdata, ydata, test_size=test_size, random_state=42)

    return train_X, train_y, test_X, test_y
