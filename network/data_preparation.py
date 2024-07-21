import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from ..utils.data_loader import load_melted_mi_data, sf_events_upd, sf_exp_upd

def data_preparation(specific_gene: str = "AR", event: str = None, test_size: Optional[float] = 0.3, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare machine learning data for predicting splicing events associated with splicing factors.

    Args:
        specific_gene (str): Specific gene to filter events for. Defaults to "AR".
        event (str): Index of the splicing event to consider. Defaults to None.
        test_size (Optional[float]): Proportion of the dataset to include in the test split. Defaults to 0.3.

    Returns:
        tuple: A tuple containing the following DataFrames:
            - train_X (pd.DataFrame): Training features for machine learning.
            - train_y (pd.DataFrame): Training labels representing specific splicing events.
            - test_X (pd.DataFrame): Testing features for validation.
            - test_y (pd.DataFrame): Testing labels representing specific splicing events.

    Raises:
        ValueError: If no events are found for the specified gene or if the event index is out of bounds.

    Notes:
        This function loads mutual information data, aggregates it into an adjacency list per splicing event,
        and prepares the necessary dataframes for training and testing machine learning models. It handles NaN values,
        filters events based on a specific gene if provided, and performs a train-test split on the prepared data.

    Example:
        To prepare data for training and testing with a specific gene 'AR' and an event index of 3:
        >>> train_X, train_y, test_X, test_y = data_preparation(event_index=3, specific_gene='AR')
    """
    # Load mutual information data
    mi_df = load_melted_mi_data()

    # Aggregate mutual information into adjacency list per splicing event
    adj_df_mi = mi_df.groupby("Splicing events").apply(lambda x: dict(zip(x["Splicing factors"], x["MI-value"]))).reset_index(name="adj_list")
    adj_df_mi.set_index("Splicing events", inplace=True)

    # Copy and prepare dataframes
    sf_events_df = sf_events_upd.copy()
    sf_exp_df = sf_exp_upd.copy()
    sf_exp_df_t = sf_exp_df.T
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])

    # Calculate number of non-NaN values per row
    na_list = [series.dropna().shape[0] for _, series in sf_events_df.iterrows()]

    # Select specific gene events
    sf_events_df_gene = sf_events_df[sf_events_df["gene"] == specific_gene]
    if sf_events_df_gene.empty:
        raise ValueError(f"No events found for the gene: {specific_gene}")
    
    sf_events_df_individual = sf_events_df_gene.loc[event, :-1]  # Not taking last column -> column: "gene"

    # Function to prepare individual dataset for a specific splicing event
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
        ix = np.random.choice(len(pat_list), size=min(na_list), replace=False)
        return X_df.loc[pat_list[ix]], y_df.loc[pat_list[ix]], pat_list

    # Prepare individual dataset
    Xdata, ydata, pat_list = individual_dataset(sf_exp_df_t, sf_events_df_individual)

    # Function to split data into training and testing sets
    def data_splitting(X, y, test_size):
        """
        Split data into training and testing sets.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target variable.
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: Training and testing sets for X and y.
        """
        return train_test_split(X, y, test_size=test_size, random_state=42)

    # Split data into training and testing sets
    train_X, test_X, train_y, test_y = data_splitting(Xdata, ydata, test_size)
    
    return train_X, train_y, test_X, test_y
