import numpy as np
import pandas as pd
import os
from typing import Tuple
from .data_dir_path import data_dir_path

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw splicing event and SF expression data.
    
    This function defines the data directory path, loads splicing event and 
    SF expression data from CSV files, processes the dataframes by setting 
    the appropriate index and removing the first column, and sorts the samples
    for both dataframes to ensure they have common columns.

    Returns:
        tuple: A tuple containing the processed SF expression dataframe and 
               the processed splicing events dataframe.
    """
    # Defining data directory path
    data_path_whole = data_dir_path(subdir="raw")
    
    # Load splicing event and SF expression data
    sf_events_path = os.path.join(data_path_whole, "correlation_gene_exp_splicing_events_cmi32_su2ce153_withNA10percent_unscaled.csv")
    sf_exp_path = os.path.join(data_path_whole, "normalized_counts_for_cmi32_su2ce153_sf_genes_upd.csv")

    sf_events_df = pd.read_csv(sf_events_path)
    sf_exp_df = pd.read_csv(sf_exp_path)

    # Process expression dataframe
    sf_exp_df.set_index(sf_exp_df.iloc[:, 0], inplace=True)
    sf_exp_df = sf_exp_df.iloc[:, 1:]

    # Process events dataframe
    sf_events_df.set_index(sf_events_df.iloc[:, 0], inplace=True)
    sf_events_df = sf_events_df.iloc[:, 1:]

    # Sort samples for both dataframes
    common_cols = np.intersect1d(sf_exp_df.columns, sf_events_df.columns)
    sf_exp_upd = sf_exp_df[common_cols]
    sf_events_upd = sf_events_df[common_cols]
    
    return sf_exp_upd, sf_events_upd

def load_mi_data() -> pd.DataFrame:
    """
    Load mutual information data.
    
    This function defines the data directory path and loads mutual information
    data from a CSV file.

    Returns:
        DataFrame: The loaded mutual information dataframe.
    """
    # Defining data directory path
    data_path_whole = data_dir_path(subdir="processed")
    
    mi_data_path = os.path.join(data_path_whole, "mutualinfo_reg_one_to_one_MI_all_melted.csv")
    
    mi_data = pd.read_csv(mi_data_path)
    
    return mi_data