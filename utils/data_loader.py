import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .data_dir_path import data_dir_path

# Global variables to store the data
sf_exp_upd: Optional[pd.DataFrame] = None
sf_events_upd: Optional[pd.DataFrame] = None

mi_raw_data: Optional[pd.DataFrame] = None
mi_melted_data: Optional[pd.DataFrame] = None


def data_files_exist(sf_events_filename: str, sf_exp_filename: str, data_path_whole: str) -> bool:
    """
    Check if the required data files exist in the specified directory path.

    Args:
        sf_events_filename (str): Filename of splicing events CSV file.
        sf_exp_filename (str): Filename of SF expression CSV file.
        data_path_whole (str): Full path to the directory containing data files.

    Returns:
        bool: True if both data files exist, False otherwise.
    """
    sf_events_path = os.path.join(data_path_whole, sf_events_filename)
    sf_exp_path = os.path.join(data_path_whole, sf_exp_filename)
    return os.path.exists(sf_events_path) and os.path.exists(sf_exp_path)

def load_raw_data(sf_events_filename: str, sf_exp_filename: str, data_path_whole: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw splicing event and SF expression data.

    Args:
        sf_events_filename (str): Filename of splicing events CSV file.
        sf_exp_filename (str): Filename of SF expression CSV file.
        data_path_whole (str): Full path to the directory containing data files.

    Returns:
        tuple: A tuple containing the processed SF expression dataframe and 
               the processed splicing events dataframe.
    """
    global sf_exp_upd, sf_events_upd
    
    if sf_exp_upd is not None and sf_events_upd is not None:
        return sf_exp_upd, sf_events_upd

    # Load splicing event and SF expression data
    sf_events_path = os.path.join(data_path_whole, sf_events_filename)
    sf_exp_path = os.path.join(data_path_whole, sf_exp_filename)

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

async def initialize_data(sf_events_filename: str = "correlation_gene_exp_splicing_events_cmi32_su2ce153_withNA10percent_unscaled.csv",
                          sf_exp_filename: str = "normalized_counts_for_cmi32_su2ce153_sf_genes_upd.csv",
                          subdir: str = "raw"):
    """
    Initialize data loading.

    Args:
        sf_events_filename (str): Filename of splicing events CSV file.
        sf_exp_filename (str): Filename of SF expression CSV file.
        subdir (str): Subdirectory name where data files are located.
    """
    data_path_whole = data_dir_path(subdir=subdir)
    if data_files_exist(sf_events_filename, sf_exp_filename, data_path_whole):
        global sf_exp_upd, sf_events_upd
        sf_exp_upd, sf_events_upd = load_raw_data(sf_events_filename, sf_exp_filename, data_path_whole)
    else:
        # You can raise an exception to handle the missing files
        raise FileNotFoundError("Data files do not exist. Please upload the data files first.")
    


def load_raw_mi_data(filename: str = "mutualinfo_reg_one_to_one_MI_all.csv") -> pd.DataFrame:
    """
    Load raw mutual information data.
    
    This function defines the data directory path and loads processed mutual information
    data from a CSV file. It sets the index and removes unnecessary columns.

    Args:
        filename (Optional[str]): The filename of the raw MI data CSV. Defaults to "mutualinfo_reg_one_to_one_MI_all.csv".

    Returns:
        DataFrame: The loaded processed mutual information dataframe.
    """
    
    global mi_raw_data
    # Defining data directory path
    data_path_whole = data_dir_path(subdir="MI")
    
    mi_data_path = os.path.join(data_path_whole, filename)
    
    mi_data = pd.read_csv(mi_data_path)
    mi_data.set_index(mi_data.iloc[:, 0], inplace=True)
    mi_data = mi_data.iloc[:, 1:]
    
    mi_raw_data = mi_data
    
    return mi_data


  
def load_melted_mi_data(filename: str = "mutualinfo_reg_one_to_one_MI_all_melted.csv") -> pd.DataFrame:
    """
    Load melted mutual information data.
    
    This function defines the data directory path and loads melted mutual information
    data from a CSV file.

    Args:
        filename (Optional[str]): The filename of the melted MI data CSV. Defaults to "mutualinfo_reg_one_to_one_MI_all_melted.csv".

    Returns:
        DataFrame: The loaded melted mutual information dataframe.
    """
    global mi_melted_data
    
    # Defining data directory path
    data_path_whole = data_dir_path(subdir="MI")
    
    mi_data_path = os.path.join(data_path_whole, filename)
    
    mi_data = pd.read_csv(mi_data_path)
    
    mi_melted_data = mi_data
    
    return mi_data


