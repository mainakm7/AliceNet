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
    """
    sf_events_path = os.path.join(data_path_whole, sf_events_filename)
    sf_exp_path = os.path.join(data_path_whole, sf_exp_filename)
    return os.path.exists(sf_events_path) and os.path.exists(sf_exp_path)

def load_raw_exp_data(sf_exp_filename: str) -> pd.DataFrame:
    """
    Load raw SF expression data.
    """
    global sf_exp_upd
    
    data_path_whole = data_dir_path(subdir="raw")
    sf_exp_path = os.path.join(data_path_whole, sf_exp_filename)
    
    if not os.path.exists(sf_exp_path):
        raise FileNotFoundError(f"File {sf_exp_path} does not exist.")
    
    try:
        sf_exp_df = pd.read_csv(sf_exp_path)
        sf_exp_df.set_index(sf_exp_df.iloc[:, 0], inplace=True)
        sf_exp_df = sf_exp_df.iloc[:, 1:]
    except Exception as e:
        raise RuntimeError(f"Error reading expression file {sf_exp_path}: {e}")
    
    sf_exp_upd = sf_exp_df
    return sf_exp_df

def load_raw_event_data(sf_events_filename: str) -> pd.DataFrame:
    """
    Load raw splicing event data.
    """
    global sf_events_upd
    
    data_path_whole = data_dir_path(subdir="raw")
    sf_events_path = os.path.join(data_path_whole, sf_events_filename)
    
    if not os.path.exists(sf_events_path):
        raise FileNotFoundError(f"File {sf_events_path} does not exist.")
    
    try:
        sf_events_df = pd.read_csv(sf_events_path)
        sf_events_df.set_index(sf_events_df.iloc[:, 0], inplace=True)
        sf_events_df = sf_events_df.iloc[:, 1:]
    except Exception as e:
        raise RuntimeError(f"Error reading event file {sf_events_path}: {e}")
    
    sf_events_upd = sf_events_df
    return sf_events_df

def intersect_exp_event(sf_exp_df: pd.DataFrame, sf_events_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common_cols = np.intersect1d(sf_exp_df.columns, sf_events_df.columns)
    global sf_exp_upd, sf_events_upd
    sf_exp_upd = sf_exp_df[common_cols]
    sf_events_upd = sf_events_df[common_cols]
    return sf_exp_upd.copy(), sf_events_upd.copy()
    


def load_raw_mi_data(filename: str = "mutualinfo_reg_one_to_one_MI_all.csv") -> pd.DataFrame:
    """
    Load raw mutual information data.
    """
    global mi_raw_data
    
    if mi_raw_data is not None:
        return mi_raw_data
    
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
    """
    global mi_melted_data
    
    if mi_melted_data is not None:
        return mi_melted_data
    
    data_path_whole = data_dir_path(subdir="MI")
    mi_data_path = os.path.join(data_path_whole, filename)
    
    mi_data = pd.read_csv(mi_data_path)
    
    mi_melted_data = mi_data
    
    return mi_data
