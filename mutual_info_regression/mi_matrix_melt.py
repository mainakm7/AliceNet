import numpy as np
import pandas as pd
import os
from ..utils.data_dir_path import data_dir_path


def mi_melt_from_df(mi_tot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt raw mutual information data and save it as a CSV file.
    
    Loads raw mutual information data, melts it into long format with
    splicing events as id_vars and splicing factors as value_vars, and
    saves the melted dataframe to a CSV file in the 'processed' directory.

    Args:
        mi_tot_df (pd.DataFrame): The raw MI data.

    Returns:
        DataFrame: The melted mutual information dataframe.
    """
        
    # Melt the dataframe
    melted_mitot_df = pd.melt(mi_tot_df,
                              id_vars='Splicing events',
                              value_vars=mi_tot_df.index,
                              var_name="Splicing factors",
                              value_name="MI-value")
    
    # Define the save path
    data_path = data_dir_path(subdir="MI")
    savefile_name = "mutualinfo_reg_one_to_one_MI_all_melted.csv"
    whole_save_path = os.path.join(data_path, savefile_name)
    
    # Save melted dataframe to CSV
    try:
        melted_mitot_df.to_csv(whole_save_path, index=False)
        print(f"Successfully saved melted mutual info data to: {whole_save_path}")
    except Exception as e:
        print(f"Error saving melted mutual info data: {e}")
    
    return melted_mitot_df
