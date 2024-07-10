import pandas as pd
import numpy as np
from ..utils.data_matrices import sf_events_upd
from ..utils.data_dir_path import data_dir_path
from typing import Optional, Tuple
import json
import os

def xgboostnet_query(event_index: int = 1, specific_gene: Optional[str] = None) -> Tuple[dict, float]:
    """
    Queries the best fit parameters for events already trained upon

    Args:
        event_index (int): Index of the splicing event to use.
        specific_gene (Optional[str]): Gene to filter the splicing events. If None, uses all genes.

    Returns:
        tuple: Best hyperparameters found by Optuna, fit_rmse.
    """
    sf_events_df = sf_events_upd.copy()
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])

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

    file_name = f"{sf_events_df_individual.name}.json"
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"File {file_name} not found")

    
    data_path = data_dir_path(subdir="network")
    file_name = f"{sf_events_df_individual.name}.json"
    whole_load_path = os.path.join(data_path, file_name)
    with open(whole_load_path, "r") as f:
        fit = json.load(f)

    best_params_custom = fit["model_best_fit_param"]
    fit_rmse = fit["fit_RMSE"]

    print("\n Best Custom XGBoostReg parameters:", best_params_custom)
    print("\n Final RMSE for custom model: ", fit_rmse)
    
    return best_params_custom, fit_rmse
