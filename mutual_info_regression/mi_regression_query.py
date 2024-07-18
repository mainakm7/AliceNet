import numpy as np
import pandas as pd
from typing import List, Optional
from ..utils.data_loader import load_melted_mi_data

def current_melted_mi_file(filename: Optional[str] = "mutualinfo_reg_one_to_one_MI_all_melted.csv") -> str:
    """
    Returns the current filename for the melted mutual information data.

    Args:
        filename (Optional[str]): The filename of the melted MI data CSV. Defaults to "mutualinfo_reg_one_to_one_MI_all_melted.csv".

    Returns:
        str: The current filename for the melted mutual information data.
    """
    return filename

def mi_regression_query_specific_gene(specific_gene: str = "AR") -> List[str]:
    """
    Query mutual information data for specific splicing events associated with a gene.
    
    Args:
        specific_gene (str): The specific gene to query for splicing events. Defaults to "AR".

    Returns:
        list: A list of unique splicing events associated with the specific gene.
    """
    
    # Load mutual information data
    mi_data = load_melted_mi_data(current_melted_mi_file())
    
    gene_events = mi_data[mi_data["spliced_genes"] == specific_gene]
    event_list = list(np.unique(gene_events["Splicing events"]))
    return event_list

def mi_regression_query_specific_event(specific_event: str, specific_gene: str = "AR") -> pd.DataFrame:
    """
    Query mutual information data for a specific splicing event associated with a gene.
    
    Args:
        specific_event (str): The specific splicing event to query.
        specific_gene (str): The specific gene to query for the splicing event. Defaults to "AR".

    Returns:
        pd.DataFrame: A dataframe containing mutual information data of all splicing factors for the specific splicing event 
                      associated with the specific gene.
    """
    
    # Load mutual information data
    mi_data = load_melted_mi_data(current_melted_mi_file())
    
    gene_events = mi_data[mi_data.index == specific_gene]
    event_mi = gene_events[gene_events["Splicing events"] == specific_event]
    return event_mi
