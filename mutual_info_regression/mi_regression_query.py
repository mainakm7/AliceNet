import numpy as np
import pandas as pd
from typing import List
from ..utils.load_data import load_melted_mi_data

def mi_regression_query_specific_gene(specific_gene: str = "AR") -> List[str]:
    """
    Query mutual information data for specific splicing events associated with a gene.
    
    Args:
        specific_gene (str): The specific gene to query for splicing events. Defaults to "AR".

    Returns:
        list: A list of unique splicing events associated with the specific gene.
    """
    
    # Load mutual information data
    mi_data = load_melted_mi_data()
    
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
    mi_data = load_melted_mi_data()
    
    gene_events = mi_data[mi_data.index == specific_gene]
    event_mi = gene_events[gene_events["Splicing events"] == specific_event]
    return event_mi
