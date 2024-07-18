import numpy as np
import pandas as pd
import requests


#Obtaining the gene and event dataframe data from endpoint
response = requests.get("http://localhost:8000/load/raw").json()

sf_exp_upd = pd.DataFrame(data=response["gene_df"]["data"], columns=response["gene_df"]["columns"])
sf_exp_upd.index = response["gene_df"]["index"]

sf_events_upd = pd.DataFrame(data=response["event_df"]["data"], columns=response["event_df"]["columns"])
sf_events_upd.index = response["event_df"]["index"]

events_mat = sf_events_upd.values
genes_mat = sf_exp_upd.values