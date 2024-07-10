import numpy as np
import pandas as pd
import os
from load_data import load_raw_data

sf_exp_upd, sf_events_upd = load_raw_data()

events_mat = sf_events_upd.values
genes_mat = sf_exp_upd.values