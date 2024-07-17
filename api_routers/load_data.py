from fastapi import APIRouter, status, Query
from ..utils.load_data import load_raw_data, load_raw_mi_data, load_melted_mi_data
from typing import Dict

router = APIRouter(prefix="/load", tags=["Load data"])

@router.get("/raw", status_code=status.HTTP_200_OK)
def load_rawdata() -> Dict[str, Dict]:
    
    sf_exp_upd, sf_events_upd = load_raw_data()
    return {"gene_df": sf_exp_upd.to_dict(orient="split"), "event_df": sf_events_upd.to_dict(orient="split")}


@router.get("/raw_mi", status_code=status.HTTP_200_OK)
def load_midata(file = Query("mutualinfo_reg_one_to_one_MI_all.csv")) -> Dict[str, Dict]:
    
    mi_data = load_raw_mi_data(filename=file)
    return {"raw_mi_data": mi_data.to_dict(orient="split")}

@router.get("/melted_mi", status_code=status.HTTP_200_OK)
def load_meltedmidata(file = Query("mutualinfo_reg_one_to_one_MI_all_melted.csv")) -> Dict[str, Dict]:
    
    mi_data = load_melted_mi_data(filename=file)
    return {"melted_mi_data": mi_data.to_dict(orient="split")}
