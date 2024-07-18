from fastapi import APIRouter, status, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from ..utils.data_loader import sf_exp_upd, sf_events_upd, initialize_data, data_files_exist
from ..utils.data_dir_path import data_dir_path
from typing import Dict
import os

router = APIRouter(prefix="/load", tags=["Load data"])

@router.post("/upload-data", status_code=status.HTTP_200_OK)
async def upload_data(file: UploadFile = File(...)):
    data_path_whole = data_dir_path(subdir="raw")
    file_location = os.path.join(data_path_whole, file.filename)

    with open(file_location, "wb") as f:
        f.write(file.file.read())

    # Check if both files are present after upload
    if data_files_exist():
        await initialize_data()
        return {"info": "Data files uploaded and data initialized successfully."}
    else:
        return {"info": "File uploaded, but both required data files are not present."}

@router.get("/raw", status_code=status.HTTP_200_OK)
async def load_rawdata() -> Dict[str, Dict]:
    try:
        await initialize_data()
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data not initialized. Please upload data files."
        )

    return {
        "gene_df": sf_exp_upd.to_dict(orient="split"), 
        "event_df": sf_events_upd.to_dict(orient="split")
    }


@router.get("/raw_mi", status_code=status.HTTP_200_OK)
def load_midata(file = Query("mutualinfo_reg_one_to_one_MI_all.csv")) -> Dict[str, Dict]:
    
    mi_data = load_raw_mi_data(filename=file)
    return {"raw_mi_data": mi_data.to_dict(orient="split")}

@router.get("/melted_mi", status_code=status.HTTP_200_OK)
def load_meltedmidata(file = Query("mutualinfo_reg_one_to_one_MI_all_melted.csv")) -> Dict[str, Dict]:
    
    mi_data = load_melted_mi_data(filename=file)
    return {"melted_mi_data": mi_data.to_dict(orient="split")}
