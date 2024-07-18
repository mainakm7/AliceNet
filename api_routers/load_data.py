from fastapi import APIRouter, Query, status, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
from ..utils.data_loader import initialize_data, data_files_exist, load_melted_mi_data, load_raw_mi_data, sf_exp_upd, sf_events_upd
from ..utils.data_dir_path import data_dir_path
from ..mutual_info_regression.mi_matrix_melt import mi_melt_from_df, mi_melt_from_file
import pandas as pd
import os
import requests


router = APIRouter(prefix="/load", tags=["Load data"])

@router.post("/upload-data", status_code=status.HTTP_200_OK)
async def upload_data(file: UploadFile = File(...)):
    data_path_whole = data_dir_path(subdir="raw")
    file_location = os.path.join(data_path_whole, file.filename)

    with open(file_location, "wb") as f:
        f.write(file.file.read())

    # Check if both files are present after upload
    if data_files_exist():
        return {"info": "Data files uploaded successfully."}
    else:
        return {"info": "File uploaded, but both required data files are not present."}

@router.get("/filenames", status_code=status.HTTP_200_OK)
async def data_filenames(subdir: str = Query("raw")) -> List[str]:
    data_path = data_dir_path(subdir=subdir)
    files = os.listdir(data_path)
    return files


@router.get("/raw", status_code=status.HTTP_200_OK)
async def load_rawdata():
    try:
        if sf_exp_upd is None or sf_exp_upd.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Gene data not found."
            )

        if sf_events_upd is None or sf_events_upd.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Event data not found."
            )

        gene_df_dict = sf_exp_upd.to_dict(orient="split")
        event_df_dict = sf_events_upd.to_dict(orient="split")

        return {
            "gene_df": gene_df_dict,
            "event_df": event_df_dict
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Exception encountered: {str(e)}"
        )

@router.post("/raw", status_code=status.HTTP_201_CREATED)
async def upload_rawdata(
    event_file: Optional[str] = Query(None, description="Select an event file from the list"),
    gene_file: Optional[str] = Query(None, description="Select a gene file from the list"),
    subdir: str = "raw"
) -> Dict[str, Dict]:
    
    
    try:
        # Fetch filenames dynamically
        response = requests.get("http://localhost:8000/load/filenames")
        if response.status_code == status.HTTP_200_OK:
            filenames = response.json()  # Assuming filenames are returned as a JSON list
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to fetch filenames. Status code: {response.status_code}"
            )

        if not filenames:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filenames available. Please check the filenames endpoint."
            )

        # Validate selected filenames
        if event_file not in filenames or gene_file not in filenames:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename selected. Please select from available filenames."
            )

        # Initialize data based on provided filenames and subdir
        await initialize_data(event_file=event_file, gene_file=gene_file, subdir=subdir)

        # Return the initialized data
        return {
            "gene_df": initialize_data.sf_exp_upd.to_dict(orient="split"),
            "event_df": initialize_data.sf_events_upd.to_dict(orient="split")
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch or process data: {str(e)}"
        )

@router.get("/raw_mi", status_code=status.HTTP_200_OK)
def load_midata(file = Query("mutualinfo_reg_one_to_one_MI_all.csv")) -> Dict[str, Dict]:
    
    mi_data = load_raw_mi_data(filename=file)
    return {"raw_mi_data": mi_data.to_dict(orient="split")}

@router.post("/melt_mi", status_code=status.HTTP_201_CREATED)
def load_midata(file: Optional[str] = Query(None, description="Choose the MI matrix to melt")) -> Dict[str, Dict]:
    try:
        mi_raw_data = requests.get("http://localhost:8000/load/raw_mi").json()
        mi_raw_data = mi_raw_data["raw_mi_data"]
        if not file:
            mi_data = mi_melt_from_df(mi_raw_data)
        else:
            mi_data = mi_melt_from_file(filename=file)
        
        return {"melted_mi_data": mi_data.to_dict(orient="split")}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error occurred: {str(e)}"
        )


@router.get("/load_melted_mi", status_code=status.HTTP_200_OK)
def load_meltedmidata(file = Query("mutualinfo_reg_one_to_one_MI_all_melted.csv")) -> Dict[str, Dict]:
    
    mi_data = load_melted_mi_data(filename=file)
    return {"melted_mi_data": mi_data.to_dict(orient="split")}
