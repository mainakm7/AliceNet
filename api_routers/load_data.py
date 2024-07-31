from fastapi import APIRouter, Query, status, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from typing import Dict, List
from ..utils.data_loader import (
    intersect_exp_event, load_melted_mi_data, load_raw_mi_data,
    load_raw_exp_data, load_raw_event_data, sf_exp_upd, sf_events_upd
)
from ..utils.data_dir_path import data_dir_path
import os

router = APIRouter(prefix="/load", tags=["Load data"])

@router.post("/upload_data", status_code=status.HTTP_201_CREATED)
async def upload_data(file: UploadFile = File(...)):
    data_path_whole = data_dir_path(subdir="raw")
    file_location = os.path.join(data_path_whole, file.filename)

    try:
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        return {"info": "Data files uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"File upload error: {e}")

@router.get("/filenames", status_code=status.HTTP_200_OK)
async def data_filenames(subdir: str = Query("raw")) -> List[str]:
    data_path = data_dir_path(subdir=subdir)
    if not os.path.exists(data_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data directory not found."
        )
    files = os.listdir(data_path)
    if not files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No files found in the data directory."
        )
    return files

@router.post("/load_expression", status_code=status.HTTP_201_CREATED)
async def load_expression_data(exp_filename: str):
    try:
        expression_df = await run_in_threadpool(load_raw_exp_data, exp_filename)
        return expression_df.to_dict(orient="split")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Expression data load error: {e}")

@router.post("/load_event", status_code=status.HTTP_201_CREATED)
async def load_event_data(event_filename: str):
    try:
        event_df = await run_in_threadpool(load_raw_event_data, event_filename)
        return event_df.to_dict(orient="split")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Splicing PSI data load error: {e}")

@router.get("/sync_data", status_code=status.HTTP_200_OK)
async def intersect_raw_data():
    try:
        await run_in_threadpool(intersect_exp_event, sf_exp_upd.copy(), sf_events_upd.copy())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error syncing up expression and event data: {e}"
        )

@router.post("/raw_mi", status_code=status.HTTP_201_CREATED)
async def load_midata(raw_mi_filename: str = "mutualinfo_reg_one_to_one_MI_all.csv") -> Dict[str, Dict]:
    try:
        mi_data = await run_in_threadpool(load_raw_mi_data, raw_mi_filename)
        return {"raw_mi_data": mi_data.to_dict(orient="split")}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading raw MI data: {str(e)}"
        )

@router.post("/load_melted_mi", status_code=status.HTTP_201_CREATED)
async def load_meltedmidata(filename: str = "mutualinfo_reg_one_to_one_MI_all_melted.csv") -> Dict[str, Dict]:
    try:
        mi_data = await run_in_threadpool(load_melted_mi_data, filename)
        return {"melted_mi_data": mi_data.to_dict(orient="split")}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading melted MI data: {str(e)}"
        )
