from fastapi import APIRouter, Query, status, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from typing import Dict, List, Optional
from ..utils.data_loader import (
    intersect_exp_event, load_melted_mi_data, load_raw_mi_data,
    load_raw_exp_data, load_raw_event_data, sf_exp_upd, sf_events_upd
)
from ..utils.data_dir_path import data_dir_path
import os
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd

router = APIRouter(prefix="/load", tags=["Load data"])

class FilenameRequest(BaseModel):
    filename: str

class DataFrameRequest(BaseModel):
    sf_exp_df: Optional[Dict] = None
    sf_events_df: Optional[Dict] = None

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
async def data_filenames(subdir: str = Query("raw")):
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
async def load_expression_data(request: FilenameRequest):
    try:
        expression_df = await run_in_threadpool(load_raw_exp_data, request.filename)
        return expression_df.to_dict(orient="split")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Expression data load error: {e}")

@router.post("/load_event", status_code=status.HTTP_201_CREATED)
async def load_event_data(request: FilenameRequest):
    try:
        event_df = await run_in_threadpool(load_raw_event_data, request.filename)
        event_df = event_df.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
        event_df = event_df.fillna(-1)
        return event_df.to_dict(orient="split")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Splicing PSI data load error: {e}")

@router.post("/sync_data", status_code=status.HTTP_201_CREATED)
async def intersect_raw_data(request: DataFrameRequest):
    try:
                
        sf_exp_df = pd.DataFrame(request.sf_exp_df['data'], columns=request.sf_exp_df['columns'], index=request.sf_exp_df['index'])
        sf_events_df = pd.DataFrame(request.sf_events_df['data'], columns=request.sf_events_df['columns'], index=request.sf_events_df['index'])
        
        if sf_exp_df is None or sf_events_df is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Expression or event data not found."
            )

        
        
        # Perform intersection
        sf_exp_df, sf_events_df = await run_in_threadpool(intersect_exp_event, sf_exp_df, sf_events_df)
        
        # Replace infinities with NaN and fill NaN with -1
        sf_exp_df = sf_exp_df.replace([np.inf, -np.inf], np.nan).fillna(-1)
        sf_events_df = sf_events_df.replace([np.inf, -np.inf], np.nan).fillna(-1)
        
        # Convert DataFrames to dictionary and send as JSON response
        return JSONResponse(content={
            "exp_df": sf_exp_df.to_dict(orient="split"),
            "event_df": sf_events_df.to_dict(orient="split")
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error syncing up expression and event data: {str(e)}"
        )

@router.post("/raw_mi", status_code=status.HTTP_201_CREATED)
async def load_midata(request: FilenameRequest):
    try:
        mi_data = await run_in_threadpool(load_raw_mi_data, request.filename)
        return mi_data.to_dict(orient="split")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading raw MI data: {str(e)}"
        )

@router.post("/load_melted_mi", status_code=status.HTTP_201_CREATED)
async def load_meltedmidata(request: FilenameRequest):
    try:
        mi_data = await run_in_threadpool(load_melted_mi_data, request.filename)
        return mi_data.to_dict(orient="split")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading melted MI data: {str(e)}"
        )
