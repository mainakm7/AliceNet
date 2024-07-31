from fastapi import APIRouter, status, Path, Query, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np
from ..database import Database
from ..mutual_info_regression.mi_regression_all import mi_regression_all
from ..mutual_info_regression.mi_matrix_melt import mi_melt_from_df
from ..utils.data_loader import sf_events_upd, sf_exp_upd, mi_melted_data, mi_raw_data

router = APIRouter(prefix="/mi", tags=["MI_regression"])

class DataFrameRequest(BaseModel):
    sf_exp_df: Optional[Dict] = None
    sf_events_df: Optional[Dict] = None
    mi_raw_data: Optional[Dict] = None
    mi_melted_data: Optional[Dict] = None

def get_db():
    db = Database.get_db()
    return db

@router.post("/compute_mi", status_code=status.HTTP_201_CREATED)
async def compute_mi_all(request: DataFrameRequest) -> Dict[str, Any]:
    try:
        # Convert dictionaries back to DataFrames
        sf_exp_df = pd.DataFrame(request.sf_exp_df['data'], columns=request.sf_exp_df['columns'], index=request.sf_exp_df['index'])
        sf_events_df = pd.DataFrame(request.sf_events_df['data'], columns=request.sf_events_df['columns'], index=request.sf_events_df['index'])
        
        mi_df = await run_in_threadpool(mi_regression_all, sf_exp_df, sf_events_df)
        
        if mi_df is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MI data not computed.")
        
        return {"raw_mi_data": mi_df.to_dict(orient="split")}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error occurred: {str(e)}"
        )

@router.post("/melt_mi", status_code=status.HTTP_201_CREATED)
async def melt_midata(request: DataFrameRequest) -> Dict[str, Dict]:
    try:
        mi_raw_data = request.mi_raw_data
        if mi_raw_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Raw MI data not found."
            )
        mi_raw_df = pd.DataFrame(mi_raw_data["data"], columns=mi_raw_data["columns"], index=mi_raw_data["index"])
        mi_data_melted = await run_in_threadpool(mi_melt_from_df, mi_raw_df)
        return {"melted_mi_data": mi_data_melted.to_dict(orient="split")}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error occurred: {str(e)}"
        )

@router.post("/melted_mi_data_to_db", status_code=status.HTTP_201_CREATED)
async def mi_data_to_db(request: DataFrameRequest, db: Database = Depends(get_db)) -> Dict[str, str]:
    try:
        mi_melted_data = request.mi_melted_data
        if mi_melted_data is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Melted MI data not found.")
        
        mi_melted_df = pd.DataFrame(mi_melted_data["data"], columns=mi_melted_data["columns"], index=mi_melted_data["index"])
        # Example: Inserting each row into MongoDB
        inserted_count = 0
        for index, row in mi_melted_df.iterrows():
            db['melted_mi_data'].insert_one({
                "spliced_genes": row["spliced_genes"],
                "events": row["Splicing events"],
                "mi": row["MI-value"]
            })
            inserted_count += 1
        
        return {"message": f"{inserted_count} records inserted successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {e}"
        )

@router.get("/event_gene_select", status_code=status.HTTP_200_OK)
async def select_specific_splicedgene() -> List[str]:
    sf_events_df = sf_events_upd.copy()
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])
    return list(np.unique(sf_events_df["gene"]))

@router.get("/specific_event_select/{gene}", status_code=status.HTTP_200_OK)
async def select_specific_splicedevent(gene: str = Path(..., description="Gene to filter")) -> List[str]:
    sf_events_df = sf_events_upd.copy()
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])
    return list(sf_events_df[sf_events_df["gene"] == gene].index)

@router.get("/{gene}", status_code=status.HTTP_200_OK)
async def mi_gene_events_query(
    gene: str = Path(..., description="Gene to query"),
    event: Optional[str] = Query(None, description="Splicing event to filter"),
    db: Database = Depends(get_db)
) -> List[Dict[str, Any]]:
    # Fetch available genes and events
    genes = await select_specific_splicedgene()
    events = await select_specific_splicedevent(gene)
    
    # Validate gene and event
    if gene not in genes:
        raise HTTPException(status_code=400, detail=f"Gene {gene} is not in the list of available genes.")
    
    if event and event not in events:
        raise HTTPException(status_code=400, detail=f"Event {event} is not in the list of available events for gene {gene}.")
    
    try:
        query = {"spliced_genes": gene}
        if event:
            query["events"] = event
        
        mi_data = list(db['melted_mi_data'].find(query))
        if not mi_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data found for the given gene and event"
            )
        return mi_data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {e}"
        )
