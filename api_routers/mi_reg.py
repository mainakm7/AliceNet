from fastapi import APIRouter, status, Path, Query, Depends, HTTPException
from ..database import Database
import requests
from ..utils.data_loader import sf_events_upd, mi_raw_data, mi_melted_data
from ..mutual_info_regression.mi_regression_all import mi_regression_all
from ..mutual_info_regression.mi_matrix_melt import mi_melt_from_df
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/mi", tags=["MI_regression"])

def get_db():
    db = Database.get_db()
    return db


@router.get("/compute_mi", status_code=status.HTTP_200_OK)
def compute_mi_all():
    try:
        mi_df = mi_regression_all()
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

@router.get("/melt_mi", status_code=status.HTTP_200_OK)
def melt_midata() -> Dict[str, Dict]:
    try:
        
        mi_raw_data_fetched = mi_raw_data
        
        if not mi_raw_data_fetched:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Raw MI data not found."
            )
        
        mi_data_melted = mi_melt_from_df(mi_raw_data_fetched)
        return {"melted_mi_data": mi_data_melted.to_dict(orient="split")}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error occurred: {str(e)}"
        )

@router.get("/melted_mi_data_to_db", status_code=status.HTTP_201_CREATED)
def mi_data_to_db(db: Database = Depends(get_db)):
    try:
        
        melted_mi_data_fetched = mi_melted_data
        
        # Example: Inserting each row into MongoDB
        inserted_count = 0
        for index, row in melted_mi_data_fetched.iterrows():
            db['melted_mi_data'].insert_one({
                "spliced_genes": row["spliced_genes"],
                "events": row["Splicing events"],
                "mi": row["MI-value"]
            })
            inserted_count += 1
        
        return {"message": f"{inserted_count} records inserted successfully"}
    
    except requests.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error fetching data from external endpoint: {e}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {e}"
        )


@router.get("/event_gene_select", status_code=status.HTTP_200_OK)
def select_specific_splicedgene() -> list[str]:
    sf_events_df = sf_events_upd.copy()
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])
    return list(np.unique(sf_events_df["gene"]))

@router.get("/specific_event_select/{gene}", status_code=status.HTTP_200_OK)
def select_specific_splicedevent(gene: str = Path()) -> list[str]:
    sf_events_df = sf_events_upd.copy()
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])
    return list(sf_events_df[sf_events_df["gene"] == gene].index)
    
    
async def get_genes() -> list[str]:
    return await select_specific_splicedgene()

async def get_events(gene: str) -> list[str]:
    return await select_specific_splicedevent(gene) 


@router.get("/{gene}", status_code=status.HTTP_200_OK)
def mi_gene_events_query(
    gene: str = Path(description="Gene to query"),
    event: Optional[str] = Query(None, description="Splicing event to filter"),
    genes: List[str] = Depends(get_genes),
    events: List[str] = Depends(lambda gene: get_events(gene)),
    db: Database = Depends(get_db)
) -> List[Dict[str, Any]]:
    
    # Validate gene and event
    if gene not in genes:
        raise HTTPException(status_code=400, detail=f"Gene {gene} is not in the list of available genes.")
    
    # Fetch events for the specific gene
    events = get_events(gene)
    
    if event not in events:
        raise HTTPException(status_code=400, detail=f"Event {event} is not in the list of available events for gene {gene}.")
    
    try:
        if not event:
            # Retrieve all events for the given gene
            all_events = list(db['melted_mi_data'].find({"spliced_genes": gene}))
            if not all_events:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No events found for the given gene"
                )
            return all_events
        
        # Retrieve specific event for the given gene
        mi = db['melted_mi_data'].find({"spliced_genes": gene, "events": event})
        mi_list = list(mi)
        if not mi_list:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data found for the given gene and event"
            )
        return mi_list

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {e}"
        )
