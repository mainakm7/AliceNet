from fastapi import APIRouter, status, Path, Query, Depends, HTTPException
from ..database import Database
import requests
from ..utils.data_loader import sf_events_upd
from ..mutual_info_regression.mi_regression_all import mi_regression_all
from ..mutual_info_regression.mi_matrix_melt import mi_melt_from_df, mi_melt_from_file
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/mi", tags=["MI_regression"])

def get_db():
    db = Database.get_db()
    return db


@router.get("/compute_mi", status_code=status.HTTP_200_OK)
def compute_mi_all():
    mi_df = mi_regression_all()
    return mi_df.to_dict(orient="split")

@router.get("/melt_mi", status_code=status.HTTP_201_CREATED)
def melt_midata(file: Optional[str] = Query(None, description="Choose the MI matrix to melt")) -> Dict[str, Dict]:
    try:
        response = requests.get("http://localhost:8000/load/raw_mi")
        response.raise_for_status()
        mi_raw_data = response.json().get("raw_mi_data")
        
        if not mi_raw_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Raw MI data not found."
            )
        
        mi_data = mi_melt_from_df(mi_raw_data) if not file else mi_melt_from_file(filename=file)
        return {"melted_mi_data": mi_data.to_dict(orient="split")}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error occurred: {str(e)}"
        )

@router.get("/mi_data", status_code=status.HTTP_201_CREATED)
async def mi_data_to_db(db: Database = Depends(get_db)):
    try:
        # Fetch melted MI data from another endpoint
        response = requests.get("http://localhost:8000/load/melted_mi")
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Process melted MI data
        melted_mi_data = pd.DataFrame(response.json().get("melted_mi_data"))
        
        # Example: Inserting each row into MongoDB
        inserted_count = 0
        for index, row in melted_mi_data.iterrows():
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
async def select_specific_splicedgene() -> list[str]:
    sf_events_df = sf_events_upd.copy()
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])
    return list(np.unique(sf_events_df["gene"]))

@router.get("/specific_event_select/{gene}", status_code=status.HTTP_200_OK)
async def select_specific_splicedevent(gene: str = Path("AR")) -> list[str]:
    sf_events_df = sf_events_upd.copy()
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])
    return list(sf_events_df[sf_events_df["gene"] == gene].index)
    
    
async def get_genes() -> list[str]:
    return await select_specific_splicedgene()

async def get_events(gene: str) -> list[str]:
    return await select_specific_splicedevent(gene) 


@router.get("/{gene}", status_code=status.HTTP_200_OK)
async def mi_gene_events_query(
    gene: str = Path("AR", description="Gene to query"),
    event: Optional[str] = Query(None, description="Splicing event to filter"),
    genes: List[str] = Depends(get_genes),
    events: List[str] = Depends(lambda gene: get_events(gene)),
    db: Database = Depends(get_db)
) -> List[Dict[str, Any]]:
    
    # Validate gene and event
    if gene not in genes:
        raise HTTPException(status_code=400, detail=f"Gene {gene} is not in the list of available genes.")
    
    # Fetch events for the specific gene
    events = await get_events(gene)
    
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
