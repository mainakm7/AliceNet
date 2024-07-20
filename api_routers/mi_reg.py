from fastapi import APIRouter, status, Path, Query, Depends, HTTPException
from ..database import Database
import requests
import pandas as pd
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/mi", tags=["MI_regression"])

def get_db():
    db = Database.get_db()
    return db

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

@router.get("/{gene}", status_code=status.HTTP_200_OK)
async def mi_gene_events_query(
    gene: str = Path("AR", description="Gene to query"),
    event: Optional[str] = Query(None, description="Splicing event to filter"),
    db: Database = Depends(get_db)
) -> List[Dict[str, Any]]:
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
