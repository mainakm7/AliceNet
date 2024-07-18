from fastapi import APIRouter, status, Path, Query, Depends
from ..database import Database
import requests
import pandas as pd


router = APIRouter(prefix="/mi", tags=["MI_regression"])

def get_db():
    db = Database.get_db()
    return db

db = Depends(get_db)

@router.post("/mi_data", status_code=status.HTTP_201_CREATED)
async def mi_data_to_db(db: Database):
    try:
        # Fetch melted MI data from another endpoint
        response = requests.get("http://localhost:8000/load/melted_mi")
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Process melted MI data
        melted_mi_data = pd.DataFrame(response.json().get("melted_mi_data"))
        
        # Example: Inserting each row into MongoDB
        for index, row in melted_mi_data.iterrows():
            db['melted_mi_data'].insert_one({
                "spliced_genes": row["spliced_genes"],
                "events": row["Splicing events"],
                "mi": row["MI-value"]
            })
        
        return {"message": "Data inserted successfully"}
    
    except requests.RequestException as e:
        return {"error": f"Error fetching data from external endpoint: {e}"}
    
    except Exception as e:
        return {"error": f"An error occurred: {e}"}



@router.get("/{gene}", status_code=status.HTTP_200_OK)
async def mi_gene_events_query(db: Database, gene: str = Path("AR"), event: str = Query(None)):
    
    if not event:
        try:
            # Retrieve all events for the given gene
            all_events = list(db['melted_mi_data'].find({"spliced_genes": gene}))
            return all_events
        
        except Exception as e:
            return {"error": f"An error occurred: {e}"}
    
    else:
        try:
            mi = db['melted_mi_data'].find({"spliced_genes": gene, "events":event})
            return mi
        
        except Exception as e:
            return {"error": f"An error occurred: {e}"}