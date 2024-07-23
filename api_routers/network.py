from fastapi import APIRouter, status, HTTPException, Query, Path, Depends
from pydantic import BaseModel
from typing import Optional, Tuple, List
from ..network.hyperparameter_tuning import hyperparameter_tuning
from ..network.data_preparation import data_preparation
from ..network.xgboostnet import xgboostnet
from ..utils.data_loader import sf_events_upd
import pandas as pd
import numpy as np
import requests
from ..database import Database

def get_db():
    db = Database.get_db()
    return db


class Hyperparameters(BaseModel):
    n_estimators: Optional[Tuple[int, int]] = (50, 200)
    max_depth: Optional[Tuple[int, int]] = (3, 9)
    learning_rate: Optional[Tuple[float, float]] = (0.01, 0.3)
    min_child_weight: Optional[Tuple[float, float]] = (1e-3, 1e1)
    gamma: Optional[Tuple[float, float]] = (1e-3, 1e1)
    subsample: Optional[Tuple[float, float]] = (0.5, 1.0)
    colsample_bytree: Optional[Tuple[float, float]] = (0.5, 1.0)
    reg_alpha: Optional[Tuple[float, float]] = (1e-3, 1e1)
    reg_lambda: Optional[Tuple[float, float]] = (1e-3, 1e1)


class DataParams(BaseModel):
    test_size: Optional[float] = 0.3


router = APIRouter(prefix="/network", tags=["Network"])

@router.get("/event_gene_select", status_code=status.HTTP_200_OK)
async def select_specific_splicedgene() -> list[str]:
    sf_events_df = sf_events_upd.copy()
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])
    return list(np.unique(sf_events_df["gene"]))

@router.get("/specific_event_select/{gene}", status_code=status.HTTP_200_OK)
async def select_specific_splicedevent(gene: str = Path()) -> list[str]:
    sf_events_df = sf_events_upd.copy()
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])
    return list(sf_events_df[sf_events_df["gene"] == gene].index)
    
    
async def get_genes() -> list[str]:
    return await select_specific_splicedgene()

async def get_events(gene: str) -> list[str]:
    return await select_specific_splicedevent(gene)   
    
@router.post("/data_prepare/{gene}", status_code=status.HTTP_201_CREATED)
async def data_prepare(
    param: DataParams,
    gene: str = Path(description="Choose a gene to analyze"),
    event: str = Query(description="Choose an event of the chosen gene"),
    genes: List[str] = Depends(get_genes),
    events: List[str] = Depends(lambda gene: get_events(gene))
):
    if gene not in genes:
        raise HTTPException(status_code=400, detail=f"Gene {gene} is not in the list of available genes.")
    
    # Fetch events for the specific gene
    events = await get_events(gene)
    
    if event not in events:
        raise HTTPException(status_code=400, detail=f"Event {event} is not in the list of available events for gene {gene}.")

    test_size = param.test_size
    try:
        train_X, train_y, test_X, test_y = data_preparation(specific_gene=gene, event_index=event, test_size=test_size)
        return {
            "train_X": train_X.to_dict(),
            "train_y": train_y.to_dict(),
            "test_X": test_X.to_dict(),
            "test_y": test_y.to_dict()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/hptuning", status_code=status.HTTP_201_CREATED)
def hp_tuning(hparams: Hyperparameters):
    try:
        response = requests.get("http://localhost:8000/network/data_prepare")
        response.raise_for_status()
        data = response.json()

        train_X = pd.DataFrame.from_dict(data["train_X"])
        train_y = pd.Series(data["train_y"])
        test_X = pd.DataFrame.from_dict(data["test_X"])
        test_y = pd.Series(data["test_y"])

        best_params, best_value = hyperparameter_tuning(train_X, train_y, test_X, test_y, **hparams.model_dump())
        return {"best_params": best_params, "best_value": best_value}
    except requests.RequestException as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Error fetching data from external endpoint: {e}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {e}")


@router.get("/xgboostnetfit/{gene}", status_code=status.HTTP_200_OK)
async def xgboostnetfit(
    gene: str = Path(...,description="Choose a gene to analyze"),
    event: str = Query(...,description="Choose an event of the chosen gene"),
    db: Database = Depends(get_db),
    genes: List[str] = Depends(get_genes)
):
    # Validate gene and event
    if gene not in genes:
        raise HTTPException(status_code=400, detail=f"Gene {gene} is not in the list of available genes.")
    
    # Fetch events for the specific gene
    events = await get_events(gene)
    
    if event not in events:
        raise HTTPException(status_code=400, detail=f"Event {event} is not in the list of available events for gene {gene}.")

    try:
        best_params, final_rmse, shap_df = xgboostnet(event_name=event)

        # Insert fit params into MongoDB
        db['xgboost_params'].insert_one({
            "spliced_gene": gene,
            "specific_event": event,
            "xgboost_params": best_params,
            "xgboost_fit_rmse": final_rmse,
            "xgboost_fit_shap": shap_df.to_dict(orient="split")
        })

        return {"best_param": best_params, "final_rmse": final_rmse}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {e}")


@router.get("/xgboostnetquery/{gene}", status_code=status.HTTP_200_OK)
async def xgboostnetquery(
    gene: str = Path(..., description="Choose a gene to analyze"),
    event: str = Query(..., description="Choose an event of the chosen gene"),
    db: Database = Depends(get_db)
):
   
    try:
        xgboost_fit = list(db['xgboost_params'].find({
            "spliced_gene": gene,
            "specific_event": event
        }))
        if not xgboost_fit:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No data found for the given gene and event")
        return xgboost_fit
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {e}")


@router.get("/cluster/{gene}", status_code=status.HTTP_200_OK)
async def feature_cluster(
    gene: str = Path(..., description="Choose a gene to analyze"),
    event: str = Query(..., description="Choose an event of the chosen gene"),
    genes: List[str] = Depends(get_genes)
):
    
    # Validate gene and event
    if gene not in genes:
        raise HTTPException(status_code=400, detail=f"Gene {gene} is not in the list of available genes.")
    
    # Fetch events for the specific gene
    gene_events = await get_events(gene)
    
    if event not in gene_events:
        raise HTTPException(status_code=400, detail=f"Event {event} is not in the list of available events for gene {gene}.")

    try:
        response = requests.get(f"http://localhost:8000/xgboostnetquery/{gene}?event={event}")
        response.raise_for_status()
        xgboost_fit_data = response.json()

        
        clustering_results = {
            "gene": gene,
            "event": event,
            "cluster": ""
        }

        return clustering_results
    except requests.RequestException as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred while querying fit results: {e}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {e}")