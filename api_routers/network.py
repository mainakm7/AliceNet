from fastapi import APIRouter, status, HTTPException, Query, Path, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional, Tuple, List, Dict, Any, Union
from ..network.hyperparameter_tuning import hyperparameter_tuning
from ..network.data_preparation import data_preparation
from ..network.xgboostnet import xgboostnet
from ..network.shap_PC_clustering import get_elbow, get_adj_matrix
from ..clustering.feature_clustering import feature_clustering, spectral_elbow
from ..utils.data_loader import sf_events_upd
import pandas as pd
import numpy as np
import requests
from ..database import Database
import pickle

router = APIRouter(prefix="/network", tags=["Network"])

def get_db() -> Database:
    return Database.get_db()

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

class AllParams(BaseModel):
    test_size: Optional[float] = 0.3
    num_cluster: Optional[int] = 10
    gene: Optional[str] = None
    event: Optional[str] = None

@router.get("/event_gene_select", response_model=List[str], status_code=status.HTTP_200_OK)
async def select_specific_splicedgene() -> List[str]:
    sf_events_df = sf_events_upd.copy()
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])
    return list(np.unique(sf_events_df["gene"]))

@router.get("/specific_event_select/{gene}", response_model=List[str], status_code=status.HTTP_200_OK)
async def select_specific_splicedevent(gene: str = Path(..., description="Gene name")) -> List[str]:
    sf_events_df = sf_events_upd.copy()
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])
    return list(sf_events_df[sf_events_df["gene"] == gene].index)

@router.post("/data_prepare", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def data_prepare(request: AllParams):
    gene = request.gene
    event = request.event
    test_size = request.test_size
    try:
        train_X, train_y, test_X, test_y = await run_in_threadpool(data_preparation, specific_gene=gene, event_index=event, test_size=test_size)
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

@router.post("/hptuning", response_model=Dict[str, Union[Dict[str, Any], float]], status_code=status.HTTP_201_CREATED)
async def hp_tuning(hparams: Hyperparameters):
    try:
        response = requests.get("http://localhost:8000/network/data_prepare")
        response.raise_for_status()
        data = response.json()

        train_X = pd.DataFrame.from_dict(data["train_X"])
        train_y = pd.Series(data["train_y"])
        test_X = pd.DataFrame.from_dict(data["test_X"])
        test_y = pd.Series(data["test_y"])

        best_params, best_value = await run_in_threadpool(hyperparameter_tuning, train_X, train_y, test_X, test_y, **hparams.model_dump())
        return {"best_params": best_params, "best_value": best_value}
    except requests.RequestException as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Error fetching data from external endpoint: {e}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {e}")

@router.post("/xgboostnetfit", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def xgboostnetfit(request: AllParams, db: Database = Depends(get_db)):
    gene = request.gene
    event = request.event
    try:
        best_params, final_rmse, final_model, train_data = await run_in_threadpool(xgboostnet, event_name=event)

        # Convert model and data to serialized format
        model_serialized = pickle.dumps(final_model)
        train_data_serialized = pickle.dumps(train_data)

        # Store parameters and serialized data in MongoDB
        db['xgboost_params'].insert_one({
            "spliced_gene": gene,
            "specific_event": event,
            "xgboost_params": best_params,
            "xgboost_fit_rmse": final_rmse,
            "xgboost_final_model": model_serialized,
            "xgboost_train_data": train_data_serialized
        })

        return {"best_params": best_params, "final_rmse": final_rmse}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {e}")

@router.post("/xgboostnetquery", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def xgboostnetquery(request: AllParams, db: Database = Depends(get_db)):
    gene = request.gene
    event = request.event
    try:
        xgboost_fit = list(db['xgboost_params'].find({
            "spliced_gene": gene,
            "specific_event": event
        }))
        if not xgboost_fit:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No data found for the given gene and event")
        return xgboost_fit[0]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {e}")

@router.get("/hcluster_elbow", response_model=List[float], status_code=status.HTTP_200_OK)
async def hcluster_elbow_dist(request: AllParams):
    gene = request.gene
    event = request.event
    try:
        response = requests.post(f"http://localhost:8000/network/xgboostnetquery", json={"gene": gene, "event": event})
        response.raise_for_status()
        xgboost_fit_data = response.json()

        final_model_serialized = xgboost_fit_data["xgboost_final_model"]
        train_data_serialized = xgboost_fit_data["xgboost_train_data"]

        final_model = pickle.loads(final_model_serialized)
        train_data = pickle.loads(train_data_serialized)

        distances = await run_in_threadpool(get_elbow, final_model_custom=final_model, train_X=train_data)
        return distances
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error encountered: {e}")

@router.post("/hcluster", response_model=Dict[str, List[Any]], status_code=status.HTTP_201_CREATED)
async def hcluster_adj_matrix(request: AllParams):
    gene = request.gene
    event = request.event
    num_cluster = request.num_cluster
    try:
        response = requests.post(f"http://localhost:8000/network/xgboostnetquery", json={"gene": gene, "event": event})
        response.raise_for_status()
        xgboost_fit_data = response.json()

        final_model_serialized = xgboost_fit_data["xgboost_final_model"]
        train_data_serialized = xgboost_fit_data["xgboost_train_data"]

        final_model = pickle.loads(final_model_serialized)
        train_data = pickle.loads(train_data_serialized)

        cluster_index, cluster_feature = await run_in_threadpool(get_adj_matrix, model=final_model, num_clusters=num_cluster, train_X=train_data)
        return {"cluster_index": cluster_index, "cluster_features": cluster_feature}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error encountered: {e}")

@router.post("/scluster", response_model=Dict[str, List[Any]], status_code=status.HTTP_201_CREATED)
async def scluster_adj_matrix(request: AllParams):
    gene = request.gene
    event = request.event
    num_cluster = request.num_cluster
    try:
        response = requests.post(f"http://localhost:8000/network/xgboostnetquery", json={"gene": gene, "event": event})
        response.raise_for_status()
        xgboost_fit_data = response.json()

        final_model_serialized = xgboost_fit_data["xgboost_final_model"]
        train_data_serialized = xgboost_fit_data["xgboost_train_data"]

        final_model = pickle.loads(final_model_serialized)
        train_data = pickle.loads(train_data_serialized)

        cluster_index, cluster_feature = await run_in_threadpool(feature_clustering, model=final_model, num_clusters=num_cluster, train_X=train_data)
        return {"cluster_index": cluster_index, "cluster_features": cluster_feature}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error encountered: {e}")
