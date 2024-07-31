from fastapi import APIRouter, status, HTTPException, Path, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional, Tuple, List, Dict, Any, Union
from ..network.hyperparameter_tuning import hyperparameter_tuning
from ..network.data_preparation import data_preparation
from ..network.xgboostnet import xgboostnet
from ..network.shap_PC_clustering import get_elbow, get_adj_matrix
from ..clustering.feature_clustering import feature_clustering
from ..database import Database
import pickle
import pandas as pd
import numpy as np

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
    specific_gene: Optional[str] = None
    event: Optional[str] = None

@router.post("/event_gene_select", response_model=List[str], status_code=status.HTTP_200_OK)
async def select_specific_splicedgene(sf_events_upd: pd.DataFrame) -> List[str]:
    sf_events_df = sf_events_upd.copy()
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])
    return list(np.unique(sf_events_df["gene"]))

@router.post("/specific_event_select/{gene}", response_model=List[str], status_code=status.HTTP_200_OK)
async def select_specific_splicedevent(
    sf_events_upd: pd.DataFrame,
    gene: str = Path(..., description="Events for specific gene")
) -> List[str]:
    sf_events_df = sf_events_upd.copy()
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])
    return list(sf_events_df[sf_events_df["gene"] == gene].index)

@router.post("/data_prepare", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def data_prepare(request: AllParams) -> Dict[str, Any]:
    specific_gene = request.specific_gene
    event = request.event
    test_size = request.test_size
    try:
        train_X, train_y, test_X, test_y = await run_in_threadpool(
            data_preparation, specific_gene=specific_gene, event=event, test_size=test_size
        )
        return {
            "train_X": train_X.to_dict(),
            "train_y": train_y.to_dict(),
            "test_X": test_X.to_dict(),
            "test_y": test_y.to_dict()
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except IndexError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/hptuning", response_model=Dict[str, Union[Dict[str, Any], float]], status_code=status.HTTP_201_CREATED)
async def hp_tuning(
    hparams: Hyperparameters, 
    request: AllParams
) -> Dict[str, Union[Dict[str, Any], float]]:
    specific_gene = request.specific_gene
    event = request.event
    test_size = request.test_size
    try:
        train_X, train_y, test_X, test_y = await run_in_threadpool(
            data_preparation, specific_gene=specific_gene, event=event, test_size=test_size
        )
        
        best_params, best_value = await run_in_threadpool(
            hyperparameter_tuning, train_X, train_y, test_X, test_y, **hparams.model_dump()
        )
        return {"best_params": best_params, "best_value": best_value}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {e}")

@router.post("/xgboostnetfit", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def xgboostnetfit(
    hparams: Hyperparameters, 
    request: AllParams, 
    db: Database = Depends(get_db)
) -> Dict[str, Any]:
    specific_gene = request.specific_gene
    event = request.event
    try:
        best_params, final_rmse, final_model, train_data = await run_in_threadpool(
            xgboostnet, hparams=hparams.model_dump(), dataparams=request.model_dump()
        )

        
        model_serialized = pickle.dumps(final_model)
        train_data_serialized = pickle.dumps(train_data)

        
        db['xgboost_params'].insert_one({
            "spliced_gene": specific_gene,
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
async def xgboostnetquery(
    request: AllParams, 
    db: Database = Depends(get_db)
) -> Dict[str, Any]:
    specific_gene = request.specific_gene
    event = request.event
    try:
        xgboost_fit = list(db['xgboost_params'].find({
            "spliced_gene": specific_gene,
            "specific_event": event
        }))
        if not xgboost_fit:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No data found for the given gene and event")
        return xgboost_fit[0]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {e}")

@router.post("/hcluster_elbow", response_model=List[float], status_code=status.HTTP_201_CREATED)
async def hcluster_elbow_dist(request: AllParams) -> List[float]:
    specific_gene = request.specific_gene
    event = request.event
    try:
        xgboost_fit_data = await xgboostnetquery(request)
        final_model_serialized = xgboost_fit_data["xgboost_final_model"]
        train_data_serialized = xgboost_fit_data["xgboost_train_data"]

        final_model = pickle.loads(final_model_serialized)
        train_data = pickle.loads(train_data_serialized)

        distances = await run_in_threadpool(get_elbow, final_model_custom=final_model, train_X=train_data)
        return distances
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error encountered: {e}")

@router.post("/hcluster", response_model=Dict[str, List[Any]], status_code=status.HTTP_201_CREATED)
async def hcluster_adj_matrix(request: AllParams) -> Dict[str, List[Any]]:
    specific_gene = request.specific_gene
    event = request.event
    num_cluster = request.num_cluster
    try:
        xgboost_fit_data = await xgboostnetquery(request)
        final_model_serialized = xgboost_fit_data["xgboost_final_model"]
        train_data_serialized = xgboost_fit_data["xgboost_train_data"]

        final_model = pickle.loads(final_model_serialized)
        train_data = pickle.loads(train_data_serialized)

        cluster_index, cluster_feature = await run_in_threadpool(
            get_adj_matrix, model=final_model, num_clusters=num_cluster, train_X=train_data
        )
        return {"cluster_index": cluster_index, "cluster_features": cluster_feature}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error encountered: {e}")

@router.post("/scluster", response_model=Dict[str, List[Any]], status_code=status.HTTP_201_CREATED)
async def scluster_adj_matrix(request: AllParams) -> Dict[str, List[Any]]:
    specific_gene = request.specific_gene
    event = request.event
    num_cluster = request.num_cluster
    try:
        xgboost_fit_data = await xgboostnetquery(request)
        final_model_serialized = xgboost_fit_data["xgboost_final_model"]
        train_data_serialized = xgboost_fit_data["xgboost_train_data"]

        final_model = pickle.loads(final_model_serialized)
        train_data = pickle.loads(train_data_serialized)

        cluster_index, cluster_feature = await run_in_threadpool(
            feature_clustering, model=final_model, num_clusters=num_cluster, train_X=train_data
        )
        return {"cluster_index": cluster_index, "cluster_features": cluster_feature}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error encountered: {e}")
