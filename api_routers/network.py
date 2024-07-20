from fastapi import APIRouter, status, HTTPException
from pydantic import BaseModel
from typing import Optional, Tuple
from ..network.hyperparameter_tuning import hyperparameter_tuning
from ..network.data_preparation import data_preparation
from ..network.xgboostnet import xgboostnet
import pandas as pd
import requests


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
    event_index: int = 1
    specific_gene: Optional[str] = None
    test_size: Optional[float] = 0.3


router = APIRouter(prefix="/network", tags=["Network"])


@router.post("/data_prepare", status_code=status.HTTP_201_CREATED)
def data_prepare(param: DataParams):
    try:
        train_X, train_y, test_X, test_y = data_preparation(**param.model_dump())
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


@router.get("/xgboostnet", status_code=status.HTTP_200_OK)
def xgboostnetfit():
    try:
        best_params, final_rmse = xgboostnet()
        return {"best_param": best_params, "final_rmse": final_rmse}
    except HTTPException as e:
        raise e  
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {e}")