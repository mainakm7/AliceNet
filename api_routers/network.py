from fastapi import APIRouter, status, HTTPException, Path, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Tuple, List, Dict, Any, Union
from ..network.hyperparameter_tuning import hyperparameter_tuning
from ..network.data_preparation import data_preparation
from ..network.xgboostnet import xgboostnet
from ..network.shap_PC_clustering import get_elbow, get_adj_matrix
from ..clustering.feature_clustering import feature_clustering, spectral_elbow
from ..utils.data_dir_path import data_dir_path
from ..database import Database
import pickle
import pandas as pd
import numpy as np
import logging
import networkx as nx
import matplotlib.pyplot as plt
import io
from bson import ObjectId
import base64
import os



logging.basicConfig(level=logging.INFO, filename="network.log", filemode="w")

router = APIRouter(prefix="/network", tags=["Network"])


def serialize_object_id(data):
    if isinstance(data, ObjectId):
        return str(data)
    elif isinstance(data, dict):
        return {k: serialize_object_id(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_object_id(item) for item in data]
    else:
        return data


def get_db() -> Database:
    return Database.get_db()


class spectralinput(BaseModel):
    adj_matrix_whole_dict: Dict
    

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
    eventname: Optional[str] = None
    best_params: Optional[Dict[str, Any]] = None

class DataFrameRequest(BaseModel):
    sf_exp_df: Optional[Dict] = None
    sf_events_df: Optional[Dict] = None
    mi_raw_data: Optional[Dict] = None
    mi_melted_data: Optional[Dict] = None

@router.post("/event_gene_select", status_code=status.HTTP_201_CREATED)
async def select_specific_splicedgene(request: DataFrameRequest):
    sf_events_dict = request.sf_events_df
    sf_events_df = pd.DataFrame(sf_events_dict["data"], columns=sf_events_dict["columns"], index=sf_events_dict["index"])
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])
    return list(np.unique(sf_events_df["gene"]))

@router.post("/specific_event_select/{gene}", status_code=status.HTTP_201_CREATED)
async def select_specific_splicedevent(
    request: DataFrameRequest,
    gene: str = Path(..., description="Events for specific gene")
):
    sf_events_dict = request.sf_events_df
    sf_events_df = pd.DataFrame(sf_events_dict["data"], columns=sf_events_dict["columns"], index=sf_events_dict["index"])
    sf_events_df["gene"] = sf_events_df.index.to_series().apply(lambda x: x.split("_")[0])
    return list(sf_events_df[sf_events_df["gene"] == gene].index)

@router.post("/data_prepare", status_code=status.HTTP_201_CREATED)
async def data_prepare(request: AllParams, datareq: DataFrameRequest):
    event = request.event
    test_size = request.test_size
    mi_melted_dict = datareq.mi_melted_data
    sf_exp_dict = datareq.sf_exp_df
    sf_event_dict = datareq.sf_events_df
    
    mi_melted_df = pd.DataFrame(mi_melted_dict["data"], columns=mi_melted_dict["columns"], index=mi_melted_dict["index"])
    sf_exp_df = pd.DataFrame(sf_exp_dict["data"], columns=sf_exp_dict["columns"], index=sf_exp_dict["index"])
    sf_event_df = pd.DataFrame(sf_event_dict["data"], columns=sf_event_dict["columns"], index=sf_event_dict["index"])
    try:
        train_X, train_y, test_X, test_y = await run_in_threadpool(
            data_preparation, event=event, test_size=test_size, 
            mi_melted_df=mi_melted_df, sf_exp_upd=sf_exp_df, sf_events_upd=sf_event_df
        )
        return {
            "train_X": train_X.to_dict(orient="split"),
            "train_y": train_y.to_dict(),
            "test_X": test_X.to_dict(orient="split"),
            "test_y": test_y.to_dict()
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except IndexError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/hptuning", status_code=status.HTTP_201_CREATED)
async def hp_tuning(paramreq: AllParams, datareq: DataFrameRequest, hparams: Hyperparameters):
    
    eventname = paramreq.eventname
    test_size = paramreq.test_size
    mi_melted_dict = datareq.mi_melted_data
    sf_exp_dict = datareq.sf_exp_df
    sf_event_dict = datareq.sf_events_df
    
    mi_melted_df = pd.DataFrame(mi_melted_dict["data"], columns=mi_melted_dict["columns"], index=mi_melted_dict["index"])
    sf_exp_df = pd.DataFrame(sf_exp_dict["data"], columns=sf_exp_dict["columns"], index=sf_exp_dict["index"])
    sf_event_df = pd.DataFrame(sf_event_dict["data"], columns=sf_event_dict["columns"], index=sf_event_dict["index"])
    
    #Data preparation
    try:
        train_X, train_y, test_X, test_y = await run_in_threadpool(
            data_preparation, event=eventname, test_size=test_size, 
            mi_melted_df=mi_melted_df, sf_exp_upd=sf_exp_df, sf_events_upd=sf_event_df
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except IndexError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
       raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred during Data preparation: {e}")
    
    
    try:
        # Perform hyperparameter tuning
        best_params, best_value = await run_in_threadpool(
            hyperparameter_tuning, train_X, train_y, test_X, test_y, **hparams.model_dump()
        )
        return {"best_params": best_params, "best_value": best_value}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred during Hyperparameter tuning: {e}")



@router.post("/xgboostnetfit", status_code=status.HTTP_201_CREATED)
async def xgboostnetfit(
    datareq: DataFrameRequest, 
    paramreq: AllParams, 
    db: Database = Depends(get_db)
):
    specific_gene = paramreq.specific_gene
    eventname = paramreq.eventname
    test_size = paramreq.test_size
    best_params = paramreq.best_params
    mi_melted_dict = datareq.mi_melted_data
    sf_exp_dict = datareq.sf_exp_df
    sf_event_dict = datareq.sf_events_df

    

    mi_melted_df = pd.DataFrame(mi_melted_dict["data"], columns=mi_melted_dict["columns"], index=mi_melted_dict["index"])
    sf_exp_df = pd.DataFrame(sf_exp_dict["data"], columns=sf_exp_dict["columns"], index=sf_exp_dict["index"])
    sf_event_df = pd.DataFrame(sf_event_dict["data"], columns=sf_event_dict["columns"], index=sf_event_dict["index"])

    try:
        train_X, train_y, test_X, test_y = await run_in_threadpool(
            data_preparation, event=eventname, test_size=test_size, 
            mi_melted_df=mi_melted_df, sf_exp_upd=sf_exp_df, sf_events_upd=sf_event_df
        )
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except IndexError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred during data preparation: {e}")
    
    data_dict = {
        "train_X": train_X,
        "train_y": train_y,
        "test_X": test_X,
        "test_y": test_y
    }

    try:
        best_params, final_rmse, final_model, train_data = await run_in_threadpool(
            xgboostnet, data_dict=data_dict, best_fit=best_params, Dataparam=paramreq.model_dump()
        )

        model_serialized = pickle.dumps(final_model)
        train_data_serialized = pickle.dumps(train_data)

        db['xgboost_params'].insert_one({
            "spliced_gene": specific_gene,
            "specific_event": eventname,
            "xgboost_params": best_params,
            "xgboost_fit_rmse": final_rmse,
            "xgboost_final_model": model_serialized,
            "xgboost_train_data": train_data_serialized
        })
        return {"message": f"For event: {eventname} - Model has been fitted with RMse: {final_rmse} and all data uploaded to Database."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred during network fitting: {e}")

@router.post("/xgboostnetquery", status_code=status.HTTP_201_CREATED)
async def xgboostnetquery(
    paramreq: AllParams, 
    db: Database = Depends(get_db)
):
    
    specific_gene = paramreq.specific_gene
    eventname = paramreq.eventname
    
    try:
        xgboost_fit = db['xgboost_params'].find_one({
            "spliced_gene": specific_gene,
            "specific_event": eventname
        })

        # Convert ObjectId to string
        converted_data = pickle.dumps(xgboost_fit)
        encoded_data = base64.b64encode(converted_data).decode("utf-8")

        if not xgboost_fit:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No data found for the given gene and event")
        
        
        return {"data":encoded_data}
    
    except HTTPException as e:
        raise e
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {e}")



@router.post("/hcluster_elbow", status_code=status.HTTP_201_CREATED)
async def hcluster_elbow_dist(paramreq: AllParams, db: Database = Depends(get_db)):
    eventname = paramreq.eventname
    try:
        logging.info(f"Received request for elbow plot with params: {paramreq}")
        
        # Call xgboostnetquery to get the XGBoost data
        xgboost_fit_data_encoded = await xgboostnetquery(paramreq, db)
        logging.info("Data retrieved from xgboostnetquery")

        # Decode and load the XGBoost data
        xgboost_fit_data = pickle.loads(base64.b64decode(xgboost_fit_data_encoded["data"]))
        logging.info("XGBoost data successfully loaded")
        

        # Extract and deserialize the model and train data
        final_model_serialized = xgboost_fit_data["xgboost_final_model"]
        train_data_serialized = xgboost_fit_data["xgboost_train_data"]

        logging.info("Model and training data deserialized")

        # Run the elbow method and plot the results
        distances = await run_in_threadpool(get_elbow, final_model_serialized=final_model_serialized, train_data_serialized=train_data_serialized)
        logging.info("Elbow method computation completed")
        dir_path = data_dir_path(subdir="plots")
        fig_path = os.path.join(dir_path,f"{eventname}_hcluster_elbow.jpeg")
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(distances) + 1), distances, marker='o')
        plt.title(f"Event = {eventname}: Elbow Plot for Column Clustering")
        plt.xlabel('Number of clusters')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.savefig(fig_path)
        
        # Save the plot to a BytesIO buffer and return it as a JPEG image
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        logging.info("Elbow plot successfully generated")

        return StreamingResponse(buf, media_type="image/jpeg")
        
    except Exception as e:
        logging.error(f"Error encountered: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error encountered: {e}")

@router.post("/hcluster", status_code=status.HTTP_201_CREATED)
async def hcluster_adj_matrix(paramreq: AllParams, db: Database = Depends(get_db)):
    specific_gene = paramreq.specific_gene
    eventname = paramreq.eventname
    num_cluster = paramreq.num_cluster

    try:
        logging.info(f"Received request for hcluster with params: {paramreq}")
        xgboost_fit_data_encoded = await xgboostnetquery(paramreq, db)
        logging.info("Data retrieved from xgboostnetquery")

        # Decode and load the XGBoost data
        xgboost_fit_data = pickle.loads(base64.b64decode(xgboost_fit_data_encoded["data"]))
        logging.info("XGBoost data successfully loaded")
        

        # Extract and deserialize the model and train data
        final_model_serialized = xgboost_fit_data["xgboost_final_model"]
        train_data_serialized = xgboost_fit_data["xgboost_train_data"]
        logging.info("Model and training data deserialized")

        adj_matrix_whole_df = await run_in_threadpool(
            get_adj_matrix, final_model_serialized=final_model_serialized, train_data_serialized=train_data_serialized, num_clusters=num_cluster
        )
        logging.info("hcluster computation completed")
        adj_matrix_whole_dict = adj_matrix_whole_df.to_dict(orient="split")

        # Query for the specific document
        query = {"spliced_gene": specific_gene, "specific_event": eventname}
        doc = db['xgboost_params'].find_one(query)

        if doc:
            # Update the document with the new adjacency matrix
            update_result = db['xgboost_params'].update_one(
                query, {"$set": {"adj_matrix_whole_dict": adj_matrix_whole_dict}}
            )

            if update_result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to update the document with the adjacency matrix."
                )
        else:
            # Insert a new document if it doesn't exist
            insert_result = db['xgboost_params'].insert_one({
                "spliced_gene": specific_gene,
                "specific_event": eventname,
                "adj_matrix_whole_dict": adj_matrix_whole_dict
            })

            if not insert_result.inserted_id:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to insert the document with the adjacency matrix."
                )
        logging.info("hcluster update completed")
        return adj_matrix_whole_dict
        
    except Exception as e:
        logging.error(f"Error encountered: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error encountered: {e}"
        )


@router.post("/scluster_elbow", status_code=status.HTTP_201_CREATED)
async def scluster_adj_matrix(paramreq: AllParams, db:Database = Depends(get_db)):
    specific_gene = paramreq.specific_gene
    eventname = paramreq.eventname
    try:
        xgboost_fit_data_encoded = await xgboostnetquery(paramreq, db)
        logging.info("Data retrieved from xgboostnetquery")

        # Decode and load the XGBoost data
        xgboost_fit_data = pickle.loads(base64.b64decode(xgboost_fit_data_encoded["data"]))
        
        adj_matrix_whole_dict = xgboost_fit_data["adj_matrix_whole_dict"]
        
        sorted_eigenvalues = await run_in_threadpool(
            spectral_elbow, adj_matrix_whole_dict
        )
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(sorted_eigenvalues) + 1), sorted_eigenvalues, marker='o')
        plt.title('Eigenvalue elbow Plot for Spectral Clustering')
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail= f"Error in plotting elbow plot for Spctral clustering: {e}")
        

        


@router.post("/scluster", status_code=status.HTTP_201_CREATED)
async def scluster_adj_matrix(paramreq: AllParams, db:Database = Depends(get_db)):
    specific_gene = paramreq.specific_gene
    eventname = paramreq.eventname
    num_cluster = paramreq.num_cluster
    try:
        xgboost_fit_data_encoded = await xgboostnetquery(paramreq, db)
        logging.info("Data retrieved from xgboostnetquery")

        # Decode and load the XGBoost data
        xgboost_fit_data = pickle.loads(base64.b64decode(xgboost_fit_data_encoded["data"]))
        
        adj_matrix_whole_dict = xgboost_fit_data["adj_matrix_whole_dict"]
        
        clustered_genes = await run_in_threadpool(
            feature_clustering, adj_matrix_whole_dict, num_cluster
        )
        
        event_node = "Event"

        G = nx.Graph()

        G.add_node(event_node)

        for cluster, cluster_genes in clustered_genes.items():
            G.add_node(cluster)
            G.add_edge(event_node, cluster)
            for gene in cluster_genes:
                G.add_node(gene)
                G.add_edge(cluster, gene)

        # Draw the graph
        pos = nx.spring_layout(G)  # Positioning for the graph

        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=250, node_color="skyblue", font_size=4, font_weight="bold", edge_color="gray")
        plt.title(f"Event = {eventname}")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error encountered during spectral clutering: {e}"
            )
