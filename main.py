from .mutual_info_regression.mi_regression_all import mi_regression_all
from .network.xgboostnet import xgboostnet
from typing import Optional
from fastapi import FastAPI, status
from .api_routers import network, mi_reg, load_data
from .database import Database
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    Database.initialize()
    yield
    Database.close()

app = FastAPI(lifespan=lifespan)

app.include_router(network)
app.include_router(mi_reg)
app.include_router(load_data)

@app.get("/healthy", status_code=status.HTTP_200_OK)
def health_check():
    return {"status":"healthy"}