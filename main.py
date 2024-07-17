from .mutual_info_regression.mi_regression_all import mi_regression_all
from .network.xgboostnet import xgboostnet
from typing import Optional
from fastapi import FastAPI, status
from .api_routers import network, mi_reg



app = FastAPI()

app.include_router(network)
app.include_router(mi_reg)

@app.get("/healthy", status_code=status.HTTP_200_OK)
def health_check():
    return {"status":"healthy"}