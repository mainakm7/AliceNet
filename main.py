from .mutual_info_regression.mi_regression_all import mi_regression_all
from .network.xgboostnet import xgboostnet
from typing import Optional
from fastapi import FastAPI



app = FastAPI()


def main(calculate_MI: str = False, view_or_calc_network: str = True, \
    xgboost_event_index: int = 1, xgboost_specific_gene: Optional[str] = None, *args, **kwargs):

    if calculate_MI == True:
        mi_reg_df = mi_regression_all()
    
    if view_or_calc_network == True:
        fit_param = xgboostnet(xgboost_event_index, xgboost_specific_gene)
    

if __name__ == "__main__":
    main()