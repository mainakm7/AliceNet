from .mutual_info_regression.mi_regression_all import mi_regression_all
from .network.xgboostnet import xgboostnet
from typing import Optional

def main(calculate_MI: str = False, xgboost_event_index: int = 1, xgboost_specific_gene: Optional[str] = None):

    if calculate_MI == True:
        mi_reg_df = mi_regression_all()
    
    fit_param = xgboostnet(xgboost_event_index, xgboost_specific_gene)
    

if __name__ == "__main__":
    main()