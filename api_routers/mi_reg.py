from fastapi import APIRouter, status, Path, Query
from ..mutual_info_regression.mi_regression_query import mi_regression_query_specific_gene, mi_regression_query_specific_event
from typing import Optional, Union, List
import asyncio


router = APIRouter(prefix="/mi", tags=["MI_regression"])


@router.get("/{gene}", status_code=status.HTTP_200_OK)
async def mi_query(gene: str = Path("AR"), event: Optional[str] = Query(None)) -> Union[List[str], float]:
    if not event:
        event_list = await asyncio.to_thread(mi_regression_query_specific_gene, gene)
        return {"event_list": event_list}
    else:
        mi = await asyncio.to_thread(mi_regression_query_specific_event, gene, event)
        return {"mi": mi}
    
    

