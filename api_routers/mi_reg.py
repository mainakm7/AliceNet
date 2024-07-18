from fastapi import APIRouter, status, Path, Query, Depends
from ..mutual_info_regression.mi_regression_query import mi_regression_query_specific_gene, mi_regression_query_specific_event, current_melted_mi_file
from typing import Optional, Union, List
import asyncio
from ..database import Database


router = APIRouter(prefix="/mi", tags=["MI_regression"])

def get_db():
    db = Database.get_db()
    return db

@router.get("/melted_mi_file")
async def get_melted_filename(file: str = Query("mutualinfo_reg_one_to_one_MI_all_melted.csv")):
    filename = await asyncio.to_thread(current_melted_mi_file, file)
    return filename

@router.get("/{gene}", status_code=status.HTTP_200_OK)
async def mi_query(gene: str = Path("AR"), event: Optional[str] = Query(None)) -> Union[List[str], float]:
    if not event:
        event_list = await asyncio.to_thread(mi_regression_query_specific_gene, gene)
        return event_list
    else:
        mi = await asyncio.to_thread(mi_regression_query_specific_event, gene, event)
        return mi



