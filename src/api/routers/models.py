# File: src/api/routers/models.py

from typing import List
from fastapi import APIRouter

from src.api.services.model_manager import get_model_manager

router = APIRouter()

@router.get("/models", response_model=List[str], summary="List available models")
async def list_models():
    """
    Return a list of available model keys (config names).
    """
    mgr = get_model_manager()
    return mgr.get_available_models()
