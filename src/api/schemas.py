# File: src/api/schemas/schemas.py

from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field

class SummariseRequest(BaseModel):
    """
    Request to generate summaries.
    Either `input_docs` or `num_random` must be provided.
    """
    input_docs: Optional[List[str]] = Field(
        None, description="List of input documents to summarize"
    )
    num_random: Optional[int] = Field(
        None, description="Number of random test documents to sample"
    )
    model_names: List[str] = Field(
        ..., description="List of model keys to use (as from GET /models)"
    )
    split: Literal["train", "validation", "test"] = Field(
        "test", description="Dataset split when using random sampling"
    )

class SummariseResponse(BaseModel):
    """
    Response containing generated summaries for each model.
    If random sampling was used, the `references` list will contain
    the ground-truth summaries corresponding to each sampled document.
    """
    results: Dict[str, List[str]] = Field(
        ..., description="Mapping from model name to list of generated summaries"
    )
    references: Optional[List[str]] = Field(
        None, description="List of reference summaries (only for random sampling)"
    )
