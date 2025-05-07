# src/api/routers/summarise.py

from fastapi import APIRouter, HTTPException
from src.api.schemas import SummariseRequest, SummariseResponse
from src.api.services.model_manager import get_model_manager
from src.api.services.sampler import get_random_examples
from src.utils import DOC_SEPARATOR

router = APIRouter()

@router.post("/summarise", response_model=SummariseResponse)
async def summarise(req: SummariseRequest):
    if not req.input_docs and not req.num_random:
        raise HTTPException(400, "Either input_docs or num_random must be provided.")

    # 1) Load documents + optional reference summaries
    if req.num_random:
        samples = get_random_examples(req.num_random, split=req.split)
        docs = [s["document"] for s in samples]
        refs = [s["summary"]  for s in samples]
    else:
        docs = req.input_docs
        refs = None

    mgr = get_model_manager()
    results = {}

    # 2) In user‐input mode, merge into one mega‐doc
    if not req.num_random:
        mega = f" {DOC_SEPARATOR} ".join(docs)
        docs_to_summarize = [mega]
    else:
        docs_to_summarize = docs

    # 3) Summarize each doc for each model
    for name in req.model_names:
        if name not in mgr.get_available_models():
            raise HTTPException(400, f"Unknown model '{name}'")
        # mgr.summarize returns one summary per entry in docs_to_summarize
        summaries = mgr.summarize(name, docs_to_summarize)
        results[name] = summaries

    return SummariseResponse(results=results, references=refs)
