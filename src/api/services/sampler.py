from datasets import load_dataset
import random

def get_random_examples(n: int, split: str = "test"):
    """
    Return n random examples from the specified split.
    Each as {'document': …, 'summary': …}.
    """
    ds = load_dataset("alexfabbri/multi_news", split=split)
    idxs = random.sample(range(len(ds)), min(n, len(ds)))
    return [{"document": ds[i]["document"], "summary": ds[i]["summary"]} for i in idxs]
