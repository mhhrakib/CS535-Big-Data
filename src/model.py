import logging

import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from src.utils import DOC_SEPARATOR

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    ddp: bool = False,

    local_rank: int = 0
):
    logger.info(f"Loading model and tokenizer: {model_name} on {device}")

    # Load tokenizer & add <doc_sep>
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
    if DOC_SEPARATOR not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': [DOC_SEPARATOR]})

    # Load model & resize embeddings
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # # Enable gradient checkpointing if requested in config
    # if getattr(config.training, "gradient_checkpointing", False):
    if 'pegasus' in model_name.lower():
        model.gradient_checkpointing_enable()

    logger.info(
        "Loaded tokenizer class %s for model %s",
        tokenizer.__class__.__name__, model_name
    )

    # Wrap in DDP if needed
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    model.eval()
    return model, tokenizer