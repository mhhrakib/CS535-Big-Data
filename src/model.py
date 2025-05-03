# import logging

# import torch
# from transformers import (
#     AutoModelForSeq2SeqLM,
#     get_linear_schedule_with_warmup
# )

# from src.utils import get_tokenizer

# # Set up logging
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO,
# )
# logger = logging.getLogger(__name__)


# def load_model_and_tokenizer(config):
#     """
#     Load model and tokenizer based on config

#     Args:
#         config: Model configuration

#     Returns:
#         model: The loaded model
#         tokenizer: The loaded tokenizer
#     """
#     model_name = config.model.model_name

#     # Determine max lengths based on model
#     if "t5" in model_name:
#         # T5 has different length limits
#         if "small" in model_name:
#             config.data.max_input_length = min(config.data.max_input_length, 512)
#         else:
#             config.data.max_input_length = min(config.data.max_input_length, 512)
#     elif "pegasus" in model_name:
#         config.data.max_input_length = min(config.data.max_input_length, 1024)
#     elif "bart" in model_name:
#         if "large" in model_name:
#             config.data.max_input_length = min(config.data.max_input_length, 1024)
#         else:
#             config.data.max_input_length = min(config.data.max_input_length, 1024)
#     elif "led" in model_name or "longformer" in model_name:
#         config.data.max_input_length = min(config.data.max_input_length, 4096)
#     else:
#         # Default
#         config.data.max_input_length = min(config.data.max_input_length, 512)


#     # For output/summary length
#     config.data.max_output_length = min(config.data.max_output_length, 256)

#     logger.info(
#         f"Using max_input_length={config.data.max_input_length}, max_output_length={config.data.max_output_length}")

#     # Load tokenizer
#     # tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer = get_tokenizer(model_name)

#     logger.info(f"Loaded model:      {model_name}")
#     logger.info(f"Loaded tokenizer:  {tokenizer.name_or_path} ({tokenizer.__class__.__name__})")
#     # Load model
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     logger.info(f"Loading model and tokenizer: {model_name}")


#     return model, tokenizer


# def setup_optimizer_and_scheduler(model, config, total_steps):
#     """
#     Setup optimizer and learning rate scheduler

#     Args:
#         model: PyTorch model
#         config: Training configuration
#         total_steps: Total training steps

#     Returns:
#         optimizer: PyTorch optimizer
#         scheduler: Learning rate scheduler
#     """
#     # Setup optimizer
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#             "weight_decay": config.model.weight_decay,
#         },
#         {
#             "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#             "weight_decay": 0.0,
#         },
#     ]
#     optimizer = torch.optim.AdamW(
#         optimizer_grouped_parameters,
#         lr=config.model.learning_rate
#     )

#     # Setup scheduler
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=config.model.warmup_steps,
#         num_training_steps=total_steps,
#     )

#     return optimizer, scheduler


import logging
import torch
from transformers import AutoModelForSeq2SeqLM
from src.utils import get_tokenizer

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    ddp: bool = False,
    local_rank: int = 0
):
    """
    Load a Seq2Seq model and its tokenizer, move to device, and optionally wrap in DDP.

    Args:
        model_name: Hugging Face model identifier (e.g. 'google/pegasus-large')
        device: torch.device to load the model onto
        ddp: whether to wrap the model in DistributedDataParallel
        local_rank: GPU index for this process when using DDP

    Returns:
        model: the loaded (and possibly DDP-wrapped) model
        tokenizer: the Hugging Face tokenizer
    """
    logger.info(f"Loading model and tokenizer: {model_name} on {device}")
    # Load tokenizer
    tokenizer = get_tokenizer(model_name)

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    # Wrap with DDP if requested
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        logger.info(f"Wrapping model in DDP on local rank {local_rank}")
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    model.eval()
    return model, tokenizer
