import logging

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)

from src.utils import get_tokenizer

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(config):
    """
    Load model and tokenizer based on config

    Args:
        config: Model configuration

    Returns:
        model: The loaded model
        tokenizer: The loaded tokenizer
    """
    model_name = config.model.model_name

    logger.info(f"Loading model and tokenizer: {model_name}")

    # Determine max lengths based on model
    if "t5" in model_name:
        # T5 has different length limits
        if "small" in model_name:
            config.data.max_input_length = min(config.data.max_input_length, 512)
        else:
            config.data.max_input_length = min(config.data.max_input_length, 512)
    elif "pegasus" in model_name:
        config.data.max_input_length = min(config.data.max_input_length, 1024)
    elif "bart" in model_name:
        if "large" in model_name:
            config.data.max_input_length = min(config.data.max_input_length, 1024)
        else:
            config.data.max_input_length = min(config.data.max_input_length, 1024)
    elif "led" in model_name or "longformer" in model_name:
        config.data.max_input_length = min(config.data.max_input_length, 4096)
    else:
        # Default
        config.data.max_input_length = min(config.data.max_input_length, 512)

    # For output/summary length
    config.data.max_output_length = min(config.data.max_output_length, 256)

    logger.info(
        f"Using max_input_length={config.data.max_input_length}, max_output_length={config.data.max_output_length}")

    # Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = get_tokenizer(model_name)

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer


def setup_optimizer_and_scheduler(model, config, total_steps):
    """
    Setup optimizer and learning rate scheduler

    Args:
        model: PyTorch model
        config: Training configuration
        total_steps: Total training steps

    Returns:
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
    """
    # Setup optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.model.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.model.learning_rate
    )

    # Setup scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.model.warmup_steps,
        num_training_steps=total_steps,
    )

    return optimizer, scheduler