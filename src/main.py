import argparse
import os
import torch
import logging
import random
import numpy as np
from src.config import Config
from src.data_processor import get_dataloaders
from src.model import load_model_and_tokenizer, setup_optimizer_and_scheduler
from src.trainer import Trainer

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Sets random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        logger.info("Using device cuda")
        torch.cuda.manual_seed_all(seed)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to the config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)

    # Update local_rank from arguments if provided
    if args.local_rank != -1:
        config.training.local_rank = args.local_rank
        config.training.distributed = True

    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)

    # Set seed for reproducibility
    set_seed(config.data.seed)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config, tokenizer)

    # Calculate total steps for scheduler
    total_steps = len(train_dataloader) * config.training.num_epochs // config.model.gradient_accumulation_steps

    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config, total_steps)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        config=config
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()