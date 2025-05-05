#!/usr/bin/env python3
import os
import sys

# ──────────────────────────────────────────────────────────────────────────────
# Make sure the project root (one level up from scripts/) is on sys.path
# So that `import src.main` will find your src/ package.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import torch

from src.main import load_config
from src.model import load_model_and_tokenizer
from src.data_processor import get_dataloaders
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(
        description="Train Pegasus, BART & LED on a small subset of Multi-News"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/pegasus.yaml",
            "configs/bart.yaml",
            "configs/led.yaml",
        ],
        help="List of YAML model config files to run"
    )
    parser.add_argument(
        "--subset_ratio",
        type=float,
        default=0.04,
        help="Fraction of the training split to use (e.g. 0.01 for 1%)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs to train on the subset"
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cfg_path in args.configs:
        # Load and override config
        config = load_config(cfg_path)
        config.data.sample_ratio  = args.subset_ratio
        config.training.epochs    = args.epochs
        config.data.batch_size = 2
        config.training.fp16 = True
        # redirect outputs to a *_subset folder
        base_out = config.output.output_dir.rstrip("/")
        config.output.output_dir  = f"{base_out}_subset"

        logger.info(
            f"=== Training {config.model.name} "
            f"on {args.subset_ratio*100:.2f}% of data for "
            f"{args.epochs} epoch(s) ==="
        )

        # Load model & tokenizer
        model, tokenizer = load_model_and_tokenizer(
            config.model.name,
            device,
            ddp=False,
            local_rank=0
        )

        model.gradient_checkpointing_enable()


        # Build small‐subset dataloaders
        train_loader, val_loader, _ = get_dataloaders(
            config, tokenizer, ddp=False
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            ddp=False
        )

        # Run training
        trainer.train()
        logger.info(f"Finished training {config.model.name}. "
                    f"Checkpoints in {config.output.output_dir}\n")

if __name__ == "__main__":
    main()
