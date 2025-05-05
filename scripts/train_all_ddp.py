#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import random

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Ensure the project root is on sys.path so `import src.*` works
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ──────────────────────────────────────────────────────────────────────────────

from src.main            import load_config
from src.model           import load_model_and_tokenizer
from src.data_processor  import get_dataloaders
from src.trainer         import Trainer
from src.dist_utils      import setup_ddp, cleanup_ddp, is_main_process

def main():
    parser = argparse.ArgumentParser(
        description="DDP‐enabled full‐data training for all models"
    )
    parser.add_argument(
        "--configs", nargs="+",
        default=["configs/pegasus.yaml", "configs/bart.yaml", "configs/led.yaml"],
        help="List of YAML config files to train"
    )
    parser.add_argument(
        "--sample_ratio", type=float, default=1.0,
        help="Fraction of each split to use (1.0 = full dataset)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="If set, override config.training.epochs for all models"
    )
    args = parser.parse_args()

    # logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger("train_all_ddp")

    # Pick up backend/init_method from the first config
    base_cfg = load_config(args.configs[0])

    # Initialize DDP (reads RANK, WORLD_SIZE, LOCAL_RANK from torchrun env)
    rank, world_size, local_rank = setup_ddp(base_cfg)
    device = torch.device("cuda", local_rank)

    # reproducibility
    random.seed(base_cfg.training.seed)
    torch.manual_seed(base_cfg.training.seed)

    for cfg_path in args.configs:
        config = load_config(cfg_path)

        # override dataset fraction
        config.data.sample_ratio = args.sample_ratio
        # optional override epochs
        if args.epochs is not None:
            config.training.epochs = args.epochs

        if is_main_process():
            logger.info(
                f"→ Training {config.model.name} "
                f"(world_size={world_size}, "
                f"sample_ratio={config.data.sample_ratio}, "
                f"epochs={config.training.epochs})"
            )

        # load model & tokenizer, wrap in DDP
        model, tokenizer = load_model_and_tokenizer(
            config.model.name,
            device,
            ddp=True,
            local_rank=local_rank
        )
        # # optional memory–compute tradeoff
        # if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

        # build DDP dataloaders
        train_loader, val_loader, _ = get_dataloaders(
            config, tokenizer, ddp=True
        )

        # train
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            ddp=True
        )
        trainer.train()

        if is_main_process():
            logger.info(f"✅ Finished {config.model.name}\n")

    # teardown
    cleanup_ddp()


if __name__ == "__main__":
    main()
