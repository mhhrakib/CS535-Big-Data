# # src/main.py

# import argparse
# import logging
# import os
# import yaml
# import torch
# from types import SimpleNamespace

# from src.dist_utils import setup_ddp, cleanup_ddp, is_main_process
# from src.model import load_model_and_tokenizer
# from src.data_processor import get_dataloaders
# from src.trainer import Trainer
# from src.evaluate import evaluate_model

# # Configure root logger
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO
# )
# logger = logging.getLogger(__name__)

# def dict_to_namespace(d):
#     """
#     Recursively convert dict to SimpleNamespace
#     """
#     if isinstance(d, dict):
#         return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
#     elif isinstance(d, list):
#         return [dict_to_namespace(v) for v in d]
#     else:
#         return d

# def load_config(path: str):
#     """Load YAML config file into a SimpleNamespace object."""
#     with open(path, 'r') as f:
#         cfg_dict = yaml.safe_load(f)
#     return dict_to_namespace(cfg_dict)

# def main():
#     parser = argparse.ArgumentParser(description="Multi-Document Summarization Trainer/Evaluator")
#     subparsers = parser.add_subparsers(dest='command', required=True)
    
#     # Train subcommand
#     train_parser = subparsers.add_parser('train')
#     train_parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
#     train_parser.add_argument('--ddp', action='store_true', help='Enable distributed training')

#     # Eval subcommand
#     eval_parser = subparsers.add_parser('eval')
#     eval_parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
#     eval_parser.add_argument('--ckpt_dir', type=str, required=True, help='Path to model checkpoint')
#     eval_parser.add_argument('--split', type=str, default='test', help='Dataset split for evaluation')
#     eval_parser.add_argument('--num_samples', type=int, default=None, 
#         help="If set, randomly sample this many examples instead of full split"
#     )

#     args = parser.parse_args()
#     config = load_config(args.config)

#     if args.command == 'train':
#         # Distributed setup
#         if args.ddp:
#             rank, world_size, local_rank = setup_ddp(config)
#         else:
#             local_rank = 0
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # Load model & tokenizer
#         model, tokenizer = load_model_and_tokenizer(
#             config.model.name,
#             device,
#             ddp=args.ddp,
#             local_rank=local_rank
#         )

#         # Data loaders
#         train_loader, val_loader, _ = get_dataloaders(config, tokenizer, ddp=args.ddp)

#         # Trainer
#         trainer = Trainer(
#             model=model,
#             tokenizer=tokenizer,
#             train_loader=train_loader,
#             val_loader=val_loader,
#             config=config,
#             device=device,
#             ddp=args.ddp
#         )
#         trainer.train()

#         # Cleanup DDP
#         if args.ddp:
#             cleanup_ddp()

#     elif args.command == 'eval':
#         # Single-process evaluation
#         metrics = evaluate_model(
#             config=config,
#             ckpt_dir=args.ckpt_dir,
#             split=args.split,
#             num_samples = args.num_samples
#         )
#         if is_main_process():
#             logger.info("Evaluation metrics:\n" +
#                         "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

# if __name__ == '__main__':
#     main()


# src/main.py

import argparse
import logging
import os
import yaml
import torch

from src.dist_utils import setup_ddp, cleanup_ddp, is_main_process
from src.model import load_model_and_tokenizer
from src.data_processor import get_dataloaders
from src.trainer import Trainer
from src.evaluate import evaluate_model
from src.utils import load_config

# Configure root logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Multi-Document Summarization Trainer/Evaluator")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train subcommand
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--config',       type=str, required=True, help='Path to YAML config')
    train_parser.add_argument('--ddp',          action='store_true',        help='Enable distributed training')
    train_parser.add_argument('--epochs',       type=int,   default=None,    help='(opt) override config.training.epochs')
    train_parser.add_argument('--sample_ratio', type=float, default=None,    help='(opt) override config.data.sample_ratio')

    # Eval subcommand
    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('--config',      type=str, required=True, help='Path to YAML config')
    eval_parser.add_argument('--ckpt_dir',    type=str, required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--split',       type=str, default='test',   help='Dataset split for evaluation')
    eval_parser.add_argument('--num_samples', type=int, default=None,
                             help='If set, randomly sample this many examples instead of full split')

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == 'train':
        # Apply overrides
        if args.epochs is not None:
            logger.info(f"Overriding epochs: {config.training.epochs} -> {args.epochs}")
            config.training.epochs = args.epochs
        if args.sample_ratio is not None:
            logger.info(f"Overriding sample_ratio: {config.data.sample_ratio} -> {args.sample_ratio}")
            config.data.sample_ratio = args.sample_ratio

        # Distributed setup
        if args.ddp:
            rank, world_size, local_rank = setup_ddp(config)
        else:
            local_rank = 0

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model & tokenizer
        model, tokenizer = load_model_and_tokenizer(
            config.model.name,
            device,
            ddp=args.ddp,
            local_rank=local_rank
        )

    # # Enable gradient checkpointing if requested in config
    # if getattr(config.training, "gradient_checkpointing", False):
    #     model.gradient_checkpointing_enable()

        # Data loaders
        train_loader, val_loader, _ = get_dataloaders(config, tokenizer, ddp=args.ddp)

        # Trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            ddp=args.ddp
        )

        try:
            trainer.train()
        finally:
            # Cleanup DDP
            if args.ddp:
                cleanup_ddp()

    elif args.command == 'eval':
        # Single-process evaluation
        metrics = evaluate_model(
            config=config,
            ckpt_dir=args.ckpt_dir,
            split=args.split,
            num_samples=args.num_samples
        )
        if is_main_process():
            logger.info("Evaluation metrics:\n" +
                        "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

if __name__ == '__main__':
    main()
