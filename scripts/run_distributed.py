import os
import subprocess
import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to the config file")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    args = parser.parse_args()

    # Check GPU availability
    if torch.cuda.device_count() < args.num_gpus:
        print(f"WARNING: Requested {args.num_gpus} GPUs but only {torch.cuda.device_count()} available")
        args.num_gpus = torch.cuda.device_count()

    if args.num_gpus <= 0:
        print("No GPUs available. Running on CPU.")
        command = f"python -m src.main --config {args.config}"
        subprocess.run(command, shell=True)
    else:
        # Run distributed training
        print(f"Running distributed training on {args.num_gpus} GPUs")
        command = f"torchrun --nproc_per_node={args.num_gpus} -m src.main --config {args.config}"
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()