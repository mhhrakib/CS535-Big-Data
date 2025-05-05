# scripts/run_distributed.py
#!/usr/bin/env python3
#!/usr/bin/env python3
import os
import subprocess
import torch
import argparse
import logging

def main():
    parser = argparse.ArgumentParser(
        description="Helper to launch src.main in single‐GPU or DDP mode"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to your YAML config (must include model, data, training, ddp blocks)"
    )
    parser.add_argument(
        "--subset_ratio", type=float, default=0.04,
        help="(Optional) override config.data.sample_ratio"
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="(Optional) override config.training.epochs"
    )
    args = parser.parse_args()

    # set up logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger("run_distributed")

    ngpu = torch.cuda.device_count()
    if ngpu < 1:
        logger.info("No GPUs detected, will run on CPU.")
    else:
        logger.info(f"Detected {ngpu} GPU(s).")

    # build the base command
    cmd_base = ["-m", "src.main", "train", "--config", args.config]
    if args.subset_ratio is not None:
        cmd_base += ["--sample_ratio", str(args.subset_ratio)]
    if args.epochs is not None:
        cmd_base += ["--epochs", str(args.epochs)]

    # choose single‐process vs DDP
    if ngpu > 1:
        logger.info(f"Launching DDP on {ngpu} GPUs")
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={ngpu}",
        ] + cmd_base + ["--ddp"]
    else:
        backend = "GPU" if ngpu == 1 else "CPU"
        logger.info(f"Running single‐process on {backend}")
        cmd = ["python"] + cmd_base

    logger.info("Executing command:\n  %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Process failed with exit code %d", e.returncode)
        raise

if __name__ == "__main__":
    main()
