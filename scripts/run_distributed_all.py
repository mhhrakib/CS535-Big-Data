#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import logging

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Ensure project root is on PYTHONPATH so `import src.*` works
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Helper to launch src.main in single‐GPU or DDP mode for _all_ models"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            # "configs/bart.yaml",
            "configs/pegasus.yaml",
            # "configs/led.yaml",
        ],
        help="List of YAML config files to train"
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.1,
        help="Override config.data.sample_ratio (1.0 = full dataset)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger("run_distributed")

    ngpu = torch.cuda.device_count()
    if ngpu < 1:
        logger.info("No GPUs detected, falling back to CPU.")
    else:
        logger.info(f"Detected {ngpu} GPU(s).")

    for cfg in args.configs:
        logger.info(f"Preparing to train model with config: {cfg}")
        # build the base invocation
        cmd_base = ["-m", "src.main", "train", "--config", cfg]
        # pass sample_ratio override
        cmd_base += ["--sample_ratio", str(args.sample_ratio)]

        if ngpu > 1:
            logger.info(f"Launching DDP on {ngpu} GPUs for {cfg}")
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={ngpu}",
            ] + cmd_base + ["--ddp"]
        else:
            mode = "GPU" if ngpu == 1 else "CPU"
            logger.info(f"Running single‐process on {mode} for {cfg}")
            cmd = ["python"] + cmd_base

        logger.info("Executing:\n  %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("Process for config %s failed (exit code %d)", cfg, e.returncode)
            raise

if __name__ == "__main__":
    main()
