#!/bin/bash
#SBATCH --job-name=multi_doc_ddp
#SBATCH --partition=peregrine-gpu       # run on peregrine
#SBATCH --qos=gpu_short                 # up to 24 h on GPU queue
#SBATCH --nodes=1                       # single node
#SBATCH --ntasks-per-node=2             # one process per GPU (peregrine has 2)
#SBATCH --gres=gpu:2                    # request exactly 2 A100 GPUs
#SBATCH --cpus-per-task=8               # for dataloading
#SBATCH --mem=90G                       # per-node RAM
#SBATCH --time=24:00:00                 # wall time limit
#SBATCH --output=logs/ddp-%j.out        # STDOUT/STDERR

set -euo pipefail

# module purge
# module load python/anaconda/py3.10-2023.03
# conda activate mds

mkdir -p logs

# Launch 2‐GPU DDP on peregrine
torchrun \
  --standalone \
  --nproc_per_node=2 \
  -m src.main train \
    --config configs/led.yaml \
    --ddp \
    --epochs 3 \
    --sample_ratio 1.0
