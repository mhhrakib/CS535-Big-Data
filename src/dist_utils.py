import os
import logging
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

def setup_ddp(config):
    """
    Initialize torch.distributed environment for DDP.
    Expects config.ddp.init_method (e.g. 'env://') or URL,
    config.ddp.world_size, and config.training.local_rank from env vars or passed args.
    """
    # Determine backend and initialization method
    backend = config.ddp.backend if hasattr(config.ddp, 'backend') else 'nccl'
    init_method = config.ddp.init_method if hasattr(config.ddp, 'init_method') else 'env://'

    # If using torchrun, LOCAL_RANK and RANK and WORLD_SIZE are set in environment
    rank = int(os.environ.get('RANK', config.ddp.rank if hasattr(config.ddp, 'rank') else 0))
    world_size = int(os.environ.get('WORLD_SIZE', config.ddp.world_size if hasattr(config.ddp, 'world_size') else 1))

    logger.info(f"Initializing DDP: rank {rank}/{world_size}, backend={backend}, init_method={init_method}")
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    # Set device for current process
    local_rank = int(os.environ.get('LOCAL_RANK', config.training.local_rank if hasattr(config.training, 'local_rank') else 0))
    torch.cuda.set_device(local_rank)
    logger.info(f"DDP setup complete. Local rank set to {local_rank}.")
    return rank, world_size, local_rank


def cleanup_ddp():
    """Clean up distributed process group."""
    logger.info("Cleaning up DDP process group.")
    dist.destroy_process_group()


def is_main_process():
    """Returns True if the current process is rank 0."""
    return dist.get_rank() == 0 if dist.is_initialized() else True


def barrier():
    """Block until all processes reach this barrier."""
    if dist.is_initialized():
        dist.barrier()
