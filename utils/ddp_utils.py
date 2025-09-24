import torch
import torch.distributed as dist
import os
from datetime import timedelta


def ddp_setup():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    if not torch.distributed.is_initialized():
        dist.init_process_group(
            "nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=1800),
        )


def ddp_cleanup():
    dist.barrier()
    dist.destroy_process_group()


def sync_across_gpus(t, world_size):
    """
    Synchronizes predictions accross all gpus.

    Args:
        t (torch tensor): Tensor to synchronzie
        world_size (int): World size.

    Returns:
        torch tensor: Synced tensor.
    """
    torch.distributed.barrier()
    gather_t_tensor = [torch.ones_like(t) for _ in range(world_size)]
    torch.distributed.all_gather(gather_t_tensor, t)
    return torch.cat(gather_t_tensor)
