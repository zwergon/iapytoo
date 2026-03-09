import os
import torch
import torch.distributed as dist

from ddp_setup import setup_ddp


def init_process():
    rank = int(os.environ['RANK'])

    setup_ddp(rank)

    # Test simple : broadcast tensor from rank 0
    device = torch.device(
        f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    tensor = torch.zeros(1, device=device)
    if rank == 0:
        tensor += 42
    dist.broadcast(tensor, src=0)
    print(f"[Rank {rank}] Tensor after broadcast: {tensor.item()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    init_process()
