from contextlib import contextmanager

import torch


@contextmanager
def seeded_rng(seed, device):
    """Reproducible draws inside the block, without touching the global generators.

    `torch.manual_seed` reseeds both the CPU and the CUDA generators, while
    `torch.get_rng_state` / `torch.set_rng_state` only save the CPU one. Seeding
    a validation step that way left the CUDA generator reseeded afterwards, so
    on GPU every training epoch replayed the same random sequence, whatever
    `config.seed` (fixed in 0.1.4). `torch.random.fork_rng` restores the CPU
    generator and, when `device` is a CUDA device, the CUDA one too.
    """
    device = torch.device(device)
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        yield
