import os
import torch
import psutil


def is_rank_0():
    return 0 == int(os.getenv("RANK"))


def get_rank():
    return int(os.getenv("RANK"))


def get_world_size():
    return int(os.getenv("WORLD_SIZE"))


def get_local_rank():
    return int(os.getenv("LOCAL_RANK"))


# memory
def print_memory_summary(prefix, device):
    if is_rank_0:
        print(
            f"{prefix}, GPU memory allocation: {torch.cuda.max_memory_allocated(device) // 1e9}GB "
            f"CPU used memory percent: {psutil.virtual_memory().percent}, "
            f"CPU memory available: {psutil.virtual_memory().available // 1e9}GB, "
        )
        torch.cuda.reset_peak_memory_stats(device)


def calc_flop(cfg):
    """todo: this pulls layer, model and vocab size to calc flops"""
    B = args.batch_size
    s = args.block_size
    conf = get_model_config(cfg)
    l = conf.n_layer
    h = conf.n_embd
    V = conf.vocab_size
    return 96 * B * s * l * h * h * (1 + s / 6 / h + V / 16 / l / h)
