import os
import torch
import psutil

# memory
def print_memory_summary(prefix, device):
    if 0 == int(os.getenv("RANK")):
        print(
            f"{prefix}, GPU memory allocation: {torch.cuda.max_memory_allocated(device) // 1e9}GB\n "
            f"CPU used memory percent: {psutil.virtual_memory().percent},\n "
            f"CPU memory available: {psutil.virtual_memory().available // 1e9}GB,\n "
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
