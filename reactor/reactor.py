# core engine for fusing model with data and updates
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import gc
import time
import os
from posix import posix_spawn
import psutil

import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    default_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    lr: float = 0.0003  
    vocab_size: int = 3072
    block_size: int = 128
    batch_size: int = 4


class Reactor:
    
    def __init__(self,):
        pass

    def setup(self,):
        pass

    def train(self,):
        pass

    def teardown(self,):
        pass


def _is_rank_0():
    return 0 == int(os.getenv("RANK"))


def print_memory_summary(prefix, device):
    if _is_rank_0:
        print(f"{prefix}, GPU memory allocation: {torch.cuda.max_memory_allocated(device) // 1e9}GB "
              f"CPU used memory percent: {psutil.virtual_memory().percent}, "
              f"CPU memory available: {psutil.virtual_memory().available // 1e9}GB, ")
        torch.cuda.reset_peak_memory_stats(device)

