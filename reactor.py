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





def build_model(cfg):
    """ core model section"""
    rank = get_rank()

    r0_device = torch.device("cuda:0")

    cpu_offloading = False

    if cfg.cpu_offload:
        if _is_rank_0:
            print(f"CPU Offloading enabled")
        cpu_offloading = True
    
    backward_prefetch = None
    if cfg.prefetch == "prehook":
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    elif cfg.prefetch == "posthook":
        backward_prefetch = BackwardPrefetch.BACKWARD_POST
    
    ## Todo - wrap model
    model = create_model(cfg)




    for e in range(cfg.num_epochs):

        output = model(inputs)
        loss = out.sum() if isinstance(out, torch.Tensor) else out.local_value().sum()
        loss.backward()

        del loss
        del out
        gc.collect()  # fire up gc to save memory
        opt.step()
    
        torch.cuda.synchronize()

    dist.barrier()

def save_model():
    """ set barrier and save states todo"""

def setup(cfg):
    local_rank = get_local_rank()
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{local_rank}"

    world_size = get_world_size()
    rank = get_rank()

    if _is_rank_0:
        print(f"World Size = {world_size")
        print("PyTorch version = {torch.__version__}")

    
    dist.init("nccl", rank = rank, world_size = world_size)


def teardown(cfg):
    if _is_rank_0:
        print("\nTraining finished\n")

def main():
    # todo load omegaconf
    #cfg = 
    setup(cfg)
    train(cfg)
    teardown(cfg)

if __name__ == "__main__":
    print(f"Starting Fusion Reactor Training")
    main()












