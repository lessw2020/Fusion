# core engine for fusing model with data and updates
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import gc
import time
import os
from posix import posix_spawn
from black import out
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

def get_rank():
    return int(os.getenv("RANK"))

def get_world_size():
    return int(os.getenv("WORLD_SIZE"))

def get_local_rank():
    return int(os.getenv("LOCAL_RANK"))



# memory
def print_memory_summary(prefix, device):
    if _is_rank_0:
        print(f"{prefix}, GPU memory allocation: {torch.cuda.max_memory_allocated(device) // 1e9}GB "
              f"CPU used memory percent: {psutil.virtual_memory().percent}, "
              f"CPU memory available: {psutil.virtual_memory().available // 1e9}GB, ")
        torch.cuda.reset_peak_memory_stats(device)

def calc_flop(cfg):
    """ todo: this pulls layer, model and vocab size to calc flops"""
    B = args.batch_size
    s = args.block_size
    conf = get_model_config(cfg)
    l = conf.n_layer
    h = conf.n_embd
    V = conf.vocab_size
    return 96 * B * s * l * h * h * (1 + s/6/h + V/16/l/h)


def build_model(cfg):
    """ core model section"""
    rank = get_rank()

    r0_device = torch.device("cuda:0")

    cpu_offloading = False

    if cfg.cpu_offload:
        if _is_rank_0:
            print(f"CPU Offloading enabled")
        cpu_offloadin = True
    
    backward_prefetch = None
    if cfg.prefetch == "prehook":
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    elif cfg.prefecth == "posthook":
        backward_prefetch = BackwardPrefetch.BACKWARD_POST
    
    ## Todo - wrap model


def train(cfg):
    rank = get_rank()
    world_size = get_world_size()

    # setup has configured all processes to map to exclusive devices.
    for item in range(torch.cuda.device_count()):
        torch.cuda.synchronize(item)

    currFlop = calc_flop(cfg)

    if _is_rank_0:
        print(f"visible devices = {torch.cuda.device_count()}", flush=True)
        print(f"TerraFlop per iteration = {currFlop // 10**12}")

    if cfg.profile:
        init_start_event = torch.cuda.Event(enable_timing=True)
        init_end_event = torch.cuda.Event(enable_timing=True)
        before_forward_event = torch.cuda.Event(enable_timing=True)
        after_forward_event = torch.cuda.Event(enable_timing=True)
        after_backward_event = torch.cuda.Event(enable_timing=True)
        after_step_event = torch.cuda.Event(enable_timing=True)
        after_zero_grad_event = torch.cuda.Event(enable_timing=True)

        init_start_event.record()
    
    

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    # record_shapes=True, # Causes seg fault in export_chrome_trace
    # with_stack=True, # Causes seg fault with EFA
    # with_flops=True, # Causes seg fault in export_chrome_trace
    on_trace_ready=my_tensorboard_trace_handler(f"tb/{now.strftime('%Y_%m_%d_%H_%M_%S')}", rank, use_gzip=True)
) if cfg.profile else contextlib.nullcontext() as prof:
    now = datetime.now()
    model = build_model()

    )

    if cfg.profile:
        init_end_event.record()
    end = datetime.now()

    sync_all_devices()
    dist.barrier()


    if _is_rank_0:
        print(f"Model built: \n {model}")
    
    print_memory_summary("After model init", "cuda:0")

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












