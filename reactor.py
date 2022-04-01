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
import torch.multiprocessing as mp

from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)
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

from environ_utils import *
from omegaconf import OmegaConf
import model_builder


@dataclass
class TrainConfig:
    lr: float = 0.0003
    vocab_size: int = 3072
    block_size: int = 128
    batch_size: int = 4


class Reactor:
    def __init__(
        self,
    ):
        pass

    def setup(
        self,
    ):
        pass

    def train(
        self,
    ):
        pass

    def teardown(
        self,
    ):
        pass


def setup_world(verbose=True):
    """configure distributed world env"""

    # $os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"

    global_seed = 2022  # todo put into config

    torch.manual_seed(global_seed)

    WORLD_SIZE = torch.cuda.device_count()

    if verbose and is_rank_0:
        print(f"--> Configuring World Environment")

    local_rank = get_local_rank()
    # print(f"local_rank = {local_rank}")

    # set device so each process only sees it's respective GPU
    set_singleton_view_gpu(local_rank)

    rank = get_rank()
    world_size = get_world_size()
    if rank == 0:
        print(f" rank = {rank} and world_size = {world_size}")
        print(f"dist initialized? {torch.distributed.is_initialized()}")
        print(f"nccl here? {torch.distributed.is_nccl_available()}")
        print(
            f"launched from torch elastic? {torch.distributed.is_torchelastic_launched()}"
        )

    # init
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if is_rank_0 and verbose:
        print(f"World size is {world_size}")
        print(f"PyTorch version {torch.__version__}")
        print(f"\n World environ setup complete \n")


def setup_model():
    """core model section"""
    rank = get_rank()

    r0_device = torch.device("cuda:0")

    cpu_offloading = False

    """
    if cfg.cpu_offload:
        if is_rank_0:
            print(f"CPU Offloading enabled")
        cpu_offloading = True

    backward_prefetch = None
    if cfg.prefetch == "prehook":
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    elif cfg.prefetch == "posthook":
        backward_prefetch = BackwardPrefetch.BACKWARD_POST

    ## Todo - wrap model
"""

    # get model config
    # cfg = OmegaConf.load("mymodel.yaml")

    # print(cfg)

    model = model_builder.create_model()  # todo - pass in model config file
    if is_rank_0:
        print(model)
        print("\n Model building complete ")


def save_model():
    """set barrier and save states todo"""


def teardown():
    """clean up world before exiting"""
    print(f"Succes! Wrapping up")
    dist.destroy_process_group()
    if is_rank_0:
        print("\nTraining finished\n")


def spawn_world(verbose=True):
    """spawn processes within the world"""

    # mp.spawn(reactor_world_main(), args=(WORLD_SIZE), nprocs=WORLD_SIZE, join=True)


def reactor_world_main():
    """main processing function for each process"""
    setup_world()

    setup_model()
    teardown()
    return


def main():
    # load omegaconf
    cfg = OmegaConf.load("mymodel.yaml")
    # print(cfg.model.file)
    # todo in progress here...
    reactor_world_main()

    # setup_model()
    # train()


if __name__ == "__main__":
    print(f"Starting Fusion Reactor...\n")

    reactor_world_main()
