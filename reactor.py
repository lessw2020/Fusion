# core engine for fusing model with data and updates
from dataclasses import dataclass
from pathlib import Path
import datetime
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

import plasma  # training engine within reactor
import build_optimizer
import build_scheduler

# from environ_utils import *

from omegaconf import OmegaConf
import model_builder
import dataset_builder

import tqdm
import colorama as cr

cr.init(autoreset=True)  # init console


# ---------------
# Rank utils have to be inside the main file.  Importing will result in it living in
# rank 0 land, and thus all 0 == int(os.getenv("RANK")) will return as True...


def set_singleton_view_gpu(local_rank: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{local_rank}"


def is_rank_0():
    if os.getenv("RANK") == 0:
        return True
    else:
        return False

    # return 0 == int(os.getenv("RANK"))


def get_rank():
    return int(os.getenv("RANK"))


def get_world_size():
    return int(os.getenv("WORLD_SIZE"))


def get_local_rank():
    """Get local rank - this is sometimes None"""
    return int(os.getenv("LOCAL_RANK"))


# ---------------


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

    if verbose and 0 == int(os.getenv("RANK")):
        print(f"--> Configuring World Environment")

    local_rank = get_local_rank()
    # print(f"local_rank = {local_rank}")

    # set device so each process only sees it's respective GPU
    set_singleton_view_gpu(local_rank)

    rank = int(os.getenv("RANK"))
    world_size = get_world_size()

    if 0 == int(os.getenv("RANK")):

        print(f" rank = {rank} and world_size = {world_size}")
        # print(f"dist initialized? {torch.distributed.is_initialized()}")
        print(f"nccl status = {torch.distributed.is_nccl_available()}")
        print(
            f"launched from torch elastic =  {torch.distributed.is_torchelastic_launched()}"
        )

    # init
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if 0 == int(os.getenv("RANK")) and verbose:
        print(f"World size is {world_size}\n")
        print(f"PyTorch version {torch.__version__}\n")
        print(f"\n World environ setup complete \n")


def setup_model():
    """core model section"""
    rank = int(os.getenv("RANK"))

    r0_device = torch.device("cuda:0")

    cpu_offloading = False

    """
    if cfg.cpu_offload:
        if 0 == int(os.getenv("RANK")):
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
    model_filename = "mnist_model"  # no .py on end - todo - parse and check
    model_class = "Mnist_Model"

    model = model_builder.create_model(
        model_filename, model_class
    )  # todo - pass in model config file
    if 0 == int(os.getenv("RANK")):
        # print(model)
        print("\n Model building complete ")

    sharded_model = FSDP(model)
    if 0 == int(os.getenv("RANK")):
        print(f"Sharded model = {sharded_model}")

    return sharded_model


def build_datasets():
    """ "build training and val dataloaders from dataset"""

    dataloader_training = dataset_builder.build_training_dataloader()

    dataloader_val = dataset_builder.build_val_dataloader()

    if 0 == int(os.getenv("RANK")):
        print(f"Dataloaders all built!\n")

    return dataloader_training, dataloader_val


def save_model():
    """set barrier and save states todo"""


def teardown():
    """clean up world before exiting"""
    if 0 == int(os.getenv("RANK")):
        print(f"Wrapping up...")

    dist.destroy_process_group()

    if 0 == int(os.getenv("RANK")):
        print(cr.Fore.LIGHTBLUE_EX + "\nTraining finished\n")


# -----   Main ----------


def reactor_world_main():
    """main processing function for each process"""
    setup_world()
    # important - models must be placed onto device, then sharded
    model = setup_model()

    dataloader_train, dataloader_val = build_datasets()

    optimizer = build_optimizer.build_optimizer(model)
    lr_scheduler = build_scheduler.build_lr_scheduler(optimizer)

    # one training loop
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    device = torch.cuda.set_device(rank)

    num_epochs = 2

    if 0 == int(os.getenv("RANK")):
        print(cr.Fore.LIGHTBLUE_EX + "Start training")
        start_time = time.time()

    # for rzero_epoch in tqdm.tqdm(range(num_epochs)):
    for curr_epoch in range(num_epochs):

        plasma.train_one_epoch(
            rank,
            world_size,
            model,
            dataloader_train,
            optimizer,
            device,
            curr_epoch,
            max_norm=1.0,
        )

        # todo - step lr, step profiler?

        plasma.val_one_epoch(rank, world_size, model, dataloader_val)

        if 0 == int(os.getenv("RANK")):
            print(cr.Fore.GREEN + f"epoch {curr_epoch} completed ")

    # end training
    if 0 == int(os.getenv("RANK")):
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(cr.Fore.LIGHTGREEN_EX + f"Training time {total_time_str}")

    teardown()
    return


if __name__ == "__main__":
    if 0 == int(os.getenv("RANK")):
        print(f"Starting Fusion Reactor...\n")

    reactor_world_main()
