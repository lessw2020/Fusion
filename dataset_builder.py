import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.data.distributed import DistributedSampler


def build_train_transforms():

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    return transform


def build_val_transforms():
    """transforms for validatin and prediction"""
    if 0 == int(os.getenv("RANK")):
        print(f" --> using training transforms")

    return build_train_transforms()  # todo - this is only for mnist example


def build_training_dataloader(batch_size=4):
    """build out the settings and return training dataloader"""

    train_transform = build_train_transforms()

    dataset_training = datasets.MNIST(
        "./data", train=True, download=True, transform=train_transform
    )

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))

    sampler_training = DistributedSampler(
        dataset_training, rank=rank, num_replicas=world_size, shuffle=True
    )

    train_kwargs = {"batch_size": batch_size, "sampler": sampler_training}

    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)

    dataloader_training = torch.utils.data.DataLoader(dataset_training, **train_kwargs)

    return dataloader_training


def build_val_dataloader(batch_size=4):
    """validation dataset and dataloader"""
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))

    val_transforms = build_val_transforms()

    dataset_val = datasets.MNIST("./data", train=False, transform=val_transforms)

    sampler_val = DistributedSampler(dataset_val, rank=rank, num_replicas=world_size)

    val_kwargs = {"batch_size": batch_size, "sampler": sampler_val}

    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}

    val_kwargs.update(cuda_kwargs)

    dataloader_val = torch.utils.data.DataLoader(dataset_val, **val_kwargs)

    return dataloader_val
