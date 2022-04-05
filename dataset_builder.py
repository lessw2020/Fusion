import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def build_train_transforms(cfg=None):
    """build transform pipeline for training"""
    train_transform = None
    print(f"Training transform = {cfg.train_transform}")

    if cfg.train_transform == "mnist_train":
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    elif cfg.train_transform == "in_dataset":
        print(f"using transforms in dataset")
    else:
        print("unknown train transform!!")

    return train_transform


def build_val_transforms(cfg=None):
    """transforms for validation and prediction"""
    val_transform = None

    if cfg.val_transform == "mnist_val":
        val_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    elif cfg.val_transform == "in_dataset":
        print(f"using val transforms in dataset")
    else:
        print(f"Unknown validation transform!")

    return val_transform


def build_training_dataloader(cfg=None):
    """build out the settings and return training dataloader"""
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))

    dataloader_training = None

    # get transforms if applicable (will return None if included in dataset ala summarization)

    train_transform = build_train_transforms(cfg)

    if cfg.project == "mnist":
        dataset_training = datasets.MNIST(
            "./data", train=True, download=True, transform=train_transform
        )
        sampler_training = DistributedSampler(
            dataset_training, rank=rank, num_replicas=world_size, shuffle=True
        )
        train_kwargs = {"batch_size": cfg.batch_size, "sampler": sampler_training}

        cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
        train_kwargs.update(cuda_kwargs)

        dataloader_training = torch.utils.data.DataLoader(
            dataset_training, **train_kwargs
        )
    # ---- wikihow --------

    elif cfg.project == "wikihow":
        # lame but hardcoding while we flush out the layout here
        tokenizer = None
        from projects.wikihow.dataset import wikihow

        if cfg.tokenizer == "t5":
            tokenizer = T5Tokenizer.from_pretrained(cfg.model_name)
        num_samples = (
            cfg.num_train_samples
        )  # todo - this is from inspecting the dataset...needs to be more dynamic

        train_dataset = wikihow(
            tokenizer=tokenizer,
            type_path="train",
            num_samples=num_samples,
            input_length=cfg.max_input_length,
            output_length=cfg.max_output_length,
        )

        dataloader_training = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
    # if 0 == rank:
    print(f"\nTraining Dataloader built for {cfg.project}!")
    print(f"Total Training samples = {len(dataloader_training)}\n")

    return dataloader_training


def build_val_dataloader(cfg=None):
    """validation dataset and dataloader"""
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))

    val_transforms = build_val_transforms(cfg)

    dataset_val = None
    dataloader_val = None

    if cfg.project == "mnist":

        dataset_val = datasets.MNIST("./data", train=False, transform=val_transforms)

        sampler_val = DistributedSampler(
            dataset_val, rank=rank, num_replicas=world_size
        )

        val_kwargs = {"batch_size": cfg.batch_size, "sampler": sampler_val}

        cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}

        val_kwargs.update(cuda_kwargs)

        dataloader_val = torch.utils.data.DataLoader(dataset_val, **val_kwargs)

    elif cfg.project == "wikihow":
        tokenizer = None
        from projects.wikihow.dataset import wikihow

        if cfg.tokenizer == "t5":
            tokenizer = T5Tokenizer.from_pretrained(cfg.model_name)
        num_samples = (
            cfg.num_val_samples
        )  # todo - this is from inspecting the dataset...needs to be more dynamic

        val_dataset = wikihow(
            tokenizer=tokenizer,
            type_path="validation",
            num_samples=num_samples,
            input_length=cfg.max_input_length,
            output_length=cfg.max_output_length,
        )

        dataloader_val = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
    # if 0 == rank:
    print(f"\nTraining Dataloader built for {cfg.project}!")
    print(f"Total validation samples = {len(dataloader_val)}\n")

    return dataloader_val
