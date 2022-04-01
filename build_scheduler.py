import torch
from torch.optim.lr_scheduler import StepLR


def build_lr_scheduler(optimizer, step_size=1, gamma=0.7):
    """build lr scheduler"""
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    return scheduler
