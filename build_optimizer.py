import torch
import torch.optim as optim


def build_optimizer(model, lr=0.003):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    return optimizer
