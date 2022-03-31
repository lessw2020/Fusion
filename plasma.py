# handles training epochs, eval and prediction

import torch
from typing import Iterable

import torch.distributed as dist



def train_one_epoch(˙
    rank: int,
    world_size: int,
    model: torch.nn.module,
    criterion: torch.nn.module,
    train_data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    profiler=None,
):

    """train one epoch"""

    model.train()

    fsdp_loss = torch.zeros(2).to(rank)

   #if sampler:
    #   sampler.set_epoch(epoch)

   for batch_idx, (samples, target) in enumerate(train_data_loader):

       samples, target = data.to(rank), target.to(rank)
       
       optimizer.zero_grad()

       output = model(samples)
       if criterion:
           loss = criterion(output, target)
        else:
            loss = F.nll_loss(output, target, reduction='sum')

       loss.backward()
       optimizer.step()

       fsdp_loss[0] += loss.item()
       fsdp_loss[1] += len(data)

       if profiler:
           profiler.step()
    
    # reduce op
    dist.reduce(fsdp_loss, 0, op = dist.ReduceOp.SUM)

    if rank == 0:
       print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, fsdp_loss[0] / fsdp_loss[1]))
    


def val_one_epoch(rank: int, world_size: int, model: torch.nn.Module, val_data_loader: Iterable, criterion=None):
    """ validation of model """
    model.eval()
    correct = 0
    fsdp_loss = torch.zeros(3).to(rank)

    with torch.no_grad():
        for samples, target in val_data_loader:
            samples = samples.to(rank)
            target = target.to(rank)

            output = model(samples)

            fsdp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            fsdp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            fsdp_loss[2] += len(samples)
            dist.reduce(fsdp_loss, 0, op=dist.ReduceOp.SUM)

    if rank == 0:
       test_loss = fsdp_loss[0] / fsdp_loss[2]
       print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
           test_loss, int(fsdp_loss[1]), int(fsdp_loss[2]),
           100. * fsdp_loss[1] / fsdp_loss[2]))




