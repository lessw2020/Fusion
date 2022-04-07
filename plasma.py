# handles training epochs, eval and prediction

from turtle import color
from finetuners.finetuner_builder import FineTunerBase
import torch
from typing import Iterable

import torch.distributed as dist

import torch.nn.functional as F
import tqdm
import finetuners


def train_one_epoch(
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    train_data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    criterion: torch.nn.Module = None,
    profiler=None,
):

    """train one epoch"""
    is_finetuner = False
    iPointer = model  # class instance pointer...for fine tuner we will be shifting 'model' to the internal nn.Model

    if isinstance(model, FineTunerBase):
        is_finetuner = True
        model = iPointer.wrapped_model

    model.train()

    inner_pbar = None

    if rank == 0:
        print(f"--> Starting Epoch {epoch}")
        inner_pbar = tqdm.tqdm(
            range(len(train_data_loader)), colour="blue", desc="r0 Training Epoch"
        )

    fsdp_loss = torch.zeros(2).to(rank)

    # if sampler:
    #   sampler.set_epoch(epoch)

    # TODO - design issue - finetuner class right now handles data prep vs Mnist and similar simply work with direct data
    # this is_finetuner is a temp workaround but needs to be addressed in a more perm fashion
    if is_finetuner:
        print(f"Finetuning epoch")
        for batch_idx, batch in enumerate(train_data_loader):

            optimizer.zero_grad()

            print(batch.keys())

            loss = None

            outputs = iPointer.model_step(batch, rank)

            print(f"outputs_loss = {outputs}")
            if batch_idx > 3:
                break
            # loss = None
            # if labels is not None:
            # loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            # if not return_dict:
            #    output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            #    return ((loss,) + output) if loss is not None else output
            # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            if rank == 0:
                inner_pbar.update(1)
        loss = outputs
        print(f"\n--> Loss from epoch = {loss}")

        return loss

    else:
        for batch_idx, (samples, target) in enumerate(train_data_loader):

            samples, target = samples.to(rank), target.to(rank)

            optimizer.zero_grad()

            output = model(samples)

            if criterion:
                loss = criterion(output, target)
            else:
                loss = F.nll_loss(output, target, reduction="sum")

            loss.backward()
            optimizer.step()

            fsdp_loss[0] += loss.item()
            fsdp_loss[1] += len(samples)

            if profiler:
                profiler.step()

            # reduce op
            dist.reduce(fsdp_loss, 0, op=dist.ReduceOp.SUM)

            if rank == 0:
                inner_pbar.update(1)

    if rank == 0:
        inner_pbar.close()  # final update
        print(
            "Train Epoch: {} \tLoss: {:.6f}".format(epoch, fsdp_loss[0] / fsdp_loss[1])
        )


def val_one_epoch(
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    val_data_loader: Iterable,
    criterion=None,
):
    """validation of model"""
    model.eval()
    if rank == 0:
        print(f"..validation set in process: \n")

    correct = 0
    fsdp_loss = torch.zeros(3).to(rank)

    with torch.no_grad():
        for samples, target in val_data_loader:
            samples = samples.to(rank)
            target = target.to(rank)

            output = model(samples)

            fsdp_loss[0] += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            fsdp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            fsdp_loss[2] += len(samples)
            dist.reduce(fsdp_loss, 0, op=dist.ReduceOp.SUM)

    if rank == 0:
        validation_loss = fsdp_loss[0] / fsdp_loss[2]
        print(
            "Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                validation_loss,
                int(fsdp_loss[1]),
                int(fsdp_loss[2]),
                100.0 * fsdp_loss[1] / fsdp_loss[2],
            )
        )
