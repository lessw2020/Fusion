from finetuners.finetuner_builder import FineTunerBase
import torch
import torch.optim as optim
import finetuners


def build_optimizer(model, lr=0.003, verbose=True):
    built_optimizer = None

    if isinstance(model, FineTunerBase):
        built_optimizer = optim.AdamW(model.wrapped_model.parameters(), lr=lr)
        if verbose:
            print(f"Optimizer built for FineTuner instance")
    else:

        built_optimizer = optim.AdamW(model.parameters(), lr=lr)
        if verbose:
            print(f"optimizer built for nn.Module class")

    if built_optimizer is None:
        RaiseValueError("unable to build optimizer!")

    return built_optimizer
