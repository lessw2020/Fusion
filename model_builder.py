from environ_utils import *
import torch.distributed as dist
import datetime

from model_class import Model_Class


def create_model(cfg):
    rank = get_rank()
    world_size = get_world_size()

    # setup has configured all processes to map to exclusive devices.
    for item in range(torch.cuda.device_count()):
        torch.cuda.synchronize(item)

    currFlop = calc_flop(cfg)

    if is_rank_0:
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
        on_trace_ready=my_tensorboard_trace_handler(
            f"tb/{now.strftime('%Y_%m_%d_%H_%M_%S')}", rank, use_gzip=True
        ),
    ) if cfg.profile else contextlib.nullcontext() as prof:

        now = datetime.now()

        rank_model = build_model_core(rank)

    if cfg.profile:
        init_end_event.record()
    end = datetime.now()

    sync_all_devices()
    dist.barrier()

    if is_rank_0:
        print(f"Model built: \n {model}")

    print_memory_summary("After model init", "cuda:0")
    return rank_model


def build_model_core(rank):

    currModel = Model_Class().to(rank)

    return currModel
