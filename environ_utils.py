import os
import torch
import psutil


def calc_flop(cfg):
    """todo: this pulls layer, model and vocab size to calc flops"""
    B = args.batch_size
    s = args.block_size
    conf = get_model_config(cfg)
    l = conf.n_layer
    h = conf.n_embd
    V = conf.vocab_size
    return 96 * B * s * l * h * h * (1 + s / 6 / h + V / 16 / l / h)
