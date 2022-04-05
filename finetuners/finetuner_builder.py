from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

import torch 


class T5Tuner(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(cfg.model_name)
    
    def listed_map(self, f, x):
        return list(map(f,x)))
    
    




def get_finetuner(cfg):
