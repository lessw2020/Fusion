from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

import torch
import time
import numpy as np


class FineTunerBase:
    """simple tagging base class to easily check for derived finetuners"""

    pass


class T5Tuner(torch.nn.Module, FineTunerBase):
    def __init__(self, cfg):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(cfg.model_name)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        lm_labels=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
        )

    def model_step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
        )

        loss = outputs[0]

        return loss

    def generative_step(self, batch):

        t0 = time.time()

        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch["target_mask"],
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])

        gen_time = (time.time() - t0) / batch["source_ids"].shape[0]

        loss = self.model_step(batch)
        base_metrics = {"val_loss": loss}
        summ_len = np.mean(list(map(len, generated_ids)))
        base_metrics.update(
            gen_time=gen_time, gen_len=summ_len, preds=preds, target=target
        )

        return base_metrics

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return list(map(str.strip, gen_text))


def get_finetuner(cfg):
    if cfg.model_type == "t5":
        return T5Tuner(cfg)
