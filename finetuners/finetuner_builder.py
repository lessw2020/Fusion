from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

import torch
import time
import numpy as np
import nlp
import textwrap


class FineTunerBase:
    """simple tagging base class to easily check for derived finetuners"""

    pass


class T5Tuner(FineTunerBase):
    def __init__(self, cfg):
        super().__init__()
        self.wrapped_model = T5ForConditionalGeneration.from_pretrained(cfg.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(cfg.model_name)
        # self.rouge_metric = nlp.load_metric("rouge")
        total_params = sum(
            p.numel() for p in self.wrapped_model.parameters() if p.requires_grad
        )

        print(f"wrapped model = {total_params}")

    def model_step(self, rank, batch, forward_only=False):
        # print(f"--> in finetuner model step\n")
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        source_ids = batch["source_ids"].to(rank)
        source_mask = batch["source_mask"].to(rank)
        target_ids = lm_labels.to(rank)
        target_mask = batch["target_mask"].to(rank)

        outputs = self.wrapped_model.forward(
            input_ids=source_ids,
            attention_mask=source_mask,
            labels=target_ids,
            decoder_attention_mask=target_mask,
        )

        loss = outputs.get("loss")
        # logits = outputs.get("logits")
        if not forward_only:
            loss.backward()

        return loss

    def generative_step(self, rank, batch):

        t0 = time.time()

        # move items to device
        source_ids = batch["source_ids"].to(rank)
        source_mask = batch["source_mask"].to(rank)
        target_mask = batch["target_mask"].to(rank)
        target_ids = batch["target_ids"].to(rank)

        generated_ids = self.wrapped_model.generate(
            source_ids,
            attention_mask=source_mask,
            use_cache=True,
            decoder_attention_mask=target_mask,
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )

        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(target_ids)

        gen_time = (time.time() - t0) / batch["source_ids"].shape[0]

        loss = self.model_step(rank, batch, forward_only=True)
        print(f"Val Loss = {loss}")
        base_metrics = {"val_loss": loss}
        summ_len = np.mean(list(map(len, generated_ids)))
        # base_metrics.update(
        print(
            f"val step = gen_time: {gen_time}, gen_len={summ_len}"
        )  # todo -add rouge , preds=preds, target=target
        # )

        return base_metrics

    def predict_step(
        self,
        rank,
        batch,
        cfg,
    ):
        """prediction from test set"""
        self.wrapped_model.eval()

        # move items to device
        source_ids = batch["source_ids"].to(rank)
        source_mask = batch["source_mask"].to(rank)
        target_mask = batch["target_mask"].to(rank)
        target_ids = batch["target_ids"].to(rank)

        outputs = self.wrapped_model.generate(
            source_ids,
            attention_mask=source_mask,
            use_cache=True,
            decoder_attention_mask=target_mask,
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )
        # preds = self.ids_to_clean_text(outputs)
        target = self.ids_to_clean_text(batch["target_ids"])
        clean_text = self.ids_to_clean_text(batch["source_ids"])

        decoded = [self.tokenizer.decode(ids) for ids in outputs]

        texts = [self.tokenizer.decode(ids) for ids in clean_text]
        targets = [self.tokenizer.decode(ids) for ids in target]

        for i in range(cfg.test_batch_size):
            lines = textwrap.wrap("WikiHow Text:\n%s\n" % texts[i], width=100)
            print("\n".join(lines))
            print("\nActual Summary: %s" % targets[i])
            print("\nPredicted Summary: %s" % decoded[i])

            print("------------------------\n")

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return list(map(str.strip, gen_text))


def get_finetuner(cfg):
    if cfg.model_type == "t5":
        return T5Tuner(cfg)
