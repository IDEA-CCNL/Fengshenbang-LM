# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : bert_modeling_base.py
#   Last Modified : 2022-05-29 23:12
#   Describe      : 
#
# ====================================================
import torch
import numpy as np
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForMaskedLM, BertTokenizer
from base_model import BaseModel
from typing import Dict


class BertBaseModel(BaseModel):
    """
    initiates a PyTorch Lightning Bert-like base model, defines basic training and evaluation steps, offer custom train/valid/test step function for specific tasks
    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        pass

    def __init__(self, args, model=None, tokenizer=None):
        super().__init__(args)
        if model is None:
            model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.model = model
        self.tokenizer = tokenizer

    def get_inputs(self, batch):
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids'],
            'labels': batch['labels'],
        }
        return inputs

    def forward(self, input_ids, attenton_mask, token_type_ids, labels=None, **kwargs):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            **kwargs,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        """ training step """
        inputs = self.get_inputs(batch)
        input_ids = inputs["input_ids"]
        batch_size = input_ids.size(0)
        self._consumed_samples += batch_size * max(self.trainer.gpus, 1)  # batch size * data parallel size
        labels = inputs.get("labels", None)
        self._consumed_tokens += len(input_ids.flatten()) * max(self.trainer.gpus, 1)

        loss, logits = self(**inputs)

        #  if self.hparams.show_training_ex > -1 and batch_idx % self.hparams.show_training_ex == 0:
        #      self.show_training_example(input_ids=input_ids[0], labels=labels[0], logits=logits[0])

        self.log("train_loss_step", loss, prog_bar=True, logger=True, on_step=True, batch_size=batch_size)
        ts_logger = self.logger.experiment
        ts_logger.add_scalar("train_loss_vs_samples", loss.item(), self._consumed_samples)
        ts_logger.add_scalar("train_loss_vs_tokens", loss.item(), self._consumed_tokens)

        # Do custom things for your task
        custom_output_dict = self.custom_training_step(batch, batch_idx, logits)

        output_dict = {"loss": loss}
        if custom_output_dict is not None:
            output_dict.update(custom_output_dict)
        #  current_step = self.trainer.lr_schedulers[0]['scheduler']._step_count

        return output_dict

    def validation_step(self, batch, batch_idx):
        """ validation step """
        inputs = self.get_inputs(batch)
        batch_size = inputs["input_ids"].size(0)
        loss, logits = self(**inputs)

        self.log("val_loss", loss, prog_bar=False, logger=True, on_step=True, batch_size=batch_size)

        # Do custom things for your task
        custom_output_dict = self.custom_validation_step(batch, batch_idx, logits)

        output_dict = {"loss": loss}
        if custom_output_dict is not None:
            output_dict.update(custom_output_dict)

        return output_dict

    def validation_epoch_end(self, validation_step_outputs):
        _loss = [x['loss'].cpu() for x in validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(_loss)).item(),
            4,
        )
        self.log("avg_val_loss", self.average_validation_loss, prog_bar=True, logger=True, on_epoch=True)

        ts_logger = self.logger.experiment
        ts_logger.add_scalar("val_loss_vs_samples", self.average_validation_loss, self._consumed_samples)

        self.custom_validation_epoch_end(validation_step_outputs)

    def test_step(self, batch, batch_idx):
        """ test step """
        inputs = self.get_inputs(batch)
        batch_size = inputs["input_ids"].size(0)
        loss, logits = self(**inputs)

        self.log("test_loss", loss, prog_bar=False, logger=True, on_step=True, batch_size=batch_size)

        custom_output_dict = self.custom_test_step(batch, batch_idx, logits)

        output_dict = {"loss": loss}
        if custom_output_dict is not None:
            output_dict.update(custom_output_dict)

        return output_dict

    def test_epoch_end(self, test_step_outputs):
        _loss = [x['loss'].cpu() for x in test_step_outputs]
        self.average_test_loss = np.round(
            torch.mean(torch.stack(_loss)).item(),
            4,
        )
        self.log("avg_test_loss", self.average_test_loss, prog_bar=True, logger=True, on_epoch=True)

        ts_logger = self.logger.experiment
        ts_logger.add_scalar("test_loss_vs_samples", self.average_test_loss, self._consumed_samples)

        self.custom_test_epoch_end(test_step_outputs)

    @classmethod
    def from_pretrained(cls, args) -> pl.LightningModule:
        model = AutoModelForMaskedLM.from_pretrained(args.model_name, return_dict=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        return cls(args, model=model, tokenizer=tokenizer)

    def show_training_example(self, input_ids, labels, logits):
        prediction = torch.argmax(logits, dim=-1)  # (seq_len, vocab_size)
        assert input_ids.size() == labels.size() == prediction.size()  # (seq_len, )
        input_tokens = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        predicted_tokens = self.tokenizer.decode(prediction, skip_special_tokens=True)
        #  predicted_tokens = self.tokenizer.convert_ids_to_tokens(prediction)
        #  if self.tokenizer.eos_token is not None and self.tokenizer.eos_token in predicted_tokens:
        #      predicted_tokens = predicted_tokens[:predicted_tokens.index(self.tokenizer.eos_token)]
        #  predicted_tokens = self.tokenizer.convert_tokens_to_string(predicted_tokens)

        labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)
        labels_tokens = self.tokenizer.decode(labels, skip_special_tokens=True)
        print('-' * 50)
        print('input_token:     ', input_tokens)
        print('-' * 50)
        print('predicted_tokens:', predicted_tokens)
        print('-' * 50)
        print('labels_tokens:   ', labels_tokens)
        print('-' * 50)

    def custom_training_step(self, batch, batch_idx, logits) -> Dict:
        pass

    def custom_training_epoch_end(self, validation_step_outputs):
        pass

    def custom_validation_step(self, batch, batch_idx, logits) -> Dict:
        pass

    def custom_validation_epoch_end(self, validation_step_outputs):
        pass

    def custom_test_step(self, batch, batch_idx, logits) -> Dict:
        pass

    def custom_test_epoch_end(self, test_step_outputs):
        pass

