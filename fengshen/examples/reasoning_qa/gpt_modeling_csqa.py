# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : gpt_modeling_csqa.py
#   Last Modified : 2022-04-24 21:37
#   Describe      : 
#
# ====================================================
import os
import copy
import jsonlines
import itertools
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import pytorch_lightning as pl
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from base_model import BaseModel
from commonsenseqa_data_model import extract_answer, is_correct, ANS_RE, INVALID_ANS
from gpt_modeling_base import GPT2BaseModel
from pysnooper import snoop


class GPT2ModelForCSQA(GPT2BaseModel):
    """
    initiates a PyTorch Lightning GPT2 model for training on commonsense qa, defines training and evaluation steps
    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add GPT specific args
        Returns:
            parent_parser
        """
        parser = parent_parser.add_argument_group('GPT2ModelForCSQA')
        parser.add_argument('--loss_on_prefix', action="store_true", default=False, help="Compute loss on question and answer choice tokens")

        return parent_parser

    def __init__(self, args, model=None, tokenizer=None):
        super().__init__(args, model, tokenizer)

    def custom_training_step(self, batch, batch_idx, logits):
        """ custom training step """
        pass

    def custom_validation_step(self, batch, batch_idx, logits):
        """ custom validation step """
        answer = batch['answer']
        context = copy.deepcopy(batch['input_text'])
        for i in range(len(context)):
            ans_start = context[i].find("[ANS]")
            context[i] = context[i][:ans_start + 5]  #  去掉[ANS]后面所有的文本让模型生成

        self.tokenizer.padding_side = "left"
        inputs_encoding = self.tokenizer(
            context, 
            return_attention_mask=True,
            return_tensors="pt", 
            add_special_tokens=False,
            padding=True,
        )
        self.tokenizer.padding_side = "right"
        for key, val in inputs_encoding.items():
            inputs_encoding[key] = val.to(self.device)
        generated_ids = self.model.generate(
            **inputs_encoding,
            max_length=160,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # if num_return_sequences>1, then batch_decode returns batch_size * num_return_sequences results
        predicted_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        num_correct = 0
        num_total = len(answer)
        for pred, gt in zip(predicted_tokens, answer):
            num_correct += int(is_correct(pred, gt))

        if self.hparams.show_training_ex > -1 and batch_idx % self.hparams.show_training_ex == 0:
            ans_start = pred.find("[ANS]")
            pred = pred[ans_start:]
            print("prediction: ", pred)
            print("answer: ", gt)
            print("answers choices: ", batch["answers_choices"][-1])
            print("\n")
        
        return {"num_correct": num_correct, "num_total": num_total}

    def custom_validation_epoch_end(self, validation_step_outputs):
        _num_correct = sum([x['num_correct'] for x in validation_step_outputs])
        _num_total = sum([x['num_total'] for x in validation_step_outputs])
        accuracy = _num_correct / _num_total
        print("Accuracy: ", accuracy)

        ts_logger = self.logger.experiment
        ts_logger.add_scalar("val_accuracy_vs_samples", accuracy, self._consumed_samples)
        self.log("val_accuracy_epoch", accuracy, prog_bar=True, logger=True, on_epoch=True)

    def custom_test_step(self, batch, batch_idx, logits):
        answer = batch['answer']
        context = copy.deepcopy(batch['input_text'])
        for i in range(len(context)):
            ans_start = context[i].find("[ANS]")
            context[i] = context[i][:ans_start + 5]  #  去掉[ANS]后面所有的文本让模型生成

        self.tokenizer.padding_side = "left"
        inputs_encoding = self.tokenizer(
            context, 
            return_attention_mask=True,
            return_tensors="pt", 
            add_special_tokens=False,
            padding=True,
        )
        self.tokenizer.padding_side = "right"
        inputs_encoding = inputs_encoding.to(self.device)
        #  for key, val in inputs_encoding.items():
        #      inputs_encoding[key] = val.to(self.device)
        generated_ids = self.model.generate(
            **inputs_encoding,
            max_length=160,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # if num_return_sequences>1, then batch_decode returns batch_size * num_return_sequences results
        predicted_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        num_correct = 0
        num_total = len(answer)
        for pred, gt in zip(predicted_tokens, answer):
            num_correct += int(is_correct(pred, gt))

        #  ans_start = pred.find("[ANS]")
        #  pred = pred[ans_start:]
        #  print("prediction: ", pred)
        #  print("answer: ", gt)
        #  print("answers choices: ", batch["answers_choices"][-1])
        #  print("\n")
        
        return {"num_correct": num_correct, "num_total": num_total}

    def custom_test_epoch_end(self, test_step_outputs):
        _num_correct = sum([x['num_correct'] for x in test_step_outputs])
        _num_total = sum([x['num_total'] for x in test_step_outputs])
        accuracy = _num_correct / _num_total
        print("Accuracy: ", accuracy)

        ts_logger = self.logger.experiment
        ts_logger.add_scalar("test_accuracy_vs_samples", accuracy, self._consumed_samples)

