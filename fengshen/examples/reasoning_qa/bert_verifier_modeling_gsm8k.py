# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : bert_verifier_modeling_gsm8k.py
#   Last Modified : 2022-05-29 23:12
#   Describe      : 
#
# ====================================================
import os
import jsonlines
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from bert_modeling_base import BertBaseModel
from calculator import batch_calculator_sample as sample
from torchsnooper import snoop


class BertModelForVerifier(BertBaseModel):
    """
    initiates a PyTorch Lightning Bert-like base model for training Verifier, defines training and evaluation steps
    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add Bert specific args
        Returns:
            parent_parser
        """
        #  TODO  考虑BERT需不需要像GPT那样继续在语料上MLM的预训练
        parser = parent_parser.add_argument_group('BertModelForVerifier')
        parser.add_argument('--verifier_head', default=None, type=str, help="load a saved verifier head model")
        parser.add_argument('--mcts_finetune', action="store_true", default=False, help="Use samples generated by MCTS for weighted training")
        #  parser.add_argument('--lm_objective', action="store_true", default=False, help="Compute loss on thought tokens")
        #
        return parent_parser

    def __init__(self, args, model=None, tokenizer=None, verifier_head=None):
        super().__init__(args, model, tokenizer)
        self.verifier_head = verifier_head
        self.verifier_idx = self.tokenizer.convert_tokens_to_ids("[VERIFIER]")

    def get_inputs(self, batch):
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids'],
            'verifier_labels': batch['verifier_labels'],
        }
        return inputs

    def forward(self, input_ids, attention_mask, token_type_ids, verifier_labels, labels=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        #  取[CLS]token的预测的[VERIFIER]token的logits
        verifier_logits = output.logits[:, 0, self.verifier_idx]  # Expected shape = (bs, )
        verifier_predictions = self.verifier_head(verifier_logits.unsqueeze(-1)).squeeze(-1)  # Expected shape = (bs, )

        #  取[CLS]token过一个线性层为logits
        #  last_hidden_states = output.hidden_states[-1]  # Expected shape = (bs, seq_len, hidden_size)
        #  cls_hidden_states = last_hidden_states[:, 0]  # Expected shape = (bs, hidden_size)
        #  verifier_predictions = self.verifier_head(cls_hidden_states).squeeze(-1)  # Expected shape = (bs, )

        loss_fct = nn.MSELoss()
        verifier_loss = loss_fct(verifier_predictions.view(-1), verifier_labels.view(-1))
        self.log("verifier_loss", verifier_loss.item(), prog_bar=True, logger=True, on_step=True, batch_size=input_ids.size(0))
        loss = verifier_loss

        return loss, output.logits

    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        batch_size = input_ids.size(0)
        inputs = self.get_inputs(batch)
        del inputs['verifier_labels']
        output = self.model(
            **inputs,
            output_hidden_states=True,
        )
        #  取[CLS]token的预测的[VERIFIER]token的logits
        verifier_logits = output.logits[:, 0, self.verifier_idx]  # Expected shape = (bs, )
        verifier_predictions = self.verifier_head(verifier_logits.unsqueeze(-1)).squeeze(-1)  # Expected shape = (bs, )

        #  取[CLS]token过一个线性层为logits
        #  last_hidden_states = output.hidden_states[-1]  # Expected shape = (bs, seq_len, hidden_size)
        #  cls_hidden_states = last_hidden_states[:, 0]  # Expected shape = (bs, hidden_size)
        #  verifier_predictions = self.verifier_head(cls_hidden_states).squeeze(-1)  # Expected shape = (bs, )

        verifier_file = os.path.join(self.hparams.data_dir, self.hparams.predict_data) + "_verifier_scored_" + str(self.global_rank)

        with jsonlines.open(verifier_file, 'a') as f:
            for idx in range(batch_size):
                f.write({"question": batch['question'][idx], "solution": batch['solution'][idx] ,"verifier_score": str(verifier_predictions[idx].item()),
                    "is_correct": batch['is_correct'][idx], "question_id": batch['question_id'][idx], "ground_truth": batch['ground_truth'][idx]})

    def save_hf_checkpoint(self) -> None:
        #  if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
        """Save huggingface model checkpoint and tokenizer"""
        if self.global_rank == 0:
            save_path = os.path.join(
                self.trainer.checkpoint_callback.dirpath if self.trainer else self.hparams.save_dir,
                'hf_pretrained_epoch{}_step{}'.format(self.current_epoch, self.global_step))
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            torch.save(self.verifier_head, os.path.join(save_path, "verifier_head.pth"))
