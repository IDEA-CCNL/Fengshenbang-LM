# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : verifier_modeling_gsm8k.py
#   Last Modified : 2022-05-12 17:03
#   Describe      : 
#
# ====================================================
import os
import jsonlines
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from gpt_modeling_base import GPT2BaseModel
from calculator import batch_calculator_sample as sample
from torchsnooper import snoop


class GPT2ModelForVerifier(GPT2BaseModel):
    """
    initiates a PyTorch Lightning GPT2 base model for training Verifier, defines training and evaluation steps
    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add GPT specific args
        Returns:
            parent_parser
        """
        parser = parent_parser.add_argument_group('GPT2ModelForVerifier')
        parser.add_argument('--lm_objective', action="store_true", default=False, help="Compute loss on thought tokens")
        parser.add_argument('--verifier_head', default=None, type=str, help="load a saved verifier head model")

        return parent_parser

    def __init__(self, args, model=None, tokenizer=None, verifier_head=None):
        super().__init__(args, model, tokenizer)
        self.verifier_head = verifier_head
        self.verifier_idx = self.tokenizer.convert_tokens_to_ids("[VERIFIER]")

    #  TODO 这里的loss，如果on prefix，就是answer和thought的每个token的logits都要取最后一个special token的logit去过一个回归头
    #  如果no prefix就是只在最后一个token上去取[VERIFIER]的logit进行gain+bias

    def get_inputs(self, batch):
        inputs = {
            'input_ids': batch['qn_ans_input_ids'],
            'attention_mask': batch['qn_ans_mask'],
            'labels': batch['labels'],
            'qn_sol_input_ids': batch['qn_sol_input_ids'],
            'qn_sol_mask': batch['qn_sol_mask'],
            'verifier_labels': batch['verifier_labels'],
            'verifier_loss_mask': batch['verifier_loss_mask'],
        }
        return inputs

    def forward(self, input_ids, attention_mask, qn_sol_input_ids, qn_sol_mask, verifier_labels, verifier_loss_mask, labels=None):
        """ forward step """
        lm_output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        output = self.model(
            qn_sol_input_ids,
            attention_mask=qn_sol_mask,
        )
        # TODO 不知道为什么GPTJ会出现有的tensor是float16有的是32
        verifier_logits = output.logits[:, :, self.verifier_idx].half()  # Expected shape = (bs, seq_len)

        verifier_predictions = self.verifier_head(verifier_logits.unsqueeze(-1)).squeeze(-1)  # Expected shape = (bs, seq_len)
        verifier_predictions = verifier_predictions * verifier_loss_mask

        verifier_labels = verifier_labels * verifier_loss_mask  # shape = (bs, seq_len)

        loss_fct = nn.MSELoss()
        verifier_loss = loss_fct(verifier_predictions.view(-1), verifier_labels.view(-1))
        self.log("verifier_loss", verifier_loss.item(), prog_bar=True, logger=True, on_step=True, batch_size=input_ids.size(0))
        loss = verifier_loss

        if lm_output.loss:
            loss += lm_output.loss
            self.log("lm_loss", lm_output.loss.item(), prog_bar=True, logger=True, on_step=True, batch_size=input_ids.size(0))

        return loss, lm_output.logits

    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        final_token_idx = batch['final_token_idx']
        batch_size = input_ids.size(0)
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
        )
        verifier_logits = output.logits[:, :, self.verifier_idx].half()  # Expected shape = (bs, seq_len)
        verifier_logits = torch.gather(verifier_logits, 1, final_token_idx)  # Expected shape = (bs, 1)

        verifier_predictions = self.verifier_head(verifier_logits).squeeze(-1)  # Expected shape = (bs, )

        verifier_file = os.path.join(self.hparams.data_dir, self.hparams.predict_data) + "_verifier_scored_" + str(self.global_rank)
        del batch['input_ids']
        del batch['attention_mask']
        del batch['final_token_idx']

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

