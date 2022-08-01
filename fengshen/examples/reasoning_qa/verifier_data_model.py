# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : verifier_data_model.py
#   Last Modified : 2022-05-12 11:50
#   Describe      : 
#
# ====================================================
import torch
import torch.nn as nn
from base_data_model import BaseDataModel, BaseDataset
from data_preprocess import DataProcessor
from typing import List, Union, Tuple, Optional, Dict, Callable
from torchsnooper import snoop as tsnoop


class Mapper(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.lm_objective = args.lm_objective

    #  def helper(self, example, max_len, key1, key2):
    #      qns = self.tokenizer(example['question'])
    #      sol = self.tokenizer(example['solution'])
    #      qn_tokens = qns["input_ids"]
    #      sol_tokens = sol["input_ids"]
    #      pad_length = max_len - len(qn_tokens) - len(sol_tokens)
    #      pad_tokens = [self.tokenizer.pad_token_id] * pad_length
    #      tokens = qn_tokens + sol_tokens + pad_tokens
    #      mask = [1] * (len(qn_tokens) + len(sol_tokens)) + [0] * pad_length
    #      tokens = torch.tensor(tokens)
    #      mask = torch.tensor(mask)

    def __call__(self, example, ans_max_len=None, sol_max_len=None):
        qns = self.tokenizer(example['question'])
        sol = self.tokenizer(example['solution'])
        ans = self.tokenizer(example['ground_truth'])

        qn_tokens = qns["input_ids"]
        sol_tokens = sol["input_ids"]
        ans_tokens = ans["input_ids"]

        sol_max_len = len(qn_tokens) + len(sol_tokens)
        ans_max_len = len(qn_tokens) + len(ans_tokens)

        sol_pad_length = sol_max_len - len(qn_tokens) - len(sol_tokens)
        sol_pad_tokens = [self.tokenizer.pad_token_id] * sol_pad_length
        ans_pad_length = ans_max_len - len(qn_tokens) - len(ans_tokens)
        ans_pad_tokens = [self.tokenizer.pad_token_id] * ans_pad_length

        qn_sol_tokens = qn_tokens + sol_tokens + sol_pad_tokens
        qn_sol_input_ids = torch.tensor(qn_sol_tokens)
        qn_ans_tokens = qn_tokens + ans_tokens + ans_pad_tokens
        qn_ans_input_ids = torch.tensor(qn_ans_tokens)

        qn_sol_mask = torch.tensor([1] * (len(qn_tokens) + len(sol_tokens)) + [0] * sol_pad_length)
        qn_ans_mask = torch.tensor([1] * (len(qn_tokens) + len(ans_tokens)) + [0] * ans_pad_length)

        if self.lm_objective:
            labels = [-100] * len(qn_tokens) + ans_tokens + [-100] * ans_pad_length
            labels = torch.tensor(labels)
        else:
            labels = None

        verifier_labels = torch.ones_like(qn_sol_input_ids) * float(example['is_correct'])
        verifier_loss_mask = [0] * len(qn_tokens) + [1] * len(sol_tokens) + [0] * sol_pad_length
        verifier_loss_mask = torch.tensor(verifier_loss_mask)

        return dict(qn_ans_input_ids=qn_ans_input_ids, qn_ans_mask=qn_ans_mask, labels=labels, 
                    qn_sol_input_ids=qn_sol_input_ids, qn_sol_mask=qn_sol_mask, verifier_labels=verifier_labels,
                    verifier_loss_mask=verifier_loss_mask,
                    question=example['question'], solution=example['solution'],
                    )


#  class VerifierDataset(BaseDataset):
#      def __init__(self, data, tokenizer):
#          super().__init__(data, tokenizer)
#          #  self.tokenized_qns = [self.tokenizer.tokenize(ex["question"]) for ex in self.data]
#          #  self.tokenized_ans = [self.tokenizer.tokenize(ex["ground_truth"]) for ex in self.data]
#          #  self.tokenized_sol = [self.tokenizer.tokenize(ex["solution"]) for ex in self.data]
#          #  self.ans_max_len = max(
#          #      [
#          #          len(self.tokenized_qns[i]) + len(self.tokenized_ans[i])
#          #          for i in range(len(self.data))
#          #      ]
#          #  )
#          #  self.sol_max_len = max(
#          #      [
#          #          len(self.tokenized_qns[i]) + len(self.tokenized_sol[i])
#          #          for i in range(len(self.data))
#          #      ]
#          #  )
#          #  print(f"Max ans tokens length: {self.ans_max_len}")
#          #  print(f"Max sol tokens length: {self.sol_max_len}")
#
#      def __getitem__(self, index):
#          #  return self.mapper(self.data[index], ans_max_len=self.ans_max_len, sol_max_len=self.sol_max_len)
#          return self.data[index]


class GPT2VerifierDataModel(BaseDataModel):
    def __init__(self, args, tokenizer, custom_dataset=BaseDataset):
        super().__init__(args, tokenizer, custom_dataset)

    def get_examples(self, path, type):
        examples = DataProcessor._read_jsonl(path)
        print(f"{len(examples)} examples")

        return examples

    @staticmethod
    def collate_fn(batch, args, tokenizer):
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]

        qn_sol_input_ids = []
        qn_ans_input_ids = []
        qn_sol_mask = []
        qn_ans_mask = []
        labels = []
        verifier_labels = []
        verifier_loss_mask = []
        final_token_idx = []
        for example in batch:
            qns = tokenizer(example['question'])
            sol = tokenizer(example['solution'])
            ans = tokenizer(example['ground_truth'])

            qn_sol_input_ids.append(torch.LongTensor(qns.input_ids + sol.input_ids))
            qn_ans_input_ids.append(torch.LongTensor(qns.input_ids + ans.input_ids))
            qn_sol_mask.append(torch.ones_like(qn_sol_input_ids[-1]))
            qn_ans_mask.append(torch.ones_like(qn_ans_input_ids[-1]))

            final_token_idx.append(len(qn_sol_input_ids[-1]) - 1)

            if args.lm_objective:
                label = torch.LongTensor([-100] * len(qns.input_ids) + ans.input_ids)
                labels.append(label)
            else:
                labels = None

            verifier_label = torch.ones_like(qn_sol_input_ids[-1]) * float(example['is_correct'])
            verifier_labels.append(verifier_label)
            verifier_mask = [0] * len(qns.input_ids) + [1] * len(sol.input_ids) # + [0] * sol_pad_length
            verifier_loss_mask.append(torch.LongTensor(verifier_mask))
        qn_sol_input_ids = nn.utils.rnn.pad_sequence(qn_sol_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        qn_ans_input_ids = nn.utils.rnn.pad_sequence(qn_ans_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        qn_sol_mask = nn.utils.rnn.pad_sequence(qn_sol_mask, batch_first=True, padding_value=0)
        qn_ans_mask = nn.utils.rnn.pad_sequence(qn_ans_mask, batch_first=True, padding_value=0)
        verifier_labels = nn.utils.rnn.pad_sequence(verifier_labels, batch_first=True, padding_value=-100)
        verifier_loss_mask = nn.utils.rnn.pad_sequence(verifier_loss_mask, batch_first=True, padding_value=0)
        final_token_idx = torch.LongTensor(final_token_idx).view(-1, 1)
        if labels:
            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return dict(**batch_data,
                    qn_ans_input_ids=qn_ans_input_ids, qn_ans_mask=qn_ans_mask, labels=labels,
                    qn_sol_input_ids=qn_sol_input_ids, qn_sol_mask=qn_sol_mask, verifier_labels=verifier_labels,
                    verifier_loss_mask=verifier_loss_mask,
                    )


class VerifierPredictDataModel(BaseDataModel):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

    def get_examples(self, path, type):
        examples = DataProcessor._read_jsonl(path)
        print(f"{len(examples)} examples")

        return examples

    @staticmethod
    def collate_fn(batch, args, tokenizer):
        bs = len(batch)
        batch_data = {}
        max_len = 0
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]

        input_ids = []
        attention_mask = []
        final_token_idx = []

        for example in batch:
            qns = tokenizer(example['question'], return_attention_mask=False)
            sol = tokenizer(example['solution'], return_attention_mask=False)
            qn_tokens = qns["input_ids"]
            sol_tokens = sol["input_ids"]

            input_ids.append(torch.LongTensor(qn_tokens + sol_tokens))
            attention_mask.append(torch.ones_like(input_ids[-1]))
            final_token_idx.append(len(qn_tokens + sol_tokens) - 1)

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        final_token_idx = torch.LongTensor(final_token_idx).view(-1, 1)

        return dict(**batch_data, input_ids=input_ids, attention_mask=attention_mask, final_token_idx=final_token_idx)


if __name__ == '__main__':
    import argparse
    import pytorch_lightning as pl
    from transformers import GPT2Tokenizer
    from base_model import BaseModel
    from base_trainer import BaseTrainer
    from verifier_modeling_gsm8k import GPT2ModelForVerifier

    total_parser = argparse.ArgumentParser()
    # * data preprocessing args
    total_parser = BaseDataModel.add_data_specific_args(total_parser)
    # * training args
    total_parser = BaseTrainer.add_trainer_specific_args(total_parser)
    # * model specific args
    total_parser = BaseModel.add_model_specific_args(total_parser)
    # * GPT specific args
    total_parser = GPT2ModelForVerifier.add_model_specific_args(total_parser)

    args = total_parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    assert "pad_token" in tokenizer.special_tokens_map
    tokenizer.add_tokens(['[QUES]', '[ANS]', '[THOUGHT]', '[VERIFIER]'])

    mapper = Mapper(args=args, tokenizer=tokenizer)
    #  verifier_data_model = VerifierDataModel(args, tokenizer, mapper)
    verifier_data_model = GPT2VerifierDataModel(args, tokenizer)
    train_dataloader = verifier_data_model.train_dataloader()
    #  val_dataloader = verifier_data_model.val_dataloader()

    #  print(len(verifier_data_model.raw_train_data))

    batch = next(iter(train_dataloader))
    #  batch = next(iter(val_dataloader))

    keys = {
        'qn_ans_input_ids',
        'qn_ans_mask',
        'labels',
        'qn_sol_input_ids',
        'qn_sol_mask',
        'verifier_labels',
        'verifier_loss_mask',
    }
    for ex in verifier_data_model.raw_train_data[:10]:
        collate = verifier_data_model.collate_fn([ex], args, tokenizer)
        mapped = mapper(ex)
        #  print("collate:", verifier_data_model.collate_fn([ex], args, tokenizer))
        #  print("mapper: ", mapper(ex))
        for k in keys:
            #  print(k, collate[k].size())
            #  print(k, mapped[k].size())
            assert collate[k].size(0) == 1
            assert all(collate[k][0] == mapped[k])

