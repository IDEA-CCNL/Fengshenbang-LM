# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : bert_verifier_data_model.py
#   Last Modified : 2022-05-30 11:17
#   Describe      : 
#
# ====================================================
import torch
import torch.nn as nn
from base_data_model import BaseDataModel, BaseDataset
from data_preprocess import DataProcessor
from typing import List, Union, Tuple, Optional, Dict, Callable


class BertVerifierDataModel(BaseDataModel):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

    def get_examples(self, path, type):
        examples = DataProcessor._read_jsonl(path)
        print(f"{len(examples)} examples")

        return examples

    @staticmethod
    def collate_fn(batch, args, tokenizer):
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]

        inputs_encoding = tokenizer(batch_data['question'], batch_data['solution'], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True, max_length=512)

        if args.mcts_finetune:
            batch_data['verifier_score'] = torch.tensor(batch_data['verifier_score'])
            batch_data['is_correct'] = torch.BoolTensor(batch_data['is_correct'])

        return dict(**batch_data, **inputs_encoding, verifier_labels=torch.FloatTensor(batch_data['is_correct']))

if __name__ == '__main__':
    import argparse
    import pytorch_lightning as pl
    from transformers import BertTokenizer, AutoModelForMaskedLM
    from base_model import BaseModel
    from base_trainer import BaseTrainer
    from bert_verifier_modeling_gsm8k import BertModelForVerifier
    import transformers
    transformers.logging.set_verbosity_error()

    total_parser = argparse.ArgumentParser()
    # * data preprocessing args
    total_parser = BertVerifierDataModel.add_data_specific_args(total_parser)
    # * training args
    total_parser = BaseTrainer.add_trainer_specific_args(total_parser)
    # * model specific args
    total_parser = BaseModel.add_model_specific_args(total_parser)
    # * Bert specific args
    total_parser = BertModelForVerifier.add_model_specific_args(total_parser)

    args = total_parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.add_tokens(['[QUES]', '[ANS]', '[THOUGHT]', '[VERIFIER]'])

    bert = AutoModelForMaskedLM.from_pretrained(args.model_name)
    if bert.config.vocab_size < len(tokenizer):
        bert.resize_token_embeddings(new_num_tokens=len(tokenizer))
    verifier_head = nn.Linear(1, 1, bias=True)
    model = BertModelForVerifier(args, bert, tokenizer, verifier_head)

    verifier_data_model = BertVerifierDataModel(args, tokenizer)
    train_dataloader = verifier_data_model.train_dataloader()
    #  val_dataloader = verifier_data_model.val_dataloader()
    trainer = BaseTrainer(args, model)
    trainer.train(verifier_data_model)
    #  batch = next(iter(train_dataloader))
    #  model.training_step(batch, 0)
    #  batch = next(iter(train_dataloader))
    #  model.predict_step(batch, 1)
    #  print(batch)

