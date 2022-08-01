# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : verifier_training_gsm8k.py
#   Last Modified : 2022-05-09 23:22
#   Describe      : 
#
# ====================================================

import os
import time
import json
import jsonlines
import itertools
import argparse
import torch
import torch.nn as nn
from transformers import (
        GPT2Tokenizer,
        AutoTokenizer, 
        AutoModelForCausalLM,
        BertTokenizer,
        AutoModelForMaskedLM,
        DebertaV2Tokenizer, 
        DebertaV2ForMaskedLM,
        )
import pytorch_lightning as pl
from verifier_data_model import GPT2VerifierDataModel, Mapper, VerifierPredictDataModel
from bert_verifier_data_model import BertVerifierDataModel
from base_trainer import BaseTrainer
from base_model import BaseModel
from base_data_model import BaseDataModel
from verifier_modeling_gsm8k import GPT2ModelForVerifier
from bert_verifier_modeling_gsm8k import BertModelForVerifier
from pysnooper import snoop
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def save_predictions(results):
    question = list(itertools.chain.from_iterable([x['question'] for x in results]))
    answer = list(itertools.chain.from_iterable([x['answer'] for x in results]))
    is_correct = list(itertools.chain.from_iterable([x['is_correct'] for x in results]))
    solutions = list(itertools.chain.from_iterable([x['solutions'] for x in results]))

    accuracy = sum(is_correct) / len(is_correct)

    return accuracy

    #  solution_file = "model_solution.jsonl" + str(torch.distributed.get_rank())
    #  if args.prompt:
    #      solution_file = "prompt_" + solution_file
    #  with jsonlines.open(os.path.join(args.save_dir, timestamp + '-' + solution_file), 'w') as f:
    #      for q, a, s, i in zip(question, answer, solutions, is_correct):
    #          f.write({"question": q, "ground_truth": a,
    #                  "solution": s, "is_correct": bool(i)})


#  @snoop()
def main():
    torch.cuda.empty_cache()
    total_parser = argparse.ArgumentParser("Verifier Model")
    # * data preprocessing args
    total_parser = BaseDataModel.add_data_specific_args(total_parser)
    # * training args
    total_parser = BaseTrainer.add_trainer_specific_args(total_parser)
    # * model specific args
    total_parser = BaseModel.add_model_specific_args(total_parser)
    #  args = total_parser.parse_args()
    args, argv = total_parser.parse_known_args()
    if args.model_type == "gpt":
        # * GPT specific args
        total_parser = GPT2ModelForVerifier.add_model_specific_args(total_parser)
    elif args.model_type == "bert" or args.model_type == "deberta":
        # * Bert specific args
        total_parser = BertModelForVerifier.add_model_specific_args(total_parser)
    else:
        raise ValueError()

    args = total_parser.parse_args()
    pl.seed_everything(args.seed)
    # root save directory
    save_dir = args.save_dir

    # create checkpoint directory in root save directory and replace save_dir with it
    model_prefix = f"{os.path.split(args.model_name)[-1]}"
    data_prefix = "Verifier-GSM"
    timestamp = args.timestamp
    save_dir = os.path.join(save_dir, model_prefix + '-' + data_prefix + '-' + timestamp)
    args.save_dir = save_dir

    if args.model_type == "gpt":
        hf_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, use_fast=True)
    elif args.model_type == "bert":
        hf_model = AutoModelForMaskedLM.from_pretrained(args.model_name)
        tokenizer = BertTokenizer.from_pretrained(args.model_name, use_fast=True)
    elif args.model_type == "deberta":
        hf_model = DebertaV2ForMaskedLM.from_pretrained(args.model_name)
        tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_name, use_fast=True)

    #  tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, use_fast=True)
    print(f"Load pretrained model from {args.model_name}...")
    if args.predict and not args.train:
        args.save_dir = args.model_name
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    assert "pad_token" in tokenizer.special_tokens_map
    tokenizer.add_tokens(['[QUES]', '[ANS]', '[THOUGHT]', '[VERIFIER]'])


    if hf_model.config.vocab_size < len(tokenizer):
        hf_model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    if args.verifier_head is not None:
        verifier_head = torch.load(args.verifier_head)
    else:
        verifier_head = nn.Linear(1, 1, bias=True)
    # TODO 单独保存和加载linear的state dict
    #  if args.continue_train_from_ckpt is not None:
    #      ckpt_state_dict = torch.load(args.continue_train_from_ckpt)
    #      verifier_head.weight.data = ckpt_state_dict['state_dict']['verifier_head.weight']
    #      verifier_head.bias.data = ckpt_state_dict['state_dict']['verifier_head.bias']
    #      del ckpt_state_dict
    # -------------------------------------------
    if args.model_type == "gpt":
        model = GPT2ModelForVerifier(args, model=hf_model, tokenizer=tokenizer, verifier_head=verifier_head)
        verifier_data_model_class = GPT2VerifierDataModel
        #  TODO 统一predict和训练的datamodel
        verifier_test_data_model_class = VerifierPredictDataModel
    elif args.model_type == "bert" or args.model_type == "deberta":
        model = BertModelForVerifier(args, model=hf_model, tokenizer=tokenizer, verifier_head=verifier_head)
        verifier_data_model_class = BertVerifierDataModel
        verifier_test_data_model_class = BertVerifierDataModel

    torch.cuda.empty_cache()
    print('-' * 30 + 'Args' + '-' * 30)
    for k, v in vars(args).items():
        if v is not None:
            print("\t", k, ":", v)
    print('\n' + '-' * 64)

    #  prepare mapper and data model
    #  mapper = Mapper(args=args, tokenizer=tokenizer)

    trainer = BaseTrainer(args, model)
    if args.train:
        verifier_data_model = verifier_data_model_class(args, tokenizer)
        # This will create save_dir
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        # save and show args
        with open(os.path.join(save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        # start training
        torch.cuda.empty_cache()
        if args.continue_train_from_ckpt is not None:
            trainer.train(verifier_data_model, ckpt_path=args.continue_train_from_ckpt)
        else:
            trainer.train(verifier_data_model)
    elif args.predict:
        torch.cuda.empty_cache()
        verifier_test_data_model = verifier_test_data_model_class(args, tokenizer)
        trainer.predict(verifier_test_data_model)


    #  if args.predict:
    #      torch.cuda.empty_cache()
    #      results = trainer.predict(verifier_data_model)
        #  accuracy = save_predictions(results)
        #  print(f"Rank {model.global_rank} Accuracy: ", accuracy)


if __name__ == "__main__":
    import transformers
    transformers.logging.set_verbosity_error()
    main()

