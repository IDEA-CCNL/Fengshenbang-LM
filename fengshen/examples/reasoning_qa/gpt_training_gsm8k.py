# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : gpt_training_gsm8k.py
#   Last Modified : 2022-05-10 12:43
#   Describe      : 
#
# ====================================================

import os
import time
import json
import random
import jsonlines
import itertools
import argparse
import torch
from transformers import (
        GPT2Config,
        GPT2Tokenizer, 
        GPT2LMHeadModel,
        GPTJForCausalLM,
        GPTJConfig,
        AutoTokenizer, 
        AutoModelForCausalLM,
        )
import pytorch_lightning as pl
from math_data_model import GSMDataModel
from base_trainer import BaseTrainer
from base_model import BaseModel
from gpt_modeling_gsm8k import GPT2ModelForGSM8K
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
    total_parser = argparse.ArgumentParser("Reasoning GPT")
    # * data preprocessing args
    total_parser = GSMDataModel.add_data_specific_args(total_parser)
    # * training args
    total_parser = BaseTrainer.add_trainer_specific_args(total_parser)
    # * model specific args
    total_parser = BaseModel.add_model_specific_args(total_parser)
    # * GPT specific args
    total_parser = GPT2ModelForGSM8K.add_model_specific_args(total_parser)

    args = total_parser.parse_args()
    pl.seed_everything(args.seed)
    # root save directory
    save_dir = args.save_dir

    # create checkpoint directory in root save directory and replace save_dir with it
    model_prefix = f"{os.path.split(args.model_name)[-1]}"
    data_prefix = "GSM"
    timestamp = args.timestamp
    save_dir = os.path.join(save_dir, model_prefix + '-' + data_prefix + '-' + timestamp)
    if args.comment:
        save_dir += '-' + args.comment
    args.save_dir = save_dir

    gpt = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    print(f"Load pretrained model from {args.model_name}...")
    if args.predict and not args.train:
        args.save_dir = args.model_name
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    assert "pad_token" in tokenizer.special_tokens_map
    tokenizer.add_tokens(['[QUES]', '[ANS]', '[THOUGHT]', '[VERIFIER]'])
    if gpt.config.vocab_size < len(tokenizer):
        gpt.resize_token_embeddings(new_num_tokens=len(tokenizer))

    model = GPT2ModelForGSM8K(args, model=gpt, tokenizer=tokenizer)

    torch.cuda.empty_cache()
    print('-' * 30 + 'Args' + '-' * 30)
    for k, v in vars(args).items():
        if v is not None:
            print("\t", k, ":", v)
    print('\n' + '-' * 64)

    gsm_data_model = GSMDataModel(args, tokenizer)

    trainer = BaseTrainer(args, model)
    if args.train:
        # This will create save_dir
        #  tokenizer.save_pretrained(save_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        # save and show args
        with open(os.path.join(save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        # start training
        torch.cuda.empty_cache()
        if args.continue_train_from_ckpt is not None:
            trainer.train(gsm_data_model, ckpt_path=args.continue_train_from_ckpt)
        else:
            trainer.train(gsm_data_model)
        #  trainer.model.save_hf_checkpoint()

    if args.predict:
        torch.cuda.empty_cache()
        if args.generator:
            for i in range(args.num_sample):
                pl.seed_everything(time.time_ns() % 100000)
                trainer.predict(gsm_data_model)
        else:
            results = trainer.predict(gsm_data_model)
            accuracy = save_predictions(results)
            print(f"Rank {model.global_rank} Accuracy: ", accuracy)
            with open(os.path.join(args.save_dir, "accuracy.txt"), "w") as f:
                f.write(str(accuracy))


if __name__ == "__main__":
    main()

