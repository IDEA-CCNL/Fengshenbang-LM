# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : gpt_training_csqa.py
#   Last Modified : 2022-04-24 21:37
#   Describe      : 
#
# ====================================================

import os
import time
import json
import jsonlines
import argparse
import torch
from transformers import (
        GPT2Config,
        GPT2Tokenizer, 
        GPT2LMHeadModel,
        GPTJForCausalLM,
        GPTJConfig,
        )
import pytorch_lightning as pl
from commonsenseqa_data_model import CommonsenseQADataModel, Mapper
from base_trainer import BaseTrainer
from base_model import BaseModel
from gpt_modeling_csqa import GPT2ModelForCSQA
from pysnooper import snoop
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


#  @snoop()
def main():
    torch.cuda.empty_cache()
    total_parser = argparse.ArgumentParser("Reasoning GPT")
    # * data preprocessing args
    total_parser = CommonsenseQADataModel.add_data_specific_args(total_parser)
    # * training args
    total_parser = BaseTrainer.add_trainer_specific_args(total_parser)
    # * model specific args
    total_parser = BaseModel.add_model_specific_args(total_parser)
    # * GPT specific args
    total_parser = GPT2ModelForCSQA.add_model_specific_args(total_parser)

    args = total_parser.parse_args()
    pl.seed_everything(args.seed)
    # root save directory
    save_dir = args.save_dir

    # create checkpoint directory in root save directory and replace save_dir with it
    model_prefix = f"{os.path.split(args.model_name)[-1]}"
    data_prefix = "CSQA"
    timestamp = args.timestamp
    save_dir = os.path.join(save_dir, model_prefix + '-' + data_prefix + '-' + timestamp)
    args.save_dir = save_dir

    if args.continue_train_from_ckpt is not None:
        if os.path.isdir(args.continue_train_from_ckpt):
            # deepspeed checkpoint is a directory
            ckpt_path = os.path.join(args.continue_train_from_ckpt, "pytorch_model.bin")
        else:
            # ddp checkpoint is a file
            ckpt_path = args.continue_train_from_ckpt

        ckpt_state_dict = torch.load(ckpt_path)
        hf_config_path = os.path.split(args.continue_train_from_ckpt)[0]
        tokenizer = GPT2Tokenizer.from_pretrained(hf_config_path, use_fast=True)
        args.model_name = ckpt_state_dict['hyper_parameters']['model_name']
        if args.test and not args.train:
            args.save_dir = ckpt_state_dict['hyper_parameters']['save_dir']
        #TODO gpt2 -> gpt-j ------------------------------
        #  config = GPTJConfig.from_pretrained(hf_config_path)
        #  gpt = GPTJForCausalLM(config)
        #--------------------------------
        config = GPT2Config.from_pretrained(hf_config_path)
        gpt = GPT2LMHeadModel(config)
        model = GPT2ModelForCSQA(args, model=gpt, tokenizer=tokenizer)
        model.load_state_dict(ckpt_state_dict['state_dict'])
        print(f"Load saved checkpoint from {args.continue_train_from_ckpt} and continue training...")
        #  model = GPT2ModelForCSQA.load_from_checkpoint(args.continue_train_from_ckpt, monitor=args.monitor, save_dir=args.save_dir)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, use_fast=True)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.add_tokens(['[QUES]', '[ANS]', '[CHOICE]'])

        #TODO gpt2 -> gpt-j ------------------------------
        gpt = GPT2LMHeadModel.from_pretrained(args.model_name)
        gpt.resize_token_embeddings(new_num_tokens=len(tokenizer))
        #  gpt = GPTJForCausalLM.from_pretrained(args.model_name)
        model = GPT2ModelForCSQA(args, model=gpt, tokenizer=tokenizer)
        #-----------------------------------
    torch.cuda.empty_cache()
    print('-' * 30 + 'Args' + '-' * 30)
    for k, v in vars(args).items():
        if v is not None:
            print("\t", k, ":", v)
    print('\n' + '-' * 64)

    #  prepare mapper and data model
    mapper = Mapper(args=args, tokenizer=tokenizer)
    csqa_data_model = CommonsenseQADataModel(args, tokenizer, mapper)

    trainer = BaseTrainer(args, model)
    if args.train:
        # This will create save_dir
        tokenizer.save_pretrained(save_dir)
        # save and show args
        with open(os.path.join(save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        # start training
        torch.cuda.empty_cache()
        if args.continue_train_from_ckpt is not None:
            trainer.train(csqa_data_model, ckpt_path=args.continue_train_from_ckpt)
        else:
            trainer.train(csqa_data_model)
        trainer.model.model.save_pretrained(save_dir)
    if args.test:
        # clear origin file content
        with jsonlines.open(os.path.join(args.save_dir, "model_solution.jsonl"), 'w') as f:
            pass
        torch.cuda.empty_cache()
        trainer.test(csqa_data_model)


if __name__ == "__main__":
    main()

