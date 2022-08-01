# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : math_data_model.py
#   Last Modified : 2022-04-13 15:31
#   Describe      : 
#
# ====================================================
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from base_data_model import BaseDataModel
from data_preprocess import DataProcessor
from typing import List, Union, Tuple, Optional, Dict, Callable
from pysnooper import snoop


class Mapper(object):
    """Used for convert examples to features"""
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def __call__(self, example, max_len):
        args = self.args
        tokenizer = self.tokenizer
        pad_token = ['[PAD]']
        pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        eos_token = tokenizer.eos_token
        eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

        question = example['question']
        answer = example['answer']
        target = ''
        
        # # * delete all numbers and operators
        # explain = re.sub(r'(\d|\+|-|\*|/|=|<|>)', '', answer)
        # explain = ' '.join(explain.split())
        # question = question + ' ' + explain
        
        # # * delete all expressions
        # reason = re.sub(r'<<.*>>', '', answer)      
        # reason = re.sub(r'\n####(.*)', '', reason)    
        # reason = '[REA] ' + reason

        # * delete results | expressions | numbers for clear reasoning
        reason = answer
        reason = re.sub(r'\n####(.*)', '[ANS]', reason)
        reason = re.sub(r'<<.*>>', '[EXP]', reason)      
        reason = re.sub('\d+', '[NUM]', reason)
        reason = re.sub(r'\+|-|\*|/|=|<|>', '[OPE]', reason)
        reason = '[REA] ' + reason

        number = re.match(r'(.*)(\n####(.*))', answer, re.S)
        number = number.group(2)
        pattern = re.compile(r'<<.*>>')
        results = pattern.findall(answer)
        expr = ' '.join(results)
        expr = '[EXP] ' + expr + ' ' + number            

        if args.text:
            target = '[ANS] ' + answer
            if args.expr:
                question = question + ' ' + expr
        else:
            if args.reason:
                if args.oracle: 
                    question = question + ' ' + reason
                    target = expr
                else:
                    target = reason + ' ' + expr
            else:
                target = expr
        
        question_tokens = tokenizer.tokenize(question)
            
        target_tokens = tokenizer.tokenize(target)
        input_tokens = question_tokens + target_tokens
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1] * len(input_ids)
        labels = [-100] * (len(question_tokens) - 1) + tokenizer.convert_tokens_to_ids(target_tokens) + [eos_token_id]

        # modified ------------------------------
        pad_length = max_len - len(input_ids)
        pad_tokens = [tokenizer.pad_token_id] * pad_length
        input_ids += pad_tokens
        attention_mask += [0] * pad_length
        labels += [-100] * pad_length
        # ------------------------------
        #  teacher_question = question + ' ' + reason
        #  teacher_question_tokens = tokenizer.tokenize(teacher_question)
        #  teacher_input_tokens = teacher_question_tokens + target_tokens
        #  teacher_input_ids = tokenizer.convert_tokens_to_ids(teacher_input_tokens)
        #  teacher_attn_mask = [1] * len(teacher_input_ids)

        feature = {
            #  **example,
            'question': question,
            'answer': answer,
            #  'target_tokens': target_tokens,
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels),
            #  'teacher_input_ids': teacher_input_ids,
            #  'teacher_attn_mask': teacher_attn_mask,
            #  'teacher_input_tokens': teacher_input_tokens,
            #  'input_tokens': input_tokens,
        }

        return feature


#  ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
ANS_RE = re.compile(r"\[ANS\] (\-?[0-9\.\,]+)\<\|endoftext\|\>")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            match_str = float(match_str)
            match_str = round(match_str, 3)
            match_str = str(match_str)
        except:
            print("matched but not a float", match_str)
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


class Mapper_openai(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.loss_on_prefix = args.loss_on_prefix

    def __call__(self, example, max_len):
        qns = self.tokenizer(example['question'])
        ans = self.tokenizer(example['answer'])
        qn_tokens = qns["input_ids"]
        ans_tokens = ans["input_ids"]
        pad_length = max_len - len(qn_tokens) - len(ans_tokens)
        pad_tokens = [self.tokenizer.pad_token_id] * pad_length
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = [1] * (len(qn_tokens) + len(ans_tokens)) + [0] * pad_length
        tokens = torch.tensor(tokens)
        mask = torch.tensor(mask)
        if self.loss_on_prefix:
            labels = tokens.clone()
        else:
            labels = [-100] * len(qn_tokens) + ans_tokens + [-100] * pad_length
            labels = torch.tensor(labels)
        #  if self.loss_on_prefix:
        #      labels = qn_tokens[1:] + ans_tokens + [self.tokenizer.eos_token_id] + pad_tokens
        #  else:
        #      labels = [-100] * (len(qn_tokens) - 1) + ans_tokens + [self.tokenizer.eos_token_id] + pad_tokens
        #  labels[labels == self.tokenizer.pad_token_id] = -100


        return dict(input_ids=tokens, attention_mask=mask, labels=labels, 
                    #  qn_tokens=qn_tokens, ans_tokens=ans_tokens,
                    question=example['question'], answer=example['answer'])


class GSMDataModel(BaseDataModel):
    def __init__(self, args, tokenizer): 
        super().__init__(args, tokenizer)

    def get_examples(self, path, type):
        examples = DataProcessor._read_jsonl(path)
        for idx, ex in enumerate(examples):
            ex.update(question="[QUES]" + ex["question"] + "\n")
            ex.update(answer="[THOUGHT]" + ex["answer"] + self.tokenizer.eos_token)
            ex.update(answer=ex["answer"].replace("####", "[ANS]"))
            ex.update(question_id=str(idx))

        #  TODO 这个暂时不用，要统一它的prompt成我的形式
        #  if type == "predict" and self.hparams.prompt:
        #      prompt = DataProcessor._read_jsonl("/cognitive_comp/zhuxinyu/datasets/chain-of-thought-prompting/gsm8k/modified_prompt.jsonl")[0]
        #      prompt_text = prompt["prompt"]
        #      for ex in examples:
        #          ex.update(question=prompt_text + ex["question"])

        print(f"{len(examples)} examples")
        return examples

    @staticmethod
    def collate_fn(batch, args, tokenizer):
        bs = len(batch)
        batch_data = {}
        max_len = 0
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]
        mcts_finetune = args.mcts_finetune
        if mcts_finetune:
            verifier_score = []
            is_correct = []
        input_ids = []
        attention_mask = []
        labels = []

        for example in batch:
            qns = tokenizer(example['question'], return_attention_mask=False, max_length=args.source_max_token_len, truncation=True)
            ans = tokenizer(example['answer'], return_attention_mask=False, max_length=args.target_max_token_len, truncation=True)
            qn_tokens = qns["input_ids"]
            ans_tokens = ans["input_ids"]
            input_ids.append(torch.LongTensor(qn_tokens + ans_tokens))
            attention_mask.append(torch.ones_like(input_ids[-1]))
            if args.loss_on_prefix:
                label = input_ids[-1].clone()
                labels.append(label)
            else:
                label = [-100] * len(qn_tokens) + ans_tokens
                labels.append(torch.LongTensor(label))
            if mcts_finetune and 'is_correct' in example:
                is_correct.append(bool(example['is_correct'])) 
                verifier_score.append(float(example['verifier_score']))
                if args.loss_on_prefix and is_correct[-1] == False:
                    labels.pop()
                    label = [-100] * len(qn_tokens) + ans_tokens
                    labels.append(torch.LongTensor(label))

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        if mcts_finetune and 'is_correct' in example:
            batch_data['verifier_score'] = torch.tensor(verifier_score)
            batch_data['is_correct'] = torch.BoolTensor(is_correct)

        return dict(**batch_data, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def predict_dataloader(self):
        return DataLoader(
            self.custom_dataset(self.raw_predict_data, tokenizer=self.tokenizer),
            batch_size=self.hparams.predict_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        

if __name__ == '__main__':
    #  import argparse
    #  import pytorch_lightning as pl
    #  from transformers import GPT2Tokenizer
    #  from base_model import BaseModel
    #  from base_trainer import BaseTrainer
    #  from gpt_modeling_gsm8k import GPT2ModelForGSM8K
    #
    #  total_parser = argparse.ArgumentParser()
    #  # * data preprocessing args
    #  total_parser = GSMDataModel.add_data_specific_args(total_parser)
    #  # * training args
    #  total_parser = BaseTrainer.add_trainer_specific_args(total_parser)
    #  # * model specific args
    #  total_parser = BaseModel.add_model_specific_args(total_parser)
    #  # * GPT specific args
    #  total_parser = GPT2ModelForGSM8K.add_model_specific_args(total_parser)
    #
    #  args = total_parser.parse_args()
    #
    #  tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, use_fast=True)
    #  tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    #  assert "pad_token" in tokenizer.special_tokens_map
    #  tokenizer.add_tokens(['[QUES]', '[ANS]', '[THOUGHT]'])
    #
    #  #  mapper = Mapper_openai(args, tokenizer)
    #  gsm_data_model = GSMDataModel(args, tokenizer)
    #
    #  train_dataloader = gsm_data_model.train_dataloader()
    #  predict_dataloader = gsm_data_model.predict_dataloader()
    #
    #  print(len(gsm_data_model.raw_train_data))
    #
    #  batch = next(iter(train_dataloader))
    #  batch = next(iter(predict_dataloader))
    #  print(batch)
    #

    a = "[THOUGHT] There are 16 x 3 = <<16*3=48>>48 eggs per day.\n [THOUGHT] Janet’s ducks lay 48 eggs per day. She eats 48 - 3 = <<48-3=45>>45 per day.\nThere are 45 x 4 = <<45*4=180>>180 muffin ingredients.\nShe bakes 180 - 45 = <<180-45=135>>135 muffins.\nShe sells 135 - 48 = <<135-48=87>>87 eggs per day at the farmers' market.\nJanet makes 87 x 2 = $<<87*2=174>>174 every day at the farmers' market.\n[ANS] 174<|endoftext|>"
    print(extract_answer(a))
