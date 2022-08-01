# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : commonsenseqa_data_model.py
#   Last Modified : 2022-04-24 20:24
#   Describe      : 
#
# ====================================================
import re
import copy
import torch
from base_data_model import BaseDataModel, BaseDataset
from data_preprocess import DataProcessor
from typing import List, Union, Tuple, Optional, Dict, Callable
from torchsnooper import snoop


ANS_RE = re.compile("\[ANS\] The answer is \(([A-E])\)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        #  match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_answer):
    gt_answer = extract_answer(gt_answer)
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


class Mapper(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.loss_on_prefix = args.loss_on_prefix

    def __call__(self, example, max_len):
        if self.loss_on_prefix:
            inputs_encoding = self.tokenizer(
                example['input_text'],
                max_length=max_len,
                padding='max_length',
                return_attention_mask=True,
                return_tensors="pt",
            )
            for key, val in inputs_encoding.items():
                inputs_encoding[key] = val[0]
            labels = inputs_encoding['input_ids'].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            inputs_encoding = {}
            qns = self.tokenizer(example['question'])
            ans_cho = self.tokenizer(example['answers_choices'])
            ans = self.tokenizer(example['answer'])
            qn_tokens = qns["input_ids"]
            ans_cho_tokens = ans_cho["input_ids"]
            ans_tokens = ans["input_ids"]
            pad_length = max_len - len(qn_tokens) - len(ans_cho_tokens) - len(ans_tokens) - 1
            pad_tokens = [self.tokenizer.pad_token_id] * pad_length
            input_ids = qn_tokens + ans_cho_tokens + ans_tokens + [self.tokenizer.eos_token_id] + pad_tokens
            labels = torch.tensor([-100] * (len(qn_tokens) + len(ans_cho_tokens)) + ans_tokens + [self.tokenizer.eos_token_id] + [-100] * pad_length)
            inputs_encoding['input_ids'] = torch.tensor(input_ids)
            inputs_encoding['attention_mask'] = torch.tensor([1] * (max_len - pad_length) + [0] * pad_length)

        return dict(**inputs_encoding, labels=labels, **example)


class CommonsenseQAChainofThoughtPromptDataset(BaseDataset):
    def __init__(self, data, tokenizer, mapper):
        super().__init__(data, tokenizer, mapper)
        {"prompt": "Q: What do people use to absorb extra ink from a fountain pen?\nAnswer Choices:\n(a) shirt pocket\n(b) calligrapher's hand\n(c) inkwell\n(d) desk drawer\n(e) blotter\nA: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is (e).\n\nQ: What home entertainment equipment requires cable?\nAnswer Choices:\n(a) radio shack\n(b) substation\n(c) television\n(d) cabinet\nA: The answer must require cable. Of the above choices, only television requires cable. So the answer is (c).\n\nQ: The fox walked from the city into the forest, what was it looking for?\nAnswer Choices:\n(a) pretty flowers\n(b) hen house\n(c) natural habitat\n(d) storybook\nA: Answer: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is (b).\n\nQ: Sammy wanted to go to where the people were. Where might he go?\nAnswer Choices:\n(a) populated areas\n(b) race track\n(c) desert\n(d) apartment\n(e) roadblock\nA: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is (a).\n\nQ: Where do you put your grapes just before checking out?\nAnswer Choices:\n(a) mouth\n(b) grocery cart\n(c)super market\n(d) fruit basket\n(e) fruit market\nA: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is (b).\n\nQ: Google Maps and other highway and street GPS services have replaced what?\nAnswer Choices:\n(a) united states\n(b) mexico\n(c) countryside\n(d) atlas\nA: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is (d).\n\nQ: Before getting a divorce, what did the wife feel who was doing all the work?\nAnswer Choices:\n(a) harder\n(b) anguish\n(c) bitterness\n(d) tears\n(e) sadness\nA: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is (c).\n\n"}
        self.examples = []
        self.labels = ['A', 'B', 'C', 'D', 'E']
        for line in self.data:
            qid = line['id']
            question = "Q: " + line['question']['stem'] + "\n"
            label_index = self.labels.index(line.get('answerKey', 'A'))
            answers = [choice['text'] for choice in sorted(line['question']['choices'], key=lambda c: c['label'])]
            answers_choices = " Answer Choices:\n"
            for label, answer_text in zip(self.labels, answers):
                answers_choices += f"\n({label}) {answer_text} "

            answer = f"[ANS] So the answer is ({self.labels[label_index]})."
            self.examples.append({"input_text": question + answers_choices + answer + "<|endoftext|>", 
                "question": question, "answers_choices": answers_choices, "answer": answer})

        self.tokenized_input_text = [self.tokenizer.tokenize(ex["input_text"]) for ex in self.examples]
        self.max_len = max(
            [
                len(self.tokenized_input_text[i])
                for i in range(len(self.examples))
            ]
        ) + 1
        print(f"Max tokens length: {self.max_len}")

    def __getitem__(self, index):
        return self.mapper(self.examples[index], max_len=self.max_len)


class CommonsenseQADataset(BaseDataset):
    def __init__(self, data, tokenizer, mapper):
        super().__init__(data, tokenizer, mapper)
        self.examples = []
        self.labels = ['A', 'B', 'C', 'D', 'E']
        for line in self.data:
            qid = line['id']
            question = "[QUES]" + line['question']['stem']
            label_index = self.labels.index(line.get('answerKey', 'A'))
            answers = [choice['text'] for choice in sorted(line['question']['choices'], key=lambda c: c['label'])]
            answers_choices = " Answer Choices: "
            for label, answer_text in zip(self.labels, answers):
                answers_choices += f"[CHOICE] ({label}) {answer_text} "
            answer = f"[ANS] The answer is ({self.labels[label_index]}) {answers[label_index]}."
            self.examples.append({"input_text": question + answers_choices + answer + "<|endoftext|>", 
              "question": question, "answers_choices": answers_choices, "answer": answer})

        self.tokenized_input_text = [self.tokenizer.tokenize(ex["input_text"]) for ex in self.examples]
        self.max_len = max(
            [
                len(self.tokenized_input_text[i])
                for i in range(len(self.examples))
            ]
        ) + 1
        print(f"Max tokens length: {self.max_len}")

    def __getitem__(self, index):
        return self.mapper(self.examples[index], max_len=self.max_len)


class CommonsenseQADataModel(BaseDataModel):
    def __init__(self, args, tokenizer, mapper, custom_dataset=CommonsenseQADataset):
        super().__init__(args, tokenizer, mapper, custom_dataset)

    def get_examples(self, path):
        examples = DataProcessor._read_jsonl(path)

        return examples

if __name__ == '__main__':
    import argparse
    import pytorch_lightning as pl
    from transformers import GPT2Tokenizer
    from base_model import BaseModel
    from base_trainer import BaseTrainer
    from gpt_modeling_csqa import GPT2ModelForCSQA

    total_parser = argparse.ArgumentParser()
    # * data preprocessing args
    total_parser = CommonsenseQADataModel.add_data_specific_args(total_parser)
    # * training args
    total_parser = BaseTrainer.add_trainer_specific_args(total_parser)
    # * model specific args
    total_parser = BaseModel.add_model_specific_args(total_parser)
    # * task model specific args
    total_parser = GPT2ModelForCSQA.add_model_specific_args(total_parser)

    args = total_parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.add_tokens(['[QUES]', '[ANS]', '[CHOICE]'])

    mapper = Mapper(args=args, tokenizer=tokenizer)
    commonsenseqa_data_model = CommonsenseQADataModel(args, tokenizer, mapper)

    train_dataloader = commonsenseqa_data_model.train_dataloader()
    val_dataloader = commonsenseqa_data_model.val_dataloader()
    test_dataloader = commonsenseqa_data_model.test_dataloader()

    #  print(len(commonsenseqa_data_model.raw_train_data))

    batch = next(iter(train_dataloader))
    print("batch input text", batch['input_text'])
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_inputs = tokenizer.batch_decode(input_ids)
    decoded_labels = tokenizer.batch_decode(labels)
    for inp, lab in zip(decoded_inputs, decoded_labels):
        print("inputs", inp)
        print("labels", lab)
    #  is_correct(batch['answer'][0], batch['answer'][0])
    #  batch = next(iter(val_dataloader))
    #  print(batch)
    #  batch = next(iter(test_dataloader))
    #  print(batch)

