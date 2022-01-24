# coding=utf-8
# Copyright 2021 The IDEA Authors. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import json
from tqdm import tqdm
import os
import numpy as np
from transformers import EvalPrediction
from label_desc import task2label
import sys
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    AutoTokenizer,
    # AutoModelForSequenceClassification
)

from auto.modeling_auto import AutoModelForSequenceClassification

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

def load_schema(task_name):
    label2id, id2label = {}, {}
    for i, k in enumerate(task2label[task_name].keys()):
        label2id[k] = i
        id2label[i] = k
    return label2id, id2label

def load_data(file_path, task_name, label2id):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        result = []
        for line in tqdm(lines):
            data = json.loads(line)
            text_id = int(data['id']) if 'id' in data.keys() else 0
            texta, textb, labels = '', '', 0
            if task_name == 'afqmc':
                texta = data['sentence1']
                textb = data['sentence2']
                labels = label2id[data['label']
                                  ] if 'label' in data.keys() else 0
            elif task_name == 'tnews':
                texta = data['sentence']
                labels = label2id[data['label_desc']
                                  ] if 'label_desc' in data.keys() else 0
            elif task_name == 'iflytek':
                texta = data['sentence']
                labels = label2id[data['label_des']
                                  ] if 'label_des' in data.keys() else 0
            elif task_name == 'ocnli':
                texta = data['sentence1']
                textb = data['sentence2']
                labels = label2id[data['label']
                                  ] if 'label' in data.keys() else 0
            elif task_name == 'wsc':
                target = data['target']
                text = list(data['text'])
                if target['span2_index'] < target['span1_index']:
                    text.insert(target['span2_index'], '_')
                    text.insert(target['span2_index'] +
                                len(target['span2_text'])+1, '_')
                    text.insert(target['span1_index']+2, '[')
                    text.insert(target['span1_index']+2 +
                                len(target['span1_text'])+1, ']')
                else:
                    text.insert(target['span1_index'], '[')
                    text.insert(target['span1_index'] +
                                len(target['span1_text'])+1, ']')
                    text.insert(target['span2_index']+2, '_')
                    text.insert(target['span2_index']+2 +
                                len(target['span2_text'])+1, '_')
                texta = ''.join(text)
                labels = label2id[data['label']
                                  ] if 'label' in data.keys() else 0
            elif task_name == 'csl':
                texta = '； '.join(data['keyword'])
                textb = data['abst']
                labels = label2id[data['label']
                                  ] if 'label' in data.keys() else 0  
            result.append({'texta': texta, 'textb': textb,
                          'labels': labels, 'id': text_id})
        print(result[0])
        return result

class taskDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        if 'textb' in item.keys() and item['textb'] != '':
            encode_dict = self.tokenizer.encode_plus(item['texta'], item['textb'],
                                                     max_length=self.max_length,
                                                     padding='max_length',
                                                     truncation='longest_first')
        else:
            encode_dict = self.tokenizer.encode_plus(item['texta'],  #
                                                     max_length=self.max_length,
                                                     padding='max_length',
                                                     truncation='longest_first')

        return {
            "input_ids": torch.tensor(encode_dict['input_ids']).long(),
            "token_type_ids": torch.tensor(encode_dict['token_type_ids']).long(),
            "attention_mask": torch.tensor(encode_dict['attention_mask']).float(),
            "labels": torch.tensor(item['labels']).long()
        }


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(
        p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


def main():
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument("--task_name", type=str, help="test file")
    parser.add_argument("--train_data_path", type=str, help="train file")
    parser.add_argument("--dev_data_path", type=str, help="dev file")
    parser.add_argument("--test_data_path", type=str, help="test file")
    parser.add_argument("--pretrained_model_path", type=str,
                        help="pretrained_model_path")
    parser.add_argument("--max_length", type=int, help="max_length")

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        training_args, task_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        training_args, task_args = parser.parse_args_into_dataclasses()

    label2id, _ = load_schema(task_args.task_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        task_args.pretrained_model_path, num_labels=len(label2id))
    tokenizer = AutoTokenizer.from_pretrained(task_args.pretrained_model_path)

    train_data = load_data(task_args.train_data_path,
                           task_args.task_name, label2id)
    dev_data = load_data(task_args.dev_data_path,
                         task_args.task_name, label2id)
    train_dataset = taskDataset(
        train_data, tokenizer=tokenizer, max_length=task_args.max_length)
    dev_dataset = taskDataset(
        dev_data, tokenizer=tokenizer, max_length=task_args.max_length)

    trainer = Trainer(
        model=model,
        args=training_args,                  
        train_dataset=train_dataset,        
        eval_dataset=dev_dataset,        
        compute_metrics=compute_metrics
    )

    trainer.train()


if __name__ == "__main__":
    main()
