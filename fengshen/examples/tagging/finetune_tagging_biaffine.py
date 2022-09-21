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
from pathlib import Path

from dataclasses import dataclass
import copy
import logging
import numpy as np
import torch.nn.functional as F
import os
import json
import torch
import pytorch_lightning as pl
import argparse
import time
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader
from fengshen.models.model_utils import configure_optimizers
from fengshen.models.tagging_models.bert_for_tagging import (
    BertBiaffine,
)
from transformers import (
    BertTokenizer, BertConfig
)
from fengshen.metric.metric import EntityScore
from fengshen.metric.utils_ner import get_entities, bert_extract_item

torch.set_printoptions(threshold=10000,linewidth=10000)
np.set_printoptions(threshold=10000,linewidth=10000)

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, text_a, labels, subject):
        self.guid = guid
        self.text_a = text_a
        self.labels = labels
        self.subject = subject

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BiaffineInputFeatures(InputFeatures):
    def __init__(self, input_ids, input_mask, input_len, segment_ids, span_labels, span_mask):
        super(BiaffineInputFeatures, self).__init__(input_ids, input_mask, input_len, segment_ids)
        self.span_labels = span_labels
        self.span_mask = span_mask


def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer):

    features = []
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    pad_token = 0
    special_tokens_count = 2
    segment_id = 0

    for (ex_index, example) in enumerate(examples):
        subjects = copy.deepcopy(example.subject)
        tokens = copy.deepcopy(example.text_a)

        span_labels = np.zeros((max_seq_length,max_seq_length))
        span_labels[:] = label2id["O"]

        for subject in subjects:
            label = subject[0]
            start = subject[1]
            end = subject[2]
            if start < max_seq_length - special_tokens_count and end < max_seq_length - special_tokens_count:
                span_labels[start + 1, end + 1] = label2id[label]

        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        span_labels[len(tokens), :] = label2id["O"]
        span_labels[:, len(tokens)] = label2id["O"]
        segment_ids = [segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        span_labels[0, :] = label2id["O"]
        span_labels[:, 0] = label2id["O"]
        segment_ids = [segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [0] * len(input_ids)
        span_mask = np.ones(span_labels.shape)
        input_len = len(input_ids)

        padding_length = max_seq_length - len(input_ids)

        input_ids += [pad_token] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [segment_id] * padding_length
        span_labels[input_len:, :] = 0
        span_labels[:, input_len:] = 0
        span_mask[input_len:, :] = 0
        span_mask[:, input_len:] = 0
        span_mask=np.triu(span_mask,0)
        span_mask=np.tril(span_mask,10)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(span_labels) == max_seq_length
        assert len(span_labels[0]) == max_seq_length

        # if ex_index < 2:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s", example.guid)
        #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))

        features.append(
            BiaffineInputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                span_labels=span_labels,
                span_mask=span_mask,
                input_len=input_len,
            )
        )
    
    return features


class DataProcessor(object):
    def __init__(self, data_dir) -> None:
        super().__init__()
        self.data_dir = data_dir

    def get_examples(self, mode):
        return self._create_examples(self._read_text(os.path.join(self.data_dir, mode + ".all.bmes")), mode)

    def get_labels(self):
        with open(os.path.join(self.data_dir, "labels.txt")) as f:
            label_list = ["[PAD]", "[START]", "[END]"]
            for line in f.readlines():
                tag = line.strip().split("-")
                if len(tag) == 1 and tag[0] not in label_list:
                    label_list.append(tag[0])
                elif tag[1] not in label_list:
                    label_list.append(tag[1])

        label2id = {label: i for i, label in enumerate(label_list)}
        return label2id

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                else:
                    labels.append(x)
            subject = get_entities(labels, id2label=None, markup='bioes')
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels, subject=subject))
        return examples

    @classmethod
    def _read_text(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split()
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines


class TaskDataset(Dataset):
    def __init__(self, processor, mode='train'):
        super().__init__()
        self.data = self.load_data(processor, mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_data(self, processor, mode):
        examples = processor.get_examples(mode)
        return examples


@dataclass
class TaskCollator:
    args = None
    tokenizer = None
    label2id = None

    def __call__(self, samples):

        features = convert_examples_to_features(samples, self.label2id, self.args.max_seq_length, self.tokenizer)

        input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
        input_mask = torch.tensor(np.array([f.input_mask for f in features]), dtype=torch.long)
        segment_ids = torch.tensor(np.array([f.segment_ids for f in features]), dtype=torch.long)
        span_labels = torch.tensor(np.array([f.span_labels for f in features]), dtype=torch.long)
        span_mask = torch.tensor(np.array([f.span_mask for f in features]), dtype=torch.long)
        input_len = torch.tensor(np.array([f.input_len for f in features]), dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'span_labels': span_labels,
            'span_mask': span_mask,
            'input_len': input_len,
        }


class TaskDataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('DataModel')
        parser.add_argument('--data_dir', default='./data', type=str)
        parser.add_argument('--num_workers', default=0, type=int)
        parser.add_argument('--train_batchsize', default=16, type=int)
        parser.add_argument('--valid_batchsize', default=16, type=int)
        parser.add_argument('--max_seq_length', default=512, type=int)

        parser.add_argument(
            "--pretrained_model_path",
            default=None,
            type=str,
            help="Path to pre-trained model or shortcut name selected in the list: ",
        )
        parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

        return parent_args

    def __init__(self, args):
        super().__init__()
        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize

        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=args.do_lower_case)
        processor = DataProcessor(args.data_dir)
        args.label2id=processor.get_labels()

        self.collator = TaskCollator()
        self.collator.args=args
        self.collator.tokenizer=tokenizer
        self.collator.label2id=args.label2id

        # global tmpdev

        self.train_data=TaskDataset(processor=processor,mode="train")
        self.valid_data=TaskDataset(processor=processor,mode="test")
        self.test_data=TaskDataset(processor=processor,mode="test")

        # tmpdev=self.valid_data

        self.save_hyperparameters(args)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=self.train_batchsize, pin_memory=False,
                          collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, batch_size=self.valid_batchsize, pin_memory=False,
                          collate_fn=self.collator)

    def predict_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, batch_size=self.valid_batchsize, pin_memory=False,
                          collate_fn=self.collator)


class LitModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--loss_type', default='ce', type=str, choices=['lsr', 'focal', 'ce'])
        return parent_args

    def __init__(self, args):
        super().__init__()

        self.label2id = args.label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        self.config=BertConfig.from_pretrained(args.pretrained_model_path)
        self.model = BertBiaffine.from_pretrained(args.pretrained_model_path, config=self.config, num_labels=len(self.id2label), loss_type=args.loss_type)
        self.entity_score=EntityScore()

        self.save_hyperparameters(args)
        
    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
            # Calculate total steps
            if self.trainer.max_epochs > 0:
                world_size = self.trainer.world_size
                tb_size = self.hparams.train_batchsize * max(1, world_size)
                ab_size = self.trainer.accumulate_grad_batches
                self.total_steps = (len(train_loader.dataset) *
                                    self.trainer.max_epochs // tb_size) // ab_size
            else:
                self.total_steps = self.trainer.max_steps // self.trainer.accumulate_grad_batches

            print('Total steps: {}' .format(self.total_steps))

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.span_logits

        preds = np.argmax(logits.cpu().numpy(), axis=-1)
        labels = batch['span_labels'].cpu().numpy()

        for i, label in enumerate(labels):
            input_len=(batch['input_len'][i])-2
            active_label=labels[i,1:input_len+1,1:input_len+1]
            active_pred=preds[i,1:input_len+1,1:input_len+1]

            temp_1 = []
            temp_2 = []

            for j in range(input_len):
                for k in range(input_len):
                    if self.id2label[active_label[j,k]]!="O":
                        temp_1.append([self.id2label[active_label[j,k]],j,k])
                    if self.id2label[active_pred[j,k]]!="O":
                        temp_2.append([self.id2label[active_pred[j,k]],j,k])

            self.entity_score.update(pred_subject=temp_2, true_subject=temp_1)

        self.log('val_loss', loss)

    def validation_epoch_end(self, outputs):
        # compute metric for all process
        score_dict, _ = self.entity_score.result()
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
            print('score_dict:\n', score_dict)
        # reset the metric after once validation
        self.entity_score.reset()
        for k, v in score_dict.items():
            self.log('val_{}'.format(k), v)

    def configure_optimizers(self):
        return configure_optimizers(self)

class TaskModelCheckpoint:
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')

        parser.add_argument('--monitor', default='train_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--dirpath', default='./log/', type=str)
        parser.add_argument(
            '--filename', default='model-{epoch:02d}-{train_loss:.4f}', type=str)

        parser.add_argument('--save_top_k', default=3, type=float)
        parser.add_argument('--every_n_train_steps', default=100, type=float)
        parser.add_argument('--save_weights_only', default=True, type=bool)

        return parent_args

    def __init__(self, args):
        self.callbacks = ModelCheckpoint(monitor=args.monitor,
                                         save_top_k=args.save_top_k,
                                         mode=args.mode,
                                         every_n_train_steps=args.every_n_train_steps,
                                         save_weights_only=args.save_weights_only,
                                         dirpath=args.dirpath,
                                         filename=args.filename)

def main():
    total_parser = argparse.ArgumentParser("TASK NAME")

    # * Args for data preprocessing
    total_parser = TaskDataModel.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = pl.Trainer.add_argparse_args(total_parser)
    total_parser = TaskModelCheckpoint.add_argparse_args(total_parser)

    # * Args for base model
    from fengshen.models.model_utils import add_module_args
    total_parser = add_module_args(total_parser)
    total_parser = LitModel.add_model_specific_args(total_parser)

    args = total_parser.parse_args()

    checkpoint_callback = TaskModelCheckpoint(args).callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[checkpoint_callback, lr_monitor]
                                            )
    data_model = TaskDataModel(args)

    model = LitModel(args)
    print(args.label2id)
    trainer.fit(model, data_model)


if __name__ == "__main__":
    main()