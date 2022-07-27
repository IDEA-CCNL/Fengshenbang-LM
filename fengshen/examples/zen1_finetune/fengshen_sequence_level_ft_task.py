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
from fengshen.models.zen1.tokenization import BertTokenizer
from fengshen.models.zen1.modeling import ZenForSequenceClassification
from fengshen.models.zen1.ngram_utils import ZenNgramDict
from pytorch_lightning.callbacks import LearningRateMonitor
import csv
from dataclasses import dataclass
import logging
import math
import numpy as np
import os
from tqdm import tqdm
import json
import torch
import pytorch_lightning as pl
from random import shuffle
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('../../../')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ngram_ids, ngram_positions, ngram_lengths,
                 ngram_tuples, ngram_seg_ids, ngram_masks):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_path, mode):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                # if sys.version_info[0] == 2:
                #     line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """Reads a jsonl file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            samples = []
            for line in tqdm(lines):
                data = json.loads(line)
                samples.append(data)
            return samples


class TnewsProcessor(DataProcessor):
    """Processor for the tnews data set (HIT version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_examples(self, data_path, mode):
        return self._create_examples(
            self._read_json(data_path),
            set_type=mode
        )

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #    continue
            guid = "%s-%s" % (set_type, i)
            # text_a = line[0]
            text_a = line['sentence']
            label = line['label'] if 'label' in line.keys() else None
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class OcnliProcessor(DataProcessor):
    """Processor for the ocnli or cmnli data set (HIT version)."""

    def get_examples(self, data_path, mode):
        return self._create_examples(
            self._read_json(data_path),
            set_type=mode
        )

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #    continue
            guid = "%s-%s" % (set_type, i)
            # text_a = line[0]
            text_a = line['sentence1']
            text_b = line['sentence2']
            label = line['label'] if 'label' in line.keys() else None
            # 特殊处理，cmnli有label为-的
            if label == '-':
                label = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class IflytekProcessor(DataProcessor):
    """Processor for the iflytek data set (HIT version)."""

    def get_examples(self, data_path, mode):
        return self._create_examples(
            self._read_json(data_path),
            set_type=mode
        )

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #    continue
            guid = "%s-%s" % (set_type, i)
            # text_a = line[0]
            text_a = line['sentence']
            label = line['label'] if 'label' in line.keys() else None
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer, ngram_dict):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        # ----------- code for ngram BEGIN-----------
        ngram_matches = []
        #  Filter the word segment from 2 to 7 to check whether there is a word
        for p in range(2, 8):
            for q in range(0, len(tokens) - p + 1):
                character_segment = tokens[q:q + p]
                # j is the starting position of the word
                # i is the length of the current word
                character_segment = tuple(character_segment)
                if character_segment in ngram_dict.ngram_to_id_dict:
                    ngram_index = ngram_dict.ngram_to_id_dict[character_segment]
                    ngram_matches.append([ngram_index, q, p, character_segment])

        shuffle(ngram_matches)
        # max_word_in_seq_proportion = max_word_in_seq
        max_word_in_seq_proportion = math.ceil((len(tokens) / max_seq_length) * ngram_dict.max_ngram_in_seq)
        if len(ngram_matches) > max_word_in_seq_proportion:
            ngram_matches = ngram_matches[:max_word_in_seq_proportion]
        ngram_ids = [ngram[0] for ngram in ngram_matches]
        ngram_positions = [ngram[1] for ngram in ngram_matches]
        ngram_lengths = [ngram[2] for ngram in ngram_matches]
        ngram_tuples = [ngram[3] for ngram in ngram_matches]
        ngram_seg_ids = [0 if position < (len(tokens_a) + 2) else 1 for position in ngram_positions]

        ngram_mask_array = np.zeros(ngram_dict.max_ngram_in_seq, dtype=np.bool)
        ngram_mask_array[:len(ngram_ids)] = 1

        # record the masked positions
        ngram_positions_matrix = np.zeros(shape=(max_seq_length, ngram_dict.max_ngram_in_seq), dtype=np.int32)
        for i in range(len(ngram_ids)):
            ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

        # Zero-pad up to the max word in seq length.
        padding = [0] * (ngram_dict.max_ngram_in_seq - len(ngram_ids))
        ngram_ids += padding
        ngram_lengths += padding
        ngram_seg_ids += padding

        # ----------- code for ngram END-----------
        label_id = label_map[example.label] if example.label is not None else 0
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          ngram_ids=ngram_ids,
                          ngram_positions=ngram_positions_matrix,
                          ngram_lengths=ngram_lengths,
                          ngram_tuples=ngram_tuples,
                          ngram_seg_ids=ngram_seg_ids,
                          ngram_masks=ngram_mask_array))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class TaskDataset(Dataset):
    def __init__(self, data_path, processor, mode='train'):
        super().__init__()
        self.data = self.load_data(data_path, processor, mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_data(self, data_path, processor, mode):
        if mode == "train":
            examples = processor.get_examples(data_path, mode)
        elif mode == "test":
            examples = processor.get_examples(data_path, mode)
        elif mode == "dev":
            examples = processor.get_examples(data_path, mode)
        return examples


@dataclass
class TaskCollator:
    args = None
    tokenizer = None
    ngram_dict = None
    label2id = None

    def __call__(self, samples):
        features = convert_examples_to_features(samples, self.label2id, self.args.max_seq_length, self.tokenizer, self.ngram_dict)
        # logger.info("  Num examples = %d", len(samples))
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        ngram_ids = torch.tensor([f.ngram_ids for f in features], dtype=torch.long)
        ngram_positions = torch.tensor([f.ngram_positions for f in features], dtype=torch.long)
        # ngram_lengths = torch.tensor([f.ngram_lengths for f in features], dtype=torch.long)
        # ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in features], dtype=torch.long)
        # ngram_masks = torch.tensor([f.ngram_masks for f in features], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'input_ngram_ids': ngram_ids,
            'ngram_position_matrix': ngram_positions,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'labels': label_ids,

        }
        # return default_collate(sample_list)


class TaskDataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('TASK NAME DataModel')
        parser.add_argument('--data_dir', default='./data', type=str)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--train_data', default='train.json', type=str)
        parser.add_argument('--valid_data', default='dev.json', type=str)
        parser.add_argument('--test_data', default='test.json', type=str)
        parser.add_argument('--train_batchsize', default=16, type=int)
        parser.add_argument('--valid_batchsize', default=32, type=int)
        parser.add_argument('--max_seq_length', default=128, type=int)

        parser.add_argument('--texta_name', default='text', type=str)
        parser.add_argument('--textb_name', default='sentence2', type=str)
        parser.add_argument('--label_name', default='label', type=str)
        parser.add_argument('--id_name', default='id', type=str)

        parser.add_argument('--dataset_name', default=None, type=str)
        parser.add_argument('--vocab_file',
                            type=str, default=None,
                            help="Vocabulary mapping/file BERT was pretrainined on")
        parser.add_argument("--do_lower_case",
                            action='store_true',
                            help="Set this flag if you are using an uncased model.")
        parser.add_argument('--task_name', default='tnews', type=str)

        return parent_args

    def __init__(self, args):
        super().__init__()
        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize
        self.collator = TaskCollator()
        self.collator.args = args
        self.collator.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=args.do_lower_case)
        self.collator.ngram_dict = ZenNgramDict.from_pretrained(args.pretrained_model_path, tokenizer=self.collator.tokenizer)

        processors = {
            'afqmc': OcnliProcessor,
            'tnews': TnewsProcessor,
            'ocnli': OcnliProcessor,
            'cmnli': OcnliProcessor,
            'iflytek': IflytekProcessor,
        }
        if args.task_name not in processors:
            raise ValueError("Task not found: %s" % (args.task_name))
        processor = processors[args.task_name]()
        if args.dataset_name is None:
            self.label2id, self.id2label = self.load_schema(os.path.join(
                args.data_dir, args.train_data), args)
            self.train_data = TaskDataset(os.path.join(
                args.data_dir, args.train_data), processor, mode='train')
            self.valid_data = TaskDataset(os.path.join(
                args.data_dir, args.valid_data), processor, mode='dev')
            self.test_data = TaskDataset(os.path.join(
                args.data_dir, args.test_data), processor, mode='test')
            self.collator.label2id = self.label2id
        else:
            import datasets
            ds = datasets.load_dataset(args.dataset_name)
            self.train_data = ds['train']
            self.valid_data = ds['validation']
            self.test_data = ds['test']
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

    def load_schema(self, data_path, args):
        with open(data_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            label_list = []
            for line in tqdm(lines):
                data = json.loads(line)
                labels = data[args.label_name] if args.label_name in data.keys(
                ) else 0
                if labels not in label_list:
                    label_list.append(labels)

        label2id, id2label = {}, {}
        for i, k in enumerate(label_list):
            label2id[k] = i
            id2label[i] = k
        return label2id, id2label


class LitModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--num_labels', default=2, type=int)

        return parent_args

    def __init__(self, args):
        super().__init__()
        self.model = ZenForSequenceClassification.from_pretrained(args.pretrained_model_path, num_labels=args.num_labels)
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
        loss, logits = self.model(**batch)
        acc = self.comput_metrix(logits, batch['labels'])
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/labels.size()[0]
        return acc

    def validation_step(self, batch, batch_idx):
        loss, logits = self.model(**batch)
        acc = self.comput_metrix(logits, batch['labels'])
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def predict_step(self, batch, batch_idx):
        output = self.model(**batch)
        return output.logits

    def configure_optimizers(self):
        from fengshen.models.model_utils import configure_optimizers
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


def save_test(data, args, data_model):
    with open(args.output_save_path, 'w', encoding='utf-8') as f:
        idx = 0
        for i in range(len(data)):
            batch = data[i]
            for sample in batch:
                tmp_result = dict()
                label_id = np.argmax(sample.numpy())
                tmp_result['id'] = data_model.test_data.data[idx]['id']
                tmp_result['label'] = data_model.id2label[label_id]
                json_data = json.dumps(tmp_result, ensure_ascii=False)
                f.write(json_data+'\n')
                idx += 1
    print('save the result to '+args.output_save_path)


def main():
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser.add_argument('--pretrained_model_path', default='', type=str)
    total_parser.add_argument('--output_save_path',
                              default='./predict.json', type=str)
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
    trainer.fit(model, data_model)


if __name__ == "__main__":
    main()
