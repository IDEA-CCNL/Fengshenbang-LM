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
from fengshen.models.zen2.modeling import ZenForTokenClassification
from fengshen.metric.metric import SeqEntityScore
from fengshen.models.zen2.tokenization import BertTokenizer
from fengshen.models.zen2.ngram_utils import ZenNgramDict
from pytorch_lightning.callbacks import LearningRateMonitor
from dataclasses import dataclass
import logging
import math
import numpy as np
import os
import json
import torch
import pytorch_lightning as pl
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import sys

import torch.nn.functional as F
sys.path.append('../../../')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.ERROR)
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
                 ngram_tuples, ngram_seg_ids, ngram_masks, valid_ids=None, label_mask=None, b_use_valid_filter=False):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks

        self.b_use_valid_filter = b_use_valid_filter


def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer, ngram_dict):
    """Loads a data file into a list of `InputBatch`s."""

    # label_map = {label: i for i, label in enumerate(label_list, 1)}
    # label_map["[PAD]"] = 0

    features = []
    b_use_valid_filter = False
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            if len(tokens) + len(token) > max_seq_length - 2:
                break
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
                    b_use_valid_filter = True
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        # ----------- code for ngram BEGIN-----------
        ngram_matches = []
        #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
        max_gram_n = ngram_dict.max_ngram_len
        for p in range(2, max_gram_n):
            for q in range(0, len(tokens) - p + 1):
                character_segment = tokens[q:q + p]
                # j is the starting position of the ngram
                # i is the length of the current ngram
                character_segment = tuple(character_segment)
                if character_segment in ngram_dict.ngram_to_id_dict:
                    ngram_index = ngram_dict.ngram_to_id_dict[character_segment]
                    ngram_freq = ngram_dict.ngram_to_freq_dict[character_segment]
                    ngram_matches.append([ngram_index, q, p, character_segment, ngram_freq])

        ngram_matches = sorted(ngram_matches, key=lambda s: s[0])

        max_ngram_in_seq_proportion = math.ceil((len(tokens) / max_seq_length) * ngram_dict.max_ngram_in_seq)
        if len(ngram_matches) > max_ngram_in_seq_proportion:
            ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

        ngram_ids = [ngram[0] for ngram in ngram_matches]
        ngram_positions = [ngram[1] for ngram in ngram_matches]
        ngram_lengths = [ngram[2] for ngram in ngram_matches]
        ngram_tuples = [ngram[3] for ngram in ngram_matches]
        ngram_freqs = [ngram[4] for ngram in ngram_matches]
        ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

        ngram_mask_array = np.zeros(ngram_dict.max_ngram_in_seq, dtype=np.bool)
        ngram_mask_array[:len(ngram_ids)] = 1

        # record the masked positions
        ngram_positions_matrix = np.zeros(shape=(max_seq_length, ngram_dict.max_ngram_in_seq), dtype=np.int32)
        for i in range(len(ngram_ids)):
            ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = ngram_freqs[i]
        ngram_positions_matrix = torch.from_numpy(ngram_positions_matrix.astype(np.float))
        ngram_positions_matrix = torch.div(ngram_positions_matrix, torch.stack(
            [torch.sum(ngram_positions_matrix, 1)] * ngram_positions_matrix.size(1)).t() + 1e-10)
        ngram_positions_matrix = ngram_positions_matrix.numpy()

        # Zero-pad up to the max ngram in seq length.
        padding = [0] * (ngram_dict.max_ngram_in_seq - len(ngram_ids))
        ngram_ids += padding
        ngram_lengths += padding
        ngram_seg_ids += padding

        # ----------- code for ngram END-----------

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (",".join([str(x) for x in example.label]), ",".join([str(x) for x in label_ids])))
            logger.info("valid: %s" % " ".join([str(x) for x in valid]))
            logger.info("b_use_valid_filter: %s" % str(b_use_valid_filter))
            logger.info("ngram_ids: %s" % " ".join([str(x) for x in ngram_ids]))
            logger.info("ngram_positions: %s" % " ".join([str(x) for x in ngram_positions]))
            logger.info("ngram_lengths: %s" % " ".join([str(x) for x in ngram_lengths]))
            logger.info("ngram_tuples: %s" % " ".join([str(x) for x in ngram_tuples]))
            logger.info("ngram_seg_ids: %s" % " ".join([str(x) for x in ngram_seg_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          ngram_ids=ngram_ids,
                          ngram_positions=ngram_positions_matrix,
                          ngram_lengths=ngram_lengths,
                          ngram_tuples=ngram_tuples,
                          ngram_seg_ids=ngram_seg_ids,
                          ngram_masks=ngram_mask_array,
                          valid_ids=valid,
                          label_mask=label_mask,
                          b_use_valid_filter=b_use_valid_filter))
    return features


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_path, set_type, quotechar=' '):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_path, self.get_quotechar()), set_type)

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_quotechar(self):
        return ' '

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        '''
        read file
        return format :
        [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
        '''
        f = open(input_file)
        data = []
        sentence = []
        label = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            splits = line.split(quotechar)
            sentence.append(splits[0])
            label.append(splits[-1][:-1])

        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []
        return data


class MSRAProcessor(DataProcessor):
    """Processor for the msra data set."""

    def get_labels(self):
        return ['B-NR', 'B-NS', 'B-NT', 'E-NR', 'E-NS', 'E-NT', 'M-NR',
                'M-NS', 'M-NT', 'O', 'S-NR', 'S-NS', 'S-NT', '[CLS]', '[SEP]']


class OntoNotes4Processor(DataProcessor):
    """Processor for the OntoNotes4 data set."""

    def get_labels(self):
        return ['B-GPE', 'B-LOC', 'B-ORG', 'B-PER', 'E-GPE', 'E-LOC',
                'E-ORG', 'E-PER', 'M-GPE', 'M-LOC', 'M-ORG', 'M-PER', 'O',
                'S-GPE', 'S-LOC', 'S-ORG', 'S-PER', '[CLS]', '[SEP]']


class WeiboProcessor(DataProcessor):
    """Processor for the Weibo data set."""

    def get_labels(self):
        return ['B-GPE.NAM', 'B-GPE.NOM', 'B-LOC.NAM', 'B-LOC.NOM',
                'B-ORG.NAM', 'B-ORG.NOM', 'B-PER.NAM', 'B-PER.NOM', 'E-GPE.NAM',
                'E-GPE.NOM', 'E-LOC.NAM', 'E-LOC.NOM', 'E-ORG.NAM', 'E-ORG.NOM',
                'E-PER.NAM', 'E-PER.NOM', 'M-GPE.NAM', 'M-LOC.NAM', 'M-LOC.NOM',
                'M-ORG.NAM', 'M-ORG.NOM', 'M-PER.NAM', 'M-PER.NOM', 'O',
                'S-GPE.NAM', 'S-LOC.NOM', 'S-PER.NAM', 'S-PER.NOM', '[CLS]', '[SEP]']


class ResumeProcessor(DataProcessor):
    """Processor for the resume data set."""

    def get_labels(self):
        return ['B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO',
                'B-RACE', 'B-TITLE', 'E-CONT', 'E-EDU', 'E-LOC', 'E-NAME',
                'E-ORG', 'E-PRO', 'E-RACE', 'E-TITLE', 'M-CONT', 'M-EDU',
                'M-LOC', 'M-NAME', 'M-ORG', 'M-PRO', 'M-RACE', 'M-TITLE',
                'O', 'S-NAME', 'S-ORG', 'S-RACE', '[CLS]', '[SEP]']


class CMeEEProcessor(DataProcessor):
    """Processor for the CMeEE data set."""

    def get_quotechar(self):
        return '\t'

    def get_labels(self):
        return ['B-临床表现', 'B-医学检验项目', 'B-医疗程序', 'B-医疗设备',
                'B-微生物类', 'B-疾病', 'B-科室', 'B-药物', 'B-身体', 'I-临床表现',
                'I-医学检验项目', 'I-医疗程序', 'I-医疗设备', 'I-微生物类',
                'I-疾病', 'I-科室', 'I-药物', 'I-身体', 'O', '[CLS]', '[SEP]']


class CLUENERProcessor(DataProcessor):
    """Processor for the CLUENER data set."""

    def get_quotechar(self):
        return '\t'

    def get_labels(self):
        return ['B-书名', 'B-公司', 'B-地址', 'B-姓名', 'B-政府', 'B-景点',
                'B-游戏', 'B-电影', 'B-组织机构', 'B-职位', 'I-书名', 'I-公司',
                'I-地址', 'I-姓名', 'I-政府', 'I-景点', 'I-游戏', 'I-电影',
                'I-组织机构', 'I-职位', 'O', '[CLS]', '[SEP]']


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
        valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)

        ngram_ids = torch.tensor([f.ngram_ids for f in features], dtype=torch.long)
        ngram_positions = torch.tensor([f.ngram_positions for f in features], dtype=torch.long)
        # ngram_lengths = torch.tensor([f.ngram_lengths for f in features], dtype=torch.long)
        # ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in features], dtype=torch.long)
        # ngram_masks = torch.tensor([f.ngram_masks for f in features], dtype=torch.long)

        # label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.long)
        b_use_valid_filter = torch.tensor([f.b_use_valid_filter for f in features], dtype=torch.bool)
        # 取第一个出来？
        # b_use_valid_filter = b_use_valid_filter.detach().cpu().numpy()[0]
        b_use_valid_filter = b_use_valid_filter[0]
        return {
            'input_ids': input_ids,
            'input_ngram_ids': ngram_ids,
            'ngram_position_matrix': ngram_positions,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'labels': label_ids,
            'valid_ids': valid_ids,
            'b_use_valid_filter': b_use_valid_filter,
        }


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
        parser.add_argument('--task_name', default='weibo', type=str)

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
            'weibo': WeiboProcessor,
            'resume': ResumeProcessor,
            'msra': MSRAProcessor,
            'ontonotes4': OntoNotes4Processor,
            'cmeee': CMeEEProcessor,
            'cluener': CLUENERProcessor,
        }
        if args.task_name not in processors:
            raise ValueError("Task not found: %s" % (args.task_name))
        processor = processors[args.task_name]()
        # 生成id映射
        label_list = processor.get_labels()
        label2id = {label: i for i, label in enumerate(label_list, 1)}
        label2id["[PAD]"] = 0
        self.id2label = {v: k for k, v in label2id.items()}
        self.collator.label2id = label2id

        if args.dataset_name is None:
            self.train_data = TaskDataset(os.path.join(
                args.data_dir, args.train_data), processor, mode='train')
            self.valid_data = TaskDataset(os.path.join(
                args.data_dir, args.valid_data), processor, mode='dev')
            self.test_data = TaskDataset(os.path.join(
                args.data_dir, args.test_data), processor, mode='test')

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


class LitModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--markup', default='bios', type=str)
        parser.add_argument('--middle_prefix', default='I-', type=str)
        return parent_args

    def __init__(self, args, id2label):
        super().__init__()
        # config = ZenConfig(os.path.join(args.pretrained_model_path, 'config.json'))
        self.model = ZenForTokenClassification.from_pretrained(args.pretrained_model_path, num_labels=len(id2label))
        self.seq_entity_score = SeqEntityScore(id2label, markup=args.markup, middle_prefix=args.middle_prefix)
        self.train_seq_entity_score = SeqEntityScore(id2label, markup=args.markup, middle_prefix=args.middle_prefix)
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
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
        # logits = outputs.logits
        # preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        # preds = preds.detach().cpu().numpy()
        # labels = batch['labels'].detach().cpu().numpy()
        # num_labels = len(self.label2id)
        # y_true = []
        # y_pred = []
        # for i, label in enumerate(labels):
        #     temp_1 = []
        #     temp_2 = []
        #     for j, m in enumerate(label):
        #         if j == 0:
        #             continue
        #         elif labels[i][j] == num_labels - 1:
        #             y_true.append(temp_1)
        #             y_pred.append(temp_2)
        #             break
        #         else:
        #             temp_1.append(self.id2label[labels[i][j]])
        #             temp_2.append(self.id2label[preds[i][j]])

        # self.train_seq_entity_score.update(y_true, y_pred)
        # result = self.train_seq_entity_score.result()
        # self.train_seq_entity_score.reset()
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        preds = preds.detach().cpu().numpy()
        labels = batch['labels'].detach().cpu().numpy()
        num_labels = len(self.label2id)
        y_true = []
        y_pred = []
        for i, label in enumerate(labels):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif labels[i][j] == num_labels - 1:
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(self.id2label[labels[i][j]])
                    temp_2.append(self.id2label[preds[i][j]])

        self.seq_entity_score.update(y_true, y_pred)
        self.log('val_loss', loss)

    def validation_epoch_end(self, outputs):
        # compute metric for all process
        score_dict, _ = self.seq_entity_score.result()
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
            print('score_dict:\n', score_dict)
        # reset the metric after once validation
        self.seq_entity_score.reset()
        for k, v in score_dict.items():
            self.log('val_{}'.format(k), v)

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
    id2label = data_model.id2label
    print('id2label:', id2label)
    model = LitModel(args, id2label)
    trainer.fit(model, data_model)


if __name__ == "__main__":
    main()
