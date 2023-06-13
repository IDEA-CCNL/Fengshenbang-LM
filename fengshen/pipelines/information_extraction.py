from logging import basicConfig
import torch
from torch import nn
import json
from tqdm import tqdm
import os
import numpy as np
from transformers import BertTokenizer
import pytorch_lightning as pl

from pytorch_lightning import trainer, loggers
from transformers import AlbertTokenizer
from transformers import AutoConfig,AutoTokenizer
from transformers.pipelines.base import Pipeline
import argparse
import copy
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
import warnings
from fengshen.models.uniex.modeling_uniex import (
    UniEXDataModel,
    TaskModelCheckpoint,
    UniEXLitModel,
    FastExtractModel,
    ExtractModel
)

class UniEXPipelines:
    @staticmethod
    def pipelines_args(parent_args):
        total_parser = parent_args.add_argument_group("piplines args")
        total_parser.add_argument(
            '--pretrained_model_path', default='', type=str)
        total_parser.add_argument('--output_path',
                                  default='./predict.json', type=str)

        total_parser.add_argument('--load_checkpoints_path',
                                  default='', type=str)

        total_parser.add_argument('--max_extract_entity_number',
                                  default=1, type=float)

        total_parser.add_argument('--train', action='store_true')
        
        total_parser.add_argument('--fast_ex_mode', action='store_true')

        total_parser.add_argument('--threshold_index',
                                  default=0.5, type=float)

        total_parser.add_argument('--threshold_entity',
                                  default=0.5, type=float)
        
        total_parser.add_argument('--threshold_event',
                                  default=0.5, type=float)

        total_parser.add_argument('--threshold_relation',
                                  default=0.5, type=float)

        total_parser = UniEXDataModel.add_data_specific_args(total_parser)
        total_parser = TaskModelCheckpoint.add_argparse_args(total_parser)
        total_parser = UniEXLitModel.add_model_specific_args(total_parser)
        total_parser = pl.Trainer.add_argparse_args(parent_args)
        return parent_args

    def __init__(self, args):

        if args.load_checkpoints_path != '':
            self.model = UniEXLitModel.load_from_checkpoint(
                args.load_checkpoints_path, args=args)
            print('导入模型成功：', args.load_checkpoints_path)
            
        else:
            self.model = UniEXLitModel(args)
            

        self.args = args
        self.checkpoint_callback = TaskModelCheckpoint(args).callbacks
        self.logger = loggers.TensorBoardLogger(save_dir=args.default_root_dir)
        self.trainer = pl.Trainer.from_argparse_args(args,
                                                     logger=self.logger,
                                                     callbacks=[self.checkpoint_callback])

        added_token = ['[unused'+str(i+1)+']' for i in range(10)]
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_path, is_split_into_words=True, add_prefix_space=True, additional_special_tokens=added_token)
        if args.fast_ex_mode:
            self.em = FastExtractModel(self.tokenizer, args)
        else:
            self.em = ExtractModel(self.tokenizer, args)

    def fit(self, train_data, dev_data,test_data=[]):
        data_model = UniEXDataModel(
            train_data, dev_data, self.tokenizer, self.args)
        self.model.num_data = len(train_data)
        self.model.dev_data = dev_data
        self.model.test_data = test_data
        self.trainer.fit(self.model, data_model)

    def predict(self, test_data, cuda=True):
        result = []
        start = 0
        if cuda:
            self.model = self.model.cuda()
        self.model.eval()
        while start < len(test_data):
            batch_data = test_data[start:start+self.args.batchsize]
            start += self.args.batchsize
            batch_result = self.em.extract(
                batch_data, self.model.model)
            result.extend(batch_result)
    
        return result
