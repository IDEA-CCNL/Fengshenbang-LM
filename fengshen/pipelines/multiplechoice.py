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
from transformers import AutoConfig
from transformers.pipelines.base import Pipeline
import argparse
import copy
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
import warnings
from fengshen.models.unimc.modeling_unimc import (
    UniMCDataModel,
    UniMCLitModel,
    UniMCPredict,
)


class UniMCPipelines(Pipeline):
    @staticmethod
    def pipelines_args(parent_args):
        total_parser = parent_args.add_argument_group("piplines args")
        total_parser.add_argument(
            '--pretrained_model_path', default='', type=str)
        total_parser.add_argument('--load_checkpoints_path',
                                  default='', type=str)
        total_parser.add_argument('--train', action='store_true')
        total_parser.add_argument('--language',
                                  default='chinese', type=str)

        total_parser = UniMCDataModel.add_data_specific_args(total_parser)
        total_parser = UniversalCheckpoint.add_argparse_args(total_parser)
        total_parser = UniMCLitModel.add_model_specific_args(total_parser)
        total_parser = pl.Trainer.add_argparse_args(parent_args)
        return parent_args

    def __init__(self, args, model_path):
        self.args = args
        self.checkpoint_callback = UniversalCheckpoint(args)
        self.logger = loggers.TensorBoardLogger(save_dir=args.default_root_dir)
        self.trainer = pl.Trainer.from_argparse_args(args,
                                                     logger=self.logger,
                                                     callbacks=[self.checkpoint_callback])
        self.config = AutoConfig.from_pretrained(model_path)
        if self.config.model_type == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained(
                model_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                model_path)

        if args.language == 'chinese':
            self.yes_token = self.tokenizer.encode('是')[1]
            self.no_token = self.tokenizer.encode('非')[1]
        else:
            self.yes_token = self.tokenizer.encode('yes')[1]
            self.no_token = self.tokenizer.encode('no')[1]

        if args.load_checkpoints_path != '':
            self.model = UniMCLitModel.load_from_checkpoint(
                args.load_checkpoints_path, args=args, yes_token=self.yes_token, model_path=model_path)
            print('load model from: ', args.load_checkpoints_path)
        else:
            self.model = UniMCLitModel(
                args, yes_token=self.yes_token, model_path=model_path)

    def train(self, train_data, dev_data, process=True):
        if process:
            train_data = self.preprocess(train_data)
            dev_data = self.preprocess(dev_data)
        data_model = UniMCDataModel(
            train_data, dev_data, self.yes_token, self.no_token, self.tokenizer, self.args)
        self.model.num_data = len(train_data)
        self.trainer.fit(self.model, data_model)

    def predict(self, test_data, cuda=True, process=True):
        if process:
            test_data = self.preprocess(test_data)

        result = []
        start = 0
        if cuda:
            self.model = self.model.cuda()
        self.model.model.eval()
        predict_model = UniMCPredict(
            self.yes_token, self.no_token, self.model, self.tokenizer, self.args)
        while start < len(test_data):
            batch_data = test_data[start:start+self.args.batchsize]
            start += self.args.batchsize
            batch_result = predict_model.predict(batch_data)
            result.extend(batch_result)
        if process:
            result = self.postprocess(result)
        return result

    def preprocess(self, data):

        for i, line in enumerate(data):
            if 'task_type' in line.keys() and line['task_type'] == '语义匹配':
                data[i]['choice'] = ['不能理解为：'+data[i]
                                     ['textb'], '可以理解为：'+data[i]['textb']]
                # data[i]['question']='怎么理解这段话？'
                data[i]['textb'] = ''

            if 'task_type' in line.keys() and line['task_type'] == '自然语言推理':
                data[i]['choice'] = ['不能推断出：'+data[i]['textb'],
                                     '很难推断出：'+data[i]['textb'], '可以推断出：'+data[i]['textb']]
                # data[i]['question']='根据这段话'
                data[i]['textb'] = ''

        return data

    def postprocess(self, data):
        for i, line in enumerate(data):
            if 'task_type' in line.keys() and line['task_type'] == '语义匹配':
                data[i]['textb'] = data[i]['choice'][0].replace('不能理解为：', '')
                data[i]['choice'] = ['不相似', '相似']
                ns = {}
                for k, v in data[i]['score'].items():
                    if '不能' in k:
                        k = '不相似'
                    if '可以' in k:
                        k = '相似'
                    ns[k] = v
                data[i]['score'] = ns
                data[i]['answer'] = data[i]['choice'][data[i]['label']]

            if 'task_type' in line.keys() and line['task_type'] == '自然语言推理':
                data[i]['textb'] = data[i]['choice'][0].replace('不能推断出：', '')
                data[i]['choice'] = ['矛盾', '自然', '蕴含']
                ns = {}
                for k, v in data[i]['score'].items():
                    if '不能' in k:
                        k = '矛盾'
                    if '很难' in k:
                        k = '自然'
                    if '可以' in k:
                        k = '蕴含'
                    ns[k] = v
                data[i]['score'] = ns
                data[i]['answer'] = data[i]['choice'][data[i]['label']]

        return data

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def _sanitize_parameters(self, return_all_scores=None, function_to_apply=None, top_k="", **tokenizer_kwargs):
        # Using "" as default argument because we're going to use `top_k=None` in user code to declare
        # "No top_k"
        preprocess_params = tokenizer_kwargs

        postprocess_params = {}
        if hasattr(self.model.config, "return_all_scores") and return_all_scores is None:
            return_all_scores = self.model.config.return_all_scores

        if isinstance(top_k, int) or top_k is None:
            postprocess_params["top_k"] = top_k
            postprocess_params["_legacy"] = False
        elif return_all_scores is not None:
            warnings.warn(
                "`return_all_scores` is now deprecated,  if want a similar funcionality use `top_k=None` instead of"
                " `return_all_scores=True` or `top_k=1` instead of `return_all_scores=False`.",
                UserWarning,
            )
            if return_all_scores:
                postprocess_params["top_k"] = None
            else:
                postprocess_params["top_k"] = 1

        if function_to_apply is not None:
            postprocess_params["function_to_apply"] = function_to_apply
        return preprocess_params, {}, postprocess_params
