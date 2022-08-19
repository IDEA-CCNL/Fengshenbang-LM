# coding: utf-8
# Copyright 2019 Sinovation Ventures AI Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""utils for ngram for ZEN model."""

import os
import logging

from transformers import cached_path

NGRAM_DICT_NAME = 'ngram.txt'

logger = logging.getLogger(__name__)
PRETRAINED_VOCAB_ARCHIVE_MAP = {'IDEA-CCNL/Erlangshen-ZEN1-224M-Chinese': 'https://huggingface.co/IDEA-CCNL/Erlangshen-ZEN1-224M-Chinese/resolve/main/ngram.txt'}


class ZenNgramDict(object):
    """
    Dict class to store the ngram
    """

    def __init__(self, ngram_freq_path, tokenizer, max_ngram_in_seq=128):
        """Constructs ZenNgramDict

        :param ngram_freq_path: ngrams with frequency
        """
        if os.path.isdir(ngram_freq_path):
            ngram_freq_path = os.path.join(ngram_freq_path, NGRAM_DICT_NAME)
        self.ngram_freq_path = ngram_freq_path
        self.max_ngram_in_seq = max_ngram_in_seq
        self.id_to_ngram_list = ["[pad]"]
        self.ngram_to_id_dict = {"[pad]": 0}
        self.ngram_to_freq_dict = {}

        logger.info("loading ngram frequency file {}".format(ngram_freq_path))
        with open(ngram_freq_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                ngram, freq = line.split(",")
                tokens = tuple(tokenizer.tokenize(ngram))
                self.ngram_to_freq_dict[ngram] = freq
                self.id_to_ngram_list.append(tokens)
                self.ngram_to_id_dict[tokens] = i + 1

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            ngram_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
            if '-cased' in pretrained_model_name_or_path and kwargs.get('do_lower_case', True):
                logger.warning("The pre-trained model you are loading is a cased model but you have not set "
                               "`do_lower_case` to False. We are setting `do_lower_case=False` for you but "
                               "you may want to check this behavior.")
                kwargs['do_lower_case'] = False
            elif '-cased' not in pretrained_model_name_or_path and not kwargs.get('do_lower_case', True):
                logger.warning("The pre-trained model you are loading is an uncased model but you have set "
                               "`do_lower_case` to False. We are setting `do_lower_case=True` for you "
                               "but you may want to check this behavior.")
                kwargs['do_lower_case'] = True
        else:
            ngram_file = pretrained_model_name_or_path
        if os.path.isdir(ngram_file):
            ngram_file = os.path.join(ngram_file, NGRAM_DICT_NAME)
        # redirect to the cache, if necessary
        try:
            resolved_ngram_file = cached_path(ngram_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
                logger.error(
                    "Couldn't reach server at '{}' to download vocabulary.".format(
                        ngram_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                        ngram_file))
            return None
        if resolved_ngram_file == ngram_file:
            logger.info("loading vocabulary file {}".format(ngram_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                ngram_file, resolved_ngram_file))
        # Instantiate ngram.
        ngram_dict = cls(resolved_ngram_file, **kwargs)
        return ngram_dict

    def save(self, ngram_freq_path):
        with open(ngram_freq_path, "w", encoding="utf-8") as fout:
            for ngram, freq in self.ngram_to_freq_dict.items():
                fout.write("{},{}\n".format(ngram, freq))
