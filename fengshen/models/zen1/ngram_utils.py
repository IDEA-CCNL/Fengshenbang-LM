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

NGRAM_DICT_NAME = 'ngram.txt'

logger = logging.getLogger(__name__)


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

    def save(self, ngram_freq_path):
        with open(ngram_freq_path, "w", encoding="utf-8") as fout:
            for ngram, freq in self.ngram_to_freq_dict.items():
                fout.write("{},{}\n".format(ngram, freq))
