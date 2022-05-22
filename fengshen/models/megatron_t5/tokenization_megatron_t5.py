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
""" T5Tokenizer """

from transformers import BertTokenizer


class T5Tokenizer():
    def __init__(self, extra_id_num=118):
        self.extra_id_num = extra_id_num

    @classmethod
    def from_pretrained(self, vocab_path):
        self.extra_id_num = 118
        self.T5_special_tokens = ['[BOS]', '[EOS]']
        for i in range(self.extra_id_num):
            self.T5_special_tokens.append(f'<extra_id_{str(i)}>')
        tokenizer = BertTokenizer.from_pretrained(vocab_path, additional_special_tokens=self.T5_special_tokens)

        return tokenizer
