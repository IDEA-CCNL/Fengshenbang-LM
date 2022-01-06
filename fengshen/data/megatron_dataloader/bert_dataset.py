# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""BERT Style dataset."""


import numpy as np
import torch

from fengshen.data.megatron_dataloader.utils import (
    print_rank_0
)
from fengshen.data.megatron_dataloader.dataset_utils import (
    get_samples_mapping,
    get_a_and_b_segments,
    truncate_segments,
    create_masked_lm_predictions,
    get_indexed_dataset_,
    get_train_valid_test_split_
)
import jieba.analyse


tokenizer = None
ignore_labels = -100


class BertDataset(torch.utils.data.Dataset):

    def __init__(self, name, indexed_dataset, data_prefix,
                 num_epochs, max_num_samples, masked_lm_prob,
                 max_seq_length, short_seq_prob, seed, binary_head):
        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.short_seq_prob = short_seq_prob
        self.binary_head = binary_head

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.
        self.samples_mapping = get_samples_mapping(self.indexed_dataset,
                                                   data_prefix,
                                                   num_epochs,
                                                   max_num_samples,
                                                   self.max_seq_length - 3,  # account for added tokens
                                                   short_seq_prob,
                                                   self.seed,
                                                   self.name,
                                                   self.binary_head)
        print_rank_0(self.samples_mapping.size)

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2**32 since numpy requres the seed to be between 0 and 2**32 - 1
        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2 ** 32))

        return build_training_sample(sample,
                                     seq_length,
                                     np_rng, self.masked_lm_prob)


def truncate_segments(source_a, source_b, masked_labels_a, masked_labels_b, max_num_tokens, np_rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    rands = np.random.random()
    while len(source_a) + len(source_b) > max_num_tokens:
        #
        if len(source_a) > len(source_b):
            if rands < 0.5:
                source_a = source_a[1:]
                masked_labels_a = masked_labels_a[1:]
            else:
                source_a = source_a[:-1]
                masked_labels_a = masked_labels_a[:-1]
        else:
            if rands < 0.5:
                source_b = source_b[1:]
                masked_labels_b = masked_labels_b[1:]
            else:
                source_b = source_b[:-1]
                masked_labels_b = masked_labels_b[:-1]

    source = []
    tokentype = []
    source.append(tokenizer.cls_token_id)
    source.extend(source_a)
    source.append(tokenizer.sep_token_id)

    token_a_length = len(source)
    tokentype.extend([0] * token_a_length)
    source.extend(source_b)
    source.append(tokenizer.sep_token_id)
    tokentype.extend([1] * (len(source) - token_a_length))

    attention_mask = [1] * len(source)

    labels = []
    labels.append(ignore_labels)
    labels.extend(masked_labels_a)
    labels.append(ignore_labels)
    labels.extend(masked_labels_b)
    labels.append(ignore_labels)

    while len(source) < 512:
        source.append(0)
        tokentype.append(0)
        attention_mask.append(0)
        labels.append(ignore_labels)

    return source, tokentype, attention_mask, labels


def build_training_sample(sample,
                          target_seq_length,
                          np_rng, masked_lm_prob):
    """Biuld training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the opper bound whereas the numpy one is exclusive.
    """

    tokens_a, tokens_b, is_next_random = get_a_and_b_segments(sample,
                                                              np_rng)

    # Masking.
    source_a, masked_labels_a = create_masked_lm_predictions(
        tokens_a, masked_lm_prob=masked_lm_prob)
    source_b, masked_labels_b = create_masked_lm_predictions(
        tokens_b, masked_lm_prob=masked_lm_prob)

    sources, tokentype, attention_mask, labels = truncate_segments(source_a,
                                                                   source_b,
                                                                   masked_labels_a,
                                                                   masked_labels_b,
                                                                   target_seq_length,
                                                                   np_rng)
    # print('source',sources)

    train_sample = {
        'input_ids': torch.tensor(sources),
        'attention_mask': torch.tensor(attention_mask),
        'token_type_ids': torch.tensor(tokentype),
        'labels': torch.tensor(labels),
        'next_sentence_label': int(is_next_random)}
    # print(train_sample)
    # for i in range(120):
    #     print(labels[i],sources[i],tokenizer.decode([sources[i]]))

    return train_sample


def token_process(token_id):
    rand = np.random.random()
    if rand <= 0.8:
        return tokenizer.mask_token_id
    elif rand <= 0.9:
        return token_id
    else:
        return np.random.randint(1, len(tokenizer))


def word_segment(text):
    return list(jieba.cut(text))


def sentence_process(text, mask_rate):
    """单个文本的处理函数
    流程：分词，然后转id，按照mask_rate构建全词mask的序列
          来指定哪些token是否要被mask
    """
    max_ngram = 3
    ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_ngram + 1)
    pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
    word_list = word_segment(text)
    # print('word_list',word_list)

    mask_ids, labels = [], []

    record = []
    for i in range(len(word_list)):
        rands = np.random.random()
        if i in record:
            continue
        if rands > mask_rate:
            word = word_list[i]
            word_encode = tokenizer.encode(word, add_special_tokens=False)
            for token in word_encode:
                mask_ids.append(token)
                labels.append(ignore_labels)
            record.append(i)
        else:
            n = np.random.choice(ngrams, p=pvals)
            for index in range(n):
                ind = index + i
                if ind in record or ind >= len(word_list):
                    continue
                record.append(ind)
                word = word_list[ind]
                word_encode = tokenizer.encode(word, add_special_tokens=False)
                for token in word_encode:
                    mask_ids.append(token_process(token))
                    labels.append(token)

    return mask_ids, labels


def create_masked_lm_predictions(tokens, masked_lm_prob):
    """Creates the predictions for the masked LM objective."""

    text = ''
    for token in tokens:
        # print('token',token)
        token = tokenizer.decode([token])
        if token[:2] == '##':  # 去掉词干前缀
            token = token[2:]
        if token[0] == '[' and token[-1] == ']':  # 去掉token
            token = '“'
        text += token
    # print('text:',text)
    output_tokens, masked_lm_labels = sentence_process(text, mask_rate=0.15)

    return output_tokens, masked_lm_labels


def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                    train_valid_test_num_samples,
                                    max_seq_length,
                                    masked_lm_prob, short_seq_prob, seed,
                                    skip_warmup, binary_head):
    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    # Get start and end indices of train/valid/train into doc-idx
    # Note that doc-idx is desinged to be num-docs + 1 so we can
    # easily iterate over it.
    total_num_of_documents = indexed_dataset.doc_idx.shape[0] - 1

    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
        start_index = indexed_dataset.doc_idx[splits[index]]
        end_index = indexed_dataset.doc_idx[splits[index + 1]]
        print_rank_0('     sentence indices in [{}, {}) total of {} '
                     'sentences'.format(start_index, end_index,
                                        end_index - start_index))

    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    splits = splits[::-1]
    for idx, s in enumerate(splits):
        if idx == 1 or idx == 2:
            splits[idx] = splits[idx - 1] - 300
    splits = splits[::-1]

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            # Get the pointer to the original doc-idx so we can set it later.
            doc_idx_ptr = indexed_dataset.get_doc_idx()
            # Slice the doc-idx
            start_index = splits[index]
            # Add +1 so we can index into the dataset to get the upper bound.
            end_index = splits[index + 1] + 1
            # New doc_idx view.
            indexed_dataset.set_doc_idx(doc_idx_ptr[start_index:end_index])

            # Build the dataset accordingly.
            kwargs = dict(
                name=name,
                data_prefix=data_prefix,
                num_epochs=1,
                max_num_samples=None,
                max_seq_length=max_seq_length,
                seed=seed,
            )

            dataset = BertDataset(
                indexed_dataset=indexed_dataset,
                masked_lm_prob=masked_lm_prob,
                short_seq_prob=short_seq_prob,
                binary_head=binary_head,
                **kwargs
            )

            # Set the original pointer so dataset remains the main dataset.
            indexed_dataset.set_doc_idx(doc_idx_ptr)
            # Checks.
            assert indexed_dataset.doc_idx[0] == 0
            assert indexed_dataset.doc_idx.shape[0] == \
                (total_num_of_documents + 1)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')

    return train_dataset, valid_dataset
