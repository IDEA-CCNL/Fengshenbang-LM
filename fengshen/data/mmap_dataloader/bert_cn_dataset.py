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

import collections

import numpy as np

from fengshen import get_tokenizer
from fengshen.data.mmap_dataloader.bert_dataset import BertDataset
from fengshen.data.mmap_dataloader.dataset_utils import (
    create_tokens_and_tokentypes,
    get_a_and_b_segments,
    get_samples_mapping,
    truncate_segments,
)


class BertCNDataset(BertDataset):
    def __init__(
        self,
        name,
        indexed_dataset,
        data_prefix,
        num_epochs,
        max_num_samples,
        masked_lm_prob,
        max_seq_length,
        short_seq_prob,
        seed,
        binary_head,
        masking_style="bert",
    ):
        # 增加masking_style为bert,bert-cn-wwm
        self.masking_style = masking_style
        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.binary_head = binary_head

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.
        self.samples_mapping = get_samples_mapping(
            self.indexed_dataset,
            data_prefix,
            num_epochs,
            max_num_samples,
            self.max_seq_length - 3,  # account for added tokens
            short_seq_prob,
            self.seed,
            self.name,
            self.binary_head,
        )

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.cls_id = tokenizer.cls
        self.sep_id = tokenizer.sep
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2**32 since numpy requres the seed to be between 0 and 2**32 - 1
        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2 ** 32))
        return build_training_sample(
            sample,
            seq_length,
            self.max_seq_length,  # needed for padding
            self.vocab_id_list,
            self.vocab_id_to_token_dict,
            self.cls_id,
            self.sep_id,
            self.mask_id,
            self.pad_id,
            self.masked_lm_prob,
            np_rng,
            self.binary_head,
            masking_style=self.masking_style,
        )


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def build_training_sample(
    sample,
    target_seq_length,
    max_seq_length,
    vocab_id_list,
    vocab_id_to_token_dict,
    cls_id,
    sep_id,
    mask_id,
    pad_id,
    masked_lm_prob,
    np_rng,
    binary_head,
    masking_style="bert",
):
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

    if binary_head:
        # We assume that we have at least two sentences in the sample
        assert len(sample) > 1
    assert target_seq_length <= max_seq_length

    # Divide sample into two segments (A and B).
    if binary_head:
        tokens_a, tokens_b, is_next_random = get_a_and_b_segments(sample, np_rng)
    else:
        tokens_a = []
        for j in range(len(sample)):
            tokens_a.extend(sample[j])
        tokens_b = []
        is_next_random = False

    # Truncate to `target_sequence_length`.
    max_num_tokens = target_seq_length
    truncated = truncate_segments(
        tokens_a, tokens_b, len(tokens_a), len(tokens_b), max_num_tokens, np_rng
    )

    # Build tokens and toketypes.
    tokens, tokentypes = create_tokens_and_tokentypes(
        tokens_a, tokens_b, cls_id, sep_id
    )

    # Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    (
        tokens,
        masked_positions,
        masked_labels,
        _,
        _,
    ) = create_masked_lm_predictions_chinese(
        tokens,
        vocab_id_list,
        vocab_id_to_token_dict,
        masked_lm_prob,
        cls_id,
        sep_id,
        mask_id,
        max_predictions_per_seq,
        np_rng,
        masking_style=masking_style,
    )

    # Padding.
    (
        tokens_np,
        tokentypes_np,
        labels_np,
        padding_mask_np,
        loss_mask_np,
    ) = pad_and_convert_to_numpy(
        tokens, tokentypes, masked_positions, masked_labels, pad_id, max_seq_length
    )

    train_sample = {
        "text": tokens_np,
        "types": tokentypes_np,
        "labels": labels_np,
        "is_random": int(is_next_random),
        "loss_mask": loss_mask_np,
        "padding_mask": padding_mask_np,
        "truncated": int(truncated),
    }
    return train_sample


def pad_and_convert_to_numpy(
    tokens, tokentypes, masked_positions, masked_labels, pad_id, max_seq_length
):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(tokentypes) == num_tokens
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)
    tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length, dtype=np.int64)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np


def create_masked_lm_predictions_chinese(
    tokens,
    vocab_id_list,
    vocab_id_to_token_dict,
    masked_lm_prob,
    cls_id,
    sep_id,
    mask_id,
    max_predictions_per_seq,
    np_rng,
    max_ngrams=3,
    do_whole_word_mask=True,
    favor_longer_ngram=False,
    do_permutation=False,
    geometric_dist=False,
    masking_style="bert",
):
    """Creates the predictions for the masked LM objective.

    Note: Tokens here are vocab ids and not text tokens.
    对中文的WWM做特殊的处理，中文的substr id为原字id+len(vocab)
    """

    cand_indexes = []  # 用来记录候选的mask的token的id
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)  # 1表示是piece开始的位置

    def _is_cn_substr(token_id):
        return token_id > len(vocab_id_list)

    def _is_en_substr(token_str):
        return token_str.startswith("##")

    valid_tokens = []  # 把中文的token转化成正常的token
    for (i, token) in enumerate(tokens):
        if token == cls_id or token == sep_id:
            token_boundary[i] = 1
            valid_tokens.append(token)
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.

        # if (do_whole_word_mask and len(cand_indexes) >= 1 and
        #         not is_start_piece(vocab_id_to_token_dict[token])):
        #     cand_indexes[-1].append(i)
        # else:
        #     cand_indexes.append([i])
        #     if is_start_piece(vocab_id_to_token_dict[token]):
        #         token_boundary[i] = 1
        if do_whole_word_mask and len(cand_indexes) >= 1:
            if _is_cn_substr(token):
                cand_indexes[-1].append(i)
                valid_tokens.append(token - len(vocab_id_list))  # 中文substr减去vocab_size
            elif _is_en_substr(vocab_id_to_token_dict[token]):
                cand_indexes[-1].append(i)
                valid_tokens.append(token)  # 英文substr保持原有的token
            else:
                cand_indexes.append([i])
                valid_tokens.append(token)
                token_boundary[i] = 1
        else:
            cand_indexes.append([i])
            if _is_cn_substr(token):
                valid_tokens.append(token - len(vocab_id_list))  # 中文substr减去vocab_size
            elif _is_en_substr(vocab_id_to_token_dict[token]):
                valid_tokens.append(token)  # 英文substr保持原有的token
            else:
                valid_tokens.append(token)
                token_boundary[i] = 1

    assert len(valid_tokens) == len(tokens)
    output_tokens = list(valid_tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary)

    num_to_predict = min(
        max_predictions_per_seq, max(1, int(round(len(valid_tokens) * masked_lm_prob)))
    )

    ngrams = np.arange(1, max_ngrams + 1, dtype=np.int64)
    if not geometric_dist:
        # Note(mingdachen):
        # By default, we set the probilities to favor shorter ngram sequences.
        pvals = 1.0 / np.arange(1, max_ngrams + 1)
        pvals /= pvals.sum(keepdims=True)
        if favor_longer_ngram:
            pvals = pvals[::-1]
    # 获取一个ngram的idx，对于每个word，记录他的ngram的word
    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx : idx + n])
        ngram_indexes.append(ngram_index)

    np_rng.shuffle(ngram_indexes)

    (masked_lms, masked_spans) = ([], [])
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        if not geometric_dist:
            n = np_rng.choice(
                ngrams[: len(cand_index_set)],
                p=pvals[: len(cand_index_set)]
                / pvals[: len(cand_index_set)].sum(keepdims=True),
            )
        else:
            # Sampling "n" from the geometric distribution and clipping it to
            # the max_ngrams. Using p=0.2 default from the SpanBERT paper
            # https://arxiv.org/pdf/1907.10529.pdf (Sec 3.1)
            n = min(np_rng.geometric(0.2), max_ngrams)

        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            if masking_style == "bert":
                # 80% of the time, replace with [MASK]
                if np_rng.random() < 0.8:
                    masked_token = mask_id
                else:
                    # 10% of the time, keep original
                    if np_rng.random() < 0.5:
                        masked_token = valid_tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_id_list[
                            np_rng.randint(0, len(vocab_id_list))
                        ]
            elif masking_style == "t5":
                masked_token = mask_id
            else:
                raise ValueError("invalid value of masking style")

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=valid_tokens[index]))

        masked_spans.append(
            MaskedLmInstance(
                index=index_set, label=[valid_tokens[index] for index in index_set]
            )
        )

    assert len(masked_lms) <= num_to_predict
    np_rng.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(
                ngrams[: len(cand_index_set)],
                p=pvals[: len(cand_index_set)]
                / pvals[: len(cand_index_set)].sum(keepdims=True),
            )
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        np_rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    # Sort the spans by the index of the first span
    masked_spans = sorted(masked_spans, key=lambda x: x.index[0])

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (
        output_tokens,
        masked_lm_positions,
        masked_lm_labels,
        token_boundary,
        masked_spans,
    )
