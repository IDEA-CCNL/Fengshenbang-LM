"""BART Style dataset. Modified from fairseq."""

import numpy as np
import torch
import math

from fengshen.data.megatron_dataloader.dataset_utils import (
    get_samples_mapping
)


class BartDataset(torch.utils.data.Dataset):
    def __init__(self, name, indexed_dataset, data_prefix,
                 num_epochs, max_num_samples, masked_lm_prob,
                 max_seq_length, short_seq_prob, seed, tokenizer, zh_tokenizer):

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length

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
                                                   False)

        # Vocab stuff.
        self.vocab_size = tokenizer.vocab_size
        inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
        self.vocab_id_list = list(inv_vocab.keys())
        self.vocab_id_to_token_dict = inv_vocab
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer

        seg_tokens = ['。', ';', '；', '!', '！', '?', '？']
        seg_token_ids = []
        for t in seg_tokens:
            if t in tokenizer.vocab:
                seg_token_ids.append(tokenizer.vocab[t])
            else:
                print('seg_token "{}" not in vocab'.format(t))
        self.seg_token_ids = set(seg_token_ids)

        self.zh_tokenizer = zh_tokenizer

        # Denoising ratios
        self.permute_sentence_ratio = 1.0
        self.mask_ratio = masked_lm_prob  # 0.15
        self.random_ratio = 0.1
        self.insert_ratio = 0.0
        self.rotate_ratio = 0.0
        self.mask_whole_word = 1
        self.item_transform_func = None

        self.mask_span_distribution = None
        if False:
            _lambda = 3  # Poisson lambda

            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2**32 since numpy requres the seed to be between 0 and 2**32 - 1
        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2**32))
        return self.build_training_sample(sample, self.max_seq_length, np_rng)

    def build_training_sample(self, sample, max_seq_length, np_rng):
        """Biuld training sample.

        Arguments:
            sample: A list of sentences in which each sentence is a list token ids.
            max_seq_length: Desired sequence length.
            np_rng: Random number genenrator. Note that this rng state should be
                numpy and not python since python randint is inclusive for
                the opper bound whereas the numpy one is exclusive.
        """
        # permute sentences
        full_stops = []
        tokens = [self.cls_id]
        for sent in sample:
            for t in sent:
                tokens.append(t)
                if t in self.seg_token_ids:
                    tokens.append(self.sep_id)
            if tokens[-1] != self.sep_id:
                tokens.append(self.sep_id)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            tokens[-1] = self.sep_id
        tokens = torch.LongTensor(tokens)
        full_stops = (tokens == self.sep_id).long()
        assert (max_seq_length - tokens.shape[0]) >= 0, (tokens.size(), tokens[-1], max_seq_length)

        source, target = tokens, tokens[1:].clone()
        use_decoder = 1
        # if torch.rand(1).item() < 0.5:
        #     use_decoder = 0

        if self.permute_sentence_ratio > 0.0 and use_decoder == 1:
            source = self.permute_sentences(source, full_stops, self.permute_sentence_ratio)

        if self.mask_ratio > 0.0:
            replace_length = 1 if use_decoder else -1
            mask_ratio = self.mask_ratio * 2 if use_decoder else self.mask_ratio
            source = self.add_whole_word_mask(source, mask_ratio, replace_length)

        if self.insert_ratio > 0.0:
            raise NotImplementedError
            source = self.add_insertion_noise(source, self.insert_ratio)

        if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
            raise NotImplementedError
            source = self.add_rolling_noise(source)

        # there can additional changes to make:
        if self.item_transform_func is not None:
            source, target = self.item_transform_func(source, target)

        assert (source >= 0).all()
        # assert (source[1:-1] >= 1).all()
        assert (source <= self.vocab_size).all()
        assert source[0] == self.cls_id
        assert source[-1] == self.sep_id

        # tokenizer = get_tokenizer()
        # print(' '.join(tokenizer.tokenizer.convert_ids_to_tokens(source)))
        # print(tokenizer.detokenize(target))
        # print(tokenizer.detokenize(source))
        # print()

        prev_output_tokens = torch.zeros_like(target)
        prev_output_tokens[0] = self.sep_id  # match the preprocessing in fairseq
        prev_output_tokens[1:] = target[:-1]

        # src_padding_length = max_seq_length - source.shape[0]
        # tgt_padding_length = max_seq_length - target.shape[0]
        # assert src_padding_length >= 0, (source.size(), source[-1], max_seq_length)
        # assert tgt_padding_length >= 0, (target.size(), target[-1], max_seq_length)
        source_ = torch.full((max_seq_length,), self.pad_id, dtype=torch.long)
        source_[:source.shape[0]] = source
        target_ = torch.full((max_seq_length,), -100, dtype=torch.long)
        # decoder not need bos in the front
        target_[:target.shape[0]] = target
        prev_output_tokens_ = torch.full((max_seq_length,), self.pad_id, dtype=torch.long)
        prev_output_tokens_[:prev_output_tokens.shape[0]] = prev_output_tokens

        return {
            "input_ids": source_,
            "labels": target_,
            # "decoder_input_ids": prev_output_tokens_,
            "attention_mask": (source_ != self.pad_id).long()
        }

    def permute_sentences(self, source, full_stops, p=1.0):
        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero(as_tuple=False) + 2
        result = source.clone()

        num_sentences = sentence_ends.size(0)
        num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
        substitutions = torch.randperm(num_sentences)[:num_to_permute]
        ordering = torch.arange(0, num_sentences)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        # Ignore <bos> at start
        index = 1
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i > 0 else 1): sentence_ends[i]]
            result[index: index + sentence.size(0)] = sentence
            index += sentence.size(0)
        return result

    def word_starts_en(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    def word_starts(self, source):
        if self.mask_whole_word is None:
            is_word_start = torch.ones(source.size())
            is_word_start[0] = 0
            is_word_start[-1] = 0
            return is_word_start
        raw_tokens = [self.vocab_id_to_token_dict[i] for i in source.tolist()]
        words = [raw_tokens[0]] + \
            self.zh_tokenizer(''.join(raw_tokens[1:-1]), HMM=True) + [raw_tokens[-1]]

        def _is_chinese_char(c):
            """Checks whether CP is the #codepoint of a CJK character."""
            # This defines a "chinese character" as anything in the CJK Unicode block:
            #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
            #
            # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
            # despite its name. The modern Korean Hangul alphabet is a different block,
            # as is Japanese Hiragana and Katakana. Those alphabets are used to write
            # space-separated words, so they are not treated specially and handled
            # like the all of the other languages.
            if len(c) > 1:
                return all([_is_chinese_char(c_i) for c_i in c])
            cp = ord(c)
            if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                    (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
                return True

            return False

        def align_linear(atokens, btokens):
            a2c = []
            c2b = []
            a2b = []
            length = 0
            for tok in atokens:
                a2c.append([length + i for i in range(len(tok))])
                length += len(tok)
            for i, tok in enumerate(btokens):
                c2b.extend([i for _ in range(len(tok))])

            for i, amap in enumerate(a2c):
                bmap = [c2b[ci] for ci in amap]
                a2b.append(list(set(bmap)))
            return a2b

        raw_to_word_align = align_linear(raw_tokens, words)
        is_word_start = torch.zeros(source.size())
        word_starts = []
        skip_cur_word = True
        for i in range(1, len(raw_to_word_align)):
            if raw_to_word_align[i-1] == raw_to_word_align[i]:
                # not a word start, as they align to the same word
                if not skip_cur_word and not _is_chinese_char(raw_tokens[i]):
                    word_starts.pop(-1)
                    skip_cur_word = True
                continue
            else:
                is_word_start[i] = 1
                if _is_chinese_char(raw_tokens[i]):
                    word_starts.append(i)
                    skip_cur_word = False
        is_word_start[0] = 0
        is_word_start[-1] = 0
        word_starts = torch.tensor(word_starts).long().view(-1, 1)
        return is_word_start, word_starts

    def add_whole_word_mask(self, source, p, replace_length=1):
        is_word_start, word_starts = self.word_starts(source)
        num_to_mask_word = int(math.ceil(word_starts.size(0) * p))
        num_to_mask_char = int(math.ceil(word_starts.size(0) * p * 0.1))
        num_to_mask = num_to_mask_word + num_to_mask_char
        if num_to_mask > word_starts.size(0):
            word_starts = is_word_start.nonzero(as_tuple=False)
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio
        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            # print(source.size(), word_starts.size(), indices.size(), mask_random.size())
            source[indices] = self.mask_id
            source[indices[mask_random]] = torch.randint(
                1, self.vocab_size, size=(mask_random.sum(),)
            )
            # sorted_indices = torch.sort(indices)[0]
            # continue_mask_pos = ((sorted_indices + 1)[:-1] == sorted_indices[1:])
            # continue_mask_indices = sorted_indices[1:][continue_mask_pos]
            # to_keep[continue_mask_indices] = 0

        # for char indices, we already masked, the following loop handles word mask
        indices = indices[:num_to_mask_word]
        mask_random = mask_random[:num_to_mask_word]
        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_id
                    source[indices[mask_random]] = torch.randint(
                        1, self.vocab_size, size=(mask_random.sum(),)
                    )
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_id
                    source[indices[mask_random]] = torch.randint(
                        1, self.vocab_size, size=(mask_random.sum(),)
                    )

                assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def add_permuted_noise(self, tokens, p):
        num_words = len(tokens)
        num_to_permute = math.ceil(((num_words * 2) * p) / 2.0)
        substitutions = torch.randperm(num_words - 2)[:num_to_permute] + 1
        tokens[substitutions] = tokens[substitutions[torch.randperm(num_to_permute)]]
        return tokens

    def add_rolling_noise(self, tokens):
        offset = np.random.randint(1, max(1, tokens.size(-1) - 1) + 1)
        tokens = torch.cat(
            (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),
            dim=0,
        )
        return tokens

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_id
        result[noise_indices[:num_random]] = torch.randint(
            low=1, high=self.vocab_size, size=(num_random,)
        )

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result
