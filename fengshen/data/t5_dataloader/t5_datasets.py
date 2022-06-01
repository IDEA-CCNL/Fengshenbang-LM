# coding=utf8
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, MT5Config, MT5Tokenizer, BatchEncoding
import torch
import pytorch_lightning as pl
import numpy as np
from itertools import chain
import sys
sys.path.append('../../')


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/
    text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(
        tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


class UnsuperviseT5Dataset(Dataset):
    '''
    Dataset Used for T5 unsuprvise pretrain.
    load_data_type = 0: load raw data from data path and save tokenized data, call function load_data
    load_data_type = 1: load tokenized data from path, call function load_tokenized_data
    load_data_type = 2: load tokenized data from memery data, call function load_tokenized_memory_data
    '''

    def __init__(self, data_path, args, load_data_type=0, data=None):
        super().__init__()

        if args.tokenizer_type == 't5_tokenizer':
            if args.new_vocab_path is not None:
                self.tokenizer = MT5Tokenizer.from_pretrained(args.new_vocab_path)
            else:
                self.tokenizer = MT5Tokenizer.from_pretrained(args.pretrained_model_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
        self.noise_density = 0.15
        self.mean_noise_span_length = 3
        self.text_column_name = args.text_column_name
        self.preprocessing_num_workers = args.preprocessing_num_workers
        self.max_seq_length = args.max_seq_length
        self.remove_columns = args.remove_columns
        # whether load tokenieze data
        self.load_data_type = load_data_type

        if self.load_data_type == 0:
            # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
            # To ensure that the input length is `max_seq_length`, we need to increase the maximum length
            # according to `mlm_probability` and `mean_noise_span_length`.
            # We can also define the label length accordingly.
            self.expanded_inputs_length, self.targets_length = compute_input_and_target_lengths(
                inputs_length=self.max_seq_length,
                noise_density=self.noise_density,
                mean_noise_span_length=self.mean_noise_span_length,
            )
            print('self.expanded_inputs_length, self.targets_length:{},{}'.format(
                self.expanded_inputs_length, self.targets_length))
            self.data = self.load_data(data_path)
        elif self.load_data_type == 1:
            self.data = self.load_tokenized_data(data_path)
        else:
            assert data is not None
            self.data = self.load_tokenized_memory_data(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_data(self, data_path):
        # TODO: large data process
        from data.fs_datasets import load_dataset
        samples = load_dataset(
            # samples = datasets.load_from_disk(data_path)['train']
            data_path, num_proc=self.preprocessing_num_workers)['train']
        # print(samples)
        tokenized_datasets = samples.map(
            self.tokenize_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            # load_from_cache_file=not data_args.overwrite_cache,
        ).map(
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=self.remove_columns)
        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            # load_from_cache_file=not data_args.overwrite_cache,
        )
        return tokenized_datasets
    '''
        The function load tokenized data saved from load_data function.
    '''

    def load_tokenized_data(self, data_path):
        from data.fs_datasets import load_dataset
        samples = load_dataset(data_path)['train']
        return samples

    def load_tokenized_memory_data(self, data):
        return data

    # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # Since we make sure that all sequences are of the same length, no attention_mask is needed.
    def tokenize_function(self, examples):
        # 这里add_special_tokens=False，避免句子中间出现eos
        return self.tokenizer(examples[self.text_column_name],
                              add_special_tokens=False,
                              return_attention_mask=False)

    # Main data processing function that will concatenate all texts from our dataset
    # and generate chunks of expanded_inputs_length.
    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.expanded_inputs_length:
            total_length = (
                total_length // self.expanded_inputs_length) * self.expanded_inputs_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.expanded_inputs_length]
                for i in range(0, total_length, self.expanded_inputs_length)]
            for k, t in concatenated_examples.items()
        }
        return result


class UnsuperviseT5DataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('UnsuperviseT5DataModel')
        parser.add_argument('--preprocessing_num_workers',
                            default=30, type=int)
        parser.add_argument(
            '--train_data_path', default='wudao_180g_mt5_tokenized', type=str)
        parser.add_argument('--train_batchsize', default=2, type=int)
        parser.add_argument('--valid_batchsize', default=2, type=int)
        parser.add_argument('--train_split_size', default=None, type=float)
        parser.add_argument('--tokenizer_type', default='t5_tokenizer', choices=['t5_tokenizer', 'bert_tokenizer'])
        parser.add_argument('--text_column_name', default='text')
        parser.add_argument('--remove_columns', nargs='+', default=[])
        return parent_args

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize
        self.preprocessing_num_workers = args.preprocessing_num_workers

        if args.train_split_size is not None:
            from data.fs_datasets import load_dataset
            data_splits = load_dataset(args.train_data_path, num_proc=args.preprocessing_num_workers)
            train_split = data_splits['train']
            test_split = data_splits['test']
            print('train:', train_split, '\ntest_data:', test_split)
            self.train_data = UnsuperviseT5Dataset(
                args.train_data_path, args, load_data_type=2, data=train_split)
            self.test_data = UnsuperviseT5Dataset(
                args.train_data_path, args, load_data_type=2, data=test_split)
        else:
            self.train_data = UnsuperviseT5Dataset(
                args.train_data_path, args, load_data_type=1)

        self.config = MT5Config.from_pretrained(args.pretrained_model_path)
        self.noise_density = 0.15
        self.mean_noise_span_length = 3
        self.pad_token_id = self.config.pad_token_id
        self.decoder_start_token_id = self.config.decoder_start_token_id
        self.eos_token_id = self.config.eos_token_id
        self.vocab_size = self.config.vocab_size
        self.max_seq_length = args.max_seq_length
        # 因为加载旧的spm里面已经包括了exrta_ids，但是T5Tokenizer会在spm的基础上再增加100个extra_ids,所以需要指定extra_ids=0
        if args.tokenizer_type == 't5_tokenizer' and args.new_vocab_path is not None:
            self.tokenizer = MT5Tokenizer.from_pretrained(args.new_vocab_path, extra_ids=0)
            # 如果是刚开始加载mt5,需要更新vocab_size为提取中英词之后的new_vocab_size
            self.vocab_size = len(self.tokenizer)

        # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
        # To ensure that the input length is `max_seq_length`, we need to increase the maximum length
        # according to `mlm_probability` and `mean_noise_span_length`. We can also define the label length accordingly.
        self.expanded_inputs_length, self.targets_length = compute_input_and_target_lengths(
            inputs_length=self.max_seq_length,
            noise_density=self.noise_density,
            mean_noise_span_length=self.mean_noise_span_length,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=self.train_batchsize,
            pin_memory=False,
            num_workers=self.preprocessing_num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_data,
            shuffle=False,
            batch_size=self.valid_batchsize,
            pin_memory=False,
            num_workers=self.preprocessing_num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_data,
            shuffle=False,
            batch_size=self.valid_batchsize,
            pin_memory=False,
            num_workers=self.preprocessing_num_workers,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, examples):
        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {k: np.array([examples[i][k] for i in range(len(examples))])
             for k, v in examples[0].items()}
        )

        input_ids = np.array(batch['input_ids'])
        batch_size, expanded_input_length = input_ids.shape
        mask_indices = np.asarray([self.random_spans_noise_mask(
            expanded_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(
            mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(
            input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.max_seq_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is \
                    {batch['input_ids'].shape[-1]}, but should be {self.targets_length}."
            )

        if batch["labels"].shape[-1] != self.targets_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is \
                    {batch['labels'].shape[-1]}, but should be {self.targets_length}."
            )

        batch["decoder_input_ids"] = self.shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        for k, v in batch.items():
            batch[k] = torch.tensor(v)
            # print(k, batch[k], self.tokenizer.batch_decode(batch[k]), '\n', flush=True)
        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - \
            np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(
            start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(
            sentinel_ids != 0, (self.vocab_size - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >=
                                   0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    # Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
    def shift_tokens_right(self, input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = np.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        shifted_input_ids[:, 0] = decoder_start_token_id

        shifted_input_ids = np.where(
            shifted_input_ids == -100, pad_token_id, shifted_input_ids)
        return shifted_input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/
        blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(
            np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(
            num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths],
                     axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]
