# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import logging
from transformers import (
    Wav2Vec2FeatureExtractor,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoTokenizer
)
import numpy as np
from fairseq.data import FairseqDataset

logger = logging.getLogger(__name__)


def add_data_specific_args(parent_args):
    parser = parent_args.add_argument_group('Wav2vec2 Dataset')
    parser.add_argument('--data', type=str)
    parser.add_argument('--sample_rate', type=float, default=16000)
    parser.add_argument('--min_sample_size', type=int)
    parser.add_argument('--max_sample_size', type=int)
    parser.add_argument('--max_tokens', type=int, default=1400000)
    parser.add_argument('--required_batch_size_multiple', type=int, default=8)
    parser.add_argument('--labels', type=str, nargs='+')
    parser.add_argument('--enable_padding', type=bool)
    parser.add_argument('--normalize', type=bool)
    parser.add_argument('--padding', type=str)
    parser.add_argument('--datatype', type=str, default="librispeech")
    return parent_args


# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: AutoProcessor
    tokenizer: AutoTokenizer
    feature_extractor: AutoFeatureExtractor
    max_sample_size: int
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        for item in features:
            sample = item["audio"]

            inputs = self.feature_extractor(
                sample["array"], sampling_rate=sample["sampling_rate"],
                max_length=self.max_sample_size, truncation=True
            )
            item["input_values"] = inputs.input_values[0]
            item["input_length"] = len(inputs.input_values[0])

            # encode targets

            item["labels"] = self.tokenizer(item["text"]).input_ids

        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


class CTCDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        lable_path,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        feature_extractor: Wav2Vec2FeatureExtractor,
        padding: Union[bool, str] = "longest",
        pad_to_multiple_of: Optional[int] = None,
        pad_to_multiple_of_labels: Optional[int] = None,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        max_tokens=1400000,
    ):
        super().__init__()
        self.collater_outside = DataCollatorCTCWithPadding(
            processor=processor,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            max_sample_size=max_sample_size,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            pad_to_multiple_of_labels=pad_to_multiple_of_labels
        )
        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize
        skipped = 0
        self.fnames = []
        self.texts = []
        sizes = []
        self.skipped_indices = set()
        self.sample_upper_bound = max_tokens//2
        # necessary for fairseq fixed token size sampler
        with open(manifest_path, "r") as f:
            with open(lable_path, "r") as g:
                self.root_dir = f.readline().strip()
                for i, (line, label) in enumerate(zip(f, g)):
                    items = line.strip().split("\t")
                    assert len(items) == 2, "{}".format(line)
                    sz = int(items[1])
                    if min_sample_size is not None and sz < min_sample_size:
                        skipped += 1
                        self.skipped_indices.add(i)
                        continue

                    if sz > self.sample_upper_bound:
                        skipped += 1
                        self.skipped_indices.add(i)
                        continue
                    self.fnames.append(items[0].encode())
                    sizes.append(sz)
                    label = label.strip("\n")
                    # label = label.strip("\n").replace(" ", "")
                    self.texts.append(label.encode())

        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
            self.texts = pyarrow.array(self.texts)
        except Exception:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

    def __getitem__(self, index):
        """
            example:
            {'chapter_id': 141231,
            'file': '/home/...flac',
            'audio': {'path': '/home/...flac',
            'array': array([-0.00048828, -0.00018311, -0.00137329, ...,  0.00079346,
                    0.00091553,  0.00085449], dtype=float32),
            'sampling_rate': 16000},
            'id': '1272-141231-0000',
            'speaker_id': 1272,
            }
            we ignore the 'text' item, which is different from the huggingface librispeech dataset
        """

        import soundfile as sf
        fn = self.fnames[index]
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        fn = fn.decode()
        text = self.texts[index]
        text = text if isinstance(self.texts, list) else text.as_py()
        text = text.decode()

        path_or_fp = os.path.join(self.root_dir, fn)

        wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")
        result = {
            'file': path_or_fp,
            'audio': {
                'path': path_or_fp,
                'array': wav,
                'sampling_rate': curr_sample_rate
            },
            'id': fn,
            'text': text
        }

        # feats = self.postprocess(feats, curr_sample_rate)
        return result

    def __len__(self):
        return len(self.sizes)

    def collater(self, samples):
        return self.collater_outside(samples)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def num_tokens(self, index):
        return self.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            order.append(
                np.minimum(
                    np.array(self.sizes),
                    self.max_sample_size,
                )
            )
            return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))
