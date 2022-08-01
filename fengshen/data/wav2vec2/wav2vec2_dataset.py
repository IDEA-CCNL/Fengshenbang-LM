# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import logging
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
import numpy as np

from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel


logger = logging.getLogger(__name__)


def add_data_specific_args(parent_args):
    parser = parent_args.add_argument_group('Wav2vec2 Dataset')
    parser.add_argument('--data', type=str)
    parser.add_argument('--sample_rate', type=float, default=16000)
    parser.add_argument('--min_sample_size', type=int)
    parser.add_argument('--max_sample_size', type=int)
    parser.add_argument('--labels', type=str, nargs='+')
    parser.add_argument('--enable_padding', type=bool)
    parser.add_argument('--normalize', type=bool)
    parser.add_argument('--padding', type=str)
    parser.add_argument('--datatype', type=str, default="librispeech")
    return parent_args


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`,
            :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`,
            defaults to :obj:`True`
        ):
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
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    model: Wav2Vec2ForPreTraining
    args: argparse.Namespace
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        # print(features)

        for item in features:
            sample = item["audio"]

            inputs = self.feature_extractor(
                sample["array"], sampling_rate=sample["sampling_rate"],
                max_length=self.args.max_sample_size, truncation=True
            )
            item["input_values"] = inputs.input_values[0]
            item["input_length"] = len(inputs.input_values[0])

        features = [{"input_values": item["input_values"]} for item in features]

        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.model.config.mask_time_prob,
            self.model.config.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Wav2vec2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        collater_outside,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        text_compression_level=TextCompressionLevel.none,
    ):
        super().__init__()
        self.collater_outside = collater_outside
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
        self.text_compressor = TextCompressor(level=text_compression_level)
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, "{}".format(line)
                sz = int(items[1])
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(self.text_compressor.compress(items[0]))
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
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
        fn = self.text_compressor.decompress(fn)
        path_or_fp = os.path.join(self.root_dir, fn)
        _path, slice_ptr = parse_path(path_or_fp)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            assert is_sf_audio_data(byte_data)
            path_or_fp = io.BytesIO(byte_data)

        if path_or_fp.split(".")[-1] == "npy":
            wav = np.load(path_or_fp)
            curr_sample_rate = int((path_or_fp.split("_")[-1]).split(".")[0])
        else:
            wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")
        # feats = torch.from_numpy(wav).float()
        fn_split = os.path.splitext(os.path.basename(fn))[0].split('-')
        result = {
            'chapter_id': int(fn_split[1]),
            'file': path_or_fp,
            'audio': {
                'path': path_or_fp,
                'array': wav,
                'sampling_rate': curr_sample_rate
            },
            'id': fn,
            'speaker_id': int(fn_split[0]),
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
