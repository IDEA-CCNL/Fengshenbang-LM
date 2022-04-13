# coding=utf-8
# Copyright 2022 IDEA-CCNL and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for TransfoXLDenoise."""

import sentencepiece as spm
from transformers.tokenization_utils import PreTrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "transformer-xl-1b-base":
            "https://huggingface.co/IDEA-CCNL/Transformer-XL-denoise-1.1B/resolve/main/spiece.model",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "transformer-xl-1b-base": 512,
}


class TransfoXLDenoiseTokenizer(PreTrainedTokenizer):
    """
    Construct a TransfoXLDenoise tokenizer. Based on pretrained sentence piece

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    SPIECE_UNDERLINE = "▁"

    def __init__(
            self,
            vocab_file,
            unk_token="<|endoftext|>",
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            **kwargs
    ):
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)
        "Initialisation"
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        "Returns vocab size"
        return len(self.sp_model)

    def _tokenize(self, text):
        """ Returns a tokenized string. """
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = "".join(tokens).replace(self.SPIECE_UNDERLINE, " ").strip()
        return out_string
