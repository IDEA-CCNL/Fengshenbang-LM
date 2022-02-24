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

from typing import TYPE_CHECKING

from transformers.file_utils import _LazyModule, is_torch_available


_import_structure = {
    "auto_factory": ["get_values"],
    "configuration_auto": ["ALL_PRETRAINED_CONFIG_ARCHIVE_MAP", "CONFIG_MAPPING", "MODEL_NAMES_MAPPING", "AutoConfig"],
    "tokenization_auto": ["TOKENIZER_MAPPING", "AutoTokenizer"],
}

if is_torch_available():
    _import_structure["modeling_auto"] = [
        "AutoModel",
        "AutoModelForMaskedLM",
        "AutoModelForMultipleChoice",
        "AutoModelForPreTraining",
        "AutoModelForQuestionAnswering",
        "AutoModelForSequenceClassification",
        "AutoModelForTokenClassification",
    ]

if TYPE_CHECKING:
    from .auto_factory import get_values
    from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, CONFIG_MAPPING, MODEL_NAMES_MAPPING, AutoConfig
    from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
    if is_torch_available():
        from .modeling_auto import (
            AutoModel,
            AutoModelForMaskedLM,
            AutoModelForMultipleChoice,
            AutoModelForPreTraining,
            AutoModelForQuestionAnswering,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
