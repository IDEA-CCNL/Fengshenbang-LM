# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""ToTTo: A Controlled Table-To-Text Generation Dataset"""


import json
from third_party.spider.preprocess.get_tables import dump_db_json_schema

import datasets

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{parikh2020totto,
  title={ToTTo: A Controlled Table-To-Text Generation Dataset},
  author={Parikh, Ankur P and Wang, Xuezhi and Gehrmann, Sebastian and Faruqui, Manaal and Dhingra, Bhuwan and Yang, Diyi and Das, Dipanjan},
  journal={arXiv preprint arXiv:2004.14373},
  year={2020}
"""

_DESCRIPTION = """\
ToTTo is an open-domain English table-to-text dataset with over 120,000 training examples that proposes a controlled generation task: given a Wikipedia table and a set of highlighted table cells, produce a one-sentence description.
"""

_HOMEPAGE = "https://github.com/google-research-datasets/ToTTo"

_LICENSE = "Creative Commons Share-Alike 3.0"

_URL = "https://storage.googleapis.com/totto-public/totto_data.zip"


class ToTTo(datasets.GeneratorBasedBuilder):
    """ToTTo dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "table_page_title": datasets.Value("string"),
                    "table_section_title": datasets.Value("string"),
                    "table": [
                        [
                            {
                                "column_span": datasets.Value("int32"),
                                "is_header": datasets.Value("bool"),
                                "row_span": datasets.Value("int32"),
                                "value": datasets.Value("string"),
                            }
                        ]
                    ],
                    "highlighted_cells": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("int32"))),
                    "final_sentences": datasets.features.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_filepath = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/totto_data/totto_train_data.jsonl",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/totto_data/totto_dev_data.jsonl",
                },
            ),
        ]

    def _generate_examples(self, data_filepath):
        with open(data_filepath, encoding="utf-8") as f:
            lines = [json.loads(line.strip()) for line in f.readlines()]
            for example_idx, example in enumerate(lines):
                yield example_idx, {
                    "table_page_title": example["table_page_title"],
                    "table_section_title": example["table_section_title"],
                    "table": example["table"],
                    "highlighted_cells": example["highlighted_cells"],
                    "final_sentences": [anno["final_sentence"] for anno in example["sentence_annotations"]],
                }
