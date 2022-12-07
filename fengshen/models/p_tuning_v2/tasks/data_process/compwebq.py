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
"""ComplexWebQuestions Dataset"""


import json
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{talmor2018web,
  title={The web as a knowledge-base for answering complex questions},
  author={Talmor, Alon and Berant, Jonathan},
  booktitle={North American Chapter of the Association for Computational Linguistics (NAACL)},
  year={2018}
}
"""

_DESCRIPTION = """\
This dataset is a processed version of the official ComplexWebQuestions Dataset release, where for each question a relevant subset of knowledge graph (a set of tuples) is retrieved from Freebase. Additionally, the original dev set is split into in-house dev and test sets.
"""

_HOMEPAGE = "https://www.tau-nlp.org/compwebq"

_LICENSE = "CC BY-SA 4.0"

_URL = "https://nlp.stanford.edu/projects/kgqa/compwebq.zip"


class ComplexWebQuestions(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="compwebq",
            version=VERSION,
            description="ComplexWebQuestions Dataset",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answers": datasets.features.Sequence(datasets.Value("string")),
                "kg_tuples": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_filepath = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/compwebq/train.jsonl",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/compwebq/dev.jsonl",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/compwebq/test.jsonl",
                },
            ),
        ]

    def _generate_examples(self, data_filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", data_filepath)
        with open(data_filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                ex = json.loads(line)
                yield idx, {
                    "id": ex["ID"],
                    "question": ex["question"],
                    "answers": ex["answers"],
                    "kg_tuples": ex["KG_tuples"],
                }
