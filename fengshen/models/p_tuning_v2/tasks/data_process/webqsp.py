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
"""WebQuestions Semantic Parses Dataset"""


import json
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{berant2013semantic,
  title={Semantic parsing on freebase from question-answer pairs},
  author={Berant, Jonathan and Chou, Andrew and Frostig, Roy and Liang, Percy},
  booktitle={Proceedings of the 2013 conference on empirical methods in natural language processing},
  pages={1533--1544},
  year={2013}
}
@inproceedings{yih2016value,
  title={The value of semantic parse labeling for knowledge base question answering},
  author={Yih, Wen-tau and Richardson, Matthew and Meek, Christopher and Chang, Ming-Wei and Suh, Jina},
  booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  pages={201--206},
  year={2016}
}
"""

_DESCRIPTION = """\
This dataset is a processed version of the official WebQuestions Semantic Parses Dataset release, where 1) for each question a relevant subset of knowledge graph (a set of tuples) is retrieved from Freebase, and 2) sparql logical form is converted to s-expression following https://github.com/salesforce/rng-kbqa.
"""

_HOMEPAGE = "https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering-2/"

_LICENSE = "CC BY-SA 4.0"

_URL = "https://nlp.stanford.edu/projects/kgqa/webqsp.zip"


class WebQSP(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="webqsp",
            version=VERSION,
            description="WebQuestions Semantic Parses Dataset",
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
                "answers": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
                "s_expression": datasets.Value("string"),
                "kg_tuples": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
                "entities": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
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
                    "data_filepath": downloaded_filepath + "/webqsp/train.jsonl",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/webqsp/dev.jsonl",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/webqsp/test.jsonl",
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
                    "s_expression": ex["s_expression"],
                    "kg_tuples": ex["kg_tuples"],
                    "entities": ex["entities"],
                }
