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
"""GrailQA: The Strongly Generalizable Question Answering Dataset"""


import json
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{gu2021beyond,
  title={Beyond IID: three levels of generalization for question answering on knowledge bases},
  author={Gu, Yu and Kase, Sue and Vanni, Michelle and Sadler, Brian and Liang, Percy and Yan, Xifeng and Su, Yu},
  booktitle={Proceedings of the Web Conference 2021},
  pages={3477--3488},
  organization={ACM}
}
"""

_DESCRIPTION = """\
This dataset is a processed version of the official GrailQA Dataset release, where for each question the retrieved schema items of Freebase are integrated (according to the GitHub release https://github.com/dki-lab/GrailQA/tree/main/cache). Additionally, the original dev set is split into in-house dev and test sets.
"""

_HOMEPAGE = "https://dki-lab.github.io/GrailQA/"

_LICENSE = "CC BY-SA 4.0"

_URL = "https://nlp.stanford.edu/projects/kgqa/grailqa.zip"


class GrailQA(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="grailqa",
            version=VERSION,
            description="GrailQA",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

    def _info(self):
        features = datasets.Features(
            {
                    "qid": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.features.Sequence(
                        {
                            "answer_type": datasets.Value("string"),
                            "answer_argument": datasets.Value("string"),
                            "entity_name": datasets.Value("string"),
                        }
                    ),
                    "function": datasets.Value("string"),
                    "num_node": datasets.Value("int32"),
                    "num_edge": datasets.Value("int32"),
                    "graph_query": {
                        "nodes": datasets.features.Sequence(
                            {
                                "nid": datasets.Value("int32"),
                                "node_type": datasets.Value("string"),
                                "id": datasets.Value("string"),
                                "class": datasets.Value("string"),
                                "friendly_name": datasets.Value("string"),
                                "question_node": datasets.Value("int32"),
                                "function": datasets.Value("string"),
                            }
                        ),
                        "edges": datasets.features.Sequence(
                            {
                                "start": datasets.Value("int32"),
                                "end": datasets.Value("int32"),
                                "relation": datasets.Value("string"),
                                "friendly_name": datasets.Value("string"),
                            }
                        ),
                    },
                    "sparql_query": datasets.Value("string"),
                    "domains": datasets.features.Sequence(datasets.Value("string")),
                    "level": datasets.Value("string"),
                    "s_expression": datasets.Value("string"),
                    "retrieved_schema": datasets.features.Sequence(datasets.Value("string"))
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
                    "data_filepath": downloaded_filepath + "/grailqa/train.jsonl",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/grailqa/dev.jsonl",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/grailqa/test.jsonl",
                },
            ),
        ]

    def _generate_examples(self, data_filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", data_filepath)
        with open(data_filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                sample = json.loads(line)
                features = {
                    "qid": str(sample["qid"]),
                    "question": sample["question"],
                    "function": sample.get("function", ""),
                    "num_node": sample.get("num_node", -1),
                    "num_edge": sample.get("num_edge", -1),
                    "graph_query": sample.get("graph_query", {"nodes": [], "edges": []}),
                    "sparql_query": sample.get("sparql_query", ""),
                    "domains": sample.get("domains", []),
                    "level": sample.get("level", ""),
                    "s_expression": sample.get("s_expression", ""),
                    "retrieved_schema": sample.get("retrieved_schema", []),
                }
                answers = sample.get("answer", [])
                for answer in answers:
                    if "entity_name" not in answer:
                        answer["entity_name"] = ""

                features["answer"] = answers

                yield idx, features
