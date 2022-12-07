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
"""CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases"""


import json
from third_party.spider.preprocess.get_tables import dump_db_json_schema

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{yu-etal-2019-cosql,
    title = "{C}o{SQL}: A Conversational Text-to-{SQL} Challenge Towards Cross-Domain Natural Language Interfaces to Databases",
    author = "Yu, Tao  and
      Zhang, Rui  and
      Er, Heyang  and
      Li, Suyi  and
      Xue, Eric  and
      Pang, Bo  and
      Lin, Xi Victoria  and
      Tan, Yi Chern  and
      Shi, Tianze  and
      Li, Zihan  and
      Jiang, Youxuan  and
      Yasunaga, Michihiro  and
      Shim, Sungrok  and
      Chen, Tao  and
      Fabbri, Alexander  and
      Li, Zifan  and
      Chen, Luyao  and
      Zhang, Yuwen  and
      Dixit, Shreya  and
      Zhang, Vincent  and
      Xiong, Caiming  and
      Socher, Richard  and
      Lasecki, Walter  and
      Radev, Dragomir",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1204",
    doi = "10.18653/v1/D19-1204",
    pages = "1962--1979",
    abstract = "We present CoSQL, a corpus for building cross-domain, general-purpose database (DB) querying dialogue systems. It consists of 30k+ turns plus 10k+ annotated SQL queries, obtained from a Wizard-of-Oz (WOZ) collection of 3k dialogues querying 200 complex DBs spanning 138 domains. Each dialogue simulates a real-world DB query scenario with a crowd worker as a user exploring the DB and a SQL expert retrieving answers with SQL, clarifying ambiguous questions, or otherwise informing of unanswerable questions. When user questions are answerable by SQL, the expert describes the SQL and execution results to the user, hence maintaining a natural interaction flow. CoSQL introduces new challenges compared to existing task-oriented dialogue datasets: (1) the dialogue states are grounded in SQL, a domain-independent executable representation, instead of domain-specific slot value pairs, and (2) because testing is done on unseen databases, success requires generalizing to new domains. CoSQL includes three tasks: SQL-grounded dialogue state tracking, response generation from query results, and user dialogue act prediction. We evaluate a set of strong baselines for each task and show that CoSQL presents significant challenges for future research. The dataset, baselines, and leaderboard will be released at https://yale-lily.github.io/cosql.",
}
"""

_DESCRIPTION = """\
CoSQL is a large-scale dataset for training and testing task oriented dialog agents with SQL
"""

_HOMEPAGE = "https://yale-lily.github.io/cosql"

_LICENSE = "CC BY-SA 4.0"

_URL = "https://drive.google.com/uc?export=download&confirm=9iBg&id=14x6lsWqlu6gR-aYxa6cemslDN3qT3zxP"


class CoSQL(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="cosql",
            version=VERSION,
            description="A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

    def _info(self):
        features = datasets.Features(
            {
                "query": datasets.Value("string"),
                "utterances": datasets.features.Sequence(datasets.Value("string")),
                "turn_idx": datasets.Value("int32"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
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
                    "data_filepath": downloaded_filepath + "/cosql_dataset/sql_state_tracking/cosql_train.json",
                    "db_path": downloaded_filepath + "/cosql_dataset/database",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/cosql_dataset/sql_state_tracking/cosql_dev.json",
                    "db_path": downloaded_filepath + "/cosql_dataset/database",
                },
            ),
        ]

    def _generate_examples(self, data_filepath, db_path):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", data_filepath)
        with open(data_filepath, encoding="utf-8") as f:
            cosql = json.load(f)
            # The picard code use the datasets 1.5.0, while we use the 1.11.0
            # so we modify the idx of the _generate_examples to avoid making
            # datasets.keyhash.DuplicatedKeysError

            # TODO: we can remind Torsten to give it a pull request to huggingface datasets package if he like.

            idx = 0
            for sample in cosql:
                db_id = sample["database_id"]
                if db_id not in self.schema_cache:
                    self.schema_cache[db_id] = dump_db_json_schema(
                        db_path + "/" + db_id + "/" + db_id + ".sqlite", db_id
                    )
                schema = self.schema_cache[db_id]

                db_stuff = {
                    "db_id": db_id,
                    "db_path": db_path,
                    "db_table_names": schema["table_names_original"],
                    "db_column_names": [
                        {"table_id": table_id, "column_name": column_name}
                        for table_id, column_name in schema["column_names_original"]
                    ],
                    "db_column_types": schema["column_types"],
                    "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                    "db_foreign_keys": [
                        {"column_id": column_id, "other_column_id": other_column_id}
                        for column_id, other_column_id in schema["foreign_keys"]
                    ],
                }

                yield idx, {
                    "utterances": [sample["final"]["utterance"]],
                    "query": sample["final"]["query"],
                    "turn_idx": -1,
                    **db_stuff,
                }
                idx += 1
                utterances = []
                for turn_idx, turn in enumerate(sample["interaction"]):
                    utterances.extend((utterance.strip() for utterance in turn["utterance"].split(sep="|")))
                    yield idx, {
                        "utterances": list(utterances),
                        "query": turn["query"],
                        "turn_idx": turn_idx,
                        **db_stuff,
                    }
                    idx += 1
