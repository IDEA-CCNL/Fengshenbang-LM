# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors.
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
# TODO： This code can be push to HuggingFace as a new contribution
#  (parse the table ino header and rows, add hardness, and deprecate the blind test since nobody uses it).
"""TabFact: A Large-scale Dataset for Table-based Fact Verification"""

import json
import os

import datasets

_CITATION = """\
@inproceedings{2019TabFactA,
  title={TabFact : A Large-scale Dataset for Table-based Fact Verification},
  author={Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou and William Yang Wang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  address = {Addis Ababa, Ethiopia},
  month = {April},
  year = {2020}
}
"""

_DESCRIPTION = """\
The problem of verifying whether a textual hypothesis holds the truth based on the given evidence, \
also known as fact verification, plays an important role in the study of natural language \
understanding and semantic representation. However, existing studies are restricted to \
dealing with unstructured textual evidence (e.g., sentences and passages, a pool of passages), \
while verification using structured forms of evidence, such as tables, graphs, and databases, remains unexplored. \
TABFACT is large scale dataset with 16k Wikipedia tables as evidence for 118k human annotated statements \
designed for fact verification with semi-structured evidence. \
The statements are labeled as either ENTAILED or REFUTED. \
TABFACT is challenging since it involves both soft linguistic reasoning and hard symbolic reasoning.
"""

_HOMEPAGE = "https://tabfact.github.io/"

_GIT_ARCHIVE_URL = (
    "https://github.com/wenhuchen/Table-Fact-Checking/archive/948b5560e2f7f8c9139bd91c7f093346a2bb56a8.zip"
)


class TabFact(datasets.GeneratorBasedBuilder):
    """TabFact: A Large-scale Dataset for Table-based Fact Verification"""

    VERSION = datasets.Version("1.0.0")

    # TODO commented by Tianbao: We deprecate the "blind_test" for now, since nobody uses it.

    def _info(self):
        features = {
            "id": datasets.Value("int32"),
            "table": {
                "id": datasets.Value("string"),
                "header": datasets.features.Sequence(datasets.Value("string")),
                "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
                "caption": datasets.Value("string"),
            },
            "statement": datasets.Value("string"),
            "label": datasets.Value("int32"),
            "hardness": datasets.Value("string"),
            "small_test": datasets.Value("bool")
        }

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.download_and_extract(_GIT_ARCHIVE_URL)

        repo_path = os.path.join(extracted_path, "Table-Fact-Checking-948b5560e2f7f8c9139bd91c7f093346a2bb56a8")
        all_csv_path = os.path.join(repo_path, "data", "all_csv")

        train_statements_file = os.path.join(repo_path, "tokenized_data", "train_examples.json")
        val_statements_file = os.path.join(repo_path, "tokenized_data", "val_examples.json")
        test_statements_file = os.path.join(repo_path, "tokenized_data", "test_examples.json")

        info_path = os.path.join(repo_path, "data")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"statements_file": train_statements_file, "all_csv_path": all_csv_path,
                            "info_path": info_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"statements_file": val_statements_file, "all_csv_path": all_csv_path,
                            "info_path": info_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"statements_file": test_statements_file, "all_csv_path": all_csv_path,
                            "info_path": info_path},
            ),
        ]

    def _generate_examples(self, statements_file, all_csv_path, info_path):

        def __convert_to_table(table_str):
            table = {
                "header": table_str.split('\n')[0].split('#'),
                "rows": [row_str.split('#') for row_str in table_str.split('\n')[1:-1]]
            }
            return table

        def __construct_hardness_dict(path):
            hardness_dict = {}
            with open(os.path.join(path, "simple_ids.json"), "r") as f:
                simple_table_list = json.load(f)
                for simple_table_id in simple_table_list:
                    hardness_dict[simple_table_id] = "simple"

            with open(os.path.join(path, "complex_ids.json"), "r") as f:
                complex_table_list = json.load(f)
                for complex_table_id in complex_table_list:
                    hardness_dict[complex_table_id] = "complex"

            with open(os.path.join(path, "all_csv_ids.json"), "r") as f:
                all_csv_list = json.load(f)
                assert len(all_csv_list) == len(hardness_dict)
                # We assert that each table either be a simple or complex one.

            return hardness_dict

        def __construct_small_test_dict(path):
            small_test_dict = {}
            with open(os.path.join(path, "all_csv_ids.json"), "r") as f:
                all_csv_list = json.load(f)

            with open(os.path.join(path, "small_test_id.json"), "r") as f:
                small_test_ids = json.load(f)

            for csv_str in all_csv_list:
                small_test_dict[csv_str] = False

            for small_test_id in small_test_ids:
                assert small_test_id in small_test_dict.keys()
                small_test_dict[small_test_id] = True

            return small_test_dict

        with open(statements_file, encoding="utf-8") as f:
            examples = json.load(f)

        hardness_dict = __construct_hardness_dict(info_path)
        small_test_dict = __construct_small_test_dict(info_path)

        for i, (table_id, example) in enumerate(examples.items()):
            table_file_path = os.path.join(all_csv_path, table_id)
            with open(table_file_path, encoding="utf-8") as f:
                table_text = f.read()

            statements, labels, caption = example

            for statement_idx, (statement, label) in enumerate(zip(statements, labels)):
                parsed_table = __convert_to_table(table_text)
                yield f"{i}_{statement_idx}", {
                    "id": i,
                    "table": {
                        "id": table_id,
                        "header": parsed_table['header'],
                        "rows": parsed_table['rows'],
                        "caption": caption
                    },
                    "statement": statement,
                    "label": label,
                    "hardness": hardness_dict[table_id],
                    "small_test": small_test_dict[table_id]
                }
