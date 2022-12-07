# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors, The Google AI Language Team Authors and the current dataset script contributor.
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
# TODO： This code can be push to HuggingFace as a new contribution.
"""FeTaQA, a Free-form Table Question Answering dataset"""
import ast
import csv
import os
import json
import pdb
import datasets

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{talmor2018commonsenseqa,
  title={Commonsenseqa: A question answering challenge targeting commonsense knowledge},
  author={Talmor, Alon and Herzig, Jonathan and Lourie, Nicholas and Berant, Jonathan},
  journal={arXiv preprint arXiv:1811.00937},
  year={2018}
}
"""

_DESCRIPTION = """
"""

_HOMEPAGE = ""

_LICENSE = "CC-BY-SA-4.0 License"

_URL = "https://github.com/Yale-LILY/FeTaQA/raw/main/data/"
_DEV_FILE = "/home/hardenhuang/fight_for_something/data_processes/dev_twoviews_csq.json"
_TRAINING_FILE = "/home/hardenhuang/fight_for_something/data_processes/train_twoviews_csq.json"
#_TEST_FILE = "fetaQA-v1_test.jsonl"

# _URLS = {
#     "train": f"{_URL}{_TRAINING_FILE}",
#     "dev": f"{_URL}{_DEV_FILE}",
#     "test": f"{_URL}{_TEST_FILE}",
# }


class CSQA(datasets.GeneratorBasedBuilder):
    """The FETAQA dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "gen_nle": datasets.Value("string"),
                    "gen_nle_gold": datasets.Value("string"),
                    "retri_nle_a": {"triples_temp": datasets.Value("string"), 
                                                            "is_freq_masked": datasets.Value("int32"),
                                                            "qc_meaning":  datasets.Value("string"),
                                                            "ac_meaning": datasets.Value("string"),
                                                            "question_text": datasets.Value("string"),
                                                            "retrieval": datasets.Value("string"),
                                                            "retri_nle_choice": datasets.Value("string")},
                    "retri_nle_b": {"triples_temp": datasets.Value("string"), 
                                                            "is_freq_masked": datasets.Value("int32"),
                                                            "qc_meaning":  datasets.Value("string"),
                                                            "ac_meaning": datasets.Value("string"),
                                                            "question_text": datasets.Value("string"),
                                                            "retrieval": datasets.Value("string"),
                                                            "retri_nle_choice": datasets.Value("string")},
                    "retri_nle_c": {"triples_temp": datasets.Value("string"), 
                                                            "is_freq_masked": datasets.Value("int32"),
                                                            "qc_meaning":  datasets.Value("string"),
                                                            "ac_meaning": datasets.Value("string"),
                                                            "question_text": datasets.Value("string"),
                                                            "retrieval": datasets.Value("string"),
                                                            "retri_nle_choice": datasets.Value("string")},
                    "retri_nle_d": {"triples_temp": datasets.Value("string"), 
                                                            "is_freq_masked": datasets.Value("int32"),
                                                            "qc_meaning":  datasets.Value("string"),
                                                            "ac_meaning": datasets.Value("string"),
                                                            "question_text": datasets.Value("string"),
                                                            "retrieval": datasets.Value("string"),
                                                            "retri_nle_choice": datasets.Value("string")},
                    "retri_nle_e": {"triples_temp": datasets.Value("string"), 
                                                            "is_freq_masked": datasets.Value("int32"),
                                                            "qc_meaning":  datasets.Value("string"),
                                                            "ac_meaning": datasets.Value("string"),
                                                            "question_text": datasets.Value("string"),
                                                            "retrieval": datasets.Value("string"),
                                                            "retri_nle_choice": datasets.Value("string")},
                    "cands": {"A": datasets.Value("string"),
                            "B": datasets.Value("string"),
                            "C": datasets.Value("string"),
                            "D": datasets.Value("string"),
                            "E": datasets.Value("string")},
                    "labels": datasets.Value("int32"),
                    # "question_concept": datasets.Value("string")
                } 
            ),
            supervised_keys=None,
            homepage=None,
            license=None,
            citation=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        #downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
          datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": _TRAINING_FILE}),
          datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": _DEV_FILE}),
        #   datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):            
                # line = f[0]
                #pdb.set_trace()
                example = json.loads(line)
                #pdb.set_trace()
                # for idx, example in enumerate(examples[0:100]):
                # pdb.set_trace()
                yield idx, {
                    "id": example["q_no"],
                    "input": example["inp"],
                    "gen_nle": example["gen_nle"],
                    "gen_nle_gold": example["gen_nle_gold"],
                    "retri_nle_a": example["retri_nle"][0],
                    "retri_nle_b": example["retri_nle"][1],
                    "retri_nle_c": example["retri_nle"][2],
                    "retri_nle_d": example["retri_nle"][3],
                    "retri_nle_e": example["retri_nle"][4],#"[SEP]".join(example["retri_nle_choice"]),
                    "cands": {"A": example["cands"][0],
                            "B": example["cands"][1],
                            "C": example["cands"][2],
                            "D": example["cands"][3],
                            "E": example["cands"][4]},
                    "labels": example["label"],
                    # "question_concept": example["question_concept"]
                }
