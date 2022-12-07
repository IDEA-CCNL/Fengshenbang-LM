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
import random
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
# _DEV_FILE = "/nvme/hardenhuang/fight_for_something/HGN/data/csqa/merged_data/dev.3views.jsonl"
# _TRAINING_FILE = "/nvme/hardenhuang/fight_for_something/HGN/data/csqa/merged_data/train.3views.jsonl"
#_TEST_FILE = "fetaQA-v1_test.jsonl"
# _TEST_FILE = "/cognitive_comp/huangyongfeng/fight_for_something/graph-soft-counter/data/obqa/statement/dev.twoviews_fact_statement.jsonl"
# _TRAINING_FILE = "/cognitive_comp/huangyongfeng/fight_for_something/graph-soft-counter/data/obqa/statement/train.twoviews_fact_statement.jsonl"
# _DEV_FILE = "/cognitive_comp/huangyongfeng/fight_for_something/graph-soft-counter/data/obqa/statement/test.twoviews_fact_statement.jsonl"

_TEST_FILE = "/cognitive_comp/huangyongfeng/fight_for_something/GreaseLM/data/medqa_usmle/statement/dev.twoview_triples.jsonl"
_TRAINING_FILE = "/cognitive_comp/huangyongfeng/fight_for_something/GreaseLM/data/medqa_usmle/statement/train.twoview_triples.jsonl"
_DEV_FILE = "/cognitive_comp/huangyongfeng/fight_for_something/GreaseLM/data/medqa_usmle/statement/test.twoview_triples.jsonl"
# few_shot_rate = 0.3
# _URLS = {
#     "train": f"{_URL}{_TRAINING_FILE}",
#     "dev": f"{_URL}{_DEV_FILE}",
#     "test": f"{_URL}{_TEST_FILE}",
# }
label2id = {'A':0, 'B':1, 'C':2, 'D':3}#, 'E':4}

class MEDQADataset(datasets.GeneratorBasedBuilder):
    """The FETAQA dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question_choice_A": datasets.Value("string"),
                    "question_choice_B": datasets.Value("string"),
                    "question_choice_C": datasets.Value("string"),
                    "question_choice_D": datasets.Value("string"),
                    # "question_choice_E": datasets.Value("string"),
                    "triplets_a": {"individual_view_triplets": datasets.Value("string"), 
                                # "latent_view_triplets":  datasets.Value("string"),
                                # "group_view_triplets": datasets.Value("string"),
                                "retri_view_triplets": datasets.Value("string"),
                                "selected_individual_view_triplets": datasets.Value("string"),
                                # "meaning_view_triplets": datasets.Value("string")
                                },
                    "triplets_b": {"individual_view_triplets": datasets.Value("string"), 
                                # "latent_view_triplets":  datasets.Value("string"),
                                # "group_view_triplets": datasets.Value("string"),
                                "retri_view_triplets": datasets.Value("string"),
                                "selected_individual_view_triplets": datasets.Value("string"),
                                # "meaning_view_triplets": datasets.Value("string")
                                },
                    "triplets_c": {"individual_view_triplets": datasets.Value("string"), 
                                # "latent_view_triplets":  datasets.Value("string"),
                                # "group_view_triplets": datasets.Value("string"),
                                "retri_view_triplets": datasets.Value("string"),
                                "selected_individual_view_triplets": datasets.Value("string"),
                                # "meaning_view_triplets": datasets.Value("string")
                                },
                    "triplets_d": {"individual_view_triplets": datasets.Value("string"), 
                                # "latent_view_triplets":  datasets.Value("string"),
                                # "group_view_triplets": datasets.Value("string"),
                                "retri_view_triplets": datasets.Value("string"),
                                "selected_individual_view_triplets": datasets.Value("string"),
                                # "meaning_view_triplets": datasets.Value("string")
                                },
                    # "triplets_e": {"individual_view_triplets": datasets.Value("string"), 
                    #             "latent_view_triplets":  datasets.Value("string"),
                    #             "group_view_triplets": datasets.Value("string"),
                    #             "retri_view_triplets": datasets.Value("string"),
                    #             "selected_individual_view_triplets": datasets.Value("string"),
                    #             "meaning_view_triplets": datasets.Value("string")},
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
          datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": _TEST_FILE}),
        #   datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            examples = json.load(f)
            #这里来控制一下few shot learning的
            # if filepath == _TRAINING_FILE:
            #     random.shuffle(examples)
            #     examples = examples[0:int(len(examples)*few_shot_rate)]
            for idx, example in enumerate(examples):            
                # line = f[0]
                #pdb.set_trace()
                # example = json.loads(line)
                # pdb.set_trace()
                # for idx, example in enumerate(examples[0:100]):
                # pdb.set_trace()
                yield idx, {
                    "id": example["id"],
                    "question_choice_A": example['question']['stem'] + ' ' + example['question']['choices'][0]['text'],
                    "question_choice_B": example['question']['stem'] + ' ' + example['question']['choices'][1]['text'],
                    "question_choice_C": example['question']['stem'] + ' ' + example['question']['choices'][2]['text'],
                    "question_choice_D": example['question']['stem'] + ' ' + example['question']['choices'][3]['text'],
                    # "question_choice_E": example['question']['stem'] + ' ' + example['question']['choices'][4]['text'],
                    "labels": label2id[example["answerKey"]],
                    "triplets_a": {'individual_view_triplets': example["statements"][0]['triplets'],
                                    # 'latent_view_triplets': '[SEP]'.join(example["statements"][0]['latent_view_triplets'][0:2]),
                                    # 'group_view_triplets': '[SEP]'.join(example["statements"][0]['group_view_triplets']),
                                    'retri_view_triplets': example['statements'][0]['retri_triplets'],
                                    'selected_individual_view_triplets': example["statements"][0]['triplets'],
                                    # 'meaning_view_triplets': example["statements"][0]['meaning_view_triplets']
                                    },
                    "triplets_b": {'individual_view_triplets': example["statements"][1]['triplets'],
                                    # 'latent_view_triplets': '[SEP]'.join(example["statements"][0]['latent_view_triplets'][0:2]),
                                    # 'group_view_triplets': '[SEP]'.join(example["statements"][0]['group_view_triplets']),
                                    'retri_view_triplets': example['statements'][1]['retri_triplets'],
                                    'selected_individual_view_triplets': example["statements"][1]['triplets'],
                                    # 'meaning_view_triplets': example["statements"][0]['meaning_view_triplets']
                                    },
                    "triplets_c": {'individual_view_triplets': example["statements"][2]['triplets'],
                                    # 'latent_view_triplets': '[SEP]'.join(example["statements"][0]['latent_view_triplets'][0:2]),
                                    # 'group_view_triplets': '[SEP]'.join(example["statements"][0]['group_view_triplets']),
                                    'retri_view_triplets': example['statements'][2]['retri_triplets'],
                                    'selected_individual_view_triplets': example["statements"][2]['triplets'],
                                    # 'meaning_view_triplets': example["statements"][0]['meaning_view_triplets']
                                    },
                    "triplets_d": {'individual_view_triplets': example["statements"][3]['triplets'],
                                    # 'latent_view_triplets': '[SEP]'.join(example["statements"][0]['latent_view_triplets'][0:2]),
                                    # 'group_view_triplets': '[SEP]'.join(example["statements"][0]['group_view_triplets']),
                                    'retri_view_triplets': example['statements'][3]['retri_triplets'],
                                    'selected_individual_view_triplets': example["statements"][3]['triplets'],
                                    # 'meaning_view_triplets': example["statements"][0]['meaning_view_triplets']
                                    },
                    # "triplets_e": {'individual_view_triplets': '[SEP]'.join(example["statements"][4]['individual_view_triplets']),
                    #                 'latent_view_triplets': '[SEP]'.join(example["statements"][4]['latent_view_triplets'][0:2]),
                    #                 'group_view_triplets': '[SEP]'.join(example["statements"][4]['group_view_triplets']),
                    #                 'retri_view_triplets': '[SEP]'.join(example["statements"][4]['retri_view_triplets']),
                    #                 'selected_individual_view_triplets': example["statements"][4]['selected_individual_view_triplets'],
                    #                 'meaning_view_triplets': example["statements"][4]['meaning_view_triplets']},
                }
# >>> import json
# >>> f=open("dev.3views.jsonl", encoding='utf-8')
# >>> xx = [json.loads(line) for line in f]
# >>> xx[0].keys()
# dict_keys(['answerKey', 'id', 'question', 'statements'])
# >>> xx[0]['question'].keys()
# dict_keys(['question_concept', 'choices', 'stem'])
# >>> xx[0]['statements'].keys()
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# AttributeError: 'list' object has no attribute 'keys'
# >>> xx[0]['statements'][0].keys()
# dict_keys(['label', 'statement', 'individual_view_triplets', 'latent_view_triplets', 'group_view_triplets'])
# >>> xx[0]['question']['choices']
# [{'label': 'A', 'text': 'bank'}, {'label': 'B', 'text': 'library'}, {'label': 'C', 'text': 'department store'}, {'label': 'D', 'text': 'mall'}, {'label': 'E', 'text': 'new york'}]
# >>>