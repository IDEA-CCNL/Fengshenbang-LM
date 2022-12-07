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
# TODO: This loader can be pushed to huggingface datasets as a new loader.
""" KVRET: A New Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset """

import json
import os
from collections import OrderedDict

import datasets

_CITATION = """\
@inproceedings{eric-etal-2017-key,
    title = "Key-Value Retrieval Networks for Task-Oriented Dialogue",
    author = "Eric, Mihail  and
      Krishnan, Lakshmi  and
      Charette, Francois  and
      Manning, Christopher D.",
    booktitle = "Proceedings of the 18th Annual {SIG}dial Meeting on Discourse and Dialogue",
    month = aug,
    year = "2017",
    address = {Saarbr{\"u}cken, Germany},
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W17-5506",
    doi = "10.18653/v1/W17-5506",
    pages = "37--49",
    abstract = "Neural task-oriented dialogue systems often struggle to smoothly interface with a knowledge base. In this work, we seek to address this problem by proposing a new neural dialogue agent that is able to effectively sustain grounded, multi-domain discourse through a novel key-value retrieval mechanism. The model is end-to-end differentiable and does not need to explicitly model dialogue state or belief trackers. We also release a new dataset of 3,031 dialogues that are grounded through underlying knowledge bases and span three distinct tasks in the in-car personal assistant space: calendar scheduling, weather information retrieval, and point-of-interest navigation. Our architecture is simultaneously trained on data from all domains and significantly outperforms a competitive rule-based system and other existing neural dialogue architectures on the provided domains according to both automatic and human evaluation metrics.",
}
"""

_DESCRIPTION = """
Task-oriented dialogue focuses on conversational agents that participate in user-initiated dialogues on domain-specific topics. Traditionally, the task-oriented dialogue community has often been hindered by a lack of sufficiently large and diverse datasets for training models across a variety of different domains. In an effort to help alleviate this problem, we release a corpus of 3,031 multi-turn dialogues in three distinct domains appropriate for an in-car assistant: calendar scheduling, weather information retrieval, and point-of-interest navigation. Our dialogues are grounded through knowledge bases ensuring that they are versatile in their natural language without being completely free form.
"""

_HOMEPAGE = "https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/"

URL = (
    "http://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip"
)


def load_entities(kvret_entity_file_path):
    """

    @param kvret_entity_file_path: the path of kvret_entities.json
    @return:
    """
    under_scored_entity_dict = OrderedDict()
    with open(kvret_entity_file_path) as f:
        entity = json.load(f)
        for sub_class_name, sub_class_entity_list in entity.items():
            if sub_class_name == 'poi':
                for entity_item in sub_class_entity_list:
                    under_scored_entity_dict[str(entity_item['address'])] = (
                        str(entity_item['address']).replace(" ", "_"))
                    under_scored_entity_dict[str(entity_item['poi'])] = (str(entity_item['poi']).replace(" ", "_"))
                    under_scored_entity_dict[str(entity_item['type'])] = (str(entity_item['type']).replace(" ", "_"))
            else:
                for entity_item in sub_class_entity_list:
                    under_scored_entity_dict[str(entity_item)] = (str(entity_item).replace(" ", "_"))
    return under_scored_entity_dict


def underscore_entities(dialogue_str, global_entities_dict):
    """
    Underscore the entities in dialogue.
    @param dialogue_str:
    @param global_entities_dict:

    @return:

    entities_in_this_turn: dict: the entities of this turn.
    processed_dialogue_str: string with entities underscored if replace=True else the same as it was.
    """
    processed_dialogue_str: str = dialogue_str

    for entity in global_entities_dict.keys():
        if entity in dialogue_str:
            processed_dialogue_str = processed_dialogue_str.replace(entity, global_entities_dict[entity])
    return processed_dialogue_str


class KVRET(datasets.GeneratorBasedBuilder):
    """ KVRET: A New Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset """

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = {
            "id": datasets.Value("int32"),
            "dialogue":
                datasets.Sequence({
                    "driver": datasets.Value("string"),
                    "assistant": datasets.Value("string")
                }),
            "kb": {
                "header": datasets.Sequence(datasets.Value("string")),
                "rows": datasets.Sequence((datasets.Sequence(datasets.Value("string"))))
            },
            "intent": datasets.Value("string"),
            "entities_file":  datasets.Value("string")
        }
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.download_and_extract(URL)

        train_file = os.path.join(extracted_path, "kvret_train_public.json")
        val_file = os.path.join(extracted_path, "kvret_dev_public.json")
        test_file = os.path.join(extracted_path, "kvret_test_public.json")
        entities_file = os.path.join(extracted_path, "kvret_entities.json")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"file_path": train_file, "entities_file": entities_file},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"file_path": val_file, "entities_file": entities_file},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"file_path": test_file, "entities_file": entities_file},
            ),
        ]

    def _generate_examples(self, file_path, entities_file):
        global_entities_dict = load_entities(entities_file)
        underscored_global_entities = list(global_entities_dict.values())

        with open(file_path, encoding="utf-8") as f:
            dialogues_scenarios = json.load(f)

        for i, dialogue_scenario in enumerate(dialogues_scenarios):
            # we only use the dialogue data.
            dialogue = {
                "driver": [],
                "assistant": []
            }
            raw_dialogue = dialogue_scenario['dialogue']

            for dialogue_item in raw_dialogue:
                dialogue_str = dialogue_item['data']['utterance']
                processed_dialogue_str = underscore_entities(dialogue_str, global_entities_dict)
                dialogue[dialogue_item['turn']].append(processed_dialogue_str)

            # we will formulate the kb into a table format
            raw_kb = dialogue_scenario['scenario']['kb']
            header = raw_kb['column_names']

            rows = []

            if raw_kb['items']:
                for row_item in raw_kb['items']:
                    row = []
                    for header_item in header:
                        row.append(row_item[header_item])
                    rows.append(row)

            intent = dialogue_scenario['scenario']['task']['intent']

            yield f"{i}", {
                "id": i,
                "dialogue": dialogue,
                "kb": {
                    "header": header,
                    "rows": rows
                },
                "intent": intent,
                "entities_file": entities_file
            }
