# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
#  The official script will get stuck in the first stage, due to the multiple links in its download instead of one.
"""MultiWOZ v2.2: Multi-domain Wizard of OZ version 2.2"""

import json
import os
from collections import OrderedDict

import datasets

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{corr/abs-2007-12720,
  author    = {Xiaoxue Zang and
               Abhinav Rastogi and
               Srinivas Sunkara and
               Raghav Gupta and
               Jianguo Zhang and
               Jindong Chen},
  title     = {MultiWOZ 2.2 : {A} Dialogue Dataset with Additional Annotation Corrections
               and State Tracking Baselines},
  journal   = {CoRR},
  volume    = {abs/2007.12720},
  year      = {2020},
  url       = {https://arxiv.org/abs/2007.12720},
  archivePrefix = {arXiv},
  eprint    = {2007.12720}
}
"""

# You can copy an official description
_DESCRIPTION = """\
Multi-Domain Wizard-of-Oz dataset (MultiWOZ), a fully-labeled collection of human-human written conversations spanning over multiple domains and topics.
MultiWOZ 2.1 (Eric et al., 2019) identified and fixed many erroneous annotations and user utterances in the original version, resulting in an
improved version of the dataset. MultiWOZ 2.2 is a yet another improved version of this dataset, which identifies and fizes dialogue state annotation errors
across 17.3% of the utterances on top of MultiWOZ 2.1 and redefines the ontology by disallowing vocabularies of slots with a large number of possible values
(e.g., restaurant name, time of booking) and introducing standardized slot span annotations for these slots.
"""

_LICENSE = "Apache License 2.0"
# git: 44f0f8479f11721831c5591b839ad78827da197b
URL = "https://github.com/budzianowski/multiwoz/archive/44f0f8479f11721831c5591b839ad78827da197b.zip"


def load_entities(multi_woz_22_entity_file_paths: list):
    """

    @param multi_woz_22_entity_file_paths: a list of .json which we can load kb/entities
    @return:
    """
    under_scored_entity_dict = OrderedDict()
    for multi_woz_22_entity_file_path in multi_woz_22_entity_file_paths:
        with open(multi_woz_22_entity_file_path) as f:
            # FIXME: ask dialogue expert for whether this mean of entities extraction is right
            entities = json.load(f)
            for entity_item in entities:
                for entity_name, entity_value in entity_item.items():
                    if isinstance(entity_value, str):
                        under_scored_entity_dict[entity_value] = entity_value.replace(" ", "_")

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


class MultiWozV22(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("2.2.0")

    def _info(self):
        features = datasets.Features(
            {
                "dialogue_id": datasets.Value("string"),
                "db_root_path": datasets.Value("string"),
                "services": datasets.Sequence(datasets.Value("string")),
                "db_paths": datasets.Sequence(datasets.Value("string")),
                "turns": datasets.Sequence(
                    {
                        "turn_id": datasets.Value("string"),
                        "speaker": datasets.ClassLabel(names=["USER", "SYSTEM"]),
                        "utterance": datasets.Value("string"),
                        "frames": datasets.Sequence(
                            {
                                "service": datasets.Value("string"),
                                "state": {
                                    "active_intent": datasets.Value("string"),
                                    "requested_slots": datasets.Sequence(datasets.Value("string")),
                                    "slots_values": datasets.Sequence(
                                        {
                                            "slots_values_name": datasets.Value("string"),
                                            "slots_values_list": datasets.Sequence(datasets.Value("string")),
                                        }
                                    ),
                                },
                                "slots": datasets.Sequence(
                                    {
                                        "slot": datasets.Value("string"),
                                        "value": datasets.Value("string"),
                                        "start": datasets.Value("int32"),
                                        "exclusive_end": datasets.Value("int32"),
                                        "copy_from": datasets.Value("string"),
                                        "copy_from_value": datasets.Sequence(datasets.Value("string")),
                                    }
                                ),
                            }
                        ),
                        "dialogue_acts": datasets.Features(
                            {
                                "dialog_act": datasets.Sequence(
                                    {
                                        "act_type": datasets.Value("string"),
                                        "act_slots": datasets.Sequence(
                                            datasets.Features(
                                                {
                                                    "slot_name": datasets.Value("string"),
                                                    "slot_value": datasets.Value("string"),
                                                }
                                            ),
                                        ),
                                    }
                                ),
                                "span_info": datasets.Sequence(
                                    {
                                        "act_type": datasets.Value("string"),
                                        "act_slot_name": datasets.Value("string"),
                                        "act_slot_value": datasets.Value("string"),
                                        "span_start": datasets.Value("int32"),
                                        "span_end": datasets.Value("int32"),
                                    }
                                ),
                            }
                        ),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=None,
            homepage="https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2",
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_path = dl_manager.download_and_extract(URL)

        def __get_file_paths_dict(root_path):
            file_paths = [
                ("dialogue_acts", "dialog_acts.json")
            ]
            file_paths += [
                (
                    f"train_{i:03d}",
                    f"train/dialogues_{i:03d}.json",
                )
                for i in range(1, 18)
            ]
            file_paths += [
                (
                    f"dev_{i:03d}",
                    f"dev/dialogues_{i:03d}.json",
                )
                for i in range(1, 3)
            ]
            file_paths += [
                (
                    f"test_{i:03d}",
                    f"test/dialogues_{i:03d}.json",
                )
                for i in range(1, 3)
            ]
            file_paths = dict(file_paths)
            for file_info, file_name in file_paths.items():
                file_paths[file_info] = os.path.join(root_path, "multiwoz-44f0f8479f11721831c5591b839ad78827da197b",
                                                     "data", "MultiWOZ_2.2", file_name)
            return dict(file_paths)

        data_files = __get_file_paths_dict(data_path)
        db_root_path = os.path.join(data_path, "multiwoz-44f0f8479f11721831c5591b839ad78827da197b", "db")
        db_paths = [path for path in os.listdir(db_root_path) if str(path).endswith(".json")]

        self.global_entities = load_entities(db_paths)
        self.stored_dialogue_acts = json.load(open(data_files["dialogue_acts"]))

        return [
            datasets.SplitGenerator(
                name=spl_enum,
                gen_kwargs={
                    "filepaths": data_files,
                    "split": spl,
                    "db_root_path": db_root_path
                },
            )
            for spl, spl_enum in [
                ("train", datasets.Split.TRAIN),
                ("dev", datasets.Split.VALIDATION),
                ("test", datasets.Split.TEST),
            ]
        ]

    def _generate_examples(self, filepaths, split, db_root_path):
        id_ = -1
        file_list = [fpath for fname, fpath in filepaths.items() if fname.startswith(split)]
        for filepath in file_list:
            dialogues = json.load(open(filepath))
            for dialogue in dialogues:
                id_ += 1
                mapped_acts = self.stored_dialogue_acts.get(dialogue["dialogue_id"], {})
                res = {
                    "dialogue_id": dialogue["dialogue_id"],
                    "db_root_path": db_root_path,
                    "services": dialogue["services"],
                    "db_paths": [os.path.join(db_root_path, "{}.json".format(service)) for service in
                                 dialogue["services"]],
                    "turns": [
                        {
                            "turn_id": turn["turn_id"],
                            "speaker": turn["speaker"],
                            "utterance": underscore_entities(turn["utterance"], self.global_entities),
                            "frames": [
                                {
                                    "service": frame["service"],
                                    "state": {
                                        "active_intent": frame["state"]["active_intent"] if "state" in frame else "",
                                        "requested_slots": frame["state"]["requested_slots"]
                                        if "state" in frame
                                        else [],
                                        "slots_values": {
                                            "slots_values_name": [
                                                sv_name for sv_name, sv_list in frame["state"]["slot_values"].items()
                                            ]
                                            if "state" in frame
                                            else [],
                                            "slots_values_list": [
                                                sv_list for sv_name, sv_list in frame["state"]["slot_values"].items()
                                            ]
                                            if "state" in frame
                                            else [],
                                        },
                                    },
                                    "slots": [
                                        {
                                            "slot": slot["slot"],
                                            "value": "" if "copy_from" in slot else slot["value"],
                                            "start": slot.get("exclusive_end", -1),
                                            "exclusive_end": slot.get("start", -1),
                                            "copy_from": slot.get("copy_from", ""),
                                            "copy_from_value": slot["value"] if "copy_from" in slot else [],
                                        }
                                        for slot in frame["slots"]
                                    ],
                                }
                                for frame in turn["frames"]
                                if (
                                        "active_only" not in self.config.name
                                        or frame.get("state", {}).get("active_intent", "NONE") != "NONE"
                                )
                            ],
                            "dialogue_acts": {
                                "dialog_act": [
                                    {
                                        "act_type": act_type,
                                        "act_slots": {
                                            "slot_name": [sl_name for sl_name, sl_val in dialog_act],
                                            "slot_value": [sl_val for sl_name, sl_val in dialog_act],
                                        },
                                    }
                                    for act_type, dialog_act in mapped_acts.get(turn["turn_id"], {})
                                        .get("dialog_act", {})
                                        .items()
                                ],
                                "span_info": [
                                    {
                                        "act_type": span_info[0],
                                        "act_slot_name": span_info[1],
                                        "act_slot_value": span_info[2],
                                        "span_start": span_info[3],
                                        "span_end": span_info[4],
                                    }
                                    for span_info in mapped_acts.get(turn["turn_id"], {}).get("span_info", [])
                                ],
                            },
                        }
                        for turn in dialogue["turns"]
                    ],
                }
                yield id_, res
