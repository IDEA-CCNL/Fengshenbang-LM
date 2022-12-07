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
""" KVRET(Mem2Seq & GLMP & DF_Net... version): A New Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset """

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
@inproceedings{wu2019global,
  title={Global-to-local Memory Pointer Networks for Task-Oriented Dialogue},
  author={Wu, Chien-Sheng and Socher, Richard and Xiong, Caiming},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2019}
}
"""
import ast
from copy import deepcopy


def read_langs(file_name, entity_file, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, kb_arr = [], []
    history = []
    max_resp_len = 0

    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1
        for line in fin:
            line = line.strip()
            if line:
                if '#' in line:
                    line = line.replace("#", "")
                    task_type = line
                    continue

                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u, r, gold_ent = line.split('\t')
                    history.append(u)

                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)
                    ent_idx_cal, ent_idx_nav, ent_idx_wet = [], [], []
                    if task_type == "weather":
                        ent_idx_wet = gold_ent
                    elif task_type == "schedule":
                        ent_idx_cal = gold_ent
                    elif task_type == "navigate":
                        ent_idx_nav = gold_ent
                    ent_index = list(set(ent_idx_cal + ent_idx_nav + ent_idx_wet))

                    data_detail = {
                        'history': deepcopy(history),
                        'response': r,
                        'ent_index': ent_index,
                        'ent_idx_cal': list(set(ent_idx_cal)),
                        'ent_idx_nav': list(set(ent_idx_nav)),
                        'ent_idx_wet': list(set(ent_idx_wet)),
                        # 'conv_arr': list(conv_arr),
                        'kb_arr': list(kb_arr),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': task_type}
                    data.append(data_detail)
                    history.append(r)

                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    r = line
                    kb_info = generate_memory(r, "", str(nid))
                    # context_arr = kb_info + context_arr
                    kb_arr += [kb_info_item for kb_info_item in kb_info if not kb_info_item == "PAD"]
            else:
                cnt_lin += 1
                history, kb_arr = [], []
                if (max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len


def generate_template(global_entity, sentence, sent_ent, kb_arr, domain):
    """
    Based on the system response and the provided entity table, the output is the sketch response.
    """
    sketch_response = []
    if sent_ent == []:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            if word not in sent_ent:
                sketch_response.append(word)
            else:
                ent_type = None
                if domain != 'weather':
                    for kb_item in kb_arr:
                        if word == kb_item[0]:
                            ent_type = kb_item[1]
                            break
                if ent_type == None:
                    for key in global_entity.keys():
                        if key != 'poi':
                            global_entity[key] = [x.lower() for x in global_entity[key]]
                            if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                                ent_type = key
                                break
                        else:
                            poi_list = [d['poi'].lower() for d in global_entity['poi']]
                            if word in poi_list or word.replace('_', ' ') in poi_list:
                                ent_type = key
                                break
                sketch_response.append('@' + ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response


def generate_memory(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    sent_new.append(sent_token)
    return sent_new



_DESCRIPTION = """
Task-oriented dialogue focuses on conversational agents that participate in user-initiated dialogues on domain-specific topics. Traditionally, the task-oriented dialogue community has often been hindered by a lack of sufficiently large and diverse datasets for training models across a variety of different domains. In an effort to help alleviate this problem, we release a corpus of 3,031 multi-turn dialogues in three distinct domains appropriate for an in-car assistant: calendar scheduling, weather information retrieval, and point-of-interest navigation. Our dialogues are grounded through knowledge bases ensuring that they are versatile in their natural language without being completely free form.
"""

_HOMEPAGE = "https://github.com/jasonwu0731/GLMP"

URL = (
    "https://github.com/jasonwu0731/GLMP/archive/refs/heads/master.zip"
)


def convert_kvr_to_kb(kb_arr, domain):
    if domain == "navigate":
        header = ["poi", "distance", "traffic_info", "poi_type", "address"]
        rows = []
        row = []
        for kb_item in kb_arr:
            if len(kb_item) == 3:
                row.append(kb_item[-1])
            else:
                if len(row) > 0:
                    rows.append(row)
                    row = []
                row.append(kb_item[-1])
        if len(row) > 0:
            rows.append(row)

    elif domain == "weather":
        header = [
          "location",
          "monday",
          "tuesday",
          "wednesday",
          "thursday",
          "friday",
          "saturday",
          "sunday",
        ]
        rows = []
        row = []
        for kb_item in kb_arr:
            if len(kb_item) == 2:
                continue
            elif len(kb_item) == 3:
                if not kb_item[0] in row:
                    if len(row) > 0:
                        rows.append(row)
                        row = []
                    row.append(kb_item[0])
                row.append(kb_item[2])
            elif len(kb_item) == 4:
                row[-1] = row[-1] + ", {} {}".format(kb_item[-2], kb_item[-1])
            else:
                raise ValueError("impossible")
        if len(row) > 0:
            rows.append(row)
    elif domain == "schedule":
        header = [
            "event",
            "time",
            "date",
            "room",
            "agenda",
            "party"
        ]
        rows = []
        row = ["-", "-", "-", "-", "-", "-"]
        for kb_item in kb_arr:
            if not kb_item[0] in row:
                if not row == ["-", "-", "-", "-", "-", "-"]:
                    rows.append(row)
                    row = ["-", "-", "-", "-", "-", "-"]
                row[0] = kb_item[0]
            pos = header.index(kb_item[1])
            row[pos] = kb_item[2]

        if not row[0]=='-':
            rows.append(row)

    for row in rows:
        if not len(row) == len(header):
            raise AssertionError
    return {"header": header, "rows": rows}


class KVRET_GLMP(datasets.GeneratorBasedBuilder):
    """ KVRET: A New Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset """

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {

                "kb": {
                    "header": datasets.Sequence(datasets.Value("string")),
                    "rows": datasets.Sequence((datasets.Sequence(datasets.Value("string"))))
                },
                "kb_arr": datasets.Sequence((datasets.Sequence(datasets.Value("string")))),
                'history': datasets.Sequence(datasets.Value("string")),
                'response': datasets.Value("string"),
                'ent_index': datasets.Sequence(datasets.Value("string")),
                "ent_idx_cal": datasets.Sequence(datasets.Value("string")),
                "ent_idx_wet": datasets.Sequence(datasets.Value("string")),
                "ent_idx_nav": datasets.Sequence(datasets.Value("string")),
                'domain': datasets.Value("string"),
                'ID': datasets.Value("int32"),
                'id': datasets.Value("int32"),
                "entities_file": datasets.Value("string"),

            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.download_and_extract(URL)

        train_file = os.path.join(extracted_path, "GLMP-master", "data", "KVR", "train.txt")
        val_file = os.path.join(extracted_path, "GLMP-master", "data", "KVR", "dev.txt")
        test_file = os.path.join(extracted_path, "GLMP-master", "data", "KVR", "test.txt")
        entities_file = os.path.join(extracted_path, "GLMP-master", "data", "KVR", "kvret_entities.json")

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
        pair_data, data_max_len = read_langs(file_name=file_path, entity_file=entities_file, max_line=None)
        for example_idx, pair_data_item in enumerate(pair_data):
            yield example_idx, {
                **pair_data_item,
                "kb": convert_kvr_to_kb(pair_data_item['kb_arr'], pair_data_item['domain']),
                "entities_file": entities_file
            }

