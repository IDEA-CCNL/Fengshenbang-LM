#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import shutil
import random
import argparse
from collections import defaultdict
import json
import sys
from universal_ie.record_schema import RecordSchema


def n_shot_smaple(source_filename, target_filename, record_schema,
                  spot_asoc_key='spot', num_shot=5, min_len=None, seed=None):

    train_data = [json.loads(line.strip()) for line in open(source_filename)]

    if seed:
        random.seed(seed)
        random.shuffle(train_data)

    # 记录每一句的类别信息
    type_to_sentence_dict = defaultdict(list)
    for index, instance in enumerate(train_data):
        for spot in instance[spot_asoc_key]:
            if spot not in record_schema.type_list:
                continue
            if min_len is not None and len(instance['tokens']) < min_len:
                continue
            type_to_sentence_dict[spot] += [index]

    sampled_data = list()
    for entity in type_to_sentence_dict:

        if len(type_to_sentence_dict[entity]) < num_shot:
            sys.stderr.write(
                f'[WARN] {entity} in {source_filename} is less than shot num {num_shot}\n'
            )
            sampled = type_to_sentence_dict[entity]
        else:
            sampled = random.sample(type_to_sentence_dict[entity], num_shot)

        sampled_data += [train_data[index] for index in sampled]

    with open(target_filename, 'w') as output:
        for instance in sampled_data:
            output.write(json.dumps(instance) + '\n')

    return sampled_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', help='Source Folder Name', required=True)
    parser.add_argument('-tgt', help='Target Folder Name, n shot sampled',
                        required=True)
    parser.add_argument('-task', help='N-Shot Task name', required=True,
                        choices=['entity', 'relation', 'event'])
    parser.add_argument('-seed', help='Default is None, no random')
    parser.add_argument('-min_len', dest='min_len', help='Default is None', type=int)
    options = parser.parse_args()

    source_folder = options.src
    target_folder = options.tgt

    task_name = options.task

    if task_name in ['relation']:
        spot_asoc_key = 'asoc'
    else:
        spot_asoc_key = 'spot'

    os.makedirs(target_folder, exist_ok=True)

    for shot in [1, 5, 10]:
        shot_folder = os.path.join(target_folder, "%sshot" % shot)

        os.makedirs(shot_folder, exist_ok=True)

        n_shot_smaple(
            source_filename=os.path.join(source_folder, 'train.json'),
            target_filename=os.path.join(shot_folder, 'train.json'),
            record_schema=RecordSchema.read_from_file(
                os.path.join(source_folder, f'{task_name}.schema'),
            ),
            spot_asoc_key=spot_asoc_key,
            num_shot=shot,
            seed=options.seed,
            min_len=options.min_len
        )

        for filename in os.listdir(source_folder):
            if filename != 'train.json':
                shutil.copy(
                    os.path.join(source_folder, filename),
                    os.path.join(shot_folder, filename),
                )


if __name__ == "__main__":
    main()
