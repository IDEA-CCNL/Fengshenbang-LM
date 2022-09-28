#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import os
import random
import json

import soundfile


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--file_home", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--tgt_home", metavar="DIR"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.01,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")

    return parser


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    json_path = os.path.realpath(args.json)
    file_home = os.path.realpath(args.file_home)
    tgt_home = args.tgt_home
    with open(json_path, 'r') as f:
        data = json.load(f)["audios"]

    rand = random.Random(args.seed)
    train_f = dict()
    valid_f = dict()
    count_f = dict()
    for flag in ["S", "M", "L"]:
        tgt_dir = os.path.join(tgt_home, flag)
        os.makedirs(tgt_dir, exist_ok=True)
        train_f[flag] = open("{}/train.tsv".format(tgt_dir), 'w')
        valid_f[flag] = (
            open("{}/valid.tsv".format(tgt_dir), 'w')
            if args.valid_percent > 0
            else None
        )
        print(file_home, file=train_f[flag])
        if valid_f[flag] is not None:
            print(file_home, file=valid_f[flag])
        count_f[flag] = 0

    for audio in data:
        audio_path = os.path.splitext(audio["path"])[0]
        for segment in audio["segments"]:
            rel_file_path = os.path.join(audio_path, "{}.{}".format(segment["sid"], args.ext))
            if "subsets" not in segment or len(segment["subsets"]) == 0 or "train" not in audio_path:
                continue
            fname = os.path.join(file_home, rel_file_path)
            frames = soundfile.info(fname).frames
            dest_dict = train_f if rand.random() > args.valid_percent else valid_f
            for flag in segment["subsets"]:
                if flag in ["S", "M", "L"]:
                    print(
                        "{}\t{}".format(rel_file_path, frames), file=dest_dict[flag]
                    )
                    count_f[flag] += segment["end_time"] - segment["begin_time"]
    for flag in ["S", "M", "L"]:
        train_f[flag].close()
        if valid_f[flag] is not None:
            valid_f[flag].close()
        print("{}: {}h".format(flag, count_f[flag]/3600))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
