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
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument(
        "--tgt_home", metavar="DIR"
    )
    return parser


def main(args):
    json_path = os.path.realpath(args.json)
    file_home = os.path.realpath(args.file_home)
    tgt_home = args.tgt_home
    with open(json_path, 'r') as f:
        data = json.load(f)["audios"]

    dir_name = dict()
    count_f = dict()
    for flag in ["dev", "test_meeting", "test_net"]:
        tgt_dir = os.path.join(tgt_home, flag)
        os.makedirs(tgt_dir, exist_ok=True)
        dir_name[flag] = open("{}/data.tsv".format(tgt_dir), 'w')
        print(file_home, file=dir_name[flag])
        count_f[flag] = 0

    for audio in data:
        audio_path = os.path.splitext(audio["path"])[0]
        for flag in ["dev", "test_meeting", "test_net"]:
            if "audio/" + flag in audio["path"]:
                for segment in audio["segments"]:
                    rel_file_path = os.path.join(audio_path, "{}.{}".format(segment["sid"], args.ext))
                    fname = os.path.join(file_home, rel_file_path)
                    frames = soundfile.info(fname).frames
                    print(
                        "{}\t{}".format(rel_file_path, frames), file=dir_name[flag]
                    )
                    count_f[flag] += segment["end_time"] - segment["begin_time"]
    for flag in ["dev", "test_meeting", "test_net"]:
        dir_name[flag].close()
        print("{}: {}h".format(flag, count_f[flag]/3600))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
