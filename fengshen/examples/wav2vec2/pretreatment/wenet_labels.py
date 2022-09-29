#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings for a flashlight (previously called wav2letter++) dataset
"""

import argparse
import os
import json
import re
from typing import Optional
import jieba


def remove_special_characters(line, chars_to_ignore_regex=None):
    if chars_to_ignore_regex is not None:
        line = re.sub(chars_to_ignore_regex, "", line)
    else:
        line = line
    return line


def create_vocabulary_from_data(
    vocab_set,
    word_delimiter_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    pad_token: Optional[str] = None,
):
    # Given training and test labels create vocabulary

    vocab_dict = {v: k for k, v in enumerate(sorted(list(vocab_set)))}

    # replace white space with delimiter token
    if word_delimiter_token is not None:
        if " " in vocab_dict:
            vocab_dict[word_delimiter_token] = vocab_dict[" "]
            if word_delimiter_token != " ":
                del vocab_dict[" "]
        elif word_delimiter_token not in vocab_dict:
            vocab_dict[word_delimiter_token] = len(vocab_dict)

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    if "</s>" not in vocab_dict:
        vocab_dict["</s>"] = len(vocab_dict)

    if "<s>" not in vocab_dict:
        vocab_dict["<s>"] = len(vocab_dict)

    return vocab_dict


def add_line_to_vocab_set(vocab_set: set, line: list):
    def extract_all_chars(line):
        vocab = set(line)
        return vocab
    return vocab_set | extract_all_chars(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--unk_token", default="<unk>")
    parser.add_argument("--pad_token", default="<pad>")
    parser.add_argument("--word_delimiter_token", default="|")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tsv")
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--chars_to_ignore", default=None, nargs='*')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    chars_to_ignore_regex = (
        f'[{"".join(args.chars_to_ignore)}]' if args.chars_to_ignore is not None else None
    )

    vocab_set = set()

    transcriptions = {}
    tsv_holder = []
    with open(args.tsv, "r") as tsv, open(
        os.path.join(args.output_dir, args.output_name + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.output_dir, args.output_name + ".wrd"), "w"
    ) as wrd_out:
        root = next(tsv).strip()
        tsv_holder.append(root)
        for line in tsv:
            line = line.strip()
            dir = os.path.dirname(line)
            if dir not in transcriptions:
                parts = dir.split(os.path.sep)
                trans_path = f"{parts[-1]}.trans.txt"
                path = os.path.join(root, dir, trans_path)
                if not os.path.exists(path):
                    continue
                texts = {}
                with open(path, "r") as trans_f:
                    for tline in trans_f:
                        items = tline.strip().split()
                        texts[items[0]] = " ".join(items[1:])
                transcriptions[dir] = texts
            part = os.path.basename(line).split(".")[0]
            if part not in transcriptions[dir]:
                continue
            tsv_holder.append(line)
            text = remove_special_characters(transcriptions[dir][part], chars_to_ignore_regex)

            if args.tokenizer_path:
                vocab_set = add_line_to_vocab_set(vocab_set, text)

            print(text, file=wrd_out)
            ltr_data = " ".join(jieba.lcut(transcriptions[dir][part]))
            print(
                ltr_data,
                file=ltr_out,
            )

    with open(args.tsv, "w") as tsv:
        tsv.write("\n".join(tsv_holder))

    # save special tokens for tokenizer
    if args.tokenizer_path:
        word_delimiter_token = args.word_delimiter_token
        unk_token = args.unk_token
        pad_token = args.pad_token
        vocab_dict = create_vocabulary_from_data(
            vocab_set,
            word_delimiter_token=word_delimiter_token,
            unk_token=unk_token,
            pad_token=pad_token,)
        tokenizer_name_or_path = args.tokenizer_path
        vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")
        os.makedirs(tokenizer_name_or_path, exist_ok=True)
        # save vocab dict to be loaded into tokenizer
        with open(vocab_file, "w") as file:
            json.dump(vocab_dict, file, ensure_ascii=False)


if __name__ == "__main__":
    main()
