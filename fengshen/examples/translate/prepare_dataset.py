#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import os


def main(file_path, src_lang, tgt_lang):

    file_list = ["train", "valid", "test"]
    for filename in file_list:
        sys.stderr.write("**** Start processing {} ... ****\n".format(filename))
        src_full_path = os.path.join(file_path, ".".join((filename, src_lang)))
        tgt_full_path = os.path.join(file_path, ".".join((filename, tgt_lang)))
        src_reader = open(src_full_path, 'r')
        tgt_reader = open(tgt_full_path, "r")

        writer_full_path = os.path.join(file_path, ".".join((filename, src_lang + "_" + tgt_lang)))
        writer = open(writer_full_path, "w")
        # combine_dict = OrderedDict()
        for row_src, row_tgt in zip(src_reader, tgt_reader):
            combine_line = {}
            combine_line["src"] = row_src.strip()
            combine_line["tgt"] = row_tgt.strip()
            json.dump(combine_line, writer, ensure_ascii=False)
            writer.write('\n')
            # print(row_src)
            # print(row_tgt)
        sys.stderr.write(f"**** Done change {filename} format **** \n")


if __name__ == "__main__":
    file_path = sys.argv[1]
    src_lang, tgt_lang = sys.argv[2].split("-")

    main(file_path, src_lang, tgt_lang)
