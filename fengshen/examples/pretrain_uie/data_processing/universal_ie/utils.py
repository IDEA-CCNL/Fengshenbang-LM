#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List
import os
import sys


global_mislabel_log = set()


def tokens_to_str(tokens: List[str], language: str = 'en') -> str:
    if language == 'en':
        return ' '.join(tokens)
    elif language == 'zh':
        return ''.join(tokens)
    else:
        raise NotImplementedError('Language %s not supported' % language)


def label_format(s):
    import re

    def uncamelize(s):
        re_outer = re.compile(r'([^A-Z ])([A-Z])')
        re_inner = re.compile(r'\b[A-Z]+(?=[A-Z][a-z])')
        sub = re_inner.sub(r'\g<0> ', re_outer.sub(r'\1 \2', s)).lower()
        return sub

    def remove(s):
        return s.replace("_", " ").replace("-", " ").replace(".", " ")

    s = remove(uncamelize(s)).split()
    if len(s) > 1 and s[0] == s[1]:
        s = s[1:]
    return " ".join(s)


def load_dict_ini_file(filename):
    print("Warning: `load_dict_ini_file` is deprecated.")
    if not os.path.exists(filename):
        sys.stderr.write(f'[warning] cannot load label mapper from {filename}\n')
        return {}
    mapper = dict()
    for line in open(filename):
        key, value = line.strip().split('=')
        mapper[key] = label_format(value)
    return mapper


def change_ptb_token_back(token):
    """将 PTBTokenized 的 Token 转换会原始字符串

    Args:
        token (str): PTBTokenize 后的 Token 字符串

    Returns:
        str: 原始 Token 字符串
    """
    ptb_token_map = {
        '``': '"',
        "''": '"',
        '-LRB-': '(',
        '-RRB-': ')',
        '-LSB-': '[',
        '-RSB-': ']',
        '-LCB-': '{',
        '-RCB-': '}',
    }
    for ptb_token, raw_token in ptb_token_map.items():
        if token == ptb_token:
            return raw_token
    return token


def change_name_using_label_mapper(label_name, label_mapper):
    if label_mapper is None or len(label_mapper) == 0:
        return label_name
    if label_name not in label_mapper:
        breakpoint()
        print(f"{label_name} not found in mapper")
        global global_mislabel_log
        if label_name not in global_mislabel_log:
            global_mislabel_log.add(label_name)
    return label_mapper.get(label_name, label_name)
