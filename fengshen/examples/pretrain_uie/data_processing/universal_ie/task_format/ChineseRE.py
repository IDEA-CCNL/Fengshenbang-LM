#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import Counter
import json
from typing import List, Dict
from universal_ie.task_format.task_format import TaskFormat
from universal_ie.utils import change_ptb_token_back, tokens_to_str
from universal_ie.ie_format import Entity, Label, Relation, Sentence, Span
from tqdm import tqdm


class ChineseRE(TaskFormat):
    """
{
    "tokens":[ "《",  "邪",  "少", "兵",  "王",  "》", "是",  "冰",  "火", "未","央",  "写", "的", "网", "络", "小", "说", "连", "载", "于", "旗", "峰", "天", "下"],
    "span_list":[
        {
            "type":"图书作品",
            "start":1,
            "end":4
        },
        {
            "type":"人物",
            "start":7,
            "end":10
        }
    ],
    "span_pair_list":[
        {
            "type":"作者",
            "head":0,
            "tail":1
        }
    ]
}
    """
    def __init__(self, instance_json: Dict, language='en') -> None:
        super().__init__(
            language=language
        )
        self.tokens = change_ptb_token_back(instance_json['tokens'])
        self.span_list = instance_json.get('span_list', [])
        self.span_pair_list = instance_json.get('span_pair_list', [])
        self.instance_id = instance_json.get('id', None)

    def generate_instance(self):
        entities = list()
        relations = list()
        # 产生实体
        for span_index, span in enumerate(self.span_list):
            tokens = self.tokens[span['start']: span['end'] + 1]
            indexes = list(range(span['start'], span['end'] + 1))
            entities += [
                Entity(
                    span=Span(
                        tokens=tokens,
                        indexes=indexes,
                        text=tokens_to_str(tokens, language=self.language),
                        text_id=self.instance_id
                    ),
                    label=Label(span['type']),
                    text_id=self.instance_id,
                    record_id=self.instance_id + "#%s" % span_index if self.instance_id else None)
            ]
        # 不同的格式修改这里
        for spanpair_index, span_pair in enumerate(self.span_pair_list):
                relations += [
                    Relation(
                        arg1=entities[span_pair['head']], # 找出头和尾实体
                        arg2=entities[span_pair['tail']],
                        label=Label(span_pair['type']),
                        text_id=self.instance_id,
                        record_id=self.instance_id + "##%s" % spanpair_index if self.instance_id else None
                    )
                ]
        return Sentence(tokens=self.tokens,
                        entities=entities,
                        relations=relations,
                        text_id=self.instance_id)

    @staticmethod
    def load_from_file(filename, language='en') -> List[Sentence]:
        sentence_list = list()
        counter = Counter()
        with open(filename) as fin:
            for line in tqdm(fin, desc="Loading {}".format(filename)):
                spannet = ChineseRE(
                    json.loads(line.strip()),
                    language=language
                )
                # 产生数据
                instance = spannet.generate_instance()
                sentence_list += [instance]
                counter.update(['sentence'])
                counter.update(['span'] * len(spannet.span_list))
        print(filename, counter)
        return sentence_list
