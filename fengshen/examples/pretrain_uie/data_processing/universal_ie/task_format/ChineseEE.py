#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json
from pydoc import doc
from typing import List
from tqdm import tqdm

from universal_ie.task_format.task_format import TaskFormat
from universal_ie.utils import tokens_to_str
from universal_ie.ie_format import Entity, Event, Label, Sentence, Span


"""DUEE Demo
{
    "id":"EE-DUEE-0",
    "text":"雀巢裁员4000人：时代抛弃你时，连招呼都不会打！",
    "tokens":[ "雀", "巢", "裁",  "员", "4", "0", "0", "0", "人", "：","时", "代","抛", "弃", "你", "时", "，","连", "招","呼", "都","不","会", "打","！"],
    "entity":[],
    "relation":[],
    "event":[
        {
            "type":"组织关系-裁员",
            "offset":[2,3],
            "text":"裁员",
            "args":[
                {
                    "type":"裁员方",
                    "offset":[ 0,1],
                    "text":"雀巢"
                },
                {
                    "type":"裁员人数",
                    "offset":[ 4, 5, 6, 7,8],
                    "text":"4000人"
                }
            ]
        }
    ]
}
"""

def get_entities(data):
    entities = []
    for sample in data:
      for arg in sample['args']:
        entities.append(arg)
    return entities

class ChineseEE(TaskFormat):
    def __init__(self, doc_json, language='en'):
        super().__init__(
            language=language
        )
        # self.doc_id = doc_json['id']
        self.sent_id = None
        self.tokens= doc_json['tokens']
        self.entities = get_entities(doc_json['event'])
        self.relations = doc_json['relation']
        self.events= doc_json['event']

    def generate_instance(self):
        events = []
        entities = dict()

        for span_index, span in enumerate(self.entities):
            tokens = list(span['text'])
            indexes = span['offset'] 
            entities[json.dumps(span,ensure_ascii=False)]= Entity(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                    text_id=self.sent_id
                ),
                label=Label(span['type']),
                text_id=self.sent_id,
                record_id=None
            )
        # breakpoint()
        for event_index, event in enumerate(self.events):
            start = event['offset'][0]
            end = event['offset'][-1]
            tokens = self.tokens[start:end+1]
            indexes = event['offset'] 
            events.append( Event(
                span=Span(
                    tokens=tokens,
                    indexes=indexes,
                    text=tokens_to_str(tokens, language=self.language),
                    text_id=self.sent_id
                ),
                label=Label(event['type']),
                args=[(Label(x['type']), entities[json.dumps(x, ensure_ascii=False)])
                      for x in event['args']],
                text_id=self.sent_id,
                record_id=None
            )
            )
        return Sentence(
            tokens=self.tokens,
            entities=list(),
            relations=list(),
            events=events,
            text_id=self.sent_id
        )

    @staticmethod
    def load_from_file(filename, language='en') -> List[Sentence]:
        sentence_list = list()
        with open(filename) as fin:
            for line in tqdm(fin, desc='Load {}'.format(filename)):
                instance = ChineseEE(
                    json.loads(line.strip()),
                    language=language
                ).generate_instance()
                sentence_list += [instance]
        return sentence_list
