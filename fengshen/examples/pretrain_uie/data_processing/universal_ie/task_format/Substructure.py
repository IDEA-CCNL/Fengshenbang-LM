#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import Counter
import json
from typing import List, Optional, Tuple, Set
from tqdm import tqdm
from universal_ie.task_format.task_format import TaskFormat
from universal_ie.utils import tokens_to_str
from universal_ie.ie_format import Entity, Label, Sentence, Span

class Cols(TaskFormat):

    def __init__(self, tokens: List[str], spans:  List[Tuple[Tuple[int, int], str]], language='en', instance_id=None) -> None:
        super().__init__(
            language=language
        )
        self.instance_id = instance_id
        self.tokens = tokens # ['对', '，', '输', '给', '一', '个', '女', '人', '，', '的', '成', '绩', '。', '失', '望']
        self.spans = spans # [{'start': 6, 'end': 7, 'type': 'PER.NOM'}]

    def generate_instance(self):
        # breakpoint()
        entities = list()
        # self.spans :[{}, {}, {'start':, 'end':, 'type}]
        for span_index, span in enumerate(self.spans):
            tokens = self.tokens[span['start']: span['end'] + 1] # text
            indexes = list(range(span['start'], span['end'] + 1)) # offsets
            ## 返回的是entity的一个对象 【Entity: -> [Span(), Label()] 】
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
        return Sentence(tokens=self.tokens,
                        entities=entities,
                        text_id=self.instance_id)

    def map_first_offset(self, text, span):
        start = text.find(span)

        end= start+len(span)-1 # attention!!

        return { 'start': start,  'end': end ,  'type': "<extrid_id_66>"}  

    @staticmethod
    def generate_sentence(filename):
        # TODO: 为什么这里使用静态方法，而上面不是静态方法呢
        sentence = list()
        with open(filename) as fin:
            for line in fin:
                yield json.loads(line)

    



class Substructure(Cols):

    # @staticmethod
    def load_from_file( filename, language='en') -> List[Sentence]:
        sentence_list = list()
        counter = Counter()
        for rows in tqdm(Cols.generate_sentence(filename), desc="Generate {}".format(filename)):
            tokens = list(rows['text'])

            spans = rows['entities']
            spans = list(filter(lambda x: x!=None, spans)) ##!

            def map_first_offset(text, span):
                start = text.find(span)
                end= start+len(span)-1 # attention!!
                # return { 'start': start,  'end': end ,  'type': '实体'}  
                return {'start': start,  'end': end ,  'type': r'<extra_id_66>'}  
                
            spans = [
                map_first_offset(rows['text'] ,span)  for span in spans
            ]

            sentence = Cols( 
                tokens=tokens,
                spans=spans,
                language=language,
            )
            counter.update(['token'] * len(tokens))
            counter.update(['sentence'])
            counter.update(['span'] * len(spans))
            sentence_list += [sentence.generate_instance()]
        print(filename, counter)
        return sentence_list

