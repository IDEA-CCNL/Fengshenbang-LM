#!/usr/bin/env python
# -*- coding:utf-8 -*-
from abc import abstractmethod
from collections import defaultdict
from typing import List, Union, Tuple
from universal_ie.utils import change_name_using_label_mapper


# All Entity Relation Events are structured records.
# They both have attributes text_id and record_id
# 所有的 Entity Relation Event 都是结构化的记录表示 （Record）
# 他们都有属性 text_id 和 record_id
class Record:
    def __init__(self,
                 text_id: Union[str, None] = None,
                 record_id: Union[str, None] = None,
                 ) -> None:
        self.text_id = text_id
        self.record_id = record_id

    @abstractmethod
    def to_offset(self):
        pass
    
# Text span
# 连续或者非连续的文本块
class Span:
    def __init__(self,
                 tokens: List[str],
                 indexes: List[int],
                 text: str,
                 text_id: Union[str, None] = None,
                 ) -> None:
        self.tokens = tokens
        self.indexes = indexes
        self.text = text
        self.text_id = text_id

    def __repr__(self) -> str:
        return "[%s](%s)" % (self.text, self.indexes)

    @staticmethod
    def get_empty_span(text_id: Union[str, None] = None,):
        return Span(
            tokens=list(),
            indexes=list(),
            text="",
            text_id=text_id
        )

    def is_empty_span(self):
        """Check is empty span.

        Returns:
            bool: True, Empty Span; False Non-Empty Span
        """
        return len(self.tokens) == 0 and len(self.indexes) == 0


# Label Name
class Label:
    def __init__(self, label_name: Union[str, List[str]]) -> None:
        self.label_name = label_name

    def __repr__(self) -> str:
        return self.label_name

    def __lt__(self, other):
        if not isinstance(other, Label):
            return NotImplemented
        return self.label_name < other.label_name


# Entity, Span
# 实体，以文本块为核心的一元结构
class Entity(Record):
    def __init__(self,
                 span: Span,
                 label: Label,
                 text_id: Union[str, None] = None,
                 record_id: Union[str, None] = None,
                 ) -> None:
        super().__init__(text_id=text_id, record_id=record_id)
        self.span = span
        self.label = label

    def __lt__(self, other):
        if not isinstance(other, Entity):
            return NotImplemented
        return self.span.indexes < other.span.indexes

    def __repr__(self) -> str:
        return self.span.__repr__() + self.label.__repr__()

    def to_offset(self, ent_label_mapper=None):
        if self.span.is_empty_span():
            # If span is empty, skip entity
            return {}
        return {'type': change_name_using_label_mapper(self.label.label_name,
                                                       ent_label_mapper),
                'offset': self.span.indexes,
                'text': self.span.text}


# Relation Span Pair
# 关系，以文本块对为核心的二元结构
class Relation(Record):
    def __init__(self,
                 arg1: Entity,
                 arg2: Entity,
                 label: Label,
                 text_id: Union[str, None] = None,
                 record_id: Union[str, None] = None,
                 ) -> None:
        super().__init__(text_id=text_id, record_id=record_id)
        self.arg1 = arg1
        self.arg2 = arg2
        self.label = label

    def __repr__(self) -> str:
        return self.arg1.__repr__() + self.label.__repr__() + self.arg2.__repr__()

    def to_offset(self, rel_label_mapper=None, ent_label_mapper=None):
        if self.arg1.span.is_empty_span() or self.arg2.span.is_empty_span():
            # If span is empty, skip relation
            return {}
        return {'type': change_name_using_label_mapper(self.label.label_name,
                                                       rel_label_mapper),
                'args': [self.arg1.to_offset(ent_label_mapper=ent_label_mapper),
                         self.arg2.to_offset(ent_label_mapper=ent_label_mapper),
                         ],
                }


# Event, Trigger-Mult-Argument
# 事件，以触发词为中心的多元(谓词论元)结构
class Event(Record):
    def __init__(self,
                 span: Span,
                 label: Label,
                 args: List[Tuple[Label, Entity]],
                 text_id: Union[str, None] = None,
                 record_id: Union[str, None] = None,
                 ) -> None:
        super().__init__(text_id=text_id, record_id=record_id)
        self.span = span
        self.label = label
        self.args = args

    def __repr__(self) -> str:
        return self.span.__repr__() + self.label.__repr__()

    def to_offset(self, evt_label_mapper=None):

        if self.span.is_empty_span():
            # If span is empty, skip relation
            return {}

        args = list()
        for role, arg in self.args:
            if arg.span.is_empty_span():
                continue
            args += [{
                    'type': change_name_using_label_mapper(
                        role.label_name,
                        evt_label_mapper,
                    ),
                    'offset': arg.span.indexes,
                    'text': arg.span.text
                }]

        return {'type': change_name_using_label_mapper(self.label.label_name,
                                                       evt_label_mapper),
                'offset': self.span.indexes,
                'text': self.span.text,
                'args': args}


class Sentence:
    def __init__(self,
                 tokens: List[str],
                 entities: List[Entity] = None,
                 relations: List[Relation] = None,
                 events: List[Event] = None,
                 text_id: Union[str, None] = None,
                 ) -> None:
        self.tokens = tokens
        self.entities = entities or list()
        self.relations = relations or list()
        self.events = events or list()
        self.text_id = text_id

    def count_entity_without_relation(self):
        entity_set = set()
        entity_counter = defaultdict(int)
        for entity in self.entities:
            entity_set.add((tuple(entity.span.indexes), entity.label.label_name))

        for relation in self.relations:
            entity1 = (tuple(relation.arg1.span.indexes), relation.arg1.label.label_name)
            entity2 = (tuple(relation.arg2.span.indexes), relation.arg2.label.label_name)
            entity_counter[entity1] += 1
            entity_counter[entity2] += 1
            entity_set.remove(entity1) if entity1 in entity_set else None
            entity_set.remove(entity2) if entity2 in entity_set else None
        overlap_entity = sum([1 if v > 1 else 0 for k, v in entity_counter.items()])
        return {'entity': len(self.entities),
                'entity_without_relation': len(entity_set),
                'overlap_entity': overlap_entity,
                }
