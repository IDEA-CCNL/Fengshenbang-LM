#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Dict, Union
from collections import defaultdict
from universal_ie.record_schema import RecordSchema
from universal_ie.generation_format.structure_marker import StructureMarker
from universal_ie.ie_format import Entity, Relation, Event, Label
import abc


class GenerationFormat:
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 structure_maker: StructureMarker,
                 label_mapper: Dict = None,
                 language: str = 'en') -> None:
        self.structure_maker = structure_maker
        self.language = language
        self.label_mapper = {} if label_mapper is None else label_mapper

        # 用于从数据中统计 Schema
        self.record_role_map = defaultdict(set)

    def get_label_str(self, label: Label):
        return self.label_mapper.get(label.__repr__(), label.__repr__())

    @abc.abstractmethod
    def annotate_entities(
        self, tokens: List[str], entities: List[Entity]): pass

    @abc.abstractmethod
    def annotate_given_entities(self, tokens: List[str], entities: Union[List[Entity], Entity]): pass

    @abc.abstractmethod
    def annotate_events(self, tokens: List[str], events: List[Event]): pass

    @abc.abstractmethod
    def annotate_event_given_predicate(self, tokens: List[str], event: Event): pass

    @abc.abstractmethod
    def annotate_relation_extraction(self, tokens: List[str],
                                     relations: List[Relation]): pass

    def output_schema(self, filename: str):
        """自动导出 Schema 文件
        每个 Schema 文件包含三行
            - 第一行为 Record 的类别名称列表
            - 第二行为 Role 的类别名称列表
            - 第三行为 Record-Role 映射关系字典
        Args:
            filename (str): [description]
        """
        record_list = list(self.record_role_map.keys())
        role_set = set()
        for record in self.record_role_map:
            role_set.update(self.record_role_map[record])
            self.record_role_map[record] = list(self.record_role_map[record])
        role_list = list(role_set)

        record_schema = RecordSchema(type_list=record_list,
                                     role_list=role_list,
                                     type_role_dict=self.record_role_map
                                     )
        record_schema.write_to_file(filename)

    def get_entity_schema(self, entities: List[Entity]):
        schema_role_map = set()
        for entity in entities:
            schema_role_map.add(self.get_label_str(entity.label))
        return RecordSchema(
            type_list=list(schema_role_map),
            role_list=list(),
            type_role_dict=dict()
        )

    def get_relation_schema(self, relations: List[Relation]):
        record_role_map = defaultdict(set)
        role_set = set()

        for relation in relations:
            record_role_map[self.get_label_str(relation.label)].add(self.get_label_str(relation.arg1.label))
            record_role_map[self.get_label_str(relation.label)].add(self.get_label_str(relation.arg2.label))

        for record in record_role_map:
            role_set.update(record_role_map[record])
            record_role_map[record] = list(self.record_role_map[record])

        return RecordSchema(
            type_list=list(record_role_map.keys()),
            role_list=list(role_set),
            type_role_dict=record_role_map
        )

    def get_event_schema(self, events: List[Event]):
        record_role_map = defaultdict(set)
        role_set = set()

        for event in events:
            for role, _ in event.args:
                record_role_map[self.get_label_str(event.label)].add(self.get_label_str(role))

        for record in record_role_map:
            role_set.update(record_role_map[record])
            record_role_map[record] = list(self.record_role_map[record])

        return RecordSchema(
            type_list=list(record_role_map.keys()),
            role_list=list(role_set),
            type_role_dict=record_role_map
        )
