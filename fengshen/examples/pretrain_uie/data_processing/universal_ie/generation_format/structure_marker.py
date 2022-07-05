#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 结构标记符


class StructureMarker:
    def __init__(self) -> None:
        pass


class BaseStructureMarker(StructureMarker):
    def __init__(self) -> None:
        super().__init__()
        self.sent_start = '<extra_id_0>'
        self.sent_end = '<extra_id_1>'
        self.record_start = '<extra_id_0>'
        self.record_end = '<extra_id_1>'
        self.span_start = '<extra_id_0>'
        self.span_end = '<extra_id_1>'
        self.sep_marker = '<extra_id_2>'
        self.source_span_start = '<extra_id_3>'
        self.source_span_end = '<extra_id_4>'
        self.target_span_start = '<extra_id_5>'


class VisualStructureMarker(StructureMarker):
    def __init__(self) -> None:
        super().__init__()
        self.sent_start = '{'
        self.sent_end = '}'
        self.record_start = '['
        self.record_end = ']'
        self.span_start = '('
        self.span_end = ')'
        self.source_span_start = '<'
        self.source_span_end = '>'
        self.target_span_start = ':'
        self.sep_marker = ':'
