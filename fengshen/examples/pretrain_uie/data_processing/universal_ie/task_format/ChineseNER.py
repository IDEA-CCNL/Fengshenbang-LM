#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import Counter
import json
from typing import List, Optional, Tuple, Set
from tqdm import tqdm
from universal_ie.task_format.task_format import TaskFormat
from universal_ie.utils import tokens_to_str
from universal_ie.ie_format import Entity, Label, Sentence, Span


# https://github.com/allenai/allennlp/blob/main/allennlp/data/dataset_readers/dataset_utils/span_utils.py
# ### Start Code
def bio_tags_to_spans(
    tag_sequence: List[str], classes_to_ignore: List[str] = None
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Given a sequence corresponding to BIO tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans. This function works properly when
    the spans are unlabeled (i.e., your labels are simply "B", "I", and "O").
    # Parameters
    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.
    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    for index, string_tag in enumerate(tag_sequence):
        # Actual BIO tag.
        bio_tag = string_tag[0]
        if bio_tag not in ["B", "I", "O"]:
            raise RuntimeError('Invalid tag sequence %s' % tag_sequence)
        conll_tag = string_tag[2:]
        if bio_tag == "O" or conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == "I" and conll_tag == active_conll_tag:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)


def _iob1_start_of_chunk(
    prev_bio_tag: Optional[str],
    prev_conll_tag: Optional[str],
    curr_bio_tag: str,
    curr_conll_tag: str,
) -> bool:
    if curr_bio_tag == "B":
        return True
    if curr_bio_tag == "I" and prev_bio_tag == "O":
        return True
    if curr_bio_tag != "O" and prev_conll_tag != curr_conll_tag:
        return True
    return False


def iob1_tags_to_spans(
    tag_sequence: List[str], classes_to_ignore: List[str] = None
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Given a sequence corresponding to IOB1 tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e., those where "B-LABEL" is not preceded
    by "I-LABEL" or "B-LABEL").
    # Parameters
    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.
    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    prev_bio_tag = None
    prev_conll_tag = None
    for index, string_tag in enumerate(tag_sequence):
        curr_bio_tag = string_tag[0]
        curr_conll_tag = string_tag[2:]

        if curr_bio_tag not in ["B", "I", "O"]:
            raise RuntimeError('Invalid tag sequence %s' % tag_sequence)
        if curr_bio_tag == "O" or curr_conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
        elif _iob1_start_of_chunk(prev_bio_tag, prev_conll_tag, curr_bio_tag, curr_conll_tag):
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = curr_conll_tag
            span_start = index
            span_end = index
        else:
            # bio_tag == "I" and curr_conll_tag == active_conll_tag
            # We're continuing a span.
            span_end += 1

        prev_bio_tag = string_tag[0]
        prev_conll_tag = string_tag[2:]
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)


def bmes_tags_to_spans(
    tag_sequence: List[str], classes_to_ignore: List[str] = None
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Given a sequence corresponding to BMES tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans.
    This function works properly when the spans are unlabeled (i.e., your labels are
    simply "B", "M", "E" and "S").
    # Parameters
    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.
    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """

    def extract_bmes_tag_label(text):
        bmes_tag = text[0]
        label = text[2:]
        return bmes_tag, label

    spans: List[Tuple[str, List[int]]] = []
    prev_bmes_tag: Optional[str] = None
    for index, tag in enumerate(tag_sequence):
        bmes_tag, label = extract_bmes_tag_label(tag)
        if bmes_tag in ("B", "S"):
            # Regardless of tag, we start a new span when reaching B & S.
            spans.append((label, [index, index]))
        elif bmes_tag in ("M", "E") and prev_bmes_tag in ("B", "M") and spans[-1][0] == label:
            # Only expand the span if
            # 1. Valid transition: B/M -> M/E.
            # 2. Matched label.
            spans[-1][1][1] = index
        else:
            # Best effort split for invalid span.
            spans.append((label, [index, index]))
        # update previous BMES tag.
        prev_bmes_tag = bmes_tag

    classes_to_ignore = classes_to_ignore or []
    return [
        # to tuple.
        (span[0], (span[1][0], span[1][1]))
        for span in spans
        if span[0] not in classes_to_ignore
    ]


def bioul_tags_to_spans(
    tag_sequence: List[str], classes_to_ignore: List[str] = None
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Given a sequence corresponding to BIOUL tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are not allowed and will raise `InvalidTagSequence`.
    This function works properly when the spans are unlabeled (i.e., your labels are
    simply "B", "I", "O", "U", and "L").
    # Parameters
    tag_sequence : `List[str]`, required.
        The tag sequence encoded in BIOUL, e.g. ["B-PER", "L-PER", "O"].
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.
    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
    """
    spans = []
    classes_to_ignore = classes_to_ignore or []
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == "U":
            spans.append((label.partition("-")[2], (index, index)))
        elif label[0] == "B":
            start = index
            while label[0] != "L":
                index += 1
                if index >= len(tag_sequence):
                    raise RuntimeError('Invalid tag sequence %s' % tag_sequence)
                    # raise InvalidTagSequence(tag_sequence)
                label = tag_sequence[index]
                if not (label[0] == "I" or label[0] == "L"):
                    raise RuntimeError('Invalid tag sequence %s' % tag_sequence)
                    # raise InvalidTagSequence(tag_sequence)
            spans.append((label.partition("-")[2], (start, index)))
        else:
            if label != "O":
                raise RuntimeError('Invalid tag sequence %s' % tag_sequence)
                # raise InvalidTagSequence(tag_sequence)
        index += 1
    return [span for span in spans if span[0] not in classes_to_ignore]


def bmeso_tags_to_spans(
    tag_sequence: List[str], classes_to_ignore: List[str] = None
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    bmeso -> bioul
    B = Beginning
    I/M = Inside / Middle
    L/E = Last / End
    O = Outside
    U/W/S = Unit-length / Whole / Singleton
    """
    new_tag = list()
    for label in tag_sequence:
        if label[0] == 'M':
            new_tag += ['I-' + label.partition("-")[2]]
        elif label[0] == 'E':
            new_tag += ['L-' + label.partition("-")[2]]
        elif label[0] == 'S':
            new_tag += ['U-' + label.partition("-")[2]]
        else:
            new_tag += [label]

    return bioul_tags_to_spans(tag_sequence=new_tag, classes_to_ignore=classes_to_ignore)


def bieso_tags_to_spans(
    tag_sequence: List[str], classes_to_ignore: List[str] = None
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    bmeso -> bioul
    B = Beginning
    I/M = Inside / Middle
    L/E = Last / End
    O = Outside
    U/W/S = Unit-length / Whole / Singleton
    """
    new_tag = list()
    for label in tag_sequence:
        if label[0] == 'E':
            new_tag += ['L-' + label.partition("-")[2]]
        elif label[0] == 'S':
            new_tag += ['U-' + label.partition("-")[2]]
        else:
            new_tag += [label]

    return bioul_tags_to_spans(tag_sequence=new_tag, classes_to_ignore=classes_to_ignore)
# ### End Code


_tagging_span_function = {
    'bioul': bioul_tags_to_spans,
    'bmes': bmes_tags_to_spans,
    'bio': bio_tags_to_spans,
    'iob1': iob1_tags_to_spans,
    'bmeso': bmeso_tags_to_spans,
    'bieso': bieso_tags_to_spans,
}


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

    @staticmethod
    def generate_sentence(filename):
        # TODO: 为什么这里使用静态方法，而上面不是静态方法呢
        sentence = list()
        with open(filename) as fin:
            for line in fin:
                # if line.strip() == '':
                if len(line.strip().split())!=2:
                    if len(sentence) != 0:
                        yield sentence
                        sentence = list() # yield 使用而不是使用 return ， 迭代器的使用的场景
                else:
                    sentence += [line.strip().split()]

            if len(sentence) != 0:
                yield sentence


class ChineseNER(Cols):
    """
        BIO/BEMS 标注形式， 需要转在yaml之中进行指定的数据标注形式
        例如：
        老 B-PER.NOM
        百 I-PER.NOM
        姓 I-PER.NOM
        心 O
        新 B-GPE.NAM
        乡 I-GPE.NAM
        新 O
        闻 O
        网 O
        话 O
        说 O
        这 O
        几 O
        天 O
        新 B-GPE.NAM
        乡 I-GPE.NAM
        天 O
        气 O
        还 O
        好 O
        吧 O
        偷 O
        笑 O
    """

    @staticmethod
    def load_from_file(filename, language='en', tagging='bio') -> List[Sentence]:
        sentence_list = list()
        counter = Counter()
        for rows in tqdm(Cols.generate_sentence(filename), desc="Generate {}".format(filename)):
            tokens = [token[0] for token in rows]
            ner = [token[1] for token in rows]
            spans = _tagging_span_function[tagging](ner)  # 根据tagging模式，选择不同的函数来解码ner
            spans = list(filter(lambda x: x[0] != "", spans))
            spans = [
                {'start': span[1][0], 'end': span[1][1], 'type': span[0]}
                for span in spans
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


class TagTokenCols(Cols):

    @staticmethod
    def load_from_file(filename, language='en', tagging='bio') -> List[Sentence]:
        sentence_list = list()
        counter = Counter()
        for rows in tqdm(Cols.generate_sentence(filename)):
            tokens = [token[1] for token in rows]
            ner = [token[0] for token in rows]
            spans = _tagging_span_function[tagging](ner)
            spans = [
                {'start': span[1][0], 'end': span[1][1], 'type': span[0]}
                for span in spans
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


class TokenTagJson(Cols):
    @staticmethod
    def load_from_file(filename, language='en', tagging='bio') -> List[Sentence]:
        sentence_list = list()
        counter = Counter()
        for line in open(filename):
            instance = json.loads(line.strip())
            tokens = instance['tokens']
            ner = instance['ner_tags']
            spans = _tagging_span_function[tagging](ner)
            spans = list(filter(lambda x: x[0] != "", spans))
            spans = [
                {'start': span[1][0], 'end': span[1][1], 'type': span[0]}
                for span in spans
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


class I2b2Conll(Cols):

    @staticmethod
    def load_from_file(filename, language='en') -> List[Sentence]:
        sentence_list = list()
        counter = Counter()
        for rows in tqdm(Cols.generate_sentence(filename)):
            tokens = [token[0] for token in rows]
            ner = [token[4] for token in rows]
            spans = bio_tags_to_spans(ner)
            spans = [
                {'start': span[1][0], 'end': span[1][1], 'type': span[0]}
                for span in spans
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


class CoNLL03(Cols):

    @staticmethod
    def load_from_file(filename, language='en') -> List[Sentence]:
        sentence_list = list()
        counter = Counter()
        for rows in tqdm(Cols.generate_sentence(filename)):
            if rows[0][0] == '-DOCSTART-':
                continue
            tokens = [token[0] for token in rows]
            ner = [token[3] for token in rows]
            spans = iob1_tags_to_spans(ner)
            spans = [
                {'start': span[1][0], 'end': span[1][1], 'type': span[0]}
                for span in spans
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


if __name__ == "__main__":
    pass
