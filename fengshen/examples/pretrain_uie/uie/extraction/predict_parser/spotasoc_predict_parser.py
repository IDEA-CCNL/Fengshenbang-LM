#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import Counter
import logging
from nltk.tree import ParentedTree
import re
from typing import Tuple, List, Dict


from uie.extraction.constants import (
    null_span,
    type_start,
    type_end,
    span_start,
)
from uie.extraction.predict_parser.predict_parser import PredictParser
from uie.extraction.predict_parser.utils import fix_unk_from_text

logger = logging.getLogger(__name__)


left_bracket = '【'
right_bracket = '】'
brackets = left_bracket + right_bracket

split_bracket = re.compile(r"<extra_id_\d>")


def add_space(text):
    """
    add space between special token
    """
    new_text_list = list()
    for item in zip(split_bracket.findall(text), split_bracket.split(text)[1:]):
        new_text_list += item
    return ' '.join(new_text_list)


def convert_bracket(text): 
    text = add_space(text) 
    for start in [type_start]:
        text = text.replace(start, left_bracket)
    for end in [type_end]:
        text = text.replace(end, right_bracket)
    return text


def find_bracket_num(tree_str):
    """
    Count Bracket Number (num_left - num_right), 0 indicates num_left = num_right
    """
    count = 0
    for char in tree_str:
        if char == left_bracket:
            count += 1
        elif char == right_bracket:
            count -= 1
        else:
            pass
    return count


def check_well_form(tree_str):
    return find_bracket_num(tree_str) == 0


def clean_text(tree_str):
    count = 0
    sum_count = 0

    tree_str_list = tree_str.split()

    for index, char in enumerate(tree_str_list):
        if char == left_bracket:
            count += 1
            sum_count += 1
        elif char == right_bracket:
            count -= 1
            sum_count += 1
        else:
            pass
        if count == 0 and sum_count > 0:
            return ' '.join(tree_str_list[:index + 1])
    return ' '.join(tree_str_list)


def resplit_label_span(label, span, split_symbol=span_start):
    """为什么需要去重新对label和span进行划分, 似乎没有任何的意义"""
    label_span = label + ' ' + span # '方法 <extra_id_5> 改进的<unk>模块'

    if split_symbol in label_span:
        splited_label_span = label_span.split(split_symbol)
        if len(splited_label_span) == 2:
            return splited_label_span[0].strip(), splited_label_span[1].strip()

    return label, span


def add_bracket(tree_str):
    """add right bracket to fix ill-formed expression
    """
    tree_str_list = tree_str.split()
    bracket_num = find_bracket_num(tree_str_list)
    tree_str_list += [right_bracket] * bracket_num
    return ' '.join(tree_str_list)


def get_tree_str(tree):
    """get str from sel tree
        仅仅只会遍历当层的一棵树的下一级的元素，如果有嵌套的模块那么就不会继续
    """
    str_list = list()
    for element in tree:
        if isinstance(element, str):
            str_list += [element]
    return ' '.join(str_list)


def rewrite_label_span(label, span, label_set=None, text=None):
    """ 可能会产生一些不在标签集之中的东西嘛 """

    # Invalid Type
    if label_set and label not in label_set:
        logger.debug('Invalid Label: %s' % label)
        return None, None

    # Fix unk using Text
    if text is not None and '<unk>' in span:
        span = fix_unk_from_text(span, text, '<unk>')

    # Invalid Text Span
    if text is not None and span not in text:
        logger.debug('Invalid Text Span: %s\n%s\n' % (span, text))
        return None, None

    return label, span


class SpotAsocPredictParser(PredictParser):

    def decode(self, gold_list, pred_list, text_list=None, raw_list=None
               ) -> Tuple[List[Dict], Counter]:
        """

        :param gold_list:
        :param pred_list:
        :param text_list:
        :param raw_list:
        :return:
            dict:
                pred_spot -> [(type1, text1), (type2, text2), ...]
                gold_spot -> [(type1, text1), (type2, text2), ...]
                pred_asoc -> [(spot type1, asoc type1, text1), (spot type2, asoc type2, text2), ...]
                gold_asoc -> [(spot type1, asoc type1, text1), (spot type2, asoc type2, text2), ...]
                pred_record -> [{'type': type1, 'text': text1, 'roles': [(spot type1, asoc type1, text1), ...]},
                                {'type': type2, 'text': text2, 'rolaes': [(spot type2, asoc type2, text2), ...]},
                                ]
                gold_record -> [{'type': type1, 'text': text1, 'roles': [(spot type1, asoc type1, text1), ...]},
                                {'type': type2, 'text': text2, 'roles': [(spot type2, asoc type2, text2), ...]},
                                ]
            Counter:
        """
        counter = Counter()
        well_formed_list = []

        if gold_list is None or len(gold_list) == 0:
            gold_list = ["%s%s" % (type_start, type_end)] * len(pred_list) # 补充为空

        if text_list is None:
            text_list = [None] * len(pred_list)

        if raw_list is None:
            raw_list = [None] * len(pred_list)

        for gold, pred, text, raw_data in zip(gold_list, pred_list, text_list,raw_list):
            gold = convert_bracket(gold)
            pred = convert_bracket(pred)
            pred = clean_text(pred)

            try:  ## 将gold转成一个nltk的parenentedtree
                gold_tree = ParentedTree.fromstring(gold, brackets=brackets)
            except ValueError:
                logger.warning(f"Ill gold: {gold}")
                logger.warning(f"Fix gold: {add_bracket(gold)}")
                gold_tree = ParentedTree.fromstring(
                    add_bracket(gold), brackets=brackets)
                counter.update(['gold_tree add_bracket'])

            instance = {
                'gold': gold,
                'pred': pred,
                'gold_tree': gold_tree,
                'text': text,
                'raw_data': raw_data
            }

            counter.update(['gold_tree' for _ in gold_tree])

            instance['gold_spot'], instance['gold_asoc'], instance['gold_record'] = self.get_record_list(
                sel_tree=instance["gold_tree"],
                text=instance['text']
            )

            try:
                if not check_well_form(pred): ## 存在没有解析的
                    pred = add_bracket(pred)
                    counter.update(['fixed'])

                pred_tree = ParentedTree.fromstring(pred, brackets=brackets)
                counter.update(['pred_tree' for _ in pred_tree])

                instance['pred_tree'] = pred_tree
                counter.update(['well-formed'])

            except ValueError:
                counter.update(['ill-formed'])
                logger.debug('ill-formed', pred) # 预测试错的
                instance['pred_tree'] = ParentedTree.fromstring(
                    left_bracket + right_bracket,
                    brackets=brackets
                )

            instance['pred_spot'], instance['pred_asoc'], instance['pred_record'] = self.get_record_list(
                sel_tree=instance["pred_tree"],
                text=instance['text']
            )
            # dict_keys(['gold', 'pred', 'gold_tree', 'text', 'raw_data', 'gold_spot', 'gold_asoc', 'gold_record', 'pred_tree', 'pred_spot', 'pred_asoc', 'pred_record']
            well_formed_list += [instance] 

        # breakpoint()
        return well_formed_list, counter

    def get_record_list(self, sel_tree, text=None):
        """ Convert single sel expression to extraction records
        Args:
            sel_tree (Tree): sel tree #将sel_tree树
            text (str, optional): _description_. Defaults to None.
        Returns:
            spot_list: list of (spot_type: str, spot_span: str)
            asoc_list: list of (spot_type: str, asoc_label: str, asoc_text: str)
            record_list: list of {'asocs': list(), 'type': spot_type, 'spot': spot_text}
        """
       
        spot_list = list()
        asoc_list = list()
        record_list = list()

        # breakpoint() ## 对于一颗seltree的结构 ParentedTree
        for spot_tree in sel_tree:
            # Drop incomplete tree
            if isinstance(spot_tree, str) or len(spot_tree) == 0:
                continue

            spot_type = spot_tree.label()
            spot_text = get_tree_str(spot_tree)

            # 1.  对于spot_type 标签进行一个
            spot_type, spot_text = resplit_label_span(
                spot_type, spot_text)
            #2.  对于spot_type, spot_text　进行一个清洗，spot_type不在标签集合r如何处理， 2. span之中包含<unk>如何处理？
            spot_type, spot_text = rewrite_label_span(
                label=spot_type,
                span=spot_text,
                label_set=self.spot_set,
                text=text
            )

            # Drop empty generated span
            if spot_text is None or spot_text == null_span:
                continue
            # Drop empty generated type
            if spot_type is None:
                continue
            # Drop invalid spot type
            if self.spot_set is not None and spot_type not in self.spot_set:
                continue

            record = {'asocs': list(),
                      'type': spot_type,
                      'spot': spot_text}

            for asoc_tree in spot_tree:
                if isinstance(asoc_tree, str) or len(asoc_tree) < 1:
                    continue

                asoc_label = asoc_tree.label()
                asoc_text = get_tree_str(asoc_tree)
                # 1. 对于<>进行重新的划分
                asoc_label, asoc_text = resplit_label_span(
                    asoc_label, asoc_text)
                # 2. 对于存在的unk的的标签进行修改
                asoc_label, asoc_text = rewrite_label_span(
                    label=asoc_label,
                    span=asoc_text,
                    label_set=self.role_set,
                    text=text
                )

                # Drop empty generated span
                if asoc_text is None or asoc_text == null_span:
                    continue
                # Drop empty generated type
                if asoc_label is None:
                    continue
                # Drop invalid asoc type
                if self.role_set is not None and asoc_label not in self.role_set:
                    continue

                asoc_list += [(spot_type, asoc_label, asoc_text)]
                record['asocs'] += [(asoc_label, asoc_text)]

            spot_list += [(spot_type, spot_text)]
            record_list += [record]

        return spot_list, asoc_list, record_list

"""
'【 【 人物 <extra_id_5> 老百姓 】 【 地理区域 <extra_id_5> 新乡 】 【 地理区域 <extra_id_5> 新乡 】 】’


ParentedTree('', [ParentedTree('人物', ['<extra_id_5>', '老百姓']), ParentedTree('地理区域', ['<extra_id_5>', '新乡']), ParentedTree('地理区域', ['<extra_id_5>', '新乡'])])

(Pdb) spot_list
[('人物', '老百姓'), ('地理区域', '新乡'), ('地理区域', '新乡')]
(Pdb) asoc_list
[]
(Pdb) ecord_list
*** NameError: name 'ecord_list' is not defined
(Pdb) record_list
[{'asocs': [], 'type': '人物', 'spot': '老百姓'}, {'asocs': [], 'type': '地理区域', 'spot': '新乡'}, {'asocs': [], 'type': '地理区域', 'spot': '新乡'}]
"""