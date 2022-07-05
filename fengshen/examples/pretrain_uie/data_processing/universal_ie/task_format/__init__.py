#!/usr/bin/env python
# -*- coding:utf-8 -*-
# from distutils.command.build_scripts import first_line_re
from universal_ie.task_format.task_format import TaskFormat

from universal_ie.task_format.ChineseEE import ChineseEE
from universal_ie.task_format.ChineseNER import (
    ChineseNER,
    I2b2Conll,
    TagTokenCols,
    TokenTagJson,
    CoNLL03
)
from universal_ie.task_format.ChineseRE import ChineseRE
from universal_ie.task_format.Substructure import Substructure