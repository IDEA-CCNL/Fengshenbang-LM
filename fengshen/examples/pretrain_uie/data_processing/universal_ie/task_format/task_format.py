#!/usr/bin/env python
# -*- coding:utf-8 -*-
import abc


class TaskFormat:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, language='en'):
        self.language = language
        # TODO:  针对不同的数据集的格式，需要去重写下面的两个方法！
        # TODO: 查资料了解abstract类的，为何如此设计代码结构

    @abc.abstractmethod
    def generate_instance(self):
        
        pass

    @staticmethod
    @abc.abstractmethod
    def load_from_file(filename, language='en'):
        pass
