from __future__ import absolute_import, division, print_function
import abc
import os

class dataholder:
    """An abstract class to manage the datasets"""
    __metaclass__ = abc.ABCMeta
    def __init__(self, name = 'DataHolder', data_folder = None):
        self.name = name
        self.data_folder = data_folder

    @abc.abstractmethod
    def evaluate(self, anom_map_list, data_str='test'):
        pass