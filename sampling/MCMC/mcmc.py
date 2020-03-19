# coding=utf-8
import Putil.sampling.sample as s
from abc import ABCMeta, abstractmethod


class MCMC(s.Sample):
    def __init__(self):
        s.Sample.__init__(self)
        #self.set_pdf_func(None)
        self._random_func = None
        self._x = None
        pass

    def set_random_func(self, func):
        '''
        this function set a function which random select a sample from the target sample space
        Important: the result of the random_func should be able to pass to the pdf_func
        '''
        self._random_func = func
        self._x = self._random_func()
        pass
    pass