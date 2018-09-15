# coding=utf-8
import random
import numpy.random as npr
import Putil.train_common as tc
import numpy  as np


class Model:
    def __init__(self):
        self._train_result_reflect = ['loss', 'loss1']
        self._val_result_reflect = ['loss', 'loss1']
        self.num = 0
        pass

    def re_init(self):
        pass

    @property
    def TrainResultReflect(self):
        return self._train_result_reflect

    @property
    def ValResultReflect(self):
        return self._val_result_reflect

    @property
    def TrainCV(self):
        return self.train

    @property
    def Val(self):
        return self.val

    def train(self, data):
        # if self.num == 20:
        #     raise RuntimeError('testError')
        # self.num += 1
        return {'loss': self.num, 'loss1': self.num + 1}
        pass

    def val(self, data):
        return {'loss': self.num, 'loss1':  self.num + 1}
    pass


class cv_generator:
    def __init__(self):
        self._total = False
        pass

    def generator(self):
        train = list(range(1, 10))
        val = list(range(1, 15))
        for i in zip(train, val):
            yield {'train': index_generator_in_cv(20).generator(),
                   'val': index_generator_in_cv(20).generator()}
        pass
    pass


class index_generator_in_cv:
    def __init__(self, cv):
        self._total = False
        self.train_data_field = list(range(0, cv))
        pass

    def reset_epoch(self):
        self._total = False
        pass

    def generator(self):
        count = 0
        while True:
            data = random.choice(self.train_data_field)
            if count == len(self.train_data_field):
                self._total = True
            count += 1
            yield {'data': data, 'total': self._total}


class index_to_data:
    def __init__(self):
        self._data_list = {'l': None, 'w': None}
        pass

    @property
    def DataListName(self):
        return self._data_list.keys()

    @property
    def DataList(self):
        return self._data_list

    # generate the real data use _train_data_index, no matter what _train_data_index is ,there is should
    # always a index_generator corresponding which generates the _traini_data_index
    def index_to_data(self, _train_data_index):
        self._data_list['l'] = 1
        self._data_list['w'] = 2
        return self._data_list
        pass
    pass


if __name__ == '__main__':
    gen_feed = {'l': 'll', 'w': 'ww'}
    re_es = {'loss': 'lo', 'loss1': 'tt'}
    cv_a = cv_generator().generator()
    cv_b = cv_generator().generator()
    da = index_to_data()
    da_b = index_to_data()
    m = Model()
    tca = tc.TrainCommon()
    tca.model_cv(m, {'a': cv_a, 'b': cv_b}, {'a': da, 'b': da_b}, 10, 2, 1, 'D:/ttt/', gdfdr=gen_feed, grerr=re_es)

