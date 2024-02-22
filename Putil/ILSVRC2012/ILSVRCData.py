# coding=utf-8
import Putil.loger as plog
import pandas as pd
import numpy as np
import os
import cv2
import threading
import sys
import copy
import random
from multiprocessing.managers import BaseManager, NamespaceProxy

root_logger = plog.PutilLogConfig('PreTrainModelData').logger()
root_logger.setLevel(plog.DEBUG)

ILSVRCLogger = root_logger.getChild('ILSVRC')
ILSVRCLogger.setLevel(plog.DEBUG)

# read the sample
def read_train_image_label_func(class_amount):
    def _read(resize_shape):
        def __read(image_path, class_id):
            img = cv2.imread(image_path)
            if img is None:
                ILSVRCLogger.error('image read : {0} failed'.format(image_path))
                sys.exit()
            img = cv2.resize(img, resize_shape, cv2.INTER_CUBIC)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            image = np.asanyarray(img.astype('float32'))
            image = image / 255.0
            label = np.zeros(shape=[class_amount], dtype=np.float32)
            label[class_id] = 1.0
            return np.array([image]), np.array([label])
        return __read
    return _read
    pass

def read_evaluate_image_label_func(class_amount):
    def _read(resize_shape):
        def __read(image_path, class_id):
            img = cv2.imread(image_path)
            if img is None:
                ILSVRCLogger.error('image read : {0} failed'.format(image_path))
                sys.exit()
                pass
            img = cv2.resize(img, resize_shape, cv2.INTER_CUBIC)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                pass
            image = np.asanyarray(img.astype('float32'))
            image = image / 255.0
            label = np.zeros(shape=[class_amount], dtype=np.float32)
            label[class_id] = 1.0
            return np.array([image]), np.array([label])
        return __read
    return _read


'''
this class generate the train and evaluate data from the ILSVRC
'''
class ILSVRC:
    def __init__(self, statistic_file, information_save_to='', load_truncate=None, split=0.8, subset_class_amount=1000, data_drop_rate=0.0,
                 device_batch=()):
        '''

        :param statistic_file: the statistic file which generate by ILSVRC_statistic
        :param load_truncate: load a part of the statistic file
        :param split: the split of the train and evaluate: split for train
        :param subset_class_amount: the amount of the target class which use to ge the sub dataset base on the class
        :param data_drop_rate: the rate of the sample which would be dropped in every class , used to get the sub dataset base on the sample amount
        :param device_batch: batch size in every device
        '''
        # a file which contain all the sample of the ImageNet
        self._statistic_file = statistic_file
        ILSVRCLogger.info(self._statistic_file)
        # the batch of the device list
        self._device_batch = device_batch
        ILSVRCLogger.info(self._device_batch)

        # the subset of the total dataset, represent the class amount, which should be smaller than 1000
        assert(subset_class_amount <= 1000), print('subset amount should be lower than 1000')
        assert(isinstance(subset_class_amount, int)), print('subset amount should be a int')
        self._subset = subset_class_amount
        ILSVRCLogger.info('subset: {0}'.format(self._subset))
        # the split for the train
        self._split = split
        ILSVRCLogger.info('split: {0}'.format(self._split))
        self._data_drop_rate = data_drop_rate
        ILSVRCLogger.info('data drop rate: {0}'.format(self._data_drop_rate))

        # : load the statistic file
        _dpd = pd.read_csv(self._statistic_file, nrows=load_truncate) if load_truncate is not None else pd.read_csv(
            self._statistic_file)
        _dpd = _dpd.drop(columns=['Unnamed: 0'])
        ILSVRCLogger.info('total sample amount: {0}'.format(_dpd.shape[0]))

        # : deal with the subset
        process = _dpd.drop_duplicates(subset='class', keep='first')
        classes = np.array(process['class'])
        np.random.shuffle(classes)

        # collection the class name
        subset_class_name_collection = classes[0: self._subset]
        ILSVRCLogger.info('sub set class : {0}'.format(subset_class_name_collection))
        self._class_reflect = dict()
        [self._class_reflect.update({class_name: np.where(subset_class_name_collection == class_name)[0][0]}) for
         class_name in subset_class_name_collection]
        ILSVRCLogger.info('name class reflect: {0}'.format(self._class_reflect))

        self._subset_df = _dpd[_dpd['class'].isin(subset_class_name_collection)]
        ILSVRCLogger.info('sub set class amount: {0}'.format(self._subset_df.index.size))
        del _dpd

        # : deal with the train valuate split
        self._train_df = None
        self._evaluate_df = None
        for sub_class in self._class_reflect.items():
            # get the target class data
            target_class = self._subset_df[self._subset_df['class'] == sub_class[0]]
            ILSVRCLogger.debug('amount of class {0}: {1}'.format(sub_class[0], target_class.index.size))
            target_class_index = np.array(target_class.index)
            np.random.shuffle(target_class_index)
            # drop sample
            target_class_index = target_class_index[
                                 0: np.floor(target_class_index.size * (1.0 - self._data_drop_rate)).astype('int64')]
            # split to train and evaluate
            train_index = target_class_index[0: (np.floor(target_class_index.size * self._split)).astype('int64')]
            evaluate_index = target_class_index[(np.floor(target_class_index.size * self._split)).astype('int64'):]
            self._train_df = self._subset_df.loc[train_index] if self._train_df is None else self._train_df.append(
                self._subset_df.loc[train_index])
            self._evaluate_df = self._subset_df.loc[
                evaluate_index] if self._evaluate_df is None else self._evaluate_df.append(
                self._subset_df.loc[evaluate_index])
            pass
        del self._subset_df


        self._train_field = list(self._train_df.index)
        np.random.shuffle(self._train_field)
        assert len(self._train_field) != 0, print('train field is zero')
        self._evaluate_field = list(self._evaluate_df.index)
        np.random.shuffle(self._evaluate_field)
        assert len(self._evaluate_field) != 0, print('evaluate field is zero')

        # the train and evaluate reader function : set the class amount
        self._train_reader = read_train_image_label_func(len(self._class_reflect.items()))
        self._evaluate_reader = read_evaluate_image_label_func(len(self._class_reflect.items()))

        self._train_epoch_done = False
        self._train_remain = copy.deepcopy(self._train_field)
        self._train_generated_epoch = None
        self._train_reset_mutex = threading.Condition()
        self._evaluate_epoch_done = False
        self._evaluate_remain = copy.deepcopy(self._evaluate_field)
        self._evaluate_generated_epoch = None
        self._evaluate_reset_mutex = threading.Condition()
        pass

    def step_in_one_training_epoch(self):
        np.ceil(self._train_df.index.size / (sum(self._device_batch)))
        pass

    def step_in_one_evaluating_epoch(self):
        np.ceil(self._train_df.index.size / sum(self._device_batch))
        pass

    def generate_train(self, ilsvrc_root):
        self._train_reset_mutex.acquire()
        self._train_reset_mutex.wait_for(lambda: ~self._train_epoch_done)
        device_image = []
        device_label = []
        for i in self._device_batch:
            # the train reader function : set shape
            _func = self._train_reader((256, 256))

            image = None
            label = None
            while (image.shape[0] if image is not None else 0) < i:
                # create a new train element index
                if len(self._train_remain) == 0:
                    self._train_epoch_done = True
                    try:
                        train_target = random.sample(self._train_field, 1)[0]
                    except Exception as ex:
                        print(ex.args)
                        sys.exit()
                    pass
                else:
                    train_target = self._train_remain.pop()
                    pass

                # get the train element index
                try:
                    element = self._train_df.loc[train_target]
                except Exception as ex:
                    print(ex.args)
                    sys.exit()
                ILSVRCLogger.debug('deal with {0} {1}'.format(element['class'], element['image_name']))

                # get the data
                class_name = element['class']
                image_name = element['image_name']
                try:
                    image_path = os.path.join(os.path.join(ilsvrc_root, class_name), image_name)
                except Exception as ex:
                    print(ex.args)
                    sys.exit()
                im, la = _func(image_path, self._class_reflect[class_name])

                # concate the data
                try:
                    image = np.concatenate([image, im], axis=0) if image is not None else im
                    label = np.concatenate([label, la], axis=0) if label is not None else la
                except Exception as ex:
                    ILSVRCLogger.info(ex.args)
                    sys.exit()

                # make the decision
                if (image.shape[0] if image is not None else 0) > i:
                    # some data would be drop
                    image = image[0: i]
                    label = label[0: i]
                    pass
                pass
            device_image.append(image)
            device_label.append(label)

        self._train_reset_mutex.release()

        return device_image, device_label

    def restart_train(self, shuffle=True):
        self._train_reset_mutex.acquire()
        np.random.shuffle(self._train_field) if shuffle is True else None
        self._train_epoch_done = False
        self._train_remain = copy.deepcopy(self._train_field)
        self._train_generated_epoch = 1 if self._train_generated_epoch is None else self._train_generated_epoch + 1
        self._train_reset_mutex.notify_all()
        self._train_reset_mutex.release()
        pass

    def train_epoch_done(self):
        return self._train_epoch_done
        pass

    def train_epoch_amount(self):
        return self._train_generated_epoch
        pass

    def generate_evaluate(self, ilsvrc_root):
        self._evaluate_reset_mutex.acquire()
        self._evaluate_reset_mutex.wait_for(lambda: ~self._evaluate_epoch_done)
        device_image = []
        device_label = []
        for i in self._device_batch:
            # the train reader function : set shape
            _func = self._evaluate_reader((256, 256))

            image = None
            label = None
            while (image.shape[0] if image is not None else 0) < i:
                # create a new train element index
                if len(self._evaluate_remain) == 0:
                    self._evaluate_epoch_done = True
                    try:
                        target = random.sample(self._evaluate_field, 1)[0]
                    except Exception as ex:
                        print(ex.args)
                        sys.exit()
                    pass
                else:
                    target = self._evaluate_remain.pop()
                    pass

                # get the train element index
                try:
                    element = self._evaluate_df.loc[target]
                except Exception as ex:
                    print(ex.args)
                    sys.exit()
                ILSVRCLogger.debug('deal with {0} {1}'.format(element['class'], element['image_name']))

                # get the data
                class_name = element['class']
                image_name = element['image_name']
                try:
                    image_path = os.path.join(os.path.join(ilsvrc_root, class_name), image_name)
                except Exception as ex:
                    print(ex.args)
                    sys.exit()
                im, la = _func(image_path, self._class_reflect[class_name])

                # concate the data
                try:
                    image = np.concatenate([image, im], axis=0) if image is not None else im
                    label = np.concatenate([label, la], axis=0) if label is not None else la
                except Exception as ex:
                    ILSVRCLogger.info(ex.args)
                    sys.exit()

                # make the decision
                if (image.shape[0] if image is not None else 0) > i:
                    # some data would be drop
                    image = image[0: i]
                    label = label[0: i]
                    pass
                pass
            device_image.append(image)
            device_label.append(label)

        self._evaluate_reset_mutex.release()

        return device_image, device_label
        pass

    def restart_evaluate(self, shuffle=False):
        self._evaluate_reset_mutex.acquire()
        np.random.shuffle(self._train_field) if shuffle is True else None
        self._evaluate_epoch_done = False
        self._evaluate_remain = copy.deepcopy(self._evaluate_field)
        self._evaluate_generated_epoch = 1 if self._evaluate_generated_epoch is None else self._evaluate_generated_epoch + 1
        self._evaluate_reset_mutex.release()
        pass

    def evaluate_epoch_done(self):
        return self._evaluate_epoch_done
        pass

    def evaluate_epoch_amount(self):
        return self._evaluate_generated_epoch
        pass

    pass

#
class ILSVRCManager(BaseManager):
    pass

'''
the proxy class for ILSVRC in multiprocess
'''
class ILSVRCProxy(NamespaceProxy):
    _exposed_ = (
    '__getattribute__',
    '__setattr__',
    '__delattr__',
    'train_epoch_done',
    'generate_train',
    'restart_train',
    'evaluate_epoch_done',
    'generate_evaluate',
    'restart_evaluate')

    def train_epoch_done(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.train_epoch_done.__name__, ())

    def generate_train(self, ilsvrc_root):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.generate_train.__name__, (ilsvrc_root, ))

    def restart_train(self, shuffle=True):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.restart_train.__name__, (shuffle, ))

    def evaluate_epoch_done(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.evaluate_epoch_done.__name__, ())

    def generate_evaluate(self, ilsvrc_root):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.generate_evaluate.__name__, (ilsvrc_root,))

    def restart_evaluate(self, shuffle=True):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.restart_evaluate.__name__, (shuffle, ))
    pass

ILSVRCManager.register('ILSVRC', ILSVRC, proxytype=ILSVRCProxy)

def train_data_extract_process(train_epoch_done_cond, stop_train_generating, tdata_queue, data, ilsvrc_root):
    '''
    the func which can be use to
    example:
    manager = pmd.ILSVRCManager()
    manager.start()
    data = manager.ILSVRC(statistic_file='/data/process_data/caojihua/ILSVRC/statistic_sample.csv', load_truncate=10000,
                      split=0.01, subset_class_amount=2, device_batch=(2, 3), data_drop_rate=0.9)
    stop_train_generating = Manager().Value(bool, False)
    tdata_queue = Manager().Queue(maxsize=10)
    train_epoch_done_cond = threading.Condition()
    pool = Pool()
    pool.apply_async(pmd.train_data_extract_process, args=(
        train_epoch_done_cond, stop_train_generating, tdata_queue, data, '/data/ILSVRC2012/train/',))
    pool.close()
    print('-----------------generate train------------------')
    while True:
        if tdata_queue.empty() and data.train_epoch_done():
            break
        else:
            il = tdata_queue.get()
            pass
        assert (len(il) == 2)
        assert (np.asarray(il[0][0]).shape == (2, 256, 256, 3))
        assert (np.asarray(il[0][1]).shape == (3, 256, 256, 3))
        assert (np.asarray(il[1][0]).shape == (2, 2))
        assert (np.asarray(il[1][1]).shape == (3, 2))
        pass

    print('-----------------restart train------------------')
    train_epoch_done_cond.acquire()
    data.restart_train()
    train_epoch_done_cond.notify()
    train_epoch_done_cond.release()
    stop_train_generating.value = True
    pool.join()
    :param train_epoch_done_cond:
    :param stop_train_generating:
    :param tdata_queue:
    :param data:
    :param ilsvrc_root:
    :return:
    '''
    # dgt = data.generate_train(dgt_root)
    while stop_train_generating.value is False:
        train_epoch_done_cond.acquire()
        train_epoch_done_cond.wait_for(lambda : ~data.train_epoch_done())
        i, l = data.generate_train(ilsvrc_root)
        tdata_queue.put((i, l))
        train_epoch_done_cond.release()
        pass
    pass

def evaluate_data_extract_process(evaluate_epoch_done_cond, stop_evaluate_generating, edata_queue, data, ilsvrc_root):
    '''
    example:
    manager = pmd.ILSVRCManager()
    manager.start()
    data = manager.ILSVRC(statistic_file='/data/process_data/caojihua/ILSVRC/statistic_sample.csv', load_truncate=10000,
                      split=0.01, subset_class_amount=2, device_batch=(2, 3), data_drop_rate=0.9)
    stop_evaluate_generating = Manager().Value(bool, False)
    edata_queue = Manager().Queue(maxsize=10)
    evaluate_epoch_done_cond = threading.Condition()
    pool = Pool()
    pool.apply_async(pmd.evaluate_data_extract_process, args=(
        evaluate_epoch_done_cond, stop_evaluate_generating, edata_queue, data,
        '/data/ILSVRC2012/train/',))
    pool.close()
    print('-----------------generate evaluate------------------')
    while True:
        if edata_queue.empty() and data.evaluate_epoch_done():
            break
        else:
            il = edata_queue.get()
            pass
        assert (len(il) == 2)
        assert (np.asarray(il[0][0]).shape == (2, 256, 256, 3))
        assert (np.asarray(il[0][1]).shape == (3, 256, 256, 3))
        assert (np.asarray(il[1][0]).shape == (2, 2))
        assert (np.asarray(il[1][1]).shape == (3, 2))
        pass
    pass
    print('-----------------restart evaluate------------------')
    evaluate_epoch_done_cond.acquire()
    data.restart_evaluate()
    evaluate_epoch_done_cond.notify()
    evaluate_epoch_done_cond.release()
    stop_evaluate_generating.value = True
    pool.join()
    :param evaluate_epoch_done_cond:
    :param stop_evaluate_generating:
    :param edata_queue:
    :param data:
    :param ilsvrc_root:
    :return:
    '''
    while stop_evaluate_generating.value is False:
        evaluate_epoch_done_cond.acquire()
        evaluate_epoch_done_cond.wait_for(lambda: ~data.evaluate_epoch_done())
        i, l = data.generate_evaluate(ilsvrc_root)
        edata_queue.put((i, l))
        evaluate_epoch_done_cond.release()
        pass
    pass
