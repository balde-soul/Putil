# coding=utf-8
import os
import queue
import numpy as np
from enum import Enum

from Putil.data.torch_151_data import dataset, dataloader
from Putil.data.torch_151_data.dataloader import _BaseDataLoaderIter

import threading
import itertools
import warnings

import multiprocessing as python_multiprocessing
import torch
import torch.multiprocessing as multiprocessing
from torch._utils import ExceptionWrapper
from torch._six import queue, string_classes

from Putil.data.torch_151_data.dataloader import IterableDataset, Sampler, SequentialSampler, RandomSampler, BatchSampler, _DatasetKind, _utils

import Putil.base.logger as plog

root_logger = plog.PutilLogConfig('data_loader').logger()
root_logger.setLevel(plog.DEBUG)
CWorkLogger = root_logger.getChild('CWork')
CWorkLogger.setLevel(plog.DEBUG)
IndexPutLogger = root_logger.getChild('IndexPut')
IndexPutLogger.setLevel(plog.DEBUG)

get_worker_info = _utils.worker.get_worker_info

default_collate = _utils.collate.default_collate


class DataLoader(object):
    #def __del__(self):
    #    if self._multi_processing_data_loader_iter is not None:
    #        del self._multi_processing_data_loader_iter 
    #    if self._signle_process_data_loader_iter is not None:
    #        del self._signle_process_data_loader_iter 
    #    pass
    __initialized = False

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
        torch._C._log_api_usage_once("python.data_loader")

        if num_workers < 0:
            raise ValueError('num_workers option should be non-negative; '
                             'use num_workers=0 to disable multiprocessing.')

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        self.dataset = dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            # NOTE [ Custom Samplers and `IterableDataset` ]
            #
            # `IterableDataset` does not support custom `batch_sampler` or
            # `sampler` since the key is irrelevant (unless we support
            # generator-style dataset one day...).
            #
            # For `sampler`, we always create a dummy sampler. This is an
            # infinite sampler even when the dataset may have an implemented
            # finite `__len__` because in multi-process data loading, naive
            # settings will return duplicated data (which may be desired), and
            # thus using a sampler with length matching that of dataset will
            # cause data lost (you may have duplicates of the first couple
            # batches, but never see anything afterwards). Therefore,
            # `Iterabledataset` always uses an infinite sampler, an instance of
            # `_InfiniteConstantSampler` defined above.
            #
            # A custom `batch_sampler` essentially only controls the batch size.
            # However, it is unclear how useful it would be since an iterable-style
            # dataset can handle that within itself. Moreover, it is pointless
            # in multi-process data loading as the assignment order of batches
            # to workers is an implementation detail so users can not control
            # how to batchify each worker's iterable. Thus, we disable this
            # option. If this turns out to be useful in future, we can re-enable
            # this, and support custom samplers that specify the assignments to
            # specific workers.
            if shuffle is not False:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "shuffle option, but got shuffle={}".format(shuffle))
            elif sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "sampler option, but got sampler={}".format(sampler))
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "batch_sampler option, but got batch_sampler={}".format(batch_sampler))
        else:
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if shuffle or drop_last:
                raise ValueError('batch_size=None option disables auto-batching '
                                 'and is mutually exclusive with '
                                 'shuffle, and drop_last')

        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.__initialized = True
        self._IterableDataset_len_called = None  # See NOTE [ IterableDataset and __len__ ]
        self._signle_process_data_loader_iter = None
        self._multi_processing_data_loader_iter = None

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if not multiprocessing._supports_context:
                    raise ValueError('multiprocessing_context relies on Python >= 3.4, with '
                                     'support for different start methods')

                if isinstance(multiprocessing_context, string_classes):
                    valid_start_methods = multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            ('multiprocessing_context option '
                             'should specify a valid start method in {}, but got '
                             'multiprocessing_context={}').format(valid_start_methods, multiprocessing_context))
                    multiprocessing_context = multiprocessing.get_context(multiprocessing_context)

                if not isinstance(multiprocessing_context, python_multiprocessing.context.BaseContext):
                    raise ValueError(('multiprocessing_context option should be a valid context '
                                      'object or a string specifying the start method, but got '
                                      'multiprocessing_context={}').format(multiprocessing_context))
            else:
                raise ValueError(('multiprocessing_context can only be used with '
                                  'multi-process loading (num_workers > 0), but got '
                                  'num_workers={}').format(self.num_workers))

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(DataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        if self.num_workers == 0:
            #if self._multi_processing_data_loader_iter is not None:
            #    del self._multi_processing_data_loader_iter
            if self._signle_process_data_loader_iter is None:
                self._signle_process_data_loader_iter = _SingleProcessDataLoaderIter(self)
            else:
                self._signle_process_data_loader_iter.restart()
            return self._signle_process_data_loader_iter
        else:
            #if self._signle_process_data_loader_iter is not None:
            #    del self._signle_process_data_loader_iter 
            if self._multi_processing_data_loader_iter is None:
                self._multi_processing_data_loader_iter = _MultiProcessingDataLoaderIter(self)
            else:
                self._multi_processing_data_loader_iter.restart()
            return self._multi_processing_data_loader_iter

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self):
        if self._dataset_kind == _DatasetKind.Iterable:
            # NOTE [ IterableDataset and __len__ ]
            #
            # For `IterableDataset`, `__len__` could be inaccurate when one naively
            # does multi-processing data loading, since the samples will be duplicated.
            # However, no real use case should be actually using that behavior, so
            # it should count as a user error. We should generally trust user
            # code to do the proper thing (e.g., configure each replica differently
            # in `__iter__`), and give us the correct `__len__` if they choose to
            # implement it (this will still throw if the dataset does not implement
            # a `__len__`).
            #
            # To provide a further warning, we track if `__len__` was called on the
            # `DataLoader`, save the returned value in `self._len_called`, and warn
            # if the iterator ends up yielding more than this number of samples.
            length = self._IterableDataset_len_called = len(self.dataset)
            return length
        else:
            return len(self._index_sampler)


def try_get_index(index_queue):
    pass


def cwork(dataset, index_queue, data_queue, cond, command, index_status):
    while True:
        index = index_queue.get()
        CWorkLogger.debug('get {} {}'.format(index, os.getpid()))
        if index is None:
            data_queue.put(None)
            _command = command.get()
            CWorkLogger.debug('get {} {}'.format(_command, os.getpid()))
            if _command == _MultiProcessingDataLoaderIter.Command.ContinueTheIteration:
                continue
            if _command == _MultiProcessingDataLoaderIter.Command.StopTheIteration:
                break
            pass
        data = list()
        for idx, _index in enumerate(index):
            _data = dataset[_index]
            if idx == 0:
                [data.append([item]) for item in _data]
                pass
            else:
                [data[item_idx].append(item) for item_idx, item in enumerate(_data)]
                pass
            pass
        for _idx, _data in enumerate(data):
            data[_idx] = np.stack(data[_idx], axis=0)
        data_queue.put(data)
        pass
    pass


def index_put(batch_sampler, index_queue, rcond, rcommand, num_workers):
    while True:
        for i in batch_sampler:
            index_queue.put(i)
            pass
        for i in range(num_workers):
            index_queue.put(None)
        _rcommand = rcommand.get()
        IndexPutLogger.debug('get {} {}'.format(_rcommand, os.getpid()))
        if _rcommand == _MultiProcessingDataLoaderIter.Command.StopTheIteration:
            break
        pass
    pass


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    class IndexStatus(Enum):
        End = 0
        Going = 1

    class Command(Enum):
        StopTheIteration = 0
        ContinueTheIteration = 1

    def __init__(self, loader, max_queue_size=8):
        _BaseDataLoaderIter.__init__(self, loader)
        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context
        self._index_queue = multiprocessing_context.Queue(max_queue_size)
        self._data_queue = multiprocessing_context.Queue(max_queue_size)
        self._command = list()
        self._cond = list()
        self._index_status = list()
        self._process = list()
        self._index_status_record = list()
        for i in range(0, self._num_workers):
            lock = multiprocessing_context.Lock()
            cond = multiprocessing_context.Condition(lock)
            self._cond.append(lock)
            command = multiprocessing_context.Queue(max_queue_size)
            self._command.append(command)
            index_status = multiprocessing_context.Queue(max_queue_size)
            self._index_status.append(index_status)
            process = multiprocessing_context.Process(target=cwork, 
            args=(self._dataset, self._index_queue, self._data_queue, lock, command, index_status)) 
            process.daemon = True
            process.start()
            self._process.append(process)
            self._index_status_record.append(_MultiProcessingDataLoaderIter.IndexStatus.Going)
            pass
        self._index_control_scond = multiprocessing_context.Condition(multiprocessing_context.Lock())
        self._index_control_scommand = multiprocessing_context.Queue(max_queue_size)
        self._index_put_process = multiprocessing_context.Process(target=index_put, 
        args=(loader.batch_sampler, self._index_queue, self._index_control_scond, self._index_control_scommand, self._num_workers))
        self._index_put_process.start()

        self._none_count = 0
        pass

    def _next_data(self):
        data = self._data_queue.get()
        while data is None:
            self._none_count += 1
            if self._none_count == self._num_workers:
                self._none_count = 0
                raise StopIteration
            data = self._data_queue.get()
            pass
        return data
        pass

    def restart(self):
        self._none_count = 0
        self._index_control_scommand.put(_MultiProcessingDataLoaderIter.Command.ContinueTheIteration)
        [command.put(_MultiProcessingDataLoaderIter.Command.ContinueTheIteration) for command in self._command]
        pass

    def __exit__(self, exc_type, exc_value, traceback):
    #def __del__(self):
        #import pdb; pdb.set_trace()
        self._index_control_scommand.put(_MultiProcessingDataLoaderIter.Command.StopTheIteration)
        #self._index_control_scommand.put(_MultiProcessingDataLoaderIter.Command.StopTheIteration)
        self._index_put_process.join()
        [command.put(_MultiProcessingDataLoaderIter.Command.StopTheIteration) for command in self._command]
        [process.join() for process in self._process]
        pass
