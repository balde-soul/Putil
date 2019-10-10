# coding=utf-8
import Putil.base.logger as plog
from abc import ABC, abstractmethod
import threading
import multiprocessing as mlp
from multiprocessing import Manager
from multiprocessing.managers import BaseManager, NamespaceProxy


logger = plog.PutilLogConfig('common_data').logger()
logger.setLevel(plog.DEBUG)
CommonDataLogger = logger.getChild('CommonData')
CommonDataLogger.setLevel(plog.DEBUG)


class CommonData(ABC):
    '''
    this class provide a common method to read the data
    '''
    def __init__(self):
        self._device_batch_mutex = threading.Lock()
        self._device_batch = None
        self._epoch_done = False
        pass

    def set_config(self, config):
        pass

    @abstractmethod
    def _restart_process(self, restart_param):
        pass

    def restart_data(self, restart_param):
        assert 'device_batch' in restart_param.keys(), CommonDataLogger.error('restart_param should contain {device_batch}')
        self._device_batch = restart_param['device_batch']
        pass

    def device_batch_operation(self):
        self._device_batch_.acquire()
        self._device_batch_operator()
        self._device_batch_mutex.release()
        pass

    @abstractmethod
    def _generate_from_one_sample(self):
        pass

    def generate_data(self):
        return self._generate_from_one_sample()
        pass

    @property
    def generate_epoch_done(self):
        return self._epoch_done
        pass
    pass


class CommonDataManager(BaseManager):
    pass


class CommonDataProxy(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__', 'restart_data', 'generate_data', 'generate_epoch_done')

    def restart_data(self, restart_param):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.restart_data(restart_param))
        pass

    def generate_data(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.generate_data())
        pass

    def generate_epoch_done(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.generate_epoch_done())
        pass
    pass


def generator(**argv):
    count = argv.pop('count')
    stop_generation = argv.pop('stop_generation')
    epoch_done_cond = argv.pop('epoch_done_cond')
    epoch_done_flag = argv.pop('epoch_done_flag')
    data = argv.pop('data')
    data_queue = argv.pop('data_queue')
    count = 0
    while stop_generation.value is False:
        epoch_done_cond.acquire()
        if data.generate_epoch_done():
            count = 0
            pass
        epoch_done_cond.wait_for(lambda: data.generate_epoch_done is False)
        get_data = data.generate_data()
        data_queue.put(get_data)
        count += 1
        epoch_done_flag.value = data.generate_epoch_done()
        epoch_done_cond.release()
        pass
    pass


class DataPutProcess(ABC):
    def __init__(self, data, manager, pool, *argc, **argv):
        '''
        data: the CommonData Object
        manager: the multiprocessing.Manager()
        pool: the multiprocessing.Manager().Pool()
        'queue_size': the size of the data queue
        '''
        self._pool = pool
        self._manager = manager
        self._data = data

        if 'queue_size' in argv.keys():
            self._queue_size = argv.pop('queue_size')
            pass
        else:
            self._queue_size = 32
            pass

        self._epoch_done_cond = self._manager.Condition()
        self._epoch_done_flag = self._manager.Value(bool, False)
        self._stop_generation = self._manager.Value(bool, False)
        self._data_queue = self._manager.Queue(maxsize=self._queue_size)

        self._count = self._manager.Value(int, 0)

        self._epoch_done_cond.acquire()
        self._first_init = True

        argv = self._manager.dict()
        argv['count'] = self._count
        argv['stop_generation'] = self._stop_generation
        argv['epoch_done_cond'] = self._epoch_done_cond
        argv['epoch_done_flag'] = self._epoch_done_flag
        argv['data'] = self._data
        argv['data_queue'] = self._data_queue
        self._ret = self._pool.apply_async(generator, args=(argv))
        pass

    def pause_queue(self):
        self._epoch_done_cond.acquire()
        self._paused = True
        pass

    def continue_queue(self):
        self._epoch_done_cond.notify_all()
        self._epoch_done_cond.release()
        self._paused = False
        pass

    def restart(self, restart_param):
        self._epoch_done_cond.acquire() if self._first_init is False else None
        self._data.restart_data(restart_param)
        self._epoch_done_flag.value = False
        self._epoch_done_cond.notify_all()
        self._epoch_done_cond.release()
        pass

    def stop_generation(self):
        self._epoch_done_cond.acquire()
        self._data.restart_data()
        self._epoch_done_flag.value = True
        self._epoch_done_cond.notify_all()
        self._epoch_done_cond.release()
        while self._data_queue.empty() is False:
            self._data_queue.get()
            pass
        pass

    @property
    def Count(self):
        return self._count
        pass

    @property
    def DataQueue(self):
        return self._data_queue
        pass

    @property
    def EpochDoneCond(self):
        return self._epoch_done_cond
        pass

    @property
    def EpochDoneFlag(self):
        return self._epoch_done_flag
        pass
    pass
