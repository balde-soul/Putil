# coding=utf-8
import colorama
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
GeneratorLogger = logger.getChild('Generator')
GeneratorLogger.setLevel(plog.DEBUG)
DataPutProcessLogger = logger.getChild('DataPutProcess')
DataPutProcessLogger.setLevel(plog.DEBUG)


class CommonDataManager(BaseManager):
    pass


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
        '''
        '''
        assert 'device_batch' in restart_param.keys(), CommonDataLogger.fatal('device_batch should be found in the restart_param vs. {0}'.format(restart_param))
        self._device_batch = restart_param['device_batch']
        self._restart_process(restart_param)
        self._epoch_done = False
        pass

    def device_batch_operation(self):
        self._device_batch_.acquire()
        self._device_batch_operator()
        self._device_batch_mutex.release()
        pass

    @abstractmethod
    def _generate_from_one_sample(self):
        '''
        this function is call in the generate_data
        generate_data from one sample
        '''
        pass

    @abstractmethod
    def _status_update(self):
        '''
        this function is call in the generate_data
        update the status of the dataset after one sample is generate
        such as update the self._epoch_done: if the sample is the last of the dataset, we should set the sale._epoch_done to True, ortherwise, set to False
        '''
        pass

    def generate_data(self):
        data = self._generate_from_one_sample()
        self._status_update()
        return data
        pass

    @property
    def generate_epoch_done(self):
        return self._epoch_done
        pass
    pass


class CommonDataProxy(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__', 'restart_data', 'generate_data', 'generate_epoch_done')

    def restart_data(self, restart_param=None):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.restart_data.__name__, (restart_param, ))
        pass

    def generate_data(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.generate_data.__name__, ())
        pass

    # @property
    # def generate_epoch_done(self):
    #     callmethod = object.__getattribute__(self, '_callmethod')
    #     return callmethod(self.generate_epoch_done.__name__)
    #     pass
    pass


# def generator(param):
def generator(count, stop_generation, epoch_done_cond, epoch_done_flag, flag_sync_mutex, data, data_queue):
    plog.api_function_log(GeneratorLogger, 'generator start')
    # count = param['count']
    # stop_generation = param['stop_generation']
    # epoch_done_cond = param['epoch_done_cond']
    # epoch_done_flag = param['epoch_done_flag']
    # data = param['data']
    # data_queue = param['data_queue']
    # flag_sync_mutex = param['flag_sync_mutex']
    count = 0
    while stop_generation.value is False:
        epoch_done_cond.acquire()
        if data.generate_epoch_done:
            count = 0
            pass
        epoch_done_cond.wait_for(lambda: data.generate_epoch_done is False)
        get_data = data.generate_data()
        flag_sync_mutex.acquire()
        data_queue.put(get_data)
        epoch_done_flag.value = data.generate_epoch_done
        flag_sync_mutex.release()
        count += 1
        epoch_done_cond.release()
        pass
    plog.api_function_log(GeneratorLogger, 'data put process end')
    pass


def a(count, stop_generation, epoch_done_cond, epoch_done_flag, data, data_queue):
    print(count)
    print('sdasdasdasda')
    pass


class DataPutProcess:
    def __init__(self, data, manager, pool, *argc, **argv):
        '''
        data: the CommonData Object
        manager: the multiprocessing.Manager()
        pool: the multiprocessing.Manager().Pool()
        'queue_size': the size of the data queue
        '''
        plog.api_function_log(DataPutProcessLogger, 'DataPutProcess')

        self._data = data

        if 'queue_size' in argv.keys():
            self._queue_size = argv.pop('queue_size')
            pass
        else:
            self._queue_size = 32
            pass

        self._epoch_done_cond = manager.Condition()
        self._epoch_done_flag = manager.Value(bool, False)
        self._flag_sync_mutex = manager.Lock()
        self._stop_generation = manager.Value(bool, False)
        self._data_queue = manager.Queue(maxsize=self._queue_size)

        self._count = manager.Value(int, 0)

        self._epoch_done_cond.acquire()
        self._first_init = True

        # param = manager.dict()
        # param['count'] = self._count
        # param['stop_generation'] = self._stop_generation
        # param['epoch_done_cond'] = self._epoch_done_cond
        # param['epoch_done_flag'] = self._epoch_done_flag
        # param['flag_sync_mutex'] = self._flag_sync_mutex
        # param['data'] = self._data
        # param['data_queue'] = self._data_queue
        # print(param)
        # self._ret = pool.apply_async(generator, args=(param,))
        self._ret = pool.apply_async(generator, args=(self._count, self._stop_generation, self._epoch_done_cond, self._epoch_done_flag, self._flag_sync_mutex, self._data, self._data_queue))
        # self._ret = pool.apply_async(generator, args=())
        # self._ret = pool.apply_async(a, args=(self._count, self._stop_generation, self._epoch_done_cond, self._epoch_done_flag, self._data, self._data_queue))
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

    @property
    def has_next(self):
        self._flag_sync_mutex.acquire()
        # print(self._data_queue.empty())
        # print(self._epoch_done_flag.value)
        flag = (self._data_queue.empty() is False or self._epoch_done_flag.value is False)
        self._flag_sync_mutex.release()
        return flag
        pass

    def restart(self, **kwargs):
        plog.api_function_log(DataPutProcessLogger, 'restart')
        restart_param = mlp.Manager().dict()
        if 'device_batch' not in kwargs:
            restart_param['device_batch'] = [1]
            pass
        else:
            restart_param['debice_batch'] = kwargs['device_batch']
            pass
        for key in kwargs.keys():
            restart_param[key] = kwargs[key]
            pass
        if self._first_init is False:
            # print('a')
            self._epoch_done_cond.acquire()
            pass
        else:
            self._first_init = False
            # print('b')
            pass
        self._data.restart_data(restart_param)
        self._epoch_done_flag.value = False
        self._epoch_done_cond.notify_all()
        self._epoch_done_cond.release()
        pass

    def stop_generation(self):
        plog.api_function_log(DataPutProcessLogger, 'stop_generation')
        restart_param = mlp.Manager().dict()
        restart_param['device_batch'] = [1]
        self._epoch_done_cond.acquire()
        self._data.restart_data(restart_param)
        self._stop_generation.value = True
        self._epoch_done_cond.notify_all()
        self._epoch_done_cond.release()
        while self._data_queue.empty() is False:
            self._data_queue.get()
            pass
        pass

    @property
    def Count(self):
        return self._count.value
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

    @property
    def queue_process_ret(self):
        return self._ret
        pass
    pass


class DataPutProcessProxy(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__', 'pause_queue', 'continue_queue', 'restart', 'stop_generation', 'Count', 'DataQueue', 'EpochDoneCond', 'EpochDoneFlag')

    def pause_queue(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.pause_queue())
        pass

    def continue_queue(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.continue_queue())
        pass

    def restart(self, restart_param):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.restart(restart_param))
        pass

    def stop_generation(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.stop_operation())
        pass

    @property
    def Count(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.Count)
        pass

    @property
    def DataQueue(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.DataQueue)
        pass

    @property
    def EpochDoneCond(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.EpochDoneCond)
        pass

    @property
    def EpochDoneFlag(self):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.EpochDoneFlag)
        pass
    pass


CommonDataManager.register('DataPutProcess', DataPutProcess, proxytype=DataPutProcessProxy)
