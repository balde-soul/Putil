# coding=utf-8
import inspect
import traceback
import copy
import numpy as np
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


class ProxyBase(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__')
    pass


class CommonData(ABC):
    '''
    this class provide a common method to read the data
    '''
    def __init__(self):
        self._device_batch_mutex = threading.Lock()
        self._device_batch = None
        self._critical_process = None
        self._epoch_done = False
        pass

    def set_config(self, config):
        pass

    @abstractmethod
    def _restart_process(self, restart_param):
        pass

    def restart_data(self, restart_param):
        '''
        restart_param:
            'device_batch': batch size for every device, list with int
            'critical_process': the method while epoch_done got during one data_batch is generating, string, support 'allow_low', 'random_fill'
        '''
        assert 'device_batch' in restart_param.keys(), CommonDataLogger.fatal('device_batch should be found in the restart_param vs. {0}'.format(restart_param))
        self._device_batch = restart_param['device_batch']
        assert 'critical_process' in restart_param.keys(), CommonDataLogger.fatal('critical_process should be found in the restart_param vs. {0}'.format(restart_param))
        self._critical_process = restart_param['critical_process']
        self._restart_process(restart_param)
        self._epoch_done = False
        pass

    def device_batch_operation(self):
        self._device_batch_.acquire()
        self._device_batch_operator()
        self._device_batch_mutex.release()
        pass

    # @abstractmethod
    # def _inject_operation(self, inject_param):
    #     pass

    # def inject_operation(self, inject_param):
    #     '''
    #     this method can be call between every data generation
    #     '''
    #     CommonDataLogger.fatal('unsupported method')
    #     assert 'type' in inject_param.keys(), CommonDataLogger.fatal('type should be found in the inject_param vs. {0}'.format(inject_param))
    #     self._inject_operation_type = inject_param['type']
    #     if self._inject_operation == 'restart':
    #         pass
    #     pass

    @abstractmethod
    def _generate_from_one_sample(self):
        '''
        this function is call in the generate_data
        generate_data from one sample
        return the ndarray
        '''
        pass

    @abstractmethod
    def _generate_from_specified(self, index):
        '''
        this function is call in the generate_data, using the specified id from the data_set_field to get the data from the id
        '''
        pass

    @abstractmethod
    def _data_set_field(self):
        '''
        this function return the data_set_field, which contain the id of all data
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
        '''
        this function return the data follow the format: 'D, B, *, *'
        the first dimesion is the device
        the second dimesion is the batch
        return the ndarray
        '''
        func_for_deal_with_epoch_done = None
        devices_data = []
        need_batch = copy.deepcopy(self._device_batch)
        [devices_data.append([]) for device in self._device_batch]
        while np.sum(need_batch) > 0:
            for ergodic_device in zip(need_batch, range(0, len(need_batch))):
                batch = ergodic_device[0]
                device_order = ergodic_device[1]
                if batch > 0:
                    # get the data
                    if self._epoch_done is False:
                        data = self._generate_from_one_sample()
                        self._status_update()
                        need_batch[device_order] = 0 if data.shape[0] > batch else batch - data.shape[0]
                        devices_data[device_order].append(data[0: batch] if data.shape[0] > batch else data)
                        pass
                    else:
                        def deal_with_epoch_done():
                            field = self._data_set_field()

                            def func():
                                random_sample = np.random.choice(field)
                                return self._generate_from_specified(random_sample)
                                pass
                            return func
                            pass
                        func_for_deal_with_epoch_done = deal_with_epoch_done() if func_for_deal_with_epoch_done is None else func_for_deal_with_epoch_done
                        if self._critical_process == 'allow_low' and batch != self._device_batch[device_order]:
                            need_batch[device_order] = 0
                            pass
                        elif self._critical_process == 'allow_low' and batch == self._device_batch[device_order]:
                            data = func_for_deal_with_epoch_done()
                            need_batch[device_order] = 0
                            devices_data[device_order].append(data[0: batch] if data.shape[0] > batch else data)
                            pass
                        elif self._critical_process == 'random_fill':
                            data = func_for_deal_with_epoch_done()
                            need_batch[device_order] = 0 if data.shape[0] > batch else batch - data.shape[0]
                            devices_data[device_order].append(data[0: batch] if data.shape[0] > batch else data)
                            pass
                        else:
                            CommonDataLogger.fatal('this should not happen')
                            pass
                        pass
                    # add the data to the
                    pass
                else:
                    pass
                pass
            pass
        data = [np.concatenate(device_data, axis=0) if len(device_data) > 1 else device_data[0] for device_data in devices_data]
        return data
        pass

    def generate_epoch_done(self):
        return self._epoch_done
        pass
    pass


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
        epoch_done_cond.wait_for(lambda: data.generate_epoch_done() is False)
        try:
            get_data = data.generate_data()
            pass
        except Exception as ex:
            GeneratorLogger.fatal(traceback.format_tb(ex.__traceback__))
            raise ex
            pass
        flag_sync_mutex.acquire()
        data_queue.put(get_data)
        epoch_done_flag.value = data.generate_epoch_done()
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

        self._ret = pool.apply_async(generator, args=(self._count, self._stop_generation, self._epoch_done_cond, self._epoch_done_flag, self._flag_sync_mutex, self._data, self._data_queue))
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
            restart_param['device_batch'] = kwargs['device_batch']
            pass
        if 'critical_process' not in kwargs:
            restart_param['critical_process'] = 'random_fill'
            pass
        else:
            restart_param['critical_process'] = kwargs['critical_process']
            pass
        for key, value in kwargs.items():
            restart_param[key] = value
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
        self._set_default_restart_param(restart_param)
        self._epoch_done_cond.acquire()
        self._data.restart_data(restart_param)
        self._stop_generation.value = True
        self._epoch_done_cond.notify_all()
        self._epoch_done_cond.release()
        while self._data_queue.empty() is False:
            self._data_queue.get()
            pass
        pass

    def _set_default_restart_param(self, restart_param):
        restart_param['device_batch'] = self._default_device_batch()
        restart_param['critical_process'] = self._default_critical_process()
        pass

    def _default_device_batch(self):
        return [1]
        pass

    def _default_critical_process(self):
        return 'random_fill'
        pass

    def Count(self):
        return self._count.value
        pass

    def DataQueue(self):
        return self._data_queue
        pass

    def EpochDoneCond(self):
        return self._epoch_done_cond
        pass

    def EpochDoneFlag(self):
        return self._epoch_done_flag
        pass

    def queue_process_ret(self):
        return self._ret
        pass
