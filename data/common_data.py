# coding=utf-8
import Putil.PutilEnvSet as penv
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
GeneratedDataLogger = logger.getChild('GeneratedData')
GeneratedDataLogger.setLevel(plog.DEBUG)


class CommonDataManager(BaseManager):
    pass


class ProxyBase(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__')
    pass


class IndexInfo:
    def __init__(self, point, type, *args, **kwargs):
        self._point = point
        self._type = type
        pass

    def point(self):
        return self._point
        pass

    def type(self):
        return self._type
        pass
    pass


class DataIndexInfo:
    def __init__(self, start, end, index_info, *args, **kwargs):
        self._range = [start, end]
        self._index_info = index_info
        pass

    def index_info(self):
        return self._index_info
        pass

    def data_range(self):
        return self._range
        pass
    pass


class GeneratedData:
    def __init__(self, data_list, index_info_list):
        self._indexs = [DataIndexInfo(0, 0, None, '')]
        for data, index_info, data_order in zip(data_list, index_info_list, range(0, len(data_list))):
            pre_range = self._indexs[data_order].data_range()
            self._indexs.append(DataIndexInfo(pre_range[0], pre_range[1] + data.shape[0], index_info))
            pass
        self._indexs.pop(0)
        self._indexs = np.array(self._indexs)
        self._data = np.concatenate(data_list, axis=0) if len(data_list) > 1 else data_list[0]
        pass

    def datas(self):
        return self._data
        pass

    def indexs(self):
        return self._indexs
        pass

    def data(self, index):
        pass
    pass

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class CommonData(ABC, Dataset):
    '''
    this class provide a common method to read the data
    '''
    def __init__(self):
        self._device_batch_mutex = threading.Lock()
        self._device_batch = None
        self._critical_process = None
        self._epoch_done = False

        # a list which contain number(int) which represent the index of the data
        self._data_field = None
        # a number(int) which the represent data read now
        self._index = None
        pass

    def data_field(self):
        return self._data_field

    def index(self):
        return self._index

    @abstractmethod
    def _restart_process(self, restart_param):
        '''
        process while restart the data, process in the derived class and called by restart_data
        restart_param: the argv which the derived class need, dict
        '''
        pass

    @abstractmethod
    def _inject_operation(self, inject_param):
        '''
        operation while the epoch_done is False, process in the derived class and called by inject_operation
        injecct_param: the argv which the derived class need, dict
        '''
        pass

    def set_seed(self, seed):
        np.random.seed(seed)
        pass

    @abstractmethod
    def _generate_from_specified(self, index):
        '''
        this function is call in the generate_data, using the specified id from the data_set_field to get the data from the id
        the format of the data returned should be: {key: data.follow(shape: [batch, **])}
        '''
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
        self._index = 0
        pass

    def reset_param(self, reset_param):
        pass

    def inject_operation(self, inject_param):
        '''
        inject some operation while whenevery, no matter the epoch_done is True or False
        inject_param: the argv which operation needed
            'device_batch': this virtual class needed
            'critical_process': this virtual class needed
        '''
        if 'device_batch' in inject_param:
            self._device_batch = inject_param['device_batch']
            pass
        if 'critical_process' in inject_param:
            self._critical_process = inject_param['critical_process']
            pass
        self._inject_operation(inject_param)
        pass

    def reset_index(self, index):
        self._index = index
        pass

    def device_batch_operation(self):
        self._device_batch_.acquire()
        self._device_batch_operator()
        self._device_batch_mutex.release()
        pass

    def _data_set_field(self):
        return self._data_field
        pass

    def _status_update(self):
        '''
        this function is call in the generate_data
        update the status of the dataset after one sample is generate
        such as update the self._epoch_done: if the sample is the last of the dataset, we should set the sale._epoch_done to True, ortherwise, set to False
        '''
        self._epoch_done = True if self._index == len(self._data_field) - 1 else False
        pass

    def generate_data(self):
        '''
        this function return the data with content follow the format: [{key: GeneratedData}.follow(data_name)].follow(device)
        in fact: the data is pack in to the GeneratedData
        the first dimesion is the device
        the second dimesion is the batch
        return the ndarray
        '''
        CommonDataLogger.debug('generate_data')
        func_for_deal_with_epoch_done = None
        devices_data = []
        index_data = []
        need_batch = copy.deepcopy(self._device_batch)
        [devices_data.append(dict()) for device in self._device_batch]
        [index_data.append(dict()) for device in self._device_batch]
        while np.sum(need_batch) > 0:
            for ergodic_device in enumerate(need_batch):
                CommonDataLogger.debug(ergodic_device)
                batch = ergodic_device[1]
                device_order = ergodic_device[0]
                if batch > 0:
                    # get the data
                    if self._epoch_done is False:
                        CommonDataLogger.debug('epoch not end')
                        data = self._generate_from_specified(self._index)
                        alignment_batch = None
                        old_item_name = None
                        for item in data.items():
                            CommonDataLogger.debug('deal with {0}'.format(item[0]))
                            alignment_batch = item[1].shape[0] if alignment_batch is None else alignment_batch
                            assert alignment_batch == item[1].shape[0], CommonDataLogger.fatal('the return data should be alignmented in batch {0} in {1} vs {2} in {3}'.format(alignment_batch, old_item_name, item[0], item[1].shape[0]))
                            old_item_name = item[0]
                            pass
                        self._status_update()
                        need_batch[device_order] = 0 if alignment_batch > batch else batch - alignment_batch
                        for key, value in data.items():
                            devices_data[device_order][key] = list() if key not in devices_data[device_order] else devices_data[device_order][key]
                            index_data[device_order][key] = list() if key not in index_data[device_order] else index_data[device_order][key]
                            devices_data[device_order][key].append(value[0: batch] if value.shape[0] > batch else value)
                            index_data[device_order][key].append(IndexInfo(self._index, 'normal'))
                            pass
                        CommonDataLogger.debug('fix over')
                        self._index += 1
                        pass
                    else:
                        def deal_with_epoch_done():
                            field = self._data_set_field()

                            def func():
                                random_sample = np.random.choice(field)
                                return self._generate_from_specified(random_sample), random_sample
                                pass
                            return func
                            pass
                        func_for_deal_with_epoch_done = deal_with_epoch_done() if func_for_deal_with_epoch_done is None else func_for_deal_with_epoch_done
                        if self._critical_process == 'allow_low' and batch != self._device_batch[device_order]:
                            CommonDataLogger.debug('method: allow_low and already data in device')
                            need_batch[device_order] = 0
                            pass
                        elif self._critical_process == 'allow_low' and batch == self._device_batch[device_order]:
                            CommonDataLogger.debug('method: allow_low and no data in device')
                            data, index = func_for_deal_with_epoch_done()
                            need_batch[device_order] = 0
                            for key, value in data.items():
                                devices_data[device_order][key] = list() if key not in devices_data[device_order] else devices_data[device_order][key]
                                index_data[device_order][key] = list() if key not in index_data[device_order] else index_data[device_order][key]
                                devices_data[device_order][key].append(value[0: batch] if value.shape[0] > batch else value)
                                index_data[device_order][key].append(IndexInfo(index, 'allow_low'))
                            pass
                        elif self._critical_process == 'random_fill':
                            CommonDataLogger.debug('method: random_fill')
                            data, index = func_for_deal_with_epoch_done()
                            alignment_batch = None
                            for item in data.items():
                                alignment_batch = item[1].shape[0] if alignment_batch is None else alignment_batch
                                assert alignment_batch == item[1].shape[0], CommonDataLogger.fatal('the return data should be alignmented in batch {0} in {1} vs {2} in {3}'.format(alignment_batch, old_item_name, item[0], item[1].shape[0]))
                                old_item_name = item[0]
                                pass
                            need_batch[device_order] = 0 if alignment_batch > batch else batch - alignment_batch
                            for key, value in data.items():
                                devices_data[device_order][key] = list() if key not in devices_data[device_order] else devices_data[device_order][key]
                                index_data[device_order][key] = list() if key not in index_data[device_order] else index_data[device_order][key]
                                devices_data[device_order][key].append(value[0: batch] if value.shape[0] > batch else value)
                                index_data[device_order][key].append(IndexInfo(index, 'random_fill'))
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
        for i in devices_data:
            for key, value in i.items():
                for d in value:
                    CommonDataLogger.debug('{0} {1}'.format(key, d.shape))
                    pass
                pass
            pass
        CommonDataLogger.debug('going to generate data')
        data = [{key: GeneratedData(datas[key], indexs[key]) for key in datas.keys()} for datas, indexs in zip(devices_data, index_data)]
        # data = [GeneratedData(datas, indexs) for datas, indexs in zip(devices_data, index_data)]
        return data
        pass

    def generate_epoch_done(self):
        return self._epoch_done

    def __getitem__(self, index):
        return self._generate_from_specified(index)

    def __len__(self):
        return len(self._data_field)

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
        if data.generate_epoch_done():
            count = 0
            pass
        epoch_done_cond.wait_for(lambda: data.generate_epoch_done() is False)
        try:
            get_data = data.generate_data()
            pass
        except Exception as ex:
            GeneratorLogger.fatal('str(Exception):\t', str(Exception))
            GeneratorLogger.fatal('str(e):\t\t', str(e))
            GeneratorLogger.fatal('repr(e):\t', repr(e))
            GeneratorLogger.fatal('e.message:\t', e.message)
            GeneratorLogger.fatal('traceback.print_exc():', traceback.print_exc())
            GeneratorLogger.fatal('traceback.format_exc():\n%s' % traceback.format_exc())
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
        self._epoch_done_flag = manager.Value(bool, True)
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
        plog.api_function_log(DataPutProcessLogger, 'pause the queue')
        self._epoch_done_cond.acquire()
        self._paused = True
        pass

    def inject_operation(self, param, **kwargs):
        '''
        this function cound be call while in the epoch processing,
        param: the argv which needed by DataPutProcess
            'recycle': bool, recycle the data in the data_queue or not, if true the data in the data_queue would be pop out and the index in the CommonData would be set to the index of the first data in the data_queue
        **kwargs: the argv which needed by the ComonData
        '''
        self.pause_queue()

        assert 'recycle' in param, DataPutProcessLogger.fatal('recycle should be foud in the param vs. {0}'.format(param))
        recycle = param['recycle']

        if (recycle is True) and (self._data_queue.empty() is False):
            get = self._data_queue.get()
            alignment_index = None
            the_first = get[0]
            for k, v in the_first.items():
                alignment_index = v.indexs()[0].index_info().point() if alignment_index is None else alignment_index
                assert alignment_index == v.indexs()[0].index_info().point()
            self._data.reset_index(alignment_index)
            while self._data_queue.empty() is False:
                self._data_queue.get()
                pass
            assert self._data_queue.empty() is True
            pass

        data_reject_param = mlp.Manager().dict()
        for key, value in kwargs.items():
            data_reject_param[key] = value
        self._data.inject_operation(data_reject_param)

        before_generate_after_inject_data_queue = self._data_queue.qsize()

        self.continue_queue()

        return before_generate_after_inject_data_queue
        pass

    def continue_queue(self):
        plog.api_function_log(DataPutProcessLogger, 'continue the queue')
        self._epoch_done_cond.release()
        self._paused = False
        pass

    def has_next(self):
        self._flag_sync_mutex.acquire()
        # print(self._data_queue.empty())
        # print(self._epoch_done_flag.value)
        flag = (self._data_queue.empty() is False or self._epoch_done_flag.value is False)
        self._flag_sync_mutex.release()
        return flag
        pass

    def paused_and_has_next(self):
        self._flag_sync_mutex.acquire()
        flag = self._data_queue.empty() is False and self._paused is True
        self._flag_sync_mutex.release()
        return flag
        pass

    def restart(self, **kwargs):
        plog.api_function_log(DataPutProcessLogger, 'restart')
        restart_param = mlp.Manager().dict()
        restart_param['device_batch'] = kwargs.get('device_batch', [1])
        # if 'device_batch' not in kwargs:
        #     restart_param['device_batch'] = [1]
        #     pass
        # else:
        #     restart_param['device_batch'] = kwargs['device_batch']
        #     pass
        restart_param['critical_process'] = kwargs.get('critical_process', 'random_fill')
        # if 'critical_process' not in kwargs:
        #     restart_param['critical_process'] = 'random_fill'
        #     pass
        # else:
        #     restart_param['critical_process'] = kwargs['critical_process']
        #     pass
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
    
    def __iter__(self):
        return self
        pass

    def __next__(self):
        if self.has_next():
            return self._data_queue.get()
        else:
            raise StopIteration()
        pass
