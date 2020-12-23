# coding=utf-8
from enum import Enum
import random
import Putil.PutilEnvSet as penv
import inspect
import traceback
import copy
import numpy as np
import colorama
import Putil.base.logger as plog
from abc import ABC, abstractmethod, ABCMeta
import threading
import multiprocessing as mlp
from multiprocessing import Manager
from multiprocessing.managers import BaseManager, NamespaceProxy


logger = plog.PutilLogConfig('common_data').logger()
logger.setLevel(plog.DEBUG)
CommonDataLogger = logger.getChild('CommonData')
CommonDataLogger.setLevel(plog.DEBUG)
CommonDataWithAugLogger = logger.getChild('CommongDataWithAug')
CommonDataWithAugLogger.setLevel(plog.DEBUG)
GeneratorLogger = logger.getChild('Generator')
GeneratorLogger.setLevel(plog.DEBUG)
DataPutProcessLogger = logger.getChild('DataPutProcess')
DataPutProcessLogger.setLevel(plog.DEBUG)
GeneratedDataLogger = logger.getChild('GeneratedData')
GeneratedDataLogger.setLevel(plog.DEBUG)
import Putil.data.convert_to_input as convert_to_input
import Putil.data.data_type_adapter as data_type_adapter
import Putil.data.fit_all_common_data as fit_all_common_data


class CommonDataManager(BaseManager):
    pass


class ProxyBase(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__', '__len__')
    pass


class IndexInfo:
    def __init__(self, point, type, *args, **kwargs):
        self._point = point
        self._type = type
        pass

    def point(self):
        return self._point

    def type(self):
        return self._type
    pass


class DataIndexInfo:
    def __init__(self, start, end, index_info, *args, **kwargs):
        self._range = [start, end]
        self._index_info = index_info
        pass

    def index_info(self):
        return self._index_info

    def data_range(self):
        return self._range
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
        self._data = np.stack(data_list, axis=0)
        pass

    def datas(self):
        return self._data

    def indexs(self):
        return self._indexs

    def data(self, index):
        pass
    pass


class ConvertPostOperation:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args):
        return args

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CommonData(metaclass=ABCMeta):
    '''
    this class provide a common method to read the data
    '''
    class RemainStrategy(Enum):
        Drop=0 # sub_data的补集直接丢弃
        UseAsNegative=1 # sub_data的补集最为负集合
        UseAsBackground=2 # sub_data的补集作为背景联合sub_data中的target生成样本
    
    @staticmethod
    def get_remain_strategy_field():
        return CommonData.RemainStrategy.__members__

    def save_result(self, *result):
        raise NotImplementedError('default save_result is not implemented')

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

    @abstractmethod
    def _generate_from_specified(self, index):
        '''
        this function is call in the generate_data, using the specified id from the data_set_field to get the data from the id
        the format of the data returned should be: {key: data.follow(shape: [batch, **])}
        '''
        pass

    def __init__(self, use_rate=1.0, sub_data=None, remain_strategy=RemainStrategy.Drop):
        #self._device_batch_mutex = threading.Lock()
        self._device_batch = None
        self._critical_process = None
        self._epoch_done = False

        # a list which contain number(int) which represent the index of the data
        self._data_field = None
        # a number(int) which the represent data read now
        self._index = None

        self._convert_to_input_method = convert_to_input.ConvertToInputNoOp()
        self._data_type_adapter = data_type_adapter.DataTypeAdapterNoOp()

        self._use_rate = use_rate
        self._sub_data = sub_data
        # 输入的remain_strategy可以是CommonData.RemainStrategy中的字段、其中的值或者直接就是引用了CommonData.RemainStrategy，
        assert remain_strategy in CommonData.RemainStrategy.__members__ or remain_strategy in CommonData.RemainStrategy._value2member_map_ or remain_strategy in CommonData.RemainStrategy
        if type(remain_strategy).__name__ == 'str':
            self._remain_strategy = CommonData.RemainStrategy.__members__[remain_strategy]
        elif type(remain_strategy).__name__ == 'int':
            self._remain_strategy = CommonData.RemainStrategy._value2member_map_[remain_strategy]
        else:
            self._remain_strategy = remain_strategy
        pass

    def _fix_field(self):
        self._data_field = random.sample(self._data_field, int(len(self._data_field) * self._use_rate))
        pass

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        pass

    def data_field(self):
        return self._data_field

    def index(self):
        return self._index

    def set_convert_to_input_method(self, method):
        self._convert_to_input_method = method

    def _convert_check(self, *args):
        '''
         @brief follow after _convert_to_input_method
         @note default : do nothing
        '''
        CommonDataLogger.debug('use default convert_check do nothing bu return the arg')
        return args

    def set_data_type_adapter(self, adapter):
        '''
         @brief 
         @note 
         set the data type adapter, set the slef._data_type_adapter
         the adapter transform the data type such as numpy.array to torch, etc
         @param[in] adapter
         the adapter is a function which receive a *args and return the transformed args
         the default self._data_type_adapter do nothing
         set
        '''
        self._data_type_adapter = adapter
        pass

    def generate_from_specified(self, index):
        return self._data_type_adapter(*self._convert_check(*self._convert_to_input_method(*self._generate_from_specified(index))))

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

    #def device_batch_operation(self):
    #    self._device_batch_.acquire()
    #    self._device_batch_operator()
    #    self._device_batch_mutex.release()
    #    pass

    def _data_set_field(self):
        return self._data_field

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
        [devices_data.append(list()) for device in self._device_batch]
        [index_data.append(list()) for device in self._device_batch]
        def data_pack(data, devices_data, device_order, data_index, index_type):
            for index, value in enumerate(data):
                devices_data[device_order].append(list()) if index >= len(devices_data[device_order]) else None
                index_data[device_order].append(list()) if index >= len(index_data[device_order]) else None
                devices_data[device_order][index].append(value)
                index_data[device_order][index].append(IndexInfo(data_index, index_type))
                pass
            pass
        while np.sum(need_batch) > 0:
            for ergodic_device in enumerate(need_batch):
                CommonDataLogger.debug(ergodic_device)
                batch = ergodic_device[1]
                device_order = ergodic_device[0]
                if batch > 0:
                    # get the data
                    if self._epoch_done is False:
                        CommonDataLogger.debug('epoch not end')
                        data = self.generate_from_specified(self._index)
                        self._status_update()
                        data_pack(data, devices_data, device_order, self._index, 'normal')
                        need_batch[device_order] -= 1
                        CommonDataLogger.debug('pack the data')
                        self._index += 1
                        pass
                    else:
                        def deal_with_epoch_done():
                            #field = self._data_set_field()
                            field = list(range(0, len(self)))

                            def func():
                                random_sample = np.random.choice(field)
                                return self.generate_from_specified(random_sample), random_sample
                            return func
                        func_for_deal_with_epoch_done = deal_with_epoch_done() if func_for_deal_with_epoch_done is None else func_for_deal_with_epoch_done
                        if self._critical_process == 'allow_low' and batch != self._device_batch[device_order]:
                            CommonDataLogger.debug('method: allow_low and already data in device, done')
                            need_batch[device_order] = 0
                            pass
                        elif self._critical_process == 'allow_low' and batch == self._device_batch[device_order]:
                            CommonDataLogger.debug('method: allow_low and no data in device, feed one')
                            data, index = func_for_deal_with_epoch_done()
                            #for key, value in data.items():
                            data_pack(data, devices_data, device_order, index, 'allow_low')
                            need_batch[device_order] = 0
                            pass
                        elif self._critical_process == 'random_fill':
                            CommonDataLogger.debug('method: random_fill')
                            data, index = func_for_deal_with_epoch_done()
                            #for key, value in data.items():
                            data_pack(data, devices_data, device_order, index, 'random_fill')
                            need_batch[device_order] -= 1
                            pass
                        else:
                            CommonDataLogger.fatal('this should not happen')
                            pass
                        pass
                    pass
                else:
                    pass
                pass
            pass
        for i in devices_data:
            for index, value in enumerate(i):
                for d in value:
                    CommonDataLogger.debug('{0} {1}'.format(index, d.shape))
                    pass
                pass
            pass
        CommonDataLogger.debug('going to generate data')
        #data = [{key: GeneratedData(datas[key], indexs[key]) for key in datas.keys()} for datas, indexs in zip(devices_data, index_data)]
        data = [[GeneratedData(data, index) for data, index in zip(datas, indexs)] for datas, indexs in zip(devices_data, index_data)]
        # data = [GeneratedData(datas, indexs) for datas, indexs in zip(devices_data, index_data)]
        return data

    def generate_epoch_done(self):
        return self._epoch_done

    def __getitem__(self, index):
        return self.generate_from_specified(index)

    def __len__(self):
        return len(self._data_field)
    pass

import Putil.data.aug as Aug

class CommonDataWithAug(CommonData, metaclass=ABCMeta):
    '''
     @brief the CommonData which support aug
     @note
        this class complete the generate_from_specified which contain aug
        generate_from_origin_index->aug->aug_check->convert->convert_check->data_type_adapter
    '''
    def __init__(self, use_rate=1.0, sub_data=None, remain_strategy=CommonData.RemainStrategy.Drop):
        CommonData.__init__(self, use_rate=use_rate, sub_data=sub_data, remain_strategy=remain_strategy)
        self._aug_node = Aug.AugNode(Aug.AugFuncNoOp())
        self._aug_node.freeze_node()
        pass

    def __len__(self):
        return len(self._data_field) * len(self._aug_node) 

    def set_aug_node_root(self, aug_node_root):
        self._aug_node = aug_node_root
        pass

    def _generate_from_specified(self, index):
        oindex = index // len(self._aug_node)
        aindex = index % len(self._aug_node)
        CommonDataWithAugLogger.debug('original index: {0}, aug index: {1}, aug_naem: {2}'.format(oindex, aindex, self._aug_node[aindex].name))
        #ret = self._generate_from_origin_index(oindex)
        ret = self._aug_node[aindex](*self._generate_from_origin_index(oindex))
        CommonDataWithAugLogger.debug('aug check:')
        ret = self._aug_check(*ret)
        return ret
    
    @abstractmethod
    def _generate_from_origin_index(self, oindex):
        '''
         @brief return the data which is origin read from the dataset(without augment), tuple
        '''
        pass

    def _aug_check(self, *args):
        '''
         @brief follow after _aug_node
         @note default : do nothing
        '''
        CommonDataWithAugLogger.debug('use default aug_check do nothing bu return the arg')
        return args
    pass


class CombineCommonData(CommonDataWithAug):
    '''
     @brief combine some common_data
     @note
     _generate_from_specified->fit_all_common_data->convert_to_input_method->convert_check->data_type_adapter
    '''
    def __init__(self, common_data_list):
        '''
         @brief combine some CommonData and provide the method to get the data
        '''
        self._common_data_list = common_data_list
        self._size_list = [len(common_data) for common_data in self._common_data_list]
        self._size_ofs = [sum(self._size_list[0: ofs]) for ofs in range(0, len(self._size_list))]
        self._len = sum(self._size_list)

        self._fit_all_common_data = fit_all_common_data.FitAllCommonDataNoOp()
        pass

    def set_fit_all_common_data(self, fit_all_common_data):
        self._fit_all_common_data = fit_all_common_data
        pass

    def which_common_data_and_offset_index(self, index):
        ofs = 0
        while(index >= self._size_ofs[ofs]):
            ofs += 1
            pass
        c_ofs = index - self._size_ofs[ofs]
        return ofs, c_ofs

    def __len__(self):
        return self._len

    def generate_from_specified(self, index):
        ofs, cofs = self.which_common_data_and_offset_index(index)
        common_data = self._common_data_list[ofs]
        return common_data._data_type_adapter(
            *common_data._convert_check(
                *common_data._convert_to_input_method(
                    self._fit_all_common_data(
                        *common_data._generate_from_specified(index), ofs=ofs, cofs=cofs))))

    def __getitem__(self, index):
        return self.generate_from_specified(index)
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
        except Exception as e:
            GeneratorLogger.fatal('str(Exception):\t', str(Exception))
            GeneratorLogger.fatal('str(e):\t\t', str(e))
            GeneratorLogger.fatal('repr(e):\t', repr(e))
            GeneratorLogger.fatal('traceback.print_exc():', traceback.print_exc())
            GeneratorLogger.fatal('traceback.format_exc():\n%s' % traceback.format_exc())
            GeneratorLogger.fatal(traceback.format_tb(e.__traceback__))
            raise e
        #flag_sync_mutex.acquire()
        GeneratorLogger.debug('got flag_sync_mutex') if DataPutProcess._running_mode is DataPutProcess.RunningMode.Debug else None
        data_queue.put(get_data)
        GeneratorLogger.debug('put data') if DataPutProcess._running_mode is DataPutProcess.RunningMode.Debug else None
        epoch_done_flag.value = data.generate_epoch_done()
        GeneratorLogger.debug('epoch_done_flag: {}'.format(epoch_done_flag.value)) if DataPutProcess._running_mode is DataPutProcess.RunningMode.Debug else None
        #flag_sync_mutex.release()
        GeneratorLogger.debug('released flag_sync_mutex') if DataPutProcess._running_mode is DataPutProcess.RunningMode.Debug else None
        count += 1
        GeneratorLogger.debug('count up') if DataPutProcess._running_mode is DataPutProcess.RunningMode.Debug else None 
        epoch_done_cond.release()
        GeneratorLogger.debug('released epoch_done_cond') if DataPutProcess._running_mode is DataPutProcess.RunningMode.Debug else None
        pass
    GeneratorLogger.debug('generator stop') if DataPutProcess._running_mode is DataPutProcess.RunningMode.Debug else None
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
            #for k, v in the_first.items():
            for _, v in enumerate(the_first):
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

    def continue_queue(self):
        plog.api_function_log(DataPutProcessLogger, 'continue the queue')
        self._epoch_done_cond.release()
        self._paused = False
        pass

    def has_next(self):
        #self._flag_sync_mutex.acquire()
        #DataPutProcessLogger.debug('got flag_sync_mutex') if DataPutProcess._running_mode is DataPutProcess.RunningMode.Debug else None
        # print(self._data_queue.empty())
        # print(self._epoch_done_flag.value)
        flag = (self._data_queue.empty() is False or self._epoch_done_flag.value is False)
        #self._flag_sync_mutex.release()
        #DataPutProcessLogger.debug('got flag_sync_mutex') if DataPutProcess._running_mode is DataPutProcess.RunningMode.Debug else None
        return flag

    def paused_and_has_next(self):
        #self._flag_sync_mutex.acquire()
        flag = self._data_queue.empty() is False and self._paused is True
        #self._flag_sync_mutex.release()
        return flag

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

    def _default_critical_process(self):
        return 'random_fill'

    def Count(self):
        return self._count.value

    def DataQueue(self):
        return self._data_queue

    def EpochDoneCond(self):
        return self._epoch_done_cond

    def EpochDoneFlag(self):
        return self._epoch_done_flag

    def queue_process_ret(self):
        return self._ret
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.has_next():
            return self._data_queue.get()
        else:
            raise StopIteration()
        pass
    
    class RunningMode(Enum):
        Debug=0
        Release=1
    
    _running_mode = RunningMode.Release

    @staticmethod
    def set_running_mode(running_mode):
        assert running_mode.name in DataPutProcess.RunningMode.__members__.keys()
        DataPutProcess._running_mode = running_mode
        pass


class StandardCommonData(CommonDataWithAug):
    def __init__(self, use_rate=1.0, sub_data=None, remain_strategy=None):
        self._use_rate = use_rate if use_rate is not None else 1.0
        self._sub_data = sub_data
        self._remain_strategy = remain_strategy if remain_strategy is not None else 'drop'

        self._data_field = self.__generate_data_field()
        # apply sub_data and remain_strategy
        self._data_field = self.__sub_data_filteration(self._sub_data, self._remain_strategy)
        # apply use_rate
        self._data_field = self.__use_rate_filteration()
        pass

    def __use_rate_filteration(self):
        self._fix_field()
        return self._data_field

    @abstractmethod
    def __sub_data_filteration(self, sub_data, remain_strategy):
        raise NotImplementedError('__sub_data_filteration is not implemented')
        pass

    @abstractmethod
    def __generate_data_field(self):
        '''
         @brief
         @note
        '''
        raise NotImplementedError('__generate_data_field is not implemented')

    @abstractmethod
    def add_result(self, result_tuple, save=False):
        raise NotImplementedError('add_result is not implemented')