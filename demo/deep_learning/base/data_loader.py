# coding=utf-8
from colorama import Fore
from importlib import reload
from torch.utils.data import DataLoader as data_loader
import Putil.base.logger as plog
from abc import abstractmethod, ABCMeta


logger = plog.PutilLogConfig('data_loader').logger()
logger.setLevel(plog.DEBUG)
torch_DataLoader_logger = logger.getChild('torch_DataLoader')
torch_DataLoader_logger.setLevel(plog.DEBUG)

from Putil.demo.deep_learning.base import util
reload(util)


class DataLoader(metaclass=ABCMeta):
    def __init__(self, args, property_type='', **kwargs):
        pass

    @abstractmethod
    def __call__(self, dataset, data_sampler, stage):
        '''
         @brief use dataset to generate the DataLoader
         @note
         @param[in] dataset
         @ret
         DataLoader
        '''
        pass
    pass


def common_data_loader_arg(parser, property_type='', **kwargs):
    def generate_arg(property_type, stage_name):
        parser.add_argument('--{}dataloader_deterministic_work_init_fn_{}'.format(property_type, stage_name), type=int, action='store', default=None, \
            help='the deterministic_work init_fn')
        parser.add_argument('--{}n_worker_per_dataset_{}'.format(property_type, stage_name), type=int, action='store', default=1, \
            help='the worker number for the data loader')
        parser.add_argument('--{}pin_memory_{}'.format(property_type, stage_name), action='store_true', default=False, \
            help='set pin memory for the data load while set, a wraper_func should be pass to the collate function, default is False')
        parser.add_argument('--{}batch_size_{}'.format(property_type, stage_name), type=int, action='store', default=8, \
            help='the batch size for train stage, default {}'.format(8))
        parser.add_argument('--{}shuffle_{}'.format(property_type, stage_name), action='store_true', default=False, \
            help='shuffle the train dataset if it is set')
        parser.add_argument('--{}drop_last_{}'.format(property_type, stage_name), action='store_true', default=True, \
            help='drop the last batch train data while the remain data less than the batch size')
        pass
    generate_arg(property_type, util.Stage.Train.name)
    generate_arg(property_type, util.Stage.Evaluate.name)
    generate_arg(property_type, util.Stage.Test.name)
    pass


class DefaultDataLoader(DataLoader):
    def __init__(self, args, property_type='', **kwargs):
        DataLoader.__init__(self, args, property_type, **kwargs)
        self._args = args
        self._property_type = property_type
        self._kwargs = dict()
        # TODO: make the determinstic pass args.dataloader_deterministic_work_init_fn as the worker_init_fn
        pass

    def __call__(self, dataset, data_sampler, stage):
        if stage in util.Stage:
            torch_DataLoader_logger.info(Fore.YELLOW + 'data loader in stage: {}'.format(stage.name) + Fore.RESET)
        else:
            raise NotImplementedError('stage: {} is not implemented'.format(stage))
        self._kwargs['worker_init_fn'] = eval('self._args.{}dataloader_deterministic_work_init_fn_{}'.format(self._property_type, stage.name))
        self._kwargs['num_workers'] = eval('self._args.{}n_worker_per_dataset_{}'.format(self._property_type, stage.name))
        self._kwargs['pin_memory'] = eval('self._args.{}pin_memory_{}'.format(self._property_type, stage.name))
        self._kwargs['batch_size'] = eval('self._args.{}batch_size_{}'.format(self._property_type, stage.name))
        self._kwargs['drop_last'] = eval('self._args.{}drop_last_{}'.format(self._property_type, stage.name))
        self._kwargs['shuffle'] = eval('self._args.{}shuffle_{}'.format(self._property_type, stage.name))
        import torch.multiprocessing as mp
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (self._kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            self._kwargs['multiprocessing_context'] = 'forkserver'
        return data_loader(dataset, sampler=data_sampler, **self._kwargs)
    pass


def DefaultDataLoaderArg(parser, property_type='', **kwargs):
    common_data_loader_arg(parser, property_type, **kwargs)
    pass