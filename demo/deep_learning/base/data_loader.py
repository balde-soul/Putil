# coding=utf-8
from torch.utils.data import DataLoader as data_loader
import Putil.base.logger as plog
from abc import abstractmethod, ABCMeta


logger = plog.PutilLogConfig('data_loader').logger()
logger.setLevel(plog.DEBUG)
torch_DataLoader_logger = logger.getChild('torch_DataLoader')
torch_DataLoader_logger.setLevel(plog.DEBUG)

from Putil.demo.deep_learning.base.util import Stage as Stage


class DataLoader(metaclass=ABCMeta):
    def __init__(self, args):
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


class DefaultDataLoader(DataLoader):
    def __init__(self, args):
        DataLoader.__init__(self, args)
        self._args = args
        self._kwargs = dict()
        # TODO: make the determinstic pass args.dataloader_deterministic_work_init_fn as the worker_init_fn
        self._kwargs['worker_init_fn'] = args.dataloader_deterministic_work_init_fn
        pass

    def __call__(self, dataset, data_sampler, stage):
        assert stage in Stage
        self._kwargs['num_workers'] = args.n_worker_per_dataset
        self._kwargs['pin_memory'] = True if args.cuda else False
        if stage == Stage.Train:
            self._kwargs['batch_size'] = self._args.batch_size_train
            self._kwargs['drop_last'] = self._args.drop_last_train
            self._kwargs['shuffle'] = self._args.shuffle_train
        elif stage == Stage.Evaluate:
            self._kwargs['batch_size'] = self._args.batch_size_evaluat
            self._kwargs['drop_last'] = self._args.drop_last_evaluate
            self._kwargs['shuffle'] = self._args.shuffle_evaluate
        elif stage == Stage.Test.name:
            self._kwargs['batch_size'] = self._args.batch_size_test
            self._kwargs['drop_last'] = self._args.drop_last_test
            self._kwargs['shuffle'] = self._args.shuffle_test
        else:
            raise NotImplementedError('stage: {} is not implemented'.format(stage))
        import torch.multiprocessing as mp
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'
        return data_loader(dataset, sampler=data_sampler, **self._kwargs)
    pass


def DefaultDataLoaderArg(parser):
    pass