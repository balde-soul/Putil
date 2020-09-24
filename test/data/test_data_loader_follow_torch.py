# coding=utf-8
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
import Putil.base.logger as plog
import numpy as np
plog.PutilLogConfig.config_format(plog.FormatRecommend)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_log_level(stream=plog.DEBUG)

root_logger = plog.PutilLogConfig('test_data_loader_follow_torch').logger()
root_logger.setLevel(plog.DEBUG)
Logger = root_logger.getChild('Logger')
Logger.setLevel(plog.DEBUG)

from Putil.data.data_loader_follow_torch import DataLoader

class DT(Dataset):
    def __init__(self):
        Dataset.__init__(self)
    
    def __len__(self):
        return 100

    def __getitem__(self, index):
        import time
        return np.array(index),

if __name__ == '__main__':
    import horovod.torch as hvd
    hvd.init()
    data = DT()
    sampler = DistributedSampler(data, num_replicas=hvd.size(), rank=hvd.rank())
    loader = DataLoader(data, 9, sampler=sampler, num_workers=2, pin_memory=True)
    sampler.set_epoch(0)
    for i in loader:
        Logger.info('get: {}'.format(i))
        pass
    sampler.set_epoch(1)
    for i in loader:
        Logger.info('get: {}'.format(i))
        pass

    del loader