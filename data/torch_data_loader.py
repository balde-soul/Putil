# coding=utf-8

from torch.utils.data import dataset, dataloader


class Dataset(dataset.Dataset):
    def __init__(self):
        dataset.Dataset.__init__(self)
        pass
    pass


class DataLoader(dataloader.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, sampler, \
            num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn):
        dataloader.DataLoader.__init__(dataset, batch_size, shuffle, sampler, \
            num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)
        self._iter_collection = list()
        pass

    def __iter__(self):
        pass