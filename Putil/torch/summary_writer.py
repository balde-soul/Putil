# coding=utf-8
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager
from tensorboardX import SummaryWriter


class Writer:
    def __init__(self):
        self._writer = SummaryWriter('./torch')
        pass
    pass


BaseManager.register('writer', Writer)
BaseManager.register('OWriter', SummaryWriter)


def main():
    manager = BaseManager()
    manager.start()
    w = manager.writer()
    mp.spawn(test, args=(w, ), nprocs=2)
    w = manager.OWriter()
    mp.spawn(test, args=(w, ), nprocs=2)


def test(n, w):
    print(w)


if __name__ == '__main__':
    main()