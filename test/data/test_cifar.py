# coding=utf-8
import numpy as np
import cv2
import pytest


from Putil.data.cifar import Cifar100
from Putil.trainer import util
from Putil.data.aug import AugFuncNoOp, AugNode
from Putil.data.convert_to_input import ConvertToInputNoOp
from Putil.data.data_type_adapter import DataTypeAdapterNoOp
from Putil.data.torch_151_data.dataloader import DataLoader
from Putil.data.torch_151_data.sampler import BatchSampler, Sampler, RandomSampler


def test_cirfar100(params):
    cifar = Cifar100(util.Stage.Train, params['cifar_root_dir'], 1.0, None, Cifar100.RemainStrategy.Drop, Cifar100.Level.FineClass)
    root_node = AugNode(AugFuncNoOp())
    root_node.add_child(AugNode(AugFuncNoOp()))
    root_node.freeze_node()
    cifar.set_aug_node_root(root_node)
    cifar.set_convert_to_input_method(ConvertToInputNoOp())
    cifar.set_data_type_adapter(DataTypeAdapterNoOp())
    d, l, = cifar[0]
    cv2.imwrite('test/data/result/test_cifar100/read_one.jpg', np.transpose(d, (1, 2, 0)), cv2.IMWRITE_PAM_FORMAT_RGB)
    sampler = BatchSampler(RandomSampler(cifar), 8, True)
    data_loader = DataLoader(cifar, sampler=sampler)
    pass