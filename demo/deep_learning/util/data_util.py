'''
 @note
 本文件数据接口类
 如果不是常规的功用数据集，或者Putil中没有实现对应的数据集，则需要继承CommonDataAug进行特化实现
 
 实现CommonData中的data_type_adapter
'''
# coding=utf-8

def generate_train_stage_evaluate_data(args):
    raise NotImplementedError('generate_train_stage_evaluate_data not implemented')


def generate_train_stage_train_data(args):
    raise NotImplementedError('generate_train_stage_train_data not implemented')


def generate_test_data(args):
    raise NotImplementedError('generate_test_data not implemented')


def generate_evaluate_data(args, train, evaluate, test):
    '''
     @ret the Dataset
    '''
    raise NotImplementedError('generate_evaluate_data not implemented')