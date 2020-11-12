'''
 @note
 本文件数据接口类
 如果不是常规的功用数据集，或者Putil中没有实现对应的数据集，则需要继承CommonDataAug进行特化实现
 
 实现CommonData中的data_type_adapter
'''
# coding=utf-8

def generate_evaluate_data(args):
    pass


def generate_train_data(args):
    pass


def generate_test_data(args):
    pass


def generate_data(args, train, evaluate, test):
    '''
     @ret the Dataset
    '''
    pass
