# coding=utf-8
from Putil.data.common_data import CommonDataWithAug


def common_dataset_arg(parser):
    parser.add_argument('--n_worker_per_dataset', action='store', type=int, default=1, \
        help='the number of worker for every dataset')
    parser.add_argument('--sub_data', type=int, nargs='+', default=None, \
        help='list with int, specified the sub dataset which would be used in train evaluate, '
        'default None(whole dataset)')
    parser.add_argument('--fake_aug', action='store', type=int, default=0, \
        help='do the sub aug with NoOp for fake_aug time, check the generate_dataset')
    parser.add_argument('--naug', action='store_true', \
        help='do not use data aug while set')
    parser.add_argument('--data_using_rate_train', action='store', type=float, default=1.0, \
        help='rate of data used in train')
    parser.add_argument('--data_using_rate_evaluate', action='store', type=float, default=1.0, \
        help='rate of data used in evaluate')
    parser.add_argument('--data_using_rate_test', action='store', type=float, default=1.0, \
        help='rate of data used in test')
    parser.add_argument('--shuffle_train', action='store_true', default=False, \
        help='shuffle the train data every epoch')
    parser.add_argument('--shuffle_evaluate', action='store_true', default=False, \
        help='shuffle the evaluate data every epoch')
    parser.add_argument('--shuffle_test', action='store_true', default=False, \
        help='shuffle the test data every epoch')
    parser.add_argument('--drop_last_train', action='store_true', default=False, \
        help='drop the last uncompleted train data while set')
    parser.add_argument('--drop_last_evaluate', action='store_true', default=False, \
        help='drop the last uncompleted evaluate data while set')
    parser.add_argument('--drop_last_test', action='store_true', default=False, \
        help='drop the last uncompleted test data while set')
    pass


def common_dd_dataset_arg(parser):
    common_dataset_arg(parser)
    parser.add_argument('--input_height', type=int, action='store', default=256, \
        help='the height of the input')
    parser.add_argument('--input_width', action='store', type=int, default=256, \
        help='the width of the input')
    pass


class Dataset(CommonDataWithAug):
    def __init__(self, args):
        CommonDataWithAug.__init__(self)
        pass
    pass


class DefaultDataset(Dataset):
    def __init__(self, args):
        Dataset.__init__(self, args)
        pass
    pass


def DefaultDatasetArg(parser):
    common_dataset_arg(parser)
    pass