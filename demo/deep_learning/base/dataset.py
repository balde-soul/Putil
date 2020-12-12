# coding=utf-8
from colorama import Fore
import argparse
import copy
from Putil.data.common_data import CommonDataWithAug
import Putil.base.logger as plog
logger = plog.PutilLogConfig('dataset').logger()
logger.setLevel(plog.DEBUG)
COCOLogger = logger.getChild('COCO')
COCOLogger.setLevel(plog.DEBUG)
import Putil.data.coco as coco
from Putil.demo.deep_learning.base.util import Stage


def common_dataset_arg(parser):
    '''
     @brief 公共通用的dataset的参数
     @note 参数说明：
      n_worker_per_dataset：dataset 都继承于Putil.data.common_data.CommonDataWithAug
      sub_data: 制定数据集子集，这在初步调试阶段很有用，在开发模型测试流程以及模型可行性时，使用数据集子集快速获得结论
      remain_data_as_negative: 除去sub_data的子集，其他子集是否作为负数据，使用可以减小样本空间的偏差
      fake_aug：当使用小子集进行测试时，重新开始一个epoch耗时比较多，我们可以设定fake_aug的数量，意思是：一个epoch的数据
        进行fake_aug次复制
      naug: 当被设置时，dataset不使用数据集扩展
    '''
    parser.add_argument('--n_worker_per_dataset', action='store', type=int, default=1, \
        help='the number of worker for every dataset')
    parser.add_argument('--sub_data', type=int, nargs='+', default=None, \
        help='list with int, specified the sub dataset which would be used in train evaluate, '
        'default None(whole dataset)')
    parser.add_argument('--remain_data_as_negative', action='store_true', \
        help='if set, the data beside $sub_data would be use as negative, otherwise the data beside' \
            '$sub_data would be abandon')
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
    parser.add_argument('--coco_remain_strategy', type=str, default='drop', \
        help='')
    pass


def common_ddd_dataset_arg(parser):
    common_dataset_arg(parser)
    parser.add_argument('--input_height', type=int, action='store', default=256, \
        help='the height of the input')
    parser.add_argument('--input_width', action='store', type=int, default=256, \
        help='the width of the input')
    parser.add_argument('--input_depth', action='store', type=int, default=256, \
        help='the depth of the input')
    pass


class Dataset(CommonDataWithAug):
    def __init__(self, args):
        CommonDataWithAug.__init__(self)
        pass
    pass


def COCODataset(args):
    temp_args = copy.deepcopy(args)
    def generate_coco_dataset(stage):
        return coco.COCOData(coco_root_dir=args.coco_root_dir,
        stage=stage,
        information_save_to_path=args.save_dir,
        detection=args.coco_detection,
        key_points=args.coco_key_points,
        stuff=args.coco_stuff,
        panoptic=args.coco_panoptic,
        dense_pose=args.coco_dense_pose,
        captions=args.coco_captions,
        cat_ids=args.coco_cat_ids,
        use_rate=args.coco_use_rate if 'coco_use_rate' in dir(args) else args.data_using_rate_train if stage == Stage.Train else args.data_using_rate_evaluate if stage == Stage.Train or stage == Stage.TrainEvaluate else args.data_using_rate_test,
        image_width=args.coco_image_width if 'coco_image_width' in dir(args) else args.image_width,
        image_height=args.coco_image_height if 'coco_image_height' in dir(args) else args.image_height,
        remain_strategy=args.coco_remain_strategy)
        pass
    pass


def COCOArg(parser):
    try:
        common_dd_dataset_arg(parser)
    except argparse.ArgumentError as e:
        COCOLogger.warning(Fore.YELLOW + e.message + Fore.RESET)
        pass
    parser.add_argument('--coco_root_dir', type=str, default='', action='store', \
        help='')
    parser.add_argument('--coco_detection', action='store_true', default=False, \
        help='generate detection data while set')
    parser.add_argument('--coco_key_points', action='store_true', default=False, \
        help='generate key points data while set')
    parser.add_argument('--coco_stuff', action='store_true', default=False, \
        help='generate stuff data while set')
    parser.add_argument('--coco_panoptic', action='store_true', default=False, \
        help='generate panoptic data while set')
    parser.add_argument('--coco_dense_pose', action='store_true', default=False, \
        help='generate dense pose data while set')
    parser.add_argument('--coco_captions', action='store_true', default=False, \
        help='generate captions data while set')
    parser.add_argument('--coco_cat_ids', type=int, default=[], nargs='+', \
        help='specify the target cat ids')
    parser.add_argument('--coco_image_width', type=int, default=256, \
        help='specify the width of the data')
    parser.add_argument('--coco_image_height', type=int, default=256, \
        help='specify the height of the data')
    pass


# DefaultDataset
class _DefaultDataset(Dataset):
    def __init__(self, args):
        Dataset.__init__(self, args)
        pass
    pass

def DefaultDataset(args):
    temp_args = copy.deepcopy(args)
    def generate_default_dataset():
        return DefaultDataset(temp_args)
    return generate_default_dataset

def DefaultDatasetArg(parser):
    common_dataset_arg(parser)
    pass


#parser = argparse.ArgumentParser()
#parser.add_argument('--t', type=int, default=1, action='store')
#try:
#    parser.add_argument('--t', type=int, default=1, action='store')
#except argparse.ArgumentError as e:
#    print('a')
#args = parser.parse_args()
#print(args)