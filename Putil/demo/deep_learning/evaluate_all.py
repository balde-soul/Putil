# coding=utf-8
import torch
torch.save
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target_path', type=str, default='', action='store', \
    help='the model in the target would be evaluate')
parser.add_argument('--target_epochs', type=int, nargs='+', default=None, \
    help='specify the epoch')
parser.add_argument('--gpu', type=list, nargs='+', default=[], help='specify the gpus')
parser.add_argument('--remain_strategy', type=str, default=None, action='store', \
    help='specify the remain data strategy')
parser.add_argument('--run_net', action='store_true', default=False, \
    help='would run the model prediction while set')
parser.add_argument('--do_visual', action='store_true', default=False, \
    help='would do the result visual part while set')
args = parser.parse_args()
import Putil.base.logger as plog

log_level = plog.LogReflect(args.Level).Level
plog.PutilLogConfig.config_handler(plog.stream_method | plog.file_method)
plog.PutilLogConfig.config_log_level(stream=log_level, file=log_level)
plog.PutilLogConfig.config_format(plog.FormatRecommend)
logger = plog.PutilLogConfig('evaluate').logger()
logger.setLevel(log_level)

from Putil.demo.deep_learning.base import BaseOperationFactory
from Putil.base.arg_operation import args_extract
from .util.data_util import generate_evaluate_data
from Putil.demo.deep_learning.base import data_loader_factory as DataLoaderFactory
from Putil.demo.deep_learning.base import data_sampler_factory as DataSamplerFactory
from Putil.demo.deep_learning.base import dataset_factory as DatasetFactory
from Putil.demo.deep_learning.base import fit_data_to_input_factory as FitDataToInputFactory
from Putil.demo.deep_learning.base import fit_decode_to_result_factory as FitDecodeToResultFactory
from Putil.demo.deep_learning.base.util import Stage
from Putil.base.arg_operation import args_merge
from Putil.base.arg_base import args_log

args = arg_merge(args, args_extract(os.path.join(args.target_path, 'args')))
args.remain_data_as_negative = True
args_log(args, MainLogger) if hvd.rank() == 0 else None
ArgsSave(args, os.path.join(args.save_dir, 'args')) if hvd.rank() == 0 else None

dataset = DatasetFactory.dataset_factory(args)
data_sampler = DataSamplerFactory.data_sampler_factory(args)(dataset)
data_loader = DataLoaderFactory.data_loader_factory(args)(dataset, data_sampler, Stage.Evaluate)

fit_data_to_input = FitDataToInputFactory.fit_data_to_input_factory(args)()
fit_decode_to_result = FitDecodeToResultFactory.fit_decode_to_result_factory(args)()

target_models = BaseOperationFactory.get_models_factory(args)(args.traget_path)
epochs = target_models['epochs'] if args.target_epochs is None else args.target_epochs
load_saved_func = BaseOperationFactory.load_saved_factory(args)
generate_save_name_func = BaseOperationFactory.generate_save_name_factory(args)

for epoch in epochs:
    model = load_saved_func(epoch, args.target_path)
    with torch.no_grad():
        model.eval()
        for index, datas in enumerate(data_loader):
            input_data = fit_data_to_input(datas) 
            decode = model(input_data)
            result = fit_decode_to_result(decode)
            # process the result
            dataset.add_result(result)