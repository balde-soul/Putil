# coding=utf-8
import torch
torch.save
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target_path', type=str, default='', action='store', \
    help='the model in the target would be evaluate')
args = parser.parse_args()

from Putil.demo.deep_learning.base import base_operation_factory
from Putil.base.arg_operation import args_extract
from .util.data_util import generate_evaluate_data
from Putil.demo.deep_learning.base.data_loader_factory import data_loader_factory as DataLoader
from Putil.demo.deep_learning.base.data_sampler_factory import data_sampler_factory as DataSampler
from Putil.demo.deep_learning.base.dataset_factory import dataset_factory as Dataset
from Putil.demo.deep_learning.base.fit_data_to_input import fit_data_to_input_factory as FitDataToInput
from Putil.demo.deep_learning.base.fit_decode_to_result import fit_decode_to_result_factory as FitDecodeToResult
from Putil.demo.deep_learning.base.util import Stage

train_args = args_extract(os.path.join(args.target_path, 'args'))

args.
dataset = Dataset(args)
data_sampler = DataSampler(args)(dataset)
data_loader = DataLoader(args)(dataset, data_sampler, Stage.Evaluate)

fit_data_to_input = FitDataToInput(train_args)()
fit_decode_to_result = FitDecodeToResult(train_args)()

target_models = base_operation_factory.get_models_factory(args)(args.traget_path)
epochs = target_models['epochs']
load_saved_func = base_operation_factory.load_saved_factory(args)
generate_save_name_func = base_operation_factory.generate_save_name_factory(args)

for epoch in epochs:
    model = load_saved_func(epoch, args.target_path)
    for index, datas in enumerate(data_loader):
        input_data = fit_data_to_input(datas) 
        decode = model(input_data)
        result = fit_decode_to_result(decode)
        dataset.save_result(result)
    pass