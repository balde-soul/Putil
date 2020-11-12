# coding=utf-8
import os
import argparse
options = argparse.ArgumentParser()
options.add_argument('--specified_epoch', type=int, nargs='+', default=None, \
    help='the epoch which would be used to evaluate')
options.add_argument('--dir', type='str', action='store', default='', \
    help='the model in the dir would be used to evaluate')
options.add_argument('--evaluate_indicator', type='str', action='store', default='', \
    help='specify the indicator name')
args = options.parse_args()

from Putil.demo.deep_learning.base.args_operation import args_extract as ArgsExtract
from Putil.demo.deep_learning.base.util import get_all_model as GetModels
# : import the Dataset
from util.data_util import generate_evaluate_data
from Putil.demo.deep_learning.base.data_loader_factory import data_loader_factory as DataLoader
from Putil.demo.deep_learning.base.data_sampler_factory import data_sampler_factory as DataSampler
from Putil.demo.deep_learning.base.model_factory import model_factory as Model
from Putil.demo.deep_learning.base.decode_factory import decode_factory as Decode
from Putil.demo.deep_learning.base.evaluate_indicator_factory import evaluate_indicator_factory as EvaluateIndicator
#from import as Dataset
trained_args = ArgsExtract(os.path.join(args.dir, 'args'))
# : generate the dataset set the encoder
dataset = generate_evaluate_data(trained_args)
data_sampler = DataSampler(trained_args)(dataset, rank_amount=hvd.size(), rank=hvd.rank())  if dataset_train is not None else None
data_loader = DataLoader(trained_args)(dataset, batch_size=args.batch_size, sampler=train_sampler) if dataset_train is not None else None

model = Model(trained_args)
decode = Decode(trained_args)
evaluate_indicator = EvaluateIndicator(args)
if args.cuda:
    model.cuda()
    decode.cuda()
    evaluate_indicator.cuda()

model_weights = GetModels(args.dir)
for model_weight in model_weights:
    os.path.join()