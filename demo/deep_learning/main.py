# coding=utf-8
from __future__ import absolute_import
import numpy as np
from importlib import reload
import re
from colorama import Fore
import sys
import json
import os
from enum import Enum
import torch
from torch import multiprocessing as mp
from tensorboardX import SummaryWriter

base_optimization_source_property_type = 'optimization_source'
base_optimization_name_property_type = 'optimization_name'
base_backbone_source_property_type = 'backbone_source'
base_backbone_name_property_type = 'backbone_name'
base_backend_source_property_type = 'backend_source'
base_backend_name_property_type = 'backend_name'
base_dataset_source_property_type = 'dataset_source'
base_dataset_name_property_type = 'dataset_name'
base_aug_source_property_type = 'aug_source'
base_aug_name_property_type = 'aug_name'
base_encode_source_property_type = 'encode_source'
base_encode_name_property_type = 'encode_name'
base_data_type_adapter_source_property_type = 'data_type_adapter_source'
base_data_type_adapter_name_property_type = 'data_type_adapter_name'
base_data_loader_source_property_type = 'data_loader_source'
base_data_loader_name_property_type = 'data_loader_name'
base_data_sampler_source_property_type = 'data_sampler_source'
base_data_sampler_name_property_type = 'data_sampler_name'
base_fit_to_loss_input_source_property_type = 'fit_to_loss_input_source'
base_fit_to_loss_input_name_property_type = 'fit_to_loss_input_name'
base_fit_to_indicator_input_source_property_type = 'fit_to_indicator_input_source'
base_fit_to_indicator_input_name_property_type = 'fit_to_indicator_input_name'
base_indicator_source_property_type = 'indicator_source'
base_indicator_name_property_type = 'indicator_name'
base_indicator_statistic_source_property_type = 'indicator_statistic_source'
base_indicator_statistic_name_property_type = 'indicator_statistic_name'
base_fit_to_decode_input_source_property_type = 'fit_to_decode_input_source'
base_fit_to_decode_input_name_property_type = 'fit_to_decode_input_name'
base_decode_source_property_type = 'decode_source'
base_decode_name_property_type = 'decode_name'
base_loss_source_property_type = 'loss_source'
base_loss_name_property_type = 'loss_name'
base_auto_stop_source_property_type = 'auto_stop_source'
base_auto_stop_name_property_type = 'auto_stop_name'
base_lr_reduce_source_property_type = 'lr_reduce_source'
base_lr_reduce_name_property_type = 'lr_reduce_name'
base_auto_save_name_property_type = 'auto_save_name'
base_auto_save_source_property_type = 'auto_save_source'

def do_save():
    MainLogger.info('run checkpoint') if args.debug else None
    eval('checkpoint(epoch, args.save_dir, {}=backbone, {}=backend, {}=lr_reduce, {}=auto_save, {}=auto_stop, {}=optimization)'.format(
        '{}-{}'.format(args.backbone_source, args.backbone_name),
        '{}-{}'.format(args.backend_source, args.backend_name),
        '{}-{}'.format(args.lr_reduce_source, args.lr_reduce_name),
        '{}-{}'.format(args.auto_save_source, args.auto_save_name),
        '{}-{}'.format(args.auto_stop_source, args.auto_stop_name),
        '{}-{}'.format(args.optimization_source, args.optimization_name)
        ))
    checkpoint(epoch, args.save_dir, backbone=backbone, lr_reduce=lr_reduce, auto_save=auto_save, \
        auto_stop=auto_stop, optimization=optimization)
    MainLogger.info('run save') if args.debug else None
    save(util.TemplateModelDecodeCombine, epoch, args.save_dir, backbone, backend, decode)
    MainLogger.info('run deploy') if args.debug else None
    deploy(util.TemplateModelDecodeCombine, \
        torch.from_numpy(np.zeros(shape=(1, 3, args.input_height, args.input_width))).cuda(), \
            recorder.epoch, args.save_dir, backbone, backend, decode)

def do_epoch_end_process(epoch_result):
    indicator = util.all_reduce(epoch_result['eloss'], 'train_indicator')
    save = auto_save.save_or_not(indicator)
    if save or args.debug:
        # :save the backbone in rank0
        do_save(epoch, util.TemplateModelDecodeCombine) if hvd.rank() == 0 else None
        # 此日志保存了保存模型的epoch数，为clear_train提供了依据
        MainLogger.info('save in epoch: {}'.format(recorder.epoch)) if hvd.rank() == 0 else None
    # :stop or not
    MainLogger.info('run ')
    stop = auto_stop.stop_or_not(indicator)
    # :lr_reduce
    _reduce = lr_reduce.reduce_or_not(indicator)
    # TODO: change the lr
    optimizer.__dict__['param_group'][0]['lr'] = lr_reduce.reduce(optimization.__dict__['param_groups'][0]['lr']) if _reduce \
        else optimization.__dict__['param_groups'][0]['lr']
    if hvd.rank() == 0:
        writer.add_scalar('lr', lr_reduce.LrNow, global_step=recorder.step)
    return stop, lr_reduce.LrNow, save

def train(epoch):
    ret = run_train_stage.train_stage_common(args, util.Stage.Train, epoch, fit_data_to_input, backbone, backend, decode, fit_decode_to_result, \
         loss, optimization, train_indicator, indicator_statistic, accumulated_opt, train_loader, recorder, writer, MainLogger)
    if args.evaluate_off:
        if args.debug:
            if epoch == 0:
                return False, 
            elif epoch == 1:
                do_epoch_end_process(ret)
                return False,
            else:
                raise RuntimeError('all_process_test would only run train two epoch')
        else:
            return do_epoch_end_process()
    else:
        return False,

def evaluate(epoch):
    ret = run_train_stage.train_stage_common(args, util.Stage.TrainEvaluate if util.evaluate_stage(args) else util.Stage.Evaluate, \
        epoch, fit_data_to_input, backbone, backend, decode, fit_decode_to_result, loss, optimization, \
            evaluate_indicator, indicator_statistic, accumulated_opt, train_loader, recorder, writer, MainLogger)
    if util.train_stage(args):
        if args.debug:
            if epoch == 0:
                # 当在all_process_test时，第二个epoch返回stop为True
                do_epoch_end_process(ret)
                return False, None, True
            else:
                # 当在all_process_test时，第一个epoch返回stop为True
                return True, None, False
        else:
            return do_epoch_end_process()
    return False, None, False

def test(epoch):
    pass

def run_test(model, data_loader, fit_data_to_input, fit_decode_to_result):
    model.eval()
    with torch.no_grad():
        for index, datas in data_loader:
            data_input = fit_data_to_input(datas)
            output = model(data_input, args)
            result = fit_decode_to_result(output)
            data_loader.dataset.save_result(prefix='test', save=False if index != len(data_loader) else True)

def test_stage():
    MainLogger.info('run test') 
    assert args.weight_path != '' and args.weight_epoch is not None, 'specify the trained weight_path and the epoch in test stage'
    MainLogger.info('load trained backbone: path: {} epoch: {}'.format(args.weight_path, args.weight_epoch))
    model = load_saved(args.weight_epoch, args.weight_path, map_location=torch.device(args.gpu))
    run_test(model, test_loader, fit_data_to_input, fit_decode_to_result)
    # release the model
    MainLogger.debug('del the model')
    del model
    torch.cuda.empty_cache()
    pass

def run_evaluate(model, data_loader, fit_data_to_input, fit_decode_to_result):
    model.eval()
    with torch.no_grad():
        for index, datas in data_loader:
            data_input = fit_data_to_input(datas)
            output = model(data_input, args)
            decode = decode(datas, output)
            result = fit_decode_to_result(decode)
            data_loader.dataset.save_result(prefix='evaluate', save=False if index != len(data_loader) else True)
            pass
        pass
    pass

def evaluate_stage():
    MainLogger.info('run evaluate')
    assert args.weight_path != '' and args.weight_epoch is not None, 'specify the trained weight_path and the epoch in test stage'
    MainLogger.info('load trained backbone: path: {} epoch: {}'.format(args.weight_path, args.weight_epoch))
    model = load_saved(args.run_evaluate_epoch, args.run_evaluate_full_path, map_location='cuda:{}'.format(args.run_evaluate_gpu))
    run_evaluate(model, evaluate_loader, fit_data_to_input, fit_decode_to_result)
    # release the model
    MainLogger.debug('del the model')
    del model
    torch.cuda.empty_cache()
    pass

def get_the_saved_epoch(train_time, path):
    pass

if __name__ == '__main__':
    import os
    import Putil.base.arg_base as pab
    import Putil.base.save_fold_base as psfb
    from Putil.demo.deep_learning.base import horovod
    framework = os.environ.get('framework')
    hvd = horovod.horovod(framework)
    hvd.init()
    ppa = pab.ProjectArg(save_dir='./result', log_level='Info', debug_mode=True, config='')
    #ppa.parser.add_argument('--data_name', action='store', type=str, default='DefaultData', \
    #    help='the name of the data, used in the data_factory, see the util.data_factory')
    #ppa.parser.add_argument('--data_source', type=str, default='standard', action='store', \
    #    help='standard: from the Putil.demo.base.deep_learning.data; project: from this project')
    #ppa.parser.add_argument('--encode_name', action='store', type=str, default='DefalutEncode', \
    #    help='the name of the encode in the encode_factory, see the util.encode_factory')
    #ppa.parser.add_argument('--encode_source', type=str, default='standard', action='store', \
    #    help='standard: from the Putil.demo.base.deep_learning.encode; project: from this project')
    #ppa.parser.add_argument('--decode_name', action='store', type=str, default='DefaultDecode', \
    #    help='the name of the decode in the decode_factory, see the util.decode_factory')
    #ppa.parser.add_argument('--decode_source', type=str, default='standard', action='store', \
    #    help='standard: from the Putil.demo.base.deep_learning.decode; project: from this project')
    #ppa.parser.add_argument('--auto_save_name', type=str, action='store', default='', \
    #    help='the name of the auto saver')
    #ppa.parser.add_argument('--auto_save_source', type=str, default='standard', action='store', \
    #    help='standard: from the Putil.demo.base.deep_learning.auto_save; project: from this project')
    #ppa.parser.add_argument('--auto_stop_name', type=str, action='store', default='', \
    #    help='the name of the auto stoper')
    #ppa.parser.add_argument('--auto_stop_source', type=str, default='standard', action='store', \
    #    help='standard: from the Putil.demo.base.deep_learning.auto_stop; project: from this project')
    #ppa.parser.add_argument('--lr_reduce_name', type=str, action='store', default='', \
    #    help='the name of the lr reducer')
    #ppa.parser.add_argument('--lr_reduce_source', type=str, default='standard', action='store', \
    #    help='standard: from the Putil.demo.base.deep_learning.lr_reduce; project: from this project')
    #ppa.parser.add_argument('--optimization_name', type=str, action='store', default=None, \
    #    help='the name of the optimization')
    #ppa.parser.add_argument('--optimization_source', type=str, default='standard', action='store', \
    #    help='standard: from the Putil.demo.base.deep_learning.optimization; project: from this project')
    # setting from the environment
    #ppa.parser.add_argument('--backbone_source', type=str, default='standard', action='store', \
    #    help='standard: from the Putil.demo.base.deep_learning.backbones; project: from this project')
    #ppa.parser.add_argument('--backbone_name', type=str, default='', action='store', \
    #    help='specify the backbone name')
    #ppa.parser.add_argument('--backbone_source', type=str, default='standard', action='store', \
    #    help='standard: from the Putil.demo.base.deep_learning.backbones; project: from this project')
    #ppa.parser.add_argument('--backbone_name', type=str, default='', action='store', \
    #    help='specify the backbone name')
    #ppa.parser.add_argument('--loss_name', type=str, default='DefaultLoss', action='store', \
    #    help='the name of the loss in the loss_factory, see the util.loss_factory')
    #ppa.parser.add_argument('--loss_source', type=str, default='standard', action='store', \
    #    help='standard: from the Putil.demo.base.deep_learning.loss; project: from this project')
    #ppa.parser.add_argument('--indicator_name', type=str, default='DefaultIndicator', action='store', \
    #    help='the name of the indicator in the indicator_factory, see the util.indicator_factory')
    #ppa.parser.add_argument('--indicator_source', type=str, default='standard', action='store', \
    #    help='standard: from the Putil.demo.base.deep_learning.indicator; project: from this project')
    #ppa.parser.add_argument('--indicator_statistic_name', type=str, default='', action='store', \
    #    help='the name of the indicator_statistic in the indicator_statistic_factory, see the util.indicator_statistic_factory')
    #ppa.parser.add_argument('--indicator_statistic_source', type=str, default='standard', action='store', \
    #    help='standard: from the Putil.demo.base.deep_learning.indicator_statistic; project: from this project')
    #<tag=====================================这些是需要reload的==============================================
    from Putil.demo.deep_learning.base import auto_save_factory as AutoSaveFactory
    from Putil.demo.deep_learning.base import auto_stop_factory as AutoStopFactory
    from Putil.demo.deep_learning.base import lr_reduce_factory as LrReduceFactory
    from Putil.demo.deep_learning.base import dataset_factory as DatasetFactory
    from Putil.demo.deep_learning.base import data_loader_factory as DataLoaderFactory
    from Putil.demo.deep_learning.base import data_sampler_factory as DataSamplerFactory
    from Putil.demo.deep_learning.base import encode_factory as EncodeFactory
    from Putil.demo.deep_learning.base import backbone_factory as BackboneFactory
    from Putil.demo.deep_learning.base import backend_factory as BackendFactory
    from Putil.demo.deep_learning.base import decode_factory as DecodeFactory
    from Putil.demo.deep_learning.base import loss_factory as LossFactory
    from Putil.demo.deep_learning.base import indicator_factory as IndicatorFactory
    from Putil.demo.deep_learning.base import indicator_statistic_factory as IndicatorStatisticFactory
    from Putil.demo.deep_learning.base import optimization_factory as OptimizationFactory
    from Putil.demo.deep_learning.base import aug_factory as AugFactory
    from Putil.demo.deep_learning.base import data_type_adapter_factory as DataTypeAdapterFactory
    from Putil.demo.deep_learning.base import fit_data_to_input_factory as FitDataToInputFactory
    from Putil.demo.deep_learning.base import fit_to_loss_input_factory as FitToLossInputFactory
    from Putil.demo.deep_learning.base import fit_to_indicator_input_factory as FitToIndicatorInputFactory
    from Putil.demo.deep_learning.base import fit_decode_to_result_factory as FitDecodeToResultFactory
    from Putil.demo.deep_learning.base import fit_to_decode_input_factory as FitToDecodeInputFactory
    from Putil.demo.deep_learning.base import model_factory as ModelFactory
    from Putil.demo.deep_learning.base import recorder_factory as RecorderFactory
    from Putil.demo.deep_learning.base import util
    from Putil.demo.deep_learning.base import base_operation_factory as BaseOperationFactory
    from Putil.demo.deep_learning.base import accumulated_opt_factory as AccumulatedOptFactory
    #======================================这些是需要reload的=============================================>
    horovod.horovod_arg(ppa.parser)
    auto_save_sources = util.get_relatived_environ(base_auto_save_source_property_type)
    auto_stop_sources = util.get_relatived_environ(base_auto_stop_source_property_type)
    lr_reduce_sources = util.get_relatived_environ(base_lr_reduce_source_property_type)
    dataset_sources = util.get_relatived_environ(base_dataset_source_property_type)
    data_loader_sources = util.get_relatived_environ(base_data_loader_source_property_type)
    data_sampler_sources = util.get_relatived_environ(base_data_sampler_source_property_type)
    encode_sources = util.get_relatived_environ(base_encode_source_property_type)
    backbone_sources = util.get_relatived_environ(base_backbone_source_property_type)
    backend_sources = util.get_relatived_environ(base_backend_source_property_type)
    decode_sources = util.get_relatived_environ(base_decode_source_property_type)
    loss_sources = util.get_relatived_environ(base_loss_source_property_type)
    indicator_sources = util.get_relatived_environ(base_indicator_source_property_type)
    indicator_statistic_sources = util.get_relatived_environ(base_indicator_statistic_source_property_type)
    ## optimization可以支持多个类型，是为了多中optimization进行优化的需求，key表示功能定向(空key表示默认功能)，name与source构成optimization的类型
    optimization_sources = util.get_relatived_environ(base_optimization_source_property_type)
    aug_sources = util.get_relatived_environ(base_aug_source_property_type)
    data_type_adapter_sources = util.get_relatived_environ(base_data_type_adapter_source_property_type)
    fit_data_to_input_source = os.environ.get('fit_data_to_input_source', 'standard')
    fit_to_loss_input_sources = util.get_relatived_environ(base_fit_to_loss_input_source_property_type)
    fit_to_indicator_input_sources = util.get_relatived_environ(base_fit_to_indicator_input_source_property_type)
    fit_to_decode_input_sources = util.get_relatived_environ(base_fit_to_decode_input_source_property_type)
    fit_decode_to_result_source = os.environ.get('fit_decode_to_result_source', 'standard')
    model_source = os.environ.get('model_source', 'standard')
    recorder_source = os.environ.get('recorder_source', 'standard')
    accumulated_opt_source = os.environ.get('accumulated_opt', 'standard')
    auto_save_names = util.get_relatived_environ(base_auto_save_name_property_type)
    util.complete_environ(auto_save_names, auto_save_sources, 'standard')
    auto_stop_names = util.get_relatived_environ(base_auto_stop_name_property_type)
    util.complete_environ(auto_stop_names, auto_stop_sources, 'standard')
    lr_reduce_names = util.get_relatived_environ(base_lr_reduce_name_property_type)
    util.complete_environ(lr_reduce_names, lr_reduce_sources, 'standard')
    dataset_names = util.get_relatived_environ(base_dataset_name_property_type)
    util.complete_environ(dataset_names, dataset_sources, 'standard')
    data_loader_names = util.get_relatived_environ(base_data_loader_name_property_type)
    util.complete_environ(data_loader_names, data_loader_sources, 'standard')
    data_sampler_names = util.get_relatived_environ(base_data_sampler_name_property_type)
    util.complete_environ(data_sampler_names, data_sampler_sources, 'standard')
    encode_names = util.get_relatived_environ(base_encode_name_property_type)
    util.complete_environ(encode_names, encode_sources, 'standard')
    backbone_names = util.get_relatived_environ(base_backbone_name_property_type)
    util.complete_environ(backbone_names, backbone_sources, 'standard')
    backend_names = util.get_relatived_environ(base_backend_name_property_type)
    util.complete_environ(backend_names, backend_sources, 'standard')
    decode_names = util.get_relatived_environ(base_decode_name_property_type)
    util.complete_environ(decode_names, decode_sources, 'standard')
    loss_names = util.get_relatived_environ(base_loss_name_property_type)
    util.complete_environ(loss_names, loss_sources, 'standard')
    indicator_names = util.get_relatived_environ(base_indicator_name_property_type)
    util.complete_environ(indicator_names, indicator_sources, 'standard')
    indicator_statistic_names = util.get_relatived_environ(base_indicator_statistic_name_property_type)
    util.complete_environ(indicator_statistic_names, indicator_statistic_sources, 'standard')
    ## optimization可以支持多个类型，是为了多中optimization进行优化的需求，key表示功能定向(空key表示默认功能)，name与source构成optimization的类型
    optimization_names = {property_type.replace(base_optimization_name_property_type, ''): os.environ[property_type] for property_type in util.find_repeatable_environ(base_optimization_name_property_type)}
    util.complete_environ(optimization_names, optimization_sources, 'standard')
    aug_names = util.get_relatived_environ(base_aug_name_property_type)
    util.complete_environ(aug_names, aug_sources, 'standard')
    data_type_adapter_names = util.get_relatived_environ(base_data_type_adapter_name_property_type)
    util.complete_environ(data_type_adapter_names, data_type_adapter_sources, 'standard')
    fit_data_to_input_name = os.environ.get('fit_data_to_input_name', 'DefaultFitDataToInput')
    fit_to_loss_input_names = util.get_relatived_environ(base_fit_to_loss_input_name_property_type)
    util.complete_environ(fit_to_loss_input_names, fit_to_loss_input_sources, 'standard')
    fit_to_indicator_input_names = util.get_relatived_environ(base_fit_to_indicator_input_name_property_type)
    util.complete_environ(fit_to_indicator_input_names, fit_to_indicator_input_sources, 'standard')
    fit_to_decode_input_names = util.get_relatived_environ(base_fit_to_decode_input_name_property_type)
    util.complete_environ(fit_to_decode_input_names, fit_to_decode_input_sources, 'standard')
    fit_decode_to_result_name = os.environ.get('fit_decode_to_result_name', 'DefaultFitDecodeToResult')
    model_name = os.environ.get('model_name', 'DefaultModel')
    recorder_name = os.environ.get('recorder_name', 'DefaultRecorder')
    accumulated_opt_name = os.environ.get('accumulated_opt_name', 'DefaultAccumulatedOpt')
    [AutoSaveFactory.auto_save_arg_factory(ppa.parser, auto_save_sources[property_type], auto_save_names[property_type], property_type) for property_type in auto_save_names.keys()]
    [AutoStopFactory.auto_stop_arg_factory(ppa.parser, auto_stop_sources[property_type], auto_stop_names[property_type], property_type) for property_type in auto_stop_names.keys()]
    [LrReduceFactory.lr_reduce_arg_factory(ppa.parser, lr_reduce_sources[property_type], lr_reduce_names[property_type], property_type) for property_type in lr_reduce_names.keys()]
    [DataLoaderFactory.data_loader_arg_factory(ppa.parser, data_loader_sources[property_type], data_loader_names[property_type], property_type) for property_type in data_loader_names.keys()]
    [DataSamplerFactory.data_sampler_arg_factory(ppa.parser, data_sampler_sources[property_type], data_sampler_names[property_type], property_type) for property_type in data_sampler_names.keys()]
    [EncodeFactory.encode_arg_factory(ppa.parser, encode_sources[property_type], encode_names[property_type], property_type) for property_type in encode_names.keys()]
    [BackboneFactory.backbone_arg_factory(ppa.parser, backbone_sources[property_type], backbone_names[property_type], property_type) for property_type in backbone_names.keys()]
    [BackendFactory.backend_arg_factory(ppa.parser, backend_sources[property_type], backend_names[property_type], property_type) for property_type in backend_names.keys()]
    [LossFactory.loss_arg_factory(ppa.parser, loss_sources[property_type], loss_names[property_type], property_type) for property_type in loss_names.keys()]
    [IndicatorFactory.indicator_arg_factory(ppa.parser, indicator_sources[property_type], indicator_names[property_type], property_type) for property_type in indicator_names.keys()]
    [IndicatorStatisticFactory.indicator_statistic_arg_factory(ppa.parser, indicator_statistic_sources[property_type], indicator_statistic_names[property_type]) for property_type in indicator_statistic_names.keys()]
    [OptimizationFactory.optimization_arg_factory(ppa.parser, optimization_sources[property_type], optimization_names[property_type], property_type) for property_type in optimization_names.keys()]
    [AugFactory.aug_arg_factory(ppa.parser, aug_sources[property_type], aug_names[property_type], property_type) for property_type in aug_names.keys()]
    [DataTypeAdapterFactory.data_type_adapter_arg_factory(ppa.parser, data_type_adapter_sources[property_type], data_type_adapter_names[property_type]) for property_type in data_type_adapter_names.keys()]
    FitDataToInputFactory.fit_data_to_input_arg_factory(ppa.parser, fit_data_to_input_source, fit_data_to_input_name)
    [FitToLossInputFactory.fit_to_loss_input_arg_factory(ppa.parser, fit_to_loss_input_sources[property_type], fit_to_loss_input_names[property_type], property_type) for property_type in fit_to_loss_input_names.keys()]
    [FitToIndicatorInputFactory.fit_to_indicator_input_arg_factory(ppa.parser, fit_to_indicator_input_sources[property_type], fit_to_indicator_input_names[property_type], property_type) for property_type in fit_to_indicator_input_names.keys()]
    [FitToDecodeInputFactory.fit_to_decode_input_arg_factory(ppa.parser, fit_to_decode_input_sources[property_type], fit_to_decode_input_names[property_type], property_type) for property_type in fit_to_decode_input_names.keys()]
    FitDecodeToResultFactory.fit_decode_to_result_arg_factory(ppa.parser, fit_decode_to_result_source, fit_decode_to_result_name)
    ModelFactory.model_arg_factory(ppa.parser, model_source, model_name)
    # data setting
    [DatasetFactory.dataset_arg_factory(ppa.parser, dataset_sources[property_type], dataset_names[property_type], property_type) for property_type in dataset_names.keys()]
    ## decode setting
    [DecodeFactory.decode_arg_factory(ppa.parser, decode_sources[property_type], decode_names[property_type], property_type) for property_type in decode_names.keys()]
    RecorderFactory.recorder_arg_factory(ppa.parser, recorder_source, recorder_name)
    AccumulatedOptFactory.accumulated_opt_arg_factory(ppa.parser, accumulated_opt_source, accumulated_opt_name)
    # : the base information set
    ppa.parser.add_argument('--seed', action='store', type=int, default=66, \
        help='the seed for the random')
    # debug
    ppa.parser.add_argument('--remote_debug', action='store_true', default=False, \
        help='setup with remote debug(blocked while not attached) or not')
    ppa.parser.add_argument('--debug', action='store_true', default=False, \
        help='run all the process in two epoch with tiny data')
    ppa.parser.add_argument('--clean_train', type=int, nargs='+', default=None, \
        help='if not None, the result in specified train time would be clean, need args.weight_path')
    # train evaluate test setting
    ### ~train_off and ~evaluate_off: run train stage with evaluate
    ### ~train_off and evaluate_off: run train stage without evaluate
    ### train_off and ~evaluate_off: run evaluate stage
    ### ~test_off: run test stage
    ppa.parser.add_argument('--train_off', action='store_true', default=False, \
        help='do not run train if set')
    ppa.parser.add_argument('--evaluate_off', action='store_true', default=False, \
        help='do not run evaluate if set')
    ppa.parser.add_argument('--test_off', action='store_true', default=False, \
        help='do not run test if set')
    ## train set
    ppa.parser.add_argument('--weight_decay', type=float, action='store', default=1e-4, \
        help='the weight decay, default is 1e-4')
    ppa.parser.add_argument('--gpus', type=str, nargs='+', default=None, \
        help='specify the gpu, this would influence the gpu set in the code')
    ppa.parser.add_argument('--epochs', type=int, default=10, metavar='N', \
        help='number of epochs to train (default: 10)')
    ppa.parser.add_argument('--batch_size', type=int, default=64, metavar='N', \
        help='input batch size for training (default: 64)')
    ppa.parser.add_argument('--summary_interval', type=int, default=100, metavar='N', \
        help='how many batchees to wait before save the summary(default: 100)')
    ppa.parser.add_argument('--evaluate_interval', type=int, default=1, metavar='N', \
        help='how many epoch to wait before evaluate the backbone(default: 1), '\
            'test the mode while the backbone is savd, would not run evaluate while -1')
    ppa.parser.add_argument('--evaluate_batch_size', type=int, default=64, metavar='N', \
        help='input batch size for evaluate (default: 64)')
    ppa.parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', \
        help='input batch size for testing (default: 1000)')
    ppa.parser.add_argument('--log_interval', type=int, default=10, metavar='N', \
        help='how many batches to wait before logging training status(default: 10)')
    ppa.parser.add_argument('--compute_efficiency', action='store_true', default=False, \
        help='evaluate the efficiency in the test or not')
    ppa.parser.add_argument('--data_rate_in_compute_efficiency', type=int, default=200, metavar='N', \
        help='how many sample used in test to evaluate the efficiency(default: 200)')
    # continue train setting
    ppa.parser.add_argument('--weight_path', type=str, default=None, action='store', \
        help='the path where the trained backbone saved')
    ppa.parser.add_argument('--weight_epoch', type=int, default=None, action='store', \
        help='the epoch when saved backbone which had been trained')
    ppa.parser.add_argument('--name', type=str, action='store', default='', \
        help='the ${backbone_name}${name} would be the name of the fold to save the result')
    ppa.parser.add_argument('--train_name', type=str, action='store', default='', \
        help='specify the name as the prefix for the continue train fold name')
    retrain_mode_header = 'retrain_mode'
    ppa.parser.add_argument('--{}_continue'.format(retrain_mode_header), default=False, action='store_true', \
        help='continue train while this is set and the weight_path, weight_epoch is specified, ' \
            'the lr_reduce, auto_save, auto_stop would be load')
    ppa.parser.add_argument('--{}_reset'.format(retrain_mode_header), default=False, action='store_true', \
        help='reset train while this is set and the weight_path, weight_epoch is specified, ' \
            'the lr_reduce, auto_save, auto_stop would not be load')
    args = ppa.parser.parse_args()
    args.auto_save_sources = auto_save_sources
    args.lr_reduce_sources = lr_reduce_sources
    args.auto_stop_sources = auto_stop_sources
    args.dataset_sources = dataset_sources
    args.data_loader_sources = data_loader_sources
    args.data_sampler_sources = data_sampler_sources
    args.encode_sources = encode_sources
    args.backbone_sources = backbone_sources
    args.backend_sources = backend_sources
    args.decode_sources = decode_sources
    args.fit_to_decode_input_sources = fit_to_decode_input_sources
    args.loss_sources = loss_sources
    args.indicator_sources = indicator_sources
    args.indicator_statistic_sources = indicator_statistic_sources
    args.optimization_sources = optimization_sources
    args.aug_sources = aug_sources
    args.data_type_adapter_sources = data_type_adapter_sources
    args.fit_data_to_input_source = fit_data_to_input_source
    args.fit_to_loss_input_sources = fit_to_loss_input_sources
    args.fit_to_indicator_input_sources = fit_to_indicator_input_sources
    args.fit_decode_to_result_source = fit_decode_to_result_source
    args.model_source = model_source
    args.recorder_source = recorder_source
    args.accumulated_opt_source = accumulated_opt_source
    args.auto_save_names = auto_save_names
    args.auto_stop_names = auto_stop_names
    args.lr_reduce_names = lr_reduce_names
    args.dataset_names = dataset_names
    args.data_loader_names = data_loader_names
    args.data_sampler_names = data_sampler_names
    args.encode_names = encode_names
    #args.backbone_name = backbone_name
    args.backbone_names = backbone_names
    args.backend_names = backend_names
    args.decode_names = decode_names
    args.fit_to_decode_input_names = fit_to_decode_input_names
    args.loss_names = loss_names
    args.indicator_names = indicator_names
    args.indicator_statistic_names = indicator_statistic_names
    args.optimization_names = optimization_names
    args.aug_names = aug_names
    args.data_type_adapter_names = data_type_adapter_names
    args.fit_data_to_input_name = fit_data_to_input_name
    args.fit_to_loss_input_names = fit_to_loss_input_names
    args.fit_to_indicator_input_names = fit_to_indicator_input_names
    args.fit_decode_to_result_name = fit_decode_to_result_name
    args.model_name = model_name
    args.recorder_name = recorder_name
    args.accumulated_opt_name = accumulated_opt_name
    args.gpus = [[int(g) for g in gpu.split('.')] for gpu in args.gpus]
    args.framework = framework

    import Putil.base.logger as plog
    reload(plog)

    empty_tensor = BaseOperationFactory.empty_tensor_factory(args)()
    # the method for remote debug
    if args.remote_debug and hvd.rank() == 0:
        import ptvsd
        host = '127.0.0.1'
        port = 12345
        ptvsd.enable_attach(address=(host, port), redirect_output=True)
        if __name__ == '__main__':
            print('waiting for remote attach')
            ptvsd.wait_for_attach()
            pass
        pass
    if args.remote_debug:
        hvd.broadcast_object(empty_tensor(), 0, 'sync_waiting_for_the_attach')
    # 生成存储位置，更新args.save_dir, 让server与worker都在同一save_dir
    util.make_sure_the_save_dir(args)
    # 删除clear_train指定的train_time结果
    if args.clean_train is not None:
        for _train_time in args.clean_train:
            hvd.broadcast_object(empty_tensor(), 0, 'sync_before_checking_clean')
            make_sure_clean_the_train_result = input(Fore.RED + 'clean the train time {} in {} (y/n):'.format(_train_time, args.save_dir) + Fore.RESET) if hvd.rank() == 0 else False
            hvd.broadcast_object(empty_tensor(), 0, 'sync_after_checking_clean')
            util.clean_train_result(_train_time, args.save_dir) if make_sure_clean_the_train_result and hvd.rank() == 0 else None
            hvd.broadcast_object(empty_tensor(), 0, 'sync_after_clean')
            pass
        pass
    # 确定当前的train_time, 需要args.save_dir，生成args.train_time
    util.make_sure_the_train_time(args)
    args.save_dir = util.subdir_base_on_train_time(args.save_dir, args.train_time, args.train_name)
    hvd.broadcast_object(empty_tensor(), 0, 'sync_before_make_save_dir')
    assert not os.path.exists(args.save_dir) if hvd.rank() == 0 else True
    os.mkdir(args.save_dir) if hvd.rank() == 0 else None
    hvd.broadcast_object(empty_tensor(), 0, 'sync_after_make_save_dir')
    print('rank {} train time {} save to {}'.format(hvd.rank(), args.train_time, args.save_dir))
    log_level = plog.LogReflect(args.log_level).Level
    plog.PutilLogConfig.config_format(plog.FormatRecommend)
    plog.PutilLogConfig.config_log_level(stream=log_level, file=log_level)
    plog.PutilLogConfig.config_file_handler(filename=os.path.join(args.save_dir, \
        'train.log' if util.train_stage(args) else 'evaluate.log' if util.evaluate_stage(args) else 'test.log'), mode='a')
    plog.PutilLogConfig.config_handler(plog.stream_method | plog.file_method)
    root_logger = plog.PutilLogConfig('train').logger()
    root_logger.setLevel(log_level)
    MainLogger = root_logger.getChild('Trainer')
    MainLogger.setLevel(log_level)
    #<tag========================================reload 部分=====================================
    reload(AutoSaveFactory)
    reload(AutoStopFactory)
    reload(LrReduceFactory)
    reload(DatasetFactory)
    reload(DataLoaderFactory)
    reload(DataSamplerFactory)
    reload(ModelFactory)
    reload(LossFactory)
    reload(IndicatorFactory)
    reload(IndicatorStatisticFactory)
    reload(OptimizationFactory)
    reload(EncodeFactory)
    reload(DecodeFactory)
    reload(FitDataToInputFactory)
    reload(FitToLossInputFactory)
    reload(FitToIndicatorInputFactory)
    reload(FitDecodeToResultFactory)
    reload(RecorderFactory)
    reload(util)
    reload(horovod)
    reload(BaseOperationFactory)
    reload(AccumulatedOptFactory)
    #========================================reload 部分=====================================>
    from Putil.base.arg_operation import args_save as ArgsSave
    from util.run_train_stage import train_stage_common 
    from Putil.data import aug as pAug
    from util import run_train_stage
    # 确定性设置
    from Putil.base import base_setting
    ## BaseOperationFactory
    def _init_fn(worker_id):
        np.random.seed(int(args.seed) + worker_id)
        pass
    args.dataloader_deterministic_work_init_fn = _init_fn
    # : set the args item
    checkpoint = BaseOperationFactory.checkpoint_factory(args)() if util.train_stage(args) else None
    save = BaseOperationFactory.save_factory(args)() if util.train_stage(args) else None
    deploy = BaseOperationFactory.deploy_factory(args)() if util.train_stage else None
    load_saved = BaseOperationFactory.load_saved_factory(args)()
    load_checkpointed = BaseOperationFactory.load_checkpointed_factory(args)()
    is_cudable = BaseOperationFactory.is_cudable_factory(args)()
    accumulated_opt = AccumulatedOptFactory.accumulated_opt_factory(args)()
    combine_optimization = BaseOperationFactory.combine_optimization_factory(args)()
    #empty_tensor = BaseOperationFactory.generate_model_element_factory(args)()

    if hvd.rank() == 0:
        writer = SummaryWriter(args.save_dir, filename_suffix='-{}'.format(args.train_time))
    # prepare the GPU
    if util.iscuda(args):
        # Horovod: pin GPU to local rank. TODO: rank is the global process index, local_rank is the local process index in a machine
        # such as -np 6 -H *.1:1 *.2:2 *.3:3 would get the rank: 0, 1, 2, 3, 4, 5 and the local rank: {[0], [0, 1], [0, 1, 2]}
        gpu_accumualation = [len(gs) for gs in args.gpus]
        for gpu_group, (gpus, amount) in enumerate(zip(args.gpus, gpu_accumualation)):
            if hvd.rank() < amount:
                break
            else:
                continue
            pass
        args.gpu = args.gpus[gpu_group][hvd.local_rank()]
        MainLogger.info("rank: {}; local_rank: {}; gpu: {}; gpu_group: {}".format( \
            hvd.rank(), hvd.local_rank(), args.gpu, gpu_group))
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    # Horovod: limnit # of CPU threads to be used per worker
    torch.set_num_threads(1)
    kwargs = {'num_workers': args.n_worker_per_dataset, 'pin_memory': True}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'
    #<tag========================================the arg would not change after this=========================================
    pab.args_log(args, MainLogger) if hvd.rank() == 0 else None
    ArgsSave(args, os.path.join(args.save_dir, 'args')) if hvd.rank() == 0 else None
    #========================================the arg would not change after this=============================================>
    fit_to_loss_input = {property_type: FitToLossInputFactory.fit_to_loss_input_factory(args, property_type)() for property_type in args.fit_to_loss_input_names.keys()} # 从data获取的datas提取loss的input（label）
    fit_to_loss_input = util.get_module(fit_to_loss_input)
    fit_to_indicator_input = {property_type: FitToIndicatorInputFactory.fit_to_indicator_input_factory(args, args.fit_to_indicator_input_sources[property_type], args.fit_to_indicator_input_names[property_type], property_type)() for property_type in args.fit_to_indicator_input_names.keys()}
    fit_to_indicator_input = util.get_module(fit_to_indicator_input)
    fit_to_decode_input = {property_type: FitToDecodeInputFactory.fit_to_decode_input_factory(args, args.fit_to_decode_input_sources[property_type], args.fit_to_decode_input_names[property_type], property_type)() for property_type in args.fit_to_decode_input_names.keys()}
    fit_to_decode_input = util.get_module(fit_to_decode_input)
    fit_decode_to_result = FitDecodeToResultFactory.fit_decode_to_result_factory(args)() # 从decode的结果生成通用的result格式，可供dataset直接保存
    if util.train_stage(args):
        # : build the backbone
        backbone = {property_type: BackboneFactory.backbone_factory(args, args.backbone_sources[property_type], args.backbone_names[property_type], property_type)() for property_type in args.backbone_names.keys()}
        backbone = util.get_module(backbone)
        # : build backend
        backend = {property_type: BackendFactory.backend_factory(args, args.backend_sources[property_type], args.backend_names[property_type], property_type)() for property_type in args.backend_names.keys()}
        backend = util.get_module(backend)
        # : build decode
        decode = {property_type: DecodeFactory.decode_factory(args, args.decode_sources[property_type], args.decode_names[property_type], property_type=property_type)() for property_type in args.decode_names.keys()}
        decode = util.get_module(decode)
        # : build the loss
        loss = {property_type: LossFactory.loss_factory(args, args.loss_sources[property_type], args.loss_names[property_type], property_type, fit_to_loss_input=fit_to_loss_input)() for property_type in args.loss_names.keys()}
        loss = util.get_module(loss)
        # : build the indicator
        train_indicator = {property_type: IndicatorFactory.indicator_factory(args, args.indicator_sources[property_type], args.indicator_names[property_type], fit_to_indicator_input=fit_to_indicator_input)() for property_type in args.indicator_names.keys()}
        train_indicator = util.get_module(train_indicator)
        evaluate_indicator = {property_type: IndicatorFactory.indicator_factory(args, args.indicator_sources[property_type], args.indicator_names[property_type], fit_to_indicator_input=fit_to_indicator_input)() for property_type in args.indicator_names.keys()} if args.evaluate_off is not True else None
        evaluate_indicator = util.get_module(evaluate_indicator) if args.evaluate_off is not True else None
        # : build the statistic indicator
        indicator_statistic = {property_type: IndicatorStatisticFactory.indicator_statistic_factory(args, args.indicator_statistic_sources[property_type], args.indicator_statistic_names[property_type], property_type=property_type)() for property_type in args.indicator_statistic_names.keys()}
        indicator_statistic = util.get_module(indicator_statistic)
        ##TODO: build the optimization, the optimization_source
        # 通过environ指定了几种属性类型的optimization，使用在哪些参数需要自己定制
        # 如果出现参数调整，则需要另外使用key，否则在load_checkpointed的时候会出现错误，可以查看load_checkpointed的代码
        # 如果出现需要增加参数，建议新增一个key-val，这用容易管理，load_checkpointed不会出现错误
        optimizations = dict()
        optimization = {property_type: OptimizationFactory.optimization_factory(args, property_type=property_type) for property_type, v in args.optimization_names.items()}
        optimizations.update({'opt-backbone': (backbone, util.get_module(optimization)(backbone.parameters()))})
        #optimizations.update({'opt-backend': (backend, util.get_module(optimization)(backend.parameters()))})
        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(backbone.state_dict(), root_rank=0)
        hvd.broadcast_parameters(backend.state_dict(), root_rank=0)
        [hvd.broadcast_optimizer_state(optimization, root_rank=0) for k, (module, optimization) in optimizations.items()]
        ##Horovod: wrap optimizer with DistributedOptimizer.
        # 影响参数：hvd_fp16_util.all_reduce、hvd_reduce_mode
        import horovod.torch as hvd
        optimizations = {k: hvd.DistributedOptimizer(optimization, named_parameters=module.named_parameters(), \
            compression=hvd.Compression.fp16 if args.hvd_compression_mode == 'fp16' else hvd.Compression.mro if args.hvd_compression_mode == 'mro' else hvd.Compression.none, \
                op=hvd.Adasum if args.hvd_reduce_mode == 'AdaSum' else hvd.Average if args.hvd_reduce_mode == 'Average' else hvd.Sum) \
                for k, (module, optimization) in optimizations.items()}
        optimization=combine_optimization(optimizations)
        #  : the auto save
        auto_save = {property_type: AutoSaveFactory.auto_save_factory(args, args.auto_save_sources[property_type], args.auto_save_names[property_type], property_type)() for property_type in args.auto_save_names.keys()}
        auto_save = util.get_module(auto_save)
        #  : the auto stop
        auto_stop = {property_type: AutoStopFactory.auto_stop_factory(args, args.auto_stop_sources[property_type], args.auto_stop_names[property_type], property_type)() for property_type in args.auto_stop_names.keys()}
        auto_stop = util.get_module(auto_stop)
        #  : the lr reduce
        lr_reduce = {property_type: LrReduceFactory.lr_reduce_factory(args, args.lr_reduce_sources[property_type], args.lr_reduce_names[property_type], property_type)() for property_type in args.lr_reduce_names.keys()}
        lr_reduce = util.get_module(lr_reduce)
        if util.iscuda(args):
            backbone.cuda() if is_cudable(backbone) else None
            backend.cuda() if is_cudable(backend) else None
            decode.cuda() if is_cudable(decode) else None
            loss.cuda() if is_cudable(loss) else None
            train_indicator.cuda() if is_cudable(train_indicator) else None
            evaluate_indicator.cuda() if args.evaluate_off is not True else None
            indicator_statistic.cuda() if is_cudable(indicator_statistic) else None
            if args.hvd_reduce_mode and hvd.nccl_built():
                lr_scaler = hvd.local_size()
                pass
            pass
        pass
    recorder = RecorderFactory.recorder_factory(args)()
    encode = {property_type: EncodeFactory.encode_factory(args, property_type)() for property_type in args.encode_names.keys()}
    encode = util.get_module(encode)
    template_model = ModelFactory.model_factory(args)()
    # : build the train dataset
    fit_data_to_input = FitDataToInputFactory.fit_data_to_input_factory(args)() # 从data获取的datas提取backbone的input
    data_type_adapter = {property_type: DataTypeAdapterFactory.data_type_adapter_factory(args, args.data_type_adapter_sources[property_type], args.data_type_adapter_names[property_type], property_type)() \
        for property_type in data_type_adapter_names.keys()}
    data_type_adapter = util.get_module(data_type_adapter)
    dataset_train = None; train_sampler = None; evaluate_loader = None
    if args.train_off is not True:
        MainLogger.info('start to generate the train dataset data_sampler data_loader')
        dataset_train = {property_type: DatasetFactory.dataset_factory(args, property_type, stage=util.Stage.Train)() for property_type, name in args.dataset_names.items()}
        ##TODO: 根据实际需求，指定使用的dataset，但一般有多种性质的dataset的情况都是要进行combine，CombineDataset框架还不完善
        dataset_train = util.get_module(dataset_train)
        root_node = pAug.AugNode(pAug.AugFuncNoOp())
        # for the fack data field, maybe cause the nan or inf
        for i in range(0, args.fake_aug):
            root_node.add_child(pAug.AugNode(pAug.AugFuncNoOp()))
        if args.naug is False:
            Original = root_node.add_child(pAug.AugNode(pAug.AugFuncNoOp()))
            [root_node.add_child(pAug.AugNode(AugFactory.aug_factory(args, property_type)())) for property_type in args.aug_names.keys()]
        root_node.freeze_node()
        dataset_train.set_aug_node_root(root_node)
        dataset_train.set_convert_to_input_method(encode)
        dataset_train.set_data_type_adapter(data_type_adapter)
        train_sampler = {property_type: DataSamplerFactory.data_sampler_factory(args, args.data_sampler_sources[property_type], args.data_sampler_names[property_type], property_type)( \
            dataset_train, rank_amount=hvd.size(), rank=hvd.rank()) for property_type in args.data_sampler_names.keys()}  if dataset_train is not None else None
        train_sampler = util.get_module(train_sampler)
        train_loader = {property_type: DataLoaderFactory.data_loader_factory(args, args.data_loader_sources[property_type], args.data_loader_names[property_type], property_type)( \
            dataset=dataset_train, data_sampler=train_sampler, stage=util.Stage.Train) for property_type in args.data_loader_names.keys()} if dataset_train is not None else None
        train_loader = util.get_module(train_loader)
        MainLogger.info('generate adataset train successful: {} sample'.format(len(dataset_train)))
    # : build the evaluate dataset
    dataset_evaluate = None; evaluate_sampler = None; evaluate_loader = None
    if args.evaluate_off is not True:
        MainLogger.info('start to generate the evaluate dataset data_sampler data_loader')
        dataset_evaluate = {property_type: DatasetFactory.dataset_factory(args, stage=util.Stage.Evaluate)() for property_type, name in args.dataset_names.items()}
        ##TODO: 根据实际需求，指定使用的dataset，但一般有多种性质的dataset的情况都是要进行combine，CombineDataset框架还不完善
        dataset_evaluate = util.get_module(dataset_evaluate)
        root_node = pAug.AugNode(pAug.AugFuncNoOp())
        root_node.add_child(pAug.AugNode(pAug.AugFuncNoOp()))
        root_node.freeze_node()
        dataset_evaluate.set_aug_node_root(root_node)
        dataset_evaluate.set_convert_to_input_method(encode)
        dataset_evaluate.set_data_type_adapter(data_type_adapter)
        evaluate_sampler = {property_type: DataSamplerFactory.data_sampler_factory(args, args.data_sampler_sources[property_type], args.data_sampler_names[property_type], property_type)( \
            dataset=dataset_evaluate, rank_amount=hvd.size(), rank=hvd.rank()) for property_type in args.data_sampler_names.keys()} if dataset_evaluate is not None else None
        evaluate_sampler = util.get_module(evaluate_sampler)
        evaluate_loader = {property_type: DataLoaderFactory.data_loader_factory(args, args.data_loader_sources[property_type], args.data_loader_names[property_type], property_type)( \
            dataset=dataset_evaluate, data_sampler=evaluate_sampler, stage=util.Stage.Evaluate) for property_type in args.data_loader_names.keys()} if dataset_evaluate is not None else None
        evaluate_loader = util.get_module(evaluate_loader)
    # : build the test dataset
    dataset_test = None; test_sampler = None; test_loader = None
    if args.test_off is not True:
        MainLogger.info('start to generate the evaluate dataset data_sampler data_loader')
        dataset_test = {property_type: DatasetFactory.dataset_factory(args, stage=util.Stage.Test)() for property_type, name in args.dataset_names.items()} if args.test_off is not True else None
        ##TODO: 根据实际需求，指定使用的dataset，但一般有多种性质的dataset的情况都是要进行combine，CombineDataset框架还不完善
        dataset_test = util.get_module(dataset_test)
        root_node = pAug.AugNode(pAug.AugFuncNoOp())
        root_node.add_child(pAug.AugNode(pAug.AugFuncNoOp()))
        root_node.freeze_node()
        dataset_test.set_aug_node_root(root_node)
        dataset_test.set_convert_to_input_method(encode)
        dataset_test.set_data_type_adapter(data_type_adapter)
        test_sampler = {property_type: DataSamplerFactory.data_sampler_factory(args, args.data_sampler_sources[property_type], args.data_sampler_names[property_type], property_type)( \
            dataset_test, rank_amount=hvd.size(), rank=hvd.rank()) for property_type in args.data_sampler_names.keys()} if dataset_test is not None else None
        test_sampler = util.get_module(test_sampler)
        test_loader = {property_type: DataLoaderFactory.data_loader_factory(args, args.data_loader_sources[property_type], args.data_loader_names[property_type], property_type)( \
            dataset_test, data_sampler=test_sampler, stage=util.Stage.Test) for property_type in args.data_loader_names.keys()} if dataset_test is not None else None
        test_loader = util.get_module(test_loader)
    test_stage() if util.test_stage(args) else None # 如果train_off为True 同时test_off为False，则为util.test_stage，util.evaluate_stage与util.test_stage可以同时存在
    evaluate_stage() if util.evaluate_stage(args) else None # 如果train_off为True 同时evaluate_off为False，则为util.evaluate_stage，util.evaluate_stage与util.test_stage可以同时存在
    if util.train_stage(args):
        # 如果train_off不为True，就是util.train_stage，只是util.train_stage中可以设定进不进行evaluate与test
        if args.weight_path != '' and args.weight_epoch is not None:
            assert np.sum([args.retrain_mode_continue_train, args.retrain_mode_reset_train])
            MainLogger.info(Fore.YELLOW + 'load trained backbone: path: {} epoch: {}'.format(args.weight_path, args.weight_epoch) + Fore.RESET)
            retrain_mode_field = dict()
            {retrain_mode_field.update({k: v}) if re.search(retrain_mode_header, k) is not None else None for k, v in args.__dict__.items()}
            assert np.sum(list(retrain_mode_field.values())) == 1
            MainLogger.info(Fore.YELLOW + 'this project contain retrain_mode: {}, set to: {}'.format(retrain_mode_field, ''.join([k if v else '' for k, v in retrain_mode_field.items()])) + Fore.RESET)
            target_dict = {}
            target_dict.update(optimization.state_dict())
            target_dict.update({'backbone': backbone})
            target_dict.update({'backend': backend})
            target_dict.update({'recorder': recorder})
            if eval('args.{}_continue'.format(retrain_mode_header)):
                target_dict.update({'lr_reduce': lr_reduce})
                target_dict.update({'auto_save': auto_save})
                target_dict.update({'auto_stop': auto_stop})
            elif eval('args.{}_reset'.format(retrain_mode_header)):
                pass
            else:
                raise NotImplementedError('continue_train_mode {} is not implemented'.format(args.continue_train_mode))
            #target_dict.update({'': }) if args.
            load_checkpointed(args.weight_epoch, args.weight_path, target_dict, map_location=torch.device(args.gpu))
        for epoch in range(recorder.epoch + 1, recorder.epoch + args.epochs + 1):
            train_ret = train(epoch)
            if train_ret[0] is True:
                break
            if ((epoch + 1) % args.evaluate_interval == 0) or (args.debug) and args.evaluate_off is False:
                evaluate_ret = evaluate(epoch) 
                if evaluate_ret[0] is True:
                    if args.debug:
                        MainLogger.debug('del the backbone, backend, decode, optimization, lr_reduce, auto_save, auto_stop')
                        del backbone, backend, decode, optimization, lr_reduce, auto_save, auto_stop
                        torch.cuda.empty_cache()
                        args.weight_path = args.save_dir
                        args.weight_epoch = 1
                        MainLogger.debug('run the evaluate stage')
                        evaluate_stage() if not args.evaluate_off else None
                        MainLogger.debug('run the test stage')
                        test_stage() if not args.test_off else None
                    break
                if evaluate_ret[2] is True and args.test_off is False:
                    MainLogger.info('run test')
                    test() if args.test_off is False else MainLogger.info('test_off, do not run test')
            else:
                MainLogger.info('evaluate_off, do not run the evaluate')
                pass
            pass
        pass
    pass