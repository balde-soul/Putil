# coding=utf-8
from colorama import Fore
import sys
import json
import os
from enum import Enum
import torch


def do_save():
    MainLogger.info('run checkpoint') if args.debug else None
    checkpoint(epoch, args.save_dir, backbone=backbone, lr_reduce=lr_reduce, auto_save=auto_save, \
        auto_stop=auto_stop, optimization=optimization)
    MainLogger.info('run save') if args.debug else None
    save(TemplateModelDecodeCombineT, epoch, args.save_dir, backbone, backend, decode)
    MainLogger.info('run deploy') if args.debug else None
    deploy(TemplateModelDecodeCombineT, \
        torch.from_numpy(np.zeros(shape=(1, 3, args.input_height, args.input_width))).cuda(), \
            epoch, args.save_dir, backbone, backend, decode)

def do_epoch_end_process():
    indicator = all_reduce(ret['eloss'], 'train_indicator')
    save = auto_save.save_or_not(indicator)
    if save or args.debug:
        # :save the backbone in rank0
        do_save() if hvd.rank() == 0 else None
    # :stop or not
    MainLogger.info('run ')
    stop = auto_stop.stop_or_not(indicator)
    # :lr_reduce
    _reduce = lr_reduce.reduce_or_not(indicator)
    # TODO: change the lr
    optimizer.__dict__['param_group'][0]['lr'] = lr_reduce.reduce(optimization.__dict__['param_groups'][0]['lr']) if _reduce \
        else optimization.__dict__['param_groups'][0]['lr']
    if hvd.rank() == 0:
        writer.add_scalar('lr', lr_reduce.LrNow, global_step=epoch * len(train_loader.dataset))
    return stop, lr_reduce.LrNow, save


def train(epoch):
    ret = train_evaluate_common(args, Stage.Train, epoch, fit_data_to_input, backbone, backend, decode, fit_decode_to_result, \
         loss, optimization, indicator, statistic_indicator, train_loader, MainLogger)
    if args.off_evaluate:
        if args.debug:
            if epoch == 0:
                return False, 
            elif epoch == 1:
                do_epoch_end_process()
                return False,
            else:
                raise RuntimeError('all_process_test would only run train two epoch')
        else:
            return do_epoch_end_process()
    else:
        return False,


def evaluate(epoch):
    ret = train_evaluate_common(args, Stage.TrainEvaluate if evaluate_stage(args) else Stage.Evaluate, \
        epoch, fit_data_to_input, backbone, backend, decode, fit_decode_to_result, loss, optimization, \
            indicator, statistic_indicator, train_loader, MainLogger)
    if train_stage(args):
        if args.debug:
            if epoch == 0:
                # 当在all_process_test时，第二个epoch返回stop为True
                do_epoch_end_process()
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
        pass
    pass


def run_test_stage():
    MainLogger.info('run test') 
    assert args.weight_path != '' and args.weight_epoch is not None, 'specify the trained weight_path and the epoch in test stage'
    MainLogger.info('load trained backbone: path: {} epoch: {}'.format(args.weight_path, args.weight_epoch))
    model = load_saved(args.weight_epoch, args.weight_path)
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
            result = fit_decode_to_result(output)
            data_loader.dataset.save_result(prefix='evaluate', save=False if index != len(data_loader) else True)
            pass
        pass
    pass


def run_evaluate_stage():
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


if __name__ == '__main__':
    import Putil.base.arg_base as pab
    import Putil.base.save_fold_base as psfb
    import os
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
    #ppa.parser.add_argument('--statistic_indicator_name', type=str, default='', action='store', \
    #    help='the name of the statistic_indicator in the statistic_indicator_factory, see the util.statistic_indicator_factory')
    #ppa.parser.add_argument('--statistic_indicator_source', type=str, default='standard', action='store', \
    #    help='standard: from the Putil.demo.base.deep_learning.statistic_indicator; project: from this project')
    from Putil.demo.deep_learning.base import auto_save_factory as AutoSaveFactory
    from Putil.demo.deep_learning.base import auto_stop_factory as AutoStopFactory
    from Putil.demo.deep_learning.base import lr_reduce_factory as LrReduceFactory
    from Putil.demo.deep_learning.base import dataset_factory as DatasetFactory
    from Putil.demo.deep_learning.base import data_loader_factory as DataLoaderFactory
    from Putil.demo.deep_learning.base import data_sampler_factory as DataSamplerFactory
    from Putil.demo.deep_learning.base import encode_factory as EncodeFactory
    #from Putil.demo.deep_learning.base import backbone_factory as ModelFactory
    from Putil.demo.deep_learning.base import backbone_factory as BackboneFactory
    from Putil.demo.deep_learning.base import backend_factory as BackendFactory
    from Putil.demo.deep_learning.base import decode_factory as DecodeFactory
    from Putil.demo.deep_learning.base import loss_factory as LossFactory
    from Putil.demo.deep_learning.base import indicator_factory as IndicatorFactory
    from Putil.demo.deep_learning.base import statistic_indicator_factory as StatisticIndicatorFactory
    from Putil.demo.deep_learning.base import optimization_factory as OptimizationFactory
    from Putil.demo.deep_learning.base import aug_factory as AugFactory
    from Putil.demo.deep_learning.base import data_type_adapter_factory as DataTypeAdapterFactory
    from Putil.demo.deep_learning.base import fit_data_to_input_factory as FitDataToInputFactory
    from Putil.demo.deep_learning.base import fit_decode_to_result_factory as FitDecodeToResultFactory
    from Putil.demo.deep_learning.base import model_factory as ModelFactory
    auto_save_source = os.environ.get('auto_save_source', 'standard')
    auto_stop_source = os.environ.get('auto_stop_source', 'standard')
    lr_reduce_source = os.environ.get('lr_reduce_source', 'standard')
    dataset_source = os.environ.get('dataset_source', 'standard')
    data_loader_source = os.environ.get('data_loader_source', 'standard')
    data_sampler_source = os.environ.get('data_sampler_source', 'standard')
    encode_source = os.environ.get('encode_source', 'standard')
    #backbone_source = os.environ.get('backbone_source', 'standard')
    backbone_source = os.environ.get('backbone_source', 'standard')
    backend_source = os.environ.get('backend_source', 'standard')
    decode_source = os.environ.get('decode_source', 'standard')
    loss_source = os.environ.get('loss_source', 'standard')
    indicator_source = os.environ.get('indicator_source', 'standard')
    statistic_indicator_source = os.environ.get('statistic_indicator_source', 'standard')
    optimization_source = os.environ.get('optimization_source', 'standard')
    aug_sources = os.environ.get('aug_sources', '').split('-')
    data_type_adapter_source = os.environ.get('data_type_adapter_source', 'standard')
    fit_data_to_input_source = os.environ.get('fit_data_to_input_source', 'standard')
    fit_decode_to_result_source = os.environ.get('fit_decode_to_result_source', 'standard')
    model_source = os.environ.get('model_source', 'standard')
    auto_save_name = os.environ.get('auto_save_name', 'DefaultAutoSave')
    auto_stop_name = os.environ.get('auto_stop_name', 'DefaultAutoStop')
    lr_reduce_name = os.environ.get('lr_reduce_name', 'DefaultLrReduce')
    dataset_name = os.environ.get('dataset_name', 'DefaultDataset')
    data_loader_name = os.environ.get('data_loader_name', 'DefaultDataLoader')
    data_sampler_name = os.environ.get('data_sampler_name', 'DefaultDataSampler')
    encode_name = os.environ.get('encode_name', 'DefaultEncode')
    #backbone_name = os.environ.get('backbone_name', 'DefaultModel')
    backbone_name = os.environ.get('backbone_name', 'DefaultBackbone')
    backend_name = os.environ.get('backend_name', 'DefaultBackend')
    decode_name = os.environ.get('decode_name', 'DefaultDecode')
    loss_name = os.environ.get('loss_name', 'DefaultLoss')
    indicator_name = os.environ.get('indicator_name', 'DefaultIndicator')
    statistic_indicator_name = os.environ.get('statistic_indicator_name', 'DefaultStatisticIndicator')
    optimization_name = os.environ.get('optimization_name', 'DefaultOptimization')
    aug_names = os.environ.get('aug_names', '').split('-')
    data_type_adapter_name = os.environ.get('data_type_adapter_name', 'DefaultDataTypeAdapter')
    fit_data_to_input_name = os.environ.get('fit_data_to_input_name', 'DefaultFitDataToInput')
    fit_decode_to_result_name = os.environ.get('fit_decode_to_result_name', 'DefaultFitDecodeToResult')
    model_name = os.environ.get('model_name', 'DefaultModel')
    AutoSaveFactory.auto_save_arg_factory(ppa.parser, auto_save_source, auto_save_name)
    AutoStopFactory.auto_stop_arg_factory(ppa.parser, auto_stop_source, auto_stop_name)
    LrReduceFactory.lr_reduce_arg_factory(ppa.parser, lr_reduce_source, lr_reduce_name)
    DataLoaderFactory.data_loader_arg_factory(ppa.parser, data_loader_source, data_loader_name)
    DataSamplerFactory.data_sampler_arg_factory(ppa.parser, data_sampler_source, data_sampler_name)
    EncodeFactory.encode_arg_factory(ppa.parser, encode_source, encode_name)
    #ModelFactory.backbone_arg_factory(ppa.parser, backbone_source, backbone_name)
    BackboneFactory.backbone_arg_factory(ppa.parser, backbone_source, backbone_name)
    BackendFactory.backend_arg_factory(ppa.parser, backend_source, backend_name)
    LossFactory.loss_arg_factory(ppa.parser, loss_source, loss_name)
    IndicatorFactory.indicator_arg_factory(ppa.parser, indicator_source, indicator_name)
    StatisticIndicatorFactory.statistic_indicator_arg_factory(ppa.parser, statistic_indicator_source, statistic_indicator_name)
    OptimizationFactory.optimization_arg_factory(ppa.parser, optimization_source, optimization_name)
    [AugFactory.aug_arg_factory(ppa.parser, aug_source, aug_name) for aug_source, aug_name in zip(aug_sources, aug_names)]
    DataTypeAdapterFactory.data_type_adapter_arg_factory(ppa.parser, data_type_adapter_source, data_type_adapter_name)
    FitDataToInputFactory.fit_data_to_input_arg_factory(ppa.parser, fit_data_to_input_source, fit_data_to_input_name)
    FitDecodeToResultFactory.fit_decode_to_result_arg_factory(ppa.parser, fit_decode_to_result_source, fit_decode_to_result_name)
    ModelFactory.model_arg_factory(ppa.parser, model_source, model_name)
    # data setting
    DatasetFactory.dataset_arg_factory(ppa.parser, dataset_source, dataset_name)
    ## decode setting
    DecodeFactory.decode_arg_factory(ppa.parser, decode_source, decode_name)
    # : the base information set
    ppa.parser.add_argument('--framework', action='store', type=str, default='torch', \
        help='specify the framework used')
    # debug
    ppa.parser.add_argument('--remote_debug', action='store_true', default=False, \
        help='setup with remote debug(blocked while not attached) or not')
    ppa.parser.add_argument('--frame_debug', action='store_true', default=False, \
        help='run all the process in two epoch with tiny data')
    # train evaluate test setting
    ### ~off_train and ~off_evaluate: run train stage with evaluate
    ### ~off_train and off_evaluate: run train stage without evaluate
    ### off_train and ~off_evaluate: run evaluate stage
    ### ~off_test: run test stage
    ppa.parser.add_argument('--off_train', action='store_true', default=False, \
        help='do not run train if set')
    ppa.parser.add_argument('--off_evaluate', action='store_true', default=False, \
        help='do not run evaluate if set')
    ppa.parser.add_argument('--off_test', action='store_true', default=False, \
        help='do not run test if set')
    ppa.parser.add_argument('--gpus', type=int, nargs='+', default=None, \
        help='specify the gpu, this would influence the gpu set in the code')
    ppa.parser.add_argument('--epochs', type=int, default=10, metavar='N', \
        help='number of epochs to train (default: 10)')
    ppa.parser.add_argument('--batch_size', type=int, default=64, metavar='N', \
        help='input batch size for training (default: 64)')
    ppa.parser.add_argument('--evaluate_batch_size', type=int, default=64, metavar='N', \
        help='input batch size for evaluate (default: 64)')
    ppa.parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', \
        help='input batch size for testing (default: 1000)')
    ppa.parser.add_argument('--log_interval', type=int, default=10, metavar='N', \
        help='how many batches to wait before logging training status(default: 10)')
    ppa.parser.add_argument('--summary_interval', type=int, default=100, metavar='N', \
        help='how many batchees to wait before save the summary(default: 100)')
    ppa.parser.add_argument('--evaluate_interval', type=int, default=1, metavar='N', \
        help='how many epoch to wait before evaluate the backbone(default: 1), '\
            'test the mode while the backbone is savd, would not run evaluate while -1')
    ppa.parser.add_argument('--compute_efficiency', action='store_true', default=False, \
        help='evaluate the efficiency in the test or not')
    ppa.parser.add_argument('--data_rate_in_compute_efficiency', type=int, default=200, metavar='N', \
        help='how many sample used in test to evaluate the efficiency(default: 200)')
    ppa.parser.add_argument('--weight_decay', type=float, action='store', default=1e-4, \
        help='the weight decay, default is 1e-4')
    # backbone setting
    ppa.parser.add_argument('--weight_path', type=str, default='', action='store', \
        help='the path where the trained backbone saved')
    ppa.parser.add_argument('--weight_epoch', type=int, default=None, action='store', \
        help='the epoch when saved backbone which had been trained')
    ppa.parser.add_argument('--name', type=str, action='store', default='', \
        help='the ${backbone_name}${name} would be the name of the fold to save the result')
    args = ppa.parser.parse_args()
    args.auto_save_name = auto_save_name
    args.auto_save_source = auto_save_source
    args.lr_reduce_source = lr_reduce_source
    args.auto_stop_source = auto_stop_source
    args.dataset_source = dataset_source
    args.data_loader_source = data_loader_source
    args.data_sampler_source = data_sampler_source
    args.encode_source = encode_source
    #args.backbone_source = backbone_source
    args.backbone_source = backbone_source
    args.backend_source = backend_source
    args.decode_source = decode_source
    args.loss_source = loss_source
    args.indicator_source = indicator_source
    args.statistic_indicator_source = statistic_indicator_source
    args.optimization_source = optimization_source
    args.aug_sources = aug_sources
    args.data_type_adapter_source = data_type_adapter_source
    args.fit_data_to_input_source = fit_data_to_input_source
    args.fit_decode_to_result_source = fit_decode_to_result_source
    args.model_source = model_source
    args.auto_save_name = auto_save_name
    args.auto_stop_name = auto_stop_name
    args.lr_reduce_name = lr_reduce_name
    args.dataset_name = dataset_name
    args.data_loader_name = data_loader_name
    args.data_sampler_name = data_sampler_name
    args.encode_name = encode_name
    #args.backbone_name = backbone_name
    args.backbone_name = backbone_name
    args.backend_name = backend_name
    args.decode_name = decode_name
    args.loss_name = loss_name
    args.indicator_name = indicator_name
    args.statistic_indicator_name = statistic_indicator_name
    args.optimization_name = optimization_name
    args.aug_names = aug_names
    args.data_type_adapter_name = data_type_adapter_name
    args.fit_data_to_input_name = fit_data_to_input_name
    args.fit_decode_to_result_name = fit_decode_to_result_name
    args.model_name = model_name
    from Putil.demo.deep_learning.base import util; train_stage = util.train_stage; evaluate_stage = util.evaluate_stage; test_stage = util.test_stage
    import Putil.base.logger as plog
    reload(plog)
    import Putil.demo.deep_learning.base.horovod as Horovod
    # the method for remote debug
    hvd = Horovod.horovod(args)
    if args.remote_debug:
        import ptvsd
        host = '127.0.0.1'
        port = 12345
        ptvsd.enable_attach(address=(host, port), redirect_output=True)
        if __name__ == '__main__':
            print('waiting for remote attach')
            ptvsd.wait_for_attach()

    if hvd.rank() == 0 and train_stage(args):
        bsf = psfb.BaseSaveFold(
            use_date=True if not args.debug else False, \
                use_git=True if not args.debug else False, \
                    should_be_new=True if not args.debug else False, \
                        base_name='{}{}{}'.format(args.backbone_name, args.name, '-debug' if args.debug else ''))
        bsf.mkdir(args.save_dir)
        args.save_dir = bsf.FullPath
    log_level = plog.LogReflect(args.Level).Level
    plog.PutilLogConfig.config_format(plog.FormatRecommend)
    plog.PutilLogConfig.config_log_level(stream=log_level, file=log_level)
    plog.PutilLogConfig.config_file_handler(filename=os.path.join(args.save_dir, \
        'train.log' if train_stage(args) else 'evaluate.log' if evaluate_stage(args) else 'test.log'), mode='a')
    plog.PutilLogConfig.config_handler(plog.stream_method | plog.file_method)
    root_logger = plog.PutilLogConfig('train').logger()
    root_logger.setLevel(log_level)
    MainLogger = root_logger.getChild('Trainer')
    MainLogger.setLevel(log_level)
    pab.args_log(args, MainLogger)
    #  TODO: the SummaryWriter
    #from import as SummaryWriter
    reload(AutoSaveFactory); AutoSave = AutoSaveFactory.auto_save_factory
    reload(AutoStopFactory); AutoStop = AutoStopFactory.auto_stop_factory
    reload(LrReduceFactory); LrReduce = LrReduceFactory.lr_reduce_factory
    reload(DatasetFactory); Dataset = DatasetFactory.dataset_factory
    reload(DataLoaderFactory); DataLoader = DataLoaderFactory.data_loader_factory
    reload(DataSamplerFactory); DataSampler = DataSamplerFactory.data_sampler_factory
    reload(ModelFactory); Model = ModelFactory.backbone_factory
    reload(LossFactory); Loss = LossFactory.loss_factory
    reload(IndicatorFactory); Indicator = IndicatorFactory.indicator_factory
    reload(StatisticIndicatorFactory); StatisticIndicator = StatisticIndicatorFactory.statistic_indicatory_factory
    reload(OptimizationFactory); Optimization = OptimizationFactory.optimization_factory
    reload(EncodeFactory); Encode = EncodeFactory.encode_factory
    reload(DecodeFactory); Decode = DecodeFactory.decode_factory
    reload(FitDataToInputFactory); FitDataDataToInput = FitDataToInputFactory.fit_data_to_input_factory
    reload(FitDecodeToResultFactory); FitDecodeToResult = FitDecodeToResultFactory.fit_decode_to_result_factory
    reload(ModelFactory); Model = ModelFactory.model_factory
    reload(util); train_stage = util.train_stage; evaluate_stage = util.evaluate_stage; test_stage = util.test_stage
    # TODO: third party import
    # TODO: set the deterministic 随机确定性可复现
    # make the result dir
    from Putil.demo.deep_learning.base.args_operation import args_save as ArgsSave
    from Putil.demo.deep_learning.base.util import Stage as Stage
    from .util.train_evaluate_common import train_evaluate_common
    from Putil.demo.deep_learning.base.base_operation_factory import load_saved_factory, \
        load_checkpointed_factory, checkpoint_factory, save_factory, deploy_factory
    from Putil.demo.deep_learning.base.util import TemplateModelDecodeCombine
    # TODO: set the args item
    ArgsSave(args, os.path.join(args.save_dir, 'args')) # after ArgsSave is called, the args should not be change
    checkpoint = checkpoint_factory(args)() if train_stage(args) else None
    save = save_factory(args)() if train_stage(args) else None
    deploy = deploy_factory(args)() if train_stage else None
    load_saved = load_saved_factory(args)()
    load_checkpointed = load_checkpointed_factory(args)()

    if hvd.rank() == 0:
        writer = SummaryWriter(args.save_dir)
    # prepare the GPU
    if args.cuda:
        # Horovod: pin GPU to local rank. TODO: rank is the global process index, local_rank is the local process index in a machine
        # such as -np 6 -H *.1:1 *.2:2 *.3:3 would get the rank: 0, 1, 2, 3, 4, 5 and the local rank: {[0], [0, 1], [0, 1, 2]}
        device_map = {
            0: {
                0: 0,
            },
            1: {
                1: 1
            },
            2: {
                2: 3
            }   
        }
        if args.gpus is not None:
            device_map[hvd.rank()] = dict() if device_map.get(hvd.rank()) is None else device_map[hvd.rank()]
            device_map[hvd.rank()][hvd.local_rank()] = args.gpus[hvd.rank()]
        MainLogger.info("rank: {}; local_rank: {}; gpu: {}".format( \
            hvd.rank(), hvd.local_rank(), device_map[hvd.local_rank()][hvd.rank()]))
        torch.cuda.set_device(device_map[hvd.local_rank()][hvd.rank()])
        torch.cuda.manual_seed(args.seed)
    # Horovod: limnit # of CPU threads to be used per worker
    torch.set_num_threads(1)
    kwargs = {'num_workers': args.n_worker_per_dataset, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'
    if train_stage(args):
        # : build the backbone
        backbone = BackboneFactory.backbone_factory(args)()
        # : build backend
        backend = BackendFactory.backend_factory(args)()
        # : build decode
        decode = DecodeFactory.decode_factory(args)()
        # : build the loss
        loss = LossFactory.loss_factory(args)()
        # : build the indicator
        train_indicator = IndicatorFactory.indicator_factory(args)()
        evaluate_indicator = IndicatorFactory.indicator_factory(args)()
        # : build the statistic indicator
        statistic_indicator = StatisticIndicator(args)()
        # TODO: build the optimization
        optimization = OptimizationFactory.optimization_factory(args)(backbone.parameters())
        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(backbone.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        # Horovod: wrap optimizer with DistributedOptimizer.
        optimizer = hvd.DistributedOptimizer(optimizer, \
            named_parameters=model.named_parameters(), \
                compression=compression, \
                    op=hvd.Adasum if args.use_adasum else hvd.Average)
        #  : the auto save
        auto_save = AutoSave(args)()
        #  : the auto stop
        auto_stop = AutoStop(args)()
        #  : the lr reduce
        lr_reduce = LrReduce(args)()
    encode = EncodeFactory.encode_factory(args)()
    # : build the train dataset
    fit_data_to_input = FitDataToInputFactory.fit_data_to_input_factory(args)
    fit_decode_to_result = FitDecodeToResultFactory.fit_decode_to_result_factory(args)
    template_model = Model(args)
    dataset_train = None; train_sampler = None; evaluate_loader = None
    if args.train_off is not True:
        MainLogger.info('start to generate the train dataset data_sampler data_loader')
        dataset_train = Dataset(args, Stage.Train)
        root_node = pAug.AugNode(pAug.AugFuncNoOp())
        # for the fack data field, maybe cause the nan or inf
        for i in range(0, args.fake_aug):
            root_node.add_child(pAug.AugNode(pAug.AugFuncNoOp()))
        if args.naug is False:
            Original = root_node.add_child(pAug.AugNode(pAug.AugFuncNoOp()))
            [root_node.add_child(pAug.AugNode(aug)) for aug in AugFactory.aug_factory(args)()]
        root_node.freeze_node()
        dataset_train.set_aug_node_root(root_node)
        dataset_train.set_convert_to_input_method(encode)
        dataset_train.set_data_type_adapter(DataTypeAdapterFactory.data_type_adapter_factory(args)())
        train_sampler = DataSampler(args)(dataset_train, rank_amount=hvd.size(), rank=hvd.rank())  if dataset_train is not None else None
        train_loader = DataLoader(args)(dataset_train, batch_size=args.batch_size, sampler=train_sampler) if dataset_train is not None else None
        MainLogger.info('generate adataset train successful: {} sample'.format(len(dataset_train)))
    # : build the evaluate dataset
    dataset_evaluate = None; evaluate_sampler = None; evaluate_loader = None
    if args.evaluate_off is not True:
        MainLogger.info('start to generate the evaluate dataset data_sampler data_loader')
        dataset_evaluate = Dataset(args, Stage.Evaluate)
        evaluate_sampler = DataSampler(args)(dataset=dataset_evaluate, rank_amount=hvd.size(), rank=hvd.rank()) if dataset_evaluate is not None else None
        evaluate_loader = DataLoader(args)(dataset=dataset_evaluate, data_sampler=evaluate_sampler) if dataset_evaluate is not None else None
    # : build the test dataset
    dataset_test = None; test_sampler = None; test_loader = None
    if args.test_off is not True:
        MainLogger.info('start to generate the evaluate dataset data_sampler data_loader')
        dataset_test = Dataset(args, Stage.Test) if args.test_off is not True else None
        test_sampler = DataSampler(args)(dataset_test, rank_amount=hvd.size(), rank=hvd.rank()) if dataset_test is not None else None
        test_loader = DataLoader(args)(dataset_test, data_sampler=test_sampler) if dataset_test is not None else None
    if args.cuda:
        backbone.cuda()
        backend.cuda()
        decode.cuda()
        loss.cuda()
        indicator.cuda()
        statistic_indicator.cuda()
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()
    if hvd.rank() == 0:
        bsf = psfb.BaseSaveFold(
            use_date=True, use_git=True, should_be_new=True, base_name='{}{}'.format(args.backbone_name, args.name))
        bsf.mkdir('./result')
        with open(os.path.join(bsf.FullPath, 'param.json'), 'w') as fp:
            fp.write(json.dumps(args.__dict__, indent=4))
        writer = SummaryWriter(bsf.FullPath)
    run_test_stage() if test_stage(args) else None # 如果train_off为True 同时test_off为False，则为test_stage，evaluate_stage与test_stage可以同时存在
    run_evaluate_stage if evaluate_stage(args) else None # 如果train_off为True 同时evaluate_off为False，则为evaluate_stage，evaluate_stage与test_stage可以同时存在
    if train_stage(args):
        # 如果train_off不为True，就是train_stage，只是train_stage中可以设定进不进行evaluate与test
        if args.weight_path != '' and args.weight_epoch is not None:
            MainLogger.info('load trained backbone: path: {} epoch: {}'.format(args.weight_path, args.weight_epoch))
            backbone = load_checkpointed(args.weight_epoch, args.weight_path, backbone=backbone, backend=backend, \
                optimization=optimization, auto_stop=auto_stop, auto_save=auto_save, lr_reduce=lr_reduce)
        for epoch in range(0, args.epochs):
            train_ret = train(epoch)
            if train_ret[0] is True:
                break
            global_step = (epoch + 1) * len(train_loader)
            # : run the val
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
                        run_evaluate_stage() if not args.evaluate_off else None
                        MainLogger.debug('run the test stage')
                        run_test_stage() if not args.test_off else None
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