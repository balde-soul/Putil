# coding=utf-8
import json
import os
from enum import Enum
import torch
torch.utils.data.distributed.DistrivutedSampler


class nothing():
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def train_evaluate_common(stage, epoch, data_loader):
    prefix = 'train' if stage == Stage.Train else 'evaluate'
    with torch.no_grad() if stage == Stage.Evaluate else nothing() as t:
        model.train() if stage == Stage.Train else model.eval()
        data_loader.data_sampler.set_epoch(epoch) if stage == Stage.Train else None

        # TODO: indicator start with zero
        loss = [0.];class_loss = [0.];wh_loss = [0.];offset_loss = [0.];iou_loss = [0.];obj_acc = [0.];otp_acc = [0.];global_acc = [0.];iou = [0.]

        TrainLogger.debug('start to {} epoch'.format(prefix))
        for batch_idx, datas in enumerate(data_loader):
            step = epoch * len(data_loader) + batch_idx + 1
            TrainLogger.debug('batch {}'.format(prefix))
            # TODO: data to cuda
            #img = torch.from_numpy(img).cuda();gt_box = torch.from_numpy(gt_box).cuda();gt_class = torch.from_numpy(gt_class).cuda();gt_obj = torch.from_numpy(gt_obj).cuda();radiance_factor = torch.from_numpy(radiance_factor).cuda()
            # time
            batch_start = time.time()
            # do the training
            optimizer.zero_grad()
            # : run the model get the output TODO:
            output = model($model_input)
            # : run the loss function get the ret
            losses = loss_func(*(output + datas))
            # TODO: get the loss item
            #_loss = ret[0]
            #_class_loss = ret[1];_wh_loss = ret[2];_offset_loss = ret[3];_iou_loss = ret[4]
            # TODO: do some simple check
            #if _iou_loss.item() < -1.0 or _iou_loss.item() > 1.0:
            #    TrainLogger.warning('giou wrong')
            #if np.isnan(_loss.item()) or np.isinf(_loss.item()) is True:
            #    TrainLogger.warning('loss in train inf occured!')
            # TODO: run the indicator function to get the indicators
            indicator_func(output, input)
            # : run the backward
            _loss.backward() if stage == Stage.Train else None
            # : do the optimize
            optimizer.step() if stage == Stage.Train else None
            # time
            batch_time = time.time() - batch_start
            # TODO: loss item and the indicator accumulation
            #obj_acc[-1] += _obj_acc.item();otp_acc[-1] += _otp_acc.item();global_acc[-1] += _global_acc.item();iou[-1] += _iou.item();loss[-1] += _loss.item();class_loss[-1] += _class_loss.item();wh_loss[-1] += _wh_loss.item();offset_loss[-1] += _offset_loss.item();iou_loss[-1] += _iou_loss.item()
            # TODO: do the regular log
            if step % args.log_interval == 0:
            #    obj_acc[-1] = obj_acc[-1] / args.log_interval;otp_acc[-1] = otp_acc[-1] / args.log_interval;global_acc[-1] = global_acc[-1] / args.log_interval;iou[-1] = iou[-1] / args.log_interval;loss[-1] = loss[-1] / args.log_interval;class_loss[-1] = class_loss[-1] / args.log_interval;wh_loss[-1] = wh_loss[-1] / args.log_interval;offset_loss[-1] = offset_loss[-1] / args.log_interval;iou_loss[-1] = iou_loss[-1] / args.log_interval
            #    TrainLogger.info('{} Epoch{}: [{}/{}]| loss(all|class|wh|offset|iou): {:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f} | acc(boj|otp|global): {:.4f}|{:.4f}|{:.4f} | iou: {:.4f} | batch time: {:.3f}s '.format( \
            #        prefix, epoch, step, len(data_loader), 
            #        loss[-1], class_loss[-1], wh_loss[-1], offset_loss[-1], iou_loss[-1], obj_acc[-1], otp_acc[-1], global_acc[-1], iou[-1], 
            #        batch_time))
            #    # set the target indicator to zero
            # TODO: the regular log would calculate the mean of the loss item and the indicator, we should append a new item to the collection
            #    loss.append(0.);class_loss.append(0.);wh_loss.append(0.);offset_loss.append(0.);iou_loss.append(0.);obj_acc.append(0.);otp_acc.append(0.);global_acc.append(0.);iou.append(0.)
            # TODO: do the regular summary
            if step % args.interval_run_evaluate == 0:
            #    gt_obj_ft = gt_obj.sum(1).gt(0.)
            #    # reduce the target_indicator
            #    rloss = all_reduce(_loss, 'loss');rclass_loss = all_reduce(_class_loss, 'class_loss');rwh_loss = all_reduce(_wh_loss, 'wh_loss');roffset_loss = all_reduce(_offset_loss, 'offset_loss');riou_loss = all_reduce(_iou_loss, 'iou_loss');mobj_acc = all_reduce(_obj_acc, 'obj_acc');motp_acc = all_reduce(_otp_acc, 'otp_acc');mglobal_acc = all_reduce(_global_acc, 'global_acc')
            #    if hvd.rank() == 0:
            #        # :decode
            #        regular_output = decode(pre_class, pre_box, args.downsample_rate, args.class_threshold)
            #        gt_output = decode(radiance_factor, gt_box, args.downsample_rate, args.class_threshold)
            #        # extract the target data
            #        boxes = regular_output[0];classes = regular_output[1];indexes = regular_output[3];gt_boxes = gt_output[0];gt_classes = gt_output[1];gt_indexes = gt_output[3];img_numpy = img.detach().cpu().numpy()
            #        pv = PointVisual();rv = RectangleVisual(2)
            #        result_visual(pv, rv, img_numpy, boxes, classes, indexes, '{}_pre'.format(prefix), 'pre', step)
            #        result_visual(pv, rv, img_numpy, gt_boxes, gt_classes, gt_indexes, '{}_gt'.format(prefix), 'gt', step)
            #        # add target indicator to sumary
            #        writer.add_scalar('{}/loss/loss'.format(prefix), rloss, global_step=step);writer.add_scalar('{}/loss/class_loss'.format(prefix), rclass_loss, global_step=step);writer.add_scalar('{}/loss/wh_loss'.format(prefix), rwh_loss, global_step=step);writer.add_scalar('{}/loss/offset_loss'.format(prefix), roffset_loss, global_step=step);writer.add_scalar('{}/loss/iou_loss'.format(prefix), riou_loss, global_step=step);writer.add_scalar('{}/acc/obj_acc'.format(prefix), mobj_acc, global_step=step);writer.add_scalar('{}/acc/otp_acc'.format(prefix), motp_acc, global_step=step);writer.add_scalar('{}/acc/global_acc'.format(prefix), mglobal_acc, global_step=step)
            #        pass
            #    pass
            TrainLogger.debug('batch {} end'.format(prefix))
            pass
        # TODO: do the summary of this epoch
        #eloss = np.mean(loss[0:-1]);eclass_loss = np.mean(class_loss[0:-1]);ewh_loss = np.mean(wh_loss[0:-1]);eoffset_loss = np.mean(offset_loss[0:-1]);eiou_loss = np.mean(iou_loss[0:-1]);eobj_acc = np.mean(obj_acc[0:-1]);eotp_acc = np.mean(otp_acc[0:-1]);eglobal_acc = np.mean(global_acc[0:-1]);eiou = np.mean(iou[0:-1])
        #TrainLogger.info('{} epoch {} done: loss(all|class|wh|offset|iou): {:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f} | acc(boj|otp|global): {:.4f}|{:.4f}|{:.4f} | iou: {:.4f}'.format( \
        #    prefix, epoch,
        #    eloss, eclass_loss, ewh_loss, eoffset_loss, eiou_loss, eobj_acc, eotp_acc, eglobal_acc, eiou))
        ##writer.add_scalar('{}/loss/rloss'.format(prefix), rloss, global_step=step);writer.add_scalar('{}/loss/class_loss'.format(prefix), rclass_loss, global_step=step);writer.add_scalar('{}/loss/wh_loss'.format(prefix), rwh_loss, global_step=step);writer.add_scalar('{}/loss/offset_loss'.format(prefix), roffset_loss, global_step=step);writer.add_scalar('{}/loss/iou_loss'.format(prefix), riou_loss, global_step=step);writer.add_scalar('{}/acc/obj_acc'.format(prefix), mobj_acc, global_step=step);writer.add_scalar('{}/acc/otp_acc'.format(prefix), motp_acc, global_step=step);writer.add_scalar('{}/acc/global_acc'.format(prefix), mglobal_acc, global_step=step)
        pass
    # TODO: ret the needed item which would be used in train evaluate and test
    #return {'eloss': eloss, 'eclass_loss': eclass_loss, 'ewh_loss': ewh_loss, 'eoffset_loss': eoffset_loss, \
    #    'eiou_loss': eiou_loss, 'eobj_acc': eobj_acc, 'eotp_acc': eotp_acc, 'eglobal_acc': eglobal_acc, 'eiou': eiou}



def train(epoch):
    train_evaluate_common(Stage.Train, epoch, train_loader):


def evaluate(epoch):
    ret = train_evaluate_common(Stage.Evaluate, epoch, evaluate_loader)
    # TODO: do all reduce of indicator
    indicator = ret['eloss']
    #indicator = hvd.allreduce(indicator)
    if auto_save.save_or_not(indicator) or args.only_test or args.debug:
        # :save the model in rank0
        if hvd.rank() == 0:
            TrainLogger.info('run checkpoint')
            checkpoint()
            # :deploy use on sample as the example input
            TrainLogger.info('run deploy')
            deploy(torch.from_numpy(np.zeros(shape=(1, 3, args.input_height, args.input_width))).cuda())
        # :run the test, if saved
        TrainLogger.info('run test')
        test() if args.test_off is False else TrainLogger.info('test_off, do not run test')
    # :stop or not
    stop = auto_stop.stop_or_not(indicator) if args.debug == False else (False if epoch == 0 else True)
    # :lr_reduce
    lr = lr_reduce.reduce_or_not(indicator)
    # TODO:
    return stop, lr


if __name__ == '__main__':
    import Putil.base.arg_base as pab
    import Putil.base.save_fold_base as psfb
    # TODO: import the controler
    #  TODO: the auto save
    #from Putil.trainer.auto_save_args import generate_args as auto_save_args
    #  TODO: the auto stop
    #from Putil.trainer.auto_stop_args import generate_args as auto_stop_args
    #  TODO: the lr_reduce
    #from Putil.trainer.lr_reduce_args import generate_args as lr_reduce_arg
        # the default arg
    ppa = pab.ProjectArg(save_dir='./result', log_level='Info', debug_mode=True, config='')
    ## :auto stop setting
    auto_stop_args(ppa.parser)
    ## :auto save setting
    auto_save_args(ppa.parser)
    ## :lr reduce setting
    lr_reduce_args(ppa.parser)
    # : the base information set
    parser.add_argument('--framework', action='store', type=str, default='torch', \
        help='specify the framework used')
    # debug
    parser.add_argument('--remote_debug', action='store_true', default=False, \
        help='setup with remote debug(blocked while not attached) or not')
    parser.add_argument('--frame_debug', action='store_true', default=False, \
        help='run all the process in two epoch with tiny data')
    # data setting
    ppa.parser.add_argument('--off_train', action='store_true', default=False, \
        help='do not run train if set')
    ppa.parser.add_argument('--off_evaluate', action='store_true', default=False, \
        help='do not run evaluate if set')
    ppa.parser.add_argument('--off_test', action='store_true', default=False, \
        help='do not run test if set')
    ppa.parser.add_argument('--only_test', action='store_true', default=False, \
        help='only run test or not')
    parser.add_argument('--n_worker_per_dataset', action='store', type=int, default=1, \
        help='the number of worker for every dataset')
    parser.add_argument('--data_using_rate_train', action='store', type=float, default=1.0, \
        help='rate of data used in train')
    ppa.parser.add_argument('--sub_data', type=int, nargs='+', default=None, \
        help='list with int, specified the sub dataset which would be used in train evaluate, '
        'default None(whole dataset)')
    parser.add_argument('--data_using_rate_evaluate', action='store', type=float, default=1.0, \
        help='rate of data used in evaluate')
    parser.add_argument('--data_using_rate_test', action='store', type=float, default=1.0, \
        help='rate of data used in test')
    ppa.parser.add_argument('--naug', action='store_true', \
        help='do not use data aug while set')
    ppa.parser.add_argument('--fake_aug', action='store', type=int, default=0, \
        help='do the sub aug with NoOp for fake_aug time, check the generate_dataset')
    ppa.parser.add_argument('--data_name', action='store', type=str, default='DefaultData' \
        help='the name of the data, used in the data_factory, see the util.data_factory')
    ppa.parser.add_argument('--encode_name', action='store', type=str, default='DefalutEncode', \
        help='the name of the encode in the encode_factory, see the util.encode_factory')
    ppa.parser.add_argument('--encode_source', type=str, default='standard', action='store', \
        help='standard: from the Putil.demo.base.deep_learning.encode; project: from this project')
    ppa.parser.add_argument('--decode_name', action='store', type=str, default='DefaultDecode', \
        help='the name of the decode in the decode_factory, see the util.decode_factory')
    ppa.parser.add_argument('--decode_source', type=str, default='standard', action='store', \
        help='standard: from the Putil.demo.base.deep_learning.decode; project: from this project')
    ppa.parser.add_argument('--input_height', type=int, action='store', default=256, \
        help='the height of the input')
    ppa.parser.add_argument('--input_width', action='store', type=int, default=256, \
        help='the width of the input')
    ppa.parser.add_argument('--shuffle_train', action='store_true', default=False, \
        help='shuffle the train data every epoch')
    ppa.parser.add_argument('--shuffle_evaluate', action='store_true', default=False, \
        help='shuffle the evaluate data every epoch')
    ppa.parser.add_argument('--shuffle_test', action='store_true', default=False, \
        help='shuffle the test data every epoch')
    ppa.parser.add_argument('--drop_last_train', action='store_true', default=False, \
        help='drop the last uncompleted train data while set')
    ppa.parser.add_argument('--drop_last_evaluate', action='store_true', default=False, \
        help='drop the last uncompleted evaluate data while set')
    ppa.parser.add_argument('--drop_last_test', action='store_true', default=False, \
        help='drop the last uncompleted test data while set')
    # train setting
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
    ppa.parser.add_argument('--interval_run_evaluate', type=int, default=1, metavar='N', \
        help='how many epoch to wait before evaluate the model(default: 1), '\
            'test the mode while the model is savd, would not run evaluate while -1')
    ppa.parser.add_argument('--compute_efficiency', action='store_true', default=False, \
        help='evaluate the efficiency in the test or not')
    ppa.parser.add_argument('--data_rate_in_compute_efficiency', type=int, default=200, metavar='N', \
        help='how many sample used in test to evaluate the efficiency(default: 200)')
    ppa.parser.add_argument('--auto_save_name', type=str, action='store', default='', \
        help='the name of the auto saver')
    ppa.parser.add_argument('--auto_save_source', type=str, default='standard', action='store', \
        help='standard: from the Putil.demo.base.deep_learning.auto_save; project: from this project')
    ppa.parser.add_argument('--auto_stop_name', type=str, action='store', default='', \
        help='the name of the auto stoper')
    ppa.parser.add_argument('--auto_stop_source', type=str, default='standard', action='store', \
        help='standard: from the Putil.demo.base.deep_learning.auto_stop; project: from this project')
    ppa.parser.add_argument('--lr_reduce_name', type=str, action='store', default='', \
        help='the name of the lr reducer')
    ppa.parser.add_argument('--lr_reduce_source', type=str, default='standard', action='store', \
        help='standard: from the Putil.demo.base.deep_learning.lr_reduce; project: from this project')
    ppa.parser.add_argument('--optimization_name', type=str, action='store', default=None, \
        help='the name of the optimization')
    ppa.parser.add_argument('--weight_decay', type=float, action='store', default=1e-4, \
        help='the weight decay, default is 1e-4')
    # model setting
    ppa.parser.add_argument('--weight', type=str, default='', action='store', \
        help='specify the pre-trained model path(default\'\')')
    ppa.parser.add_argument('--model_source', type=str, default='standard', action='store', \
        help='standard: from the Putil.demo.base.deep_learning.models; project: from this project')
    ppa.parser.add_argument('--model_name', type=str, default='', action='store', \
        help='specify the model name')
    ppa.parser.add_argument('--backbone_source', type=str, default='standard', action='store', \
        help='standard: from the Putil.demo.base.deep_learning.backbones; project: from this project')
    ppa.parser.add_argument('--backbone_name', type=str, default='', action='store', \
        help='specify the backbone name')
    ppa.parser.add_argument('--backbone_arch', type=str, default='', action='store', \
        help='specify the arch of the backbone, such 19 for backbone_name with vgg')
    ppa.parser.add_argument('--backbone_downsample_rate', type=int, default=None, action='store', \
        help='specify the downsample rate for the backbone')
    ppa.parser.add_argument('--backbone_pretrained', default=False, action='store_true', \
        help='load the pretrained backbone weight or not')
    ppa.parser.add_argument('--backbone_weight_path', type=str, default='', action='store', \
        help='specify the pre-trained model for the backbone, use while in finetune mode, '\
            'if the weight is specify, the backbone weight would be useless')
    ppa.parser.add_argument('--loss_name', type=str, default='DefaultLoss', action='store', \
        help='the name of the loss in the loss_factory, see the util.loss_factory')
    ppa.parser.add_argument('--loss_source', type=str, default='standard', action='store', \
        help='standard: from the Putil.demo.base.deep_learning.loss; project: from this project')
    ppa.parser.add_argument('--indicator_name', type=str, default='DefaultIndicator', action='store', \
        help='the name of the indicator in the indicator_factory, see the util.indicator_factory')
    ppa.parser.add_argument('--indicator_source', type=str, default='standard', action='store', \
        help='standard: from the Putil.demo.base.deep_learning.indicator; project: from this project')
    ppa.parser.add_argument('--statistic_indicator_name', type=str, default='', action='store', \
        help='the name of the statistic_indicator in the statistic_indicator_factory, see the util.statistic_indicator_factory')
    ppa.parser.add_argument('--statistic_indicator_source', type=str, default='standard', action='store', \
        help='standard: from the Putil.demo.base.deep_learning.statistic_indicator; project: from this project')
    ppa.parser.add_argument('--name', type=str, action='store', default='', \
        help='the ${backbone_name}${name} would be the name of the fold to save the result')
    args = ppa.parser.parse_args()
    # the method for remote debug
    if args.remote_debug:
        import ptvsd
        host = '127.0.0.1'
        port = 12345
        ptvsd.enable_attach(address=(host, port), redirect_output=True)
        if __name__ == '__main__':
            print('waiting for remote attach')
            ptvsd.wait_for_attach()

    import Putil.base.logger as plog
    if hvd.rank() == 0:
        bsf = psfb.BaseSaveFold(
            use_date=True, use_git=True, should_be_new=True, base_name='{}{}'.format(args.backbone_name, args.name))
        bsf.mkdir(args.save_dir)
        args.save_dir = bsf.FullPath
    log_level = plog.LogReflect(args.Level).Level
    plog.PutilLogConfig.config_format(plog.FormatRecommend)
    plog.PutilLogConfig.config_log_level(stream=log_level, file=log_level)
    plog.PutilLogConfig.config_file_handler(filename=os.path.join(args.save_dir, 'log'), mode='a')
    plog.PutilLogConfig.config_handler(plog.stream_method | plog.file_method)
    root_logger = plog.PutilLogConfig('train').logger()
    root_logger.setLevel(log_level)
    TrainLogger = root_logger.getChild('Trainer')
    TrainLogger.setLevel(log_level)
    pab.args_log(args, TrainLogger)
    # TODO: third party import
    #  TODO: the SummaryWriter
    #from import as SummaryWriter
    from Putil.demo.deep_learning.base.auto_save_factory import auto_save_factory as AutoSave
    from Putil.demo.deep_learning.base.auto_stop_factory import auto_stop_factory as AutoStop
    from Putil.demo.deep_learning.base.lr_reduce_factory import lr_reduce_factory as LrReduce
    # TODO: Putil import
    #  TODO: the Dataset
    #from import as Dataset
    from Putil.demo.deep_learning.base.data_loader_factory import data_loader_factory as DataLoader
    from Putil.demo.deep_learning.base.data_sampler_factory import data_sampler_factory as DataSampler
    from Putil.demo.deep_learning.base.util import Stage as Stage
    from Putil.demo.deep_learning.base.model_factory import model_factory as Model
    from Putil.demo.deep_learning.base.loss_factory import loss_factory as Loss
    from Putil.demo.deep_learning.base.indicator_factory import indicator_factory as Indicator
    from Putil.demo.deep_learning.base.statistic_indicator_factory import statistic_indicator_factory as StatisticIndicator
    from Putil.demo.deep_learning.base.optimization_factory import optimization_factory as Optimization
    # TODO: set the deterministic 随机确定性可复现
    # make the result dir
    from Putil.demo.deep_learning.base.args_operation import args_save as ArgsSave
    # TODO: set the args item
    ArgsSave(args, os.path.join(args.save_dir, 'args')) # after ArgsSave is called, the args should not be change

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
        TrainLogger.info("rank: {}; local_rank: {}; gpu: {}".format( \
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
    # : build the net
    model = Model(args)
    # : build the loss
    loss = Loss(args)
    # : build the indicator
    train_indicator = Indicator(args)
    evaluate_indicator = Indicator(args)
    # : build the statistic indicator
    statistic_indicator = StatisticIndicator(args)
    # TODO: build the optimization
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    #  : the auto save
    auto_save = AutoSave(args)
    #  : the auto stop
    auto_stop = AutoStop(args)
    #  : the lr reduce
    lr_reduce = LrReduce(args)
    # : build the train dataset
    dataset_train = Dataset(args, Stage.Train) if args.train_off is not True else None
    train_sampler = DataSampler(args)(dataset_train, rank_amount=hvd.size(), rank=hvd.rank())  if dataset_train is not None else None
    train_loader = DataLoader(args)(dataset_train, batch_size=args.batch_size, sampler=train_sampler) if dataset_train is not None else None
    # : build the evaluate dataset
    dataset_evaluate = Dataset(args, Stage.Evaluate) if args.evaluate_off is not True else None
    evaluate_sampler = DataSampler(args)(dataset=dataset_evaluate, rank_amount=hvd.size(), rank=hvd.rank()) if dataset_evaluate is not None else None
    evaluate_loader = DataLoader(args)(dataset=dataset_evaluate, data_sampler=evaluate_sampler) if dataset_evaluate is not None else None
    # : build the test dataset
    dataset_test = Dataset(args, Stage.Test) if args.test_off is not True else None
    test_sampler = DataSampler(args)(dataset_test, rank_amount=hvd.size(), rank=hvd.rank()) if dataset_test is not None else None
    test_loader = DataLoader(args)(dataset_test, data_sampler=test_sampler) if dataset_test is not None else None
    if args.cuda:
        # TODO: to_cuda
        model.cuda()
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
    
    if args.only_test:
        assert args.weight != '', 'should specify the pre-trained weight while run only_test, please specify the args.weight'
        state_dict = torch.load(args.weight)
        model.load_state_dict(state_dict)
        test()
    else:
        if args.weight != '':
            TrainLogger.info('load pre-trained model: {}'.format(args.weight))
            state_dict = torch.load(args.weight)
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(state_dict)
            auto_stop.load_state_dict(state_dict)
            auto_save.load_state_dict(state_dict)
            lr_reduce.load_state_dict(state_dict)
        for epoch in range(0, args.epochs):
            train_ret = train(epoch)
            global_step = (epoch + 1) * len(train_loader)
            # : run the val
            if ((epoch + 1) % args.interval_run_evaluate == 0) or (args.debug):
                if args.evaluate_off is False:
                    evaluate_ret = evaluate(epoch) 
                    if evaluate_ret[0] is True:
                        break
                    pass
                else:
                    TrainLogger.info('evaluate_off, do not run the evaluate')
                    pass
                pass
            pass
        pass
    pass