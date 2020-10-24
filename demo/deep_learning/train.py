# coding=utf-8
import json
import os
from enum import Enum


class nothing():
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Stage(Enum):
    Train=0
    Evaluate=1
    Test=2


def train_evaluate_common(stage, epoch, data_loader, data_sampler):
    prefix = 'train' if stage == Stage.Train else 'evaluate'
    with torch.no_grad() if stage == Stage.Evaluate else nothing() as t:
        model.train() if stage == Stage.Train else model.eval()
        data_sampler.set_epoch(epoch) if stage == Stage.Train else None

        TrainLogger.debug('start to {} epoch'.format(prefix))
        for batch_idx, (img, gt_box, gt_class, gt_obj, base_information, radiance_factor) in enumerate(data_loader):
            step = epoch * len(data_loader) + batch_idx + 1

            TrainLogger.debug('batch {}'.format(prefix))
            img = torch.from_numpy(img).cuda();gt_box = torch.from_numpy(gt_box).cuda();gt_class = torch.from_numpy(gt_class).cuda();gt_obj = torch.from_numpy(gt_obj).cuda();radiance_factor = torch.from_numpy(radiance_factor).cuda()
            # time
            batch_start = time.time()
            # do the training TODO:
            optimizer.zero_grad()
            pre_box, pre_class, pre_obj = model(torch.transpose(torch.transpose(img, 3, 2), 2, 1))
            ret = loss_func(pre_box, pre_class, pre_obj, gt_box, gt_class, gt_obj, radiance_factor)
            loss = ret[0];class_loss = ret[1];wh_loss = ret[2];offset_loss = ret[3];iou_loss = ret[4]
            if iou_loss.item() < -1.0 or iou_loss.item() > 1.0:
                TrainLogger.warning('giou wrong')

            obj_acc, class_acc_otp, class_acc_global, iou = indicator_func(pre_box, pre_class, pre_obj, gt_box, gt_class, gt_obj, radiance_factor)
            if np.isnan(loss.item()) or np.isinf(loss.item()) is True:
                TrainLogger.warning('loss in train inf occured!')
            loss.backward() if stage == Stage.Train else None
            optimizer.step() if stage == Stage.Train else None
            # time
            batch_time = time.time() - batch_start;obj_acc[-1] += obj_acc.item();otp_acc[-1] += class_acc_otp.item();global_acc[-1] += class_acc_global.item();iou[-1] += iou.item();loss[-1] += loss.item();class_loss[-1] += class_loss.item();wh_loss[-1] += wh_loss.item();offset_loss[-1] += offset_loss.item();iou_loss[-1] += iou_loss.item()
            if (step) % args.log_interval == 0:
                obj_acc[-1] = obj_acc[-1] / args.log_interval;otp_acc[-1] = class_acc_otp[-1] / args.log_interval;global_acc[-1] = class_acc_global[-1] / args.log_interval;iou[-1] = iou[-1] / args.log_interval;loss[-1] = loss[-1] / args.log_interval;class_loss[-1] = class_loss[-1] / args.log_interval;wh_loss[-1] = wh_loss[-1] / args.log_interval;offset_loss[-1] = offset_loss[-1] / args.log_interval;iou_loss[-1] = iou_loss[-1] / args.log_interval
                TrainLogger.info('{} Epoch{}: [{}/{}]| loss(all|class|wh|offset|iou): {:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f} | acc(boj|otp|global): {:.4f}|{:.4f}|{:.4f} | iou: {:.4f} | batch time: {:.3f}s '.format( \
                    prefix,
                    epoch, 
                    batch_idx, 
                    len(loader), 
                    loss[-1], class_loss[-1], wh_loss[-1], offset_loss[-1], iou_loss[-1], obj_acc[-1], otp_acc[-1], global_acc[-1], iou[-1], 
                    batch_time))
                # set the target indicator to zero
                loss.append(0.);class_loss.append(0.);wh_loss.append(0.);offset_loss.append(0.);iou_loss.append(0.);obj_acc.append(0.);otp_acc.append(0.);global_acc.append(0.);iou.append(0.)
            if (step) % args.summary_interval == 0:
                gt_obj_ft = gt_obj.sum(1).gt(0.)
                # reduce the target_indicator
                mloss = all_reduce(loss, 'loss');mclass_loss = all_reduce(class_loss, 'class_loss');mwh_loss = all_reduce(wh_loss, 'wh_loss');moffset_loss = all_reduce(offset_loss, 'offset_loss');miou_loss = all_reduce(iou_loss, 'iou_loss');mobj_acc = all_reduce(obj_acc, 'obj_acc');motp_acc = all_reduce(class_acc_otp, 'otp_acc');mglobal_acc = all_reduce(class_acc_global, 'global_acc')
                if hvd.rank() == 0:
                    # :decode
                    regular_output = decode(pre_class, pre_box, args.downsample_rate, args.class_threshold)
                    gt_output = decode(radiance_factor, gt_box, args.downsample_rate, args.class_threshold)
                    # extract the target data
                    boxes = regular_output[0];classes = regular_output[1];indexes = regular_output[3];gt_boxes = gt_output[0];gt_classes = gt_output[1];gt_indexes = gt_output[3];pv = PointVisual();rv = RectangleVisual(2);img_numpy = img.detach().cpu().numpy()
                    result_visual(img_numpy, boxes, classes, indexes, prefix, 'pre')
                    result_visual(img_numpy, gt_boxes, gt_classes, gt_indexes, prefix, 'gt')
                    # add target indicator to sumary
                    writer.add_scalar('{}/loss/loss'.format(prefix), mloss, global_step=step);writer.add_scalar('{}/loss/class_loss'.format(prefix), mclass_loss, global_step=step);writer.add_scalar('{}/loss/wh_loss'.format(prefix), mwh_loss, global_step=step);writer.add_scalar('{}/loss/offset_loss'.format(prefix), moffset_loss, global_step=step);writer.add_scalar('{}/loss/iou_loss'.format(prefix), miou_loss, global_step=step);writer.add_scalar('{}/acc/obj_acc'.format(prefix), mobj_acc, global_step=step);writer.add_scalar('{}/acc/otp_acc'.format(prefix), motp_acc, global_step=step);writer.add_scalar('{}/acc/global_acc'.format(prefix), mglobal_acc, global_step=step)
                    pass
                pass
            TrainLogger.debug('batch {} end'.format(prefix))
            pass
        TrainLogger.info('{} epoch {} done: loss(all|class|wh|offset|iou): {:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f} | acc(boj|otp|global): {:.4f}|{:.4f}|{:.4f} | iou: {:.4f}'.format( \
            prefix,
            epoch,
            np.mean(loss[-1]), np.mean(class_loss[-1]), np.mean(wh_loss[-1]), np.mean(offset_loss[-1]), np.mean(iou_loss[-1]), np.mean(obj_acc[-1]), np.mean(otp_acc[-1]), np.mean(global_acc[-1]), np.mean(iou[-1])))
        pass
    pass


if __name__ == '__main__':
    import Putil.base.arg_base as pab
    import Putil.base.save_fold_base as psfb
    # the auto save TODO:
    from Putil.trainer.auto_save_args import generate_args as auto_save_args
    # the auto stop TODO:
    from Putil.trainer.auto_stop_args import generate_args as auto_stop_args
    # the lr_reduce TODO:
    from Putil.trainer.lr_reduce_args import generate_args as lr_reduce_arg
        # the default arg
    ppa = pab.ProjectArg(save_dir='./result', log_level='Info', debug_mode=True, config='')
    ## :auto stop setting
    auto_stop_args(ppa.parser)
    ## :auto save setting
    auto_save_args(ppa.parser)
    ## :lr reduce setting
    lr_reduce_args(ppa.parser)
    # debug
    parser.add_argument('--remote_debug', action='store_true', default=False, \
        help='setup with remote debug(blocked while not attached) or not')
    parser.add_argument('--frame_debug', action='store_true', default=False, \
        help='run all the process in two epoch with tiny data')
    # mode
    ppa.parser.add_argument('--train_off', action='store_true', default=False, \
        help='do not run train or not')
    ppa.parser.add_argument('--only_test', action='store_true', default=False, \
        help='only run test or not')
    # data setting
    parser.add_argument('--train_data_using_rate', action='store', type=float, default=1.0, \
        help='rate of data used in train')
    parser.add_argument('--evaluate_data_using_rate', action='store', type=float, default=1.0, \
        help='rate of data used in evaluate')
    parser.add_argument('--test_data_using_rate', action='store', type=float, default=1.0, \
        help='rate of data used in test')
    ppa.parser.add_argument('--naug', action='store_true', \
        help='do not use data aug while set')
    ppa.parser.add_argument('--fake_aug', action='store', type=int, default=0, \
        help='do the sub aug with NoOp for fake_aug time, check the generate_dataset')
    ppa.parser.add_argument('--data_name', action='store', type=str, default='' \
        help='the name of the data, used in the data_factory, see the util.data_factory')
    ppa.parser.add_argument('--encode_name', action='store', type=str, default='', \
        help='the name of the encode in the encode_factory, see the util.encode_factory')
    ppa.parser.add_argument('--decode_name', action='store', type=str, default='', \
        help='the name of the decode in the decode_factory, see the util.decode_factory')
    # train setting
    ppa.parser.add_argument('--epochs', type=int, default=10, metavar='N', \
        help='number of epochs to train (default: 10)')
    ppa.parser.add_argument('--batch_size', type=int, default=64, metavar='N', \
        help='input batch size for training (default: 64)')
    ppa.parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', \
        help='input batch size for testing (default: 1000)')
    ppa.parser.add_argument('--log_interval', type=int, default=10, metavar='N', \
        help='how many batches to wait before logging training status(default: 10)')
    ppa.parser.add_argument('--summary_interval', type=int, default=100, metavar='N', \
        help='how many batchees to wait before save the summary(default: 100)')
    ppa.parser.add_argument('--evaluate_interval', type=int, default=1, metavar='N', \
        help='how many epoch to wait before evaluate the model(default: 1), '\
            'test the mode while the model is savd, would not run evaluate while -1')
    ppa.parser.add_argument('--compute_efficiency', action='store_true', default=False, \
        help='evaluate the efficiency in the test or not')
    ppa.parser.add_argument('--data_rate_in_compute_efficiency', type=int, default=200, metavar='N', \
        help='how many sample used in test to evaluate the efficiency(default: 200)')
    # model setting
    ppa.parser.add_argument('--weight', type=str, default='', action='store', \
        help='specify the pre-trained model path(default\'\')')
    ppa.parser.add_argument('--backbone_weight_path', type=str, default='', action='store', \
        help='specify the pre-trained model for the backbone, use while in finetune mode, '\
            'if the weight is specify, the backbone weight would be useless')
    ppa.parser.add_argument('--backbone_name', type=str, default='', action='store', \
        help='specify the backbone name')
    ppa.parser.add_argument('--loss_name', type=str, default='', action='store', \
        help='the name of the loss in the loss_factory, see the util.loss_factory')
    ppa.parser.add_argument('--indicator_name', type=str, default='', action='store', \
        help='the name of the indicator in the indicator_factory, see the util.indicator_factory')
    ppa.parser.add_argument('--statistic_indicator_name', type=str, default='', action='store', \
        help='the name of the statistic_indicator in the statistic_indicator_factory, see the util.statistic_indicator_factory')
    ppa.parser.add_argument('--name', type=str, action='store', default='', \
        help='the ${backbone_name}${name} would be the name of the fold to save the result')
    # image param
    ppa.parser.add_argument('--input_height', type=int, action='store', default=${the default height}, \
        help='the height of the input')
    ppa.parser.add_argument('--input_width', action='store', type=int, default=${the default width}, \
        help='the width of the input')

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
    log_level = plog.LogReflect(args.Level).Level
    plog.PutilLogConfig.config_format(plog.FormatRecommend)
    plog.PutilLogConfig.config_log_level(stream=log_level, file=log_level)
    plog.PutilLogConfig.config_file_handler(filename=os.path.join('./', 'log'), mode='a')
    plog.PutilLogConfig.config_handler(plog.stream_method | plog.file_method)
    root_logger = plog.PutilLogConfig('train').logger()
    root_logger.setLevel(log_level)
    TrainLogger = root_logger.getChild('Trainer')
    TrainLogger.setLevel(log_level)
    pab.args_log(args, TrainLogger)
    
    # TODO: build the net
    # TODO: build the loss
    # TODO: build the optimization
    # TODO: build the train
    # TODO: build the evaluate
    # TODO: build the test
    # TODO: to_cuda

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
            if ((epoch + 1) % args.evaluate_interval == 0) and (args.evaluate_interval != -1):
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