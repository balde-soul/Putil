# coding=utf-8
import numpy as np
from importlib import reload
import torch
import time
from Putil.demo.deep_learning.base import util
reload(util)
from Putil.demo.deep_learning.base import horovod
reload(horovod)


def train_stage_common(args, 
stage, 
epoch, 
fit_data_to_input, 
backbone, 
backend, 
decode, 
fit_decode_to_result, 
loss, 
optimization, 
indicator, 
indicator_statistic, 
accumulated_opt, 
data_loader, 
recorder, 
writer,
logger):
    '''
     @brief Train TrainEvaluate中使用，即在train stage中使用
     @note
      关于recorder：recorder只在Train阶段进行更新
      关于log： Train阶段会根据log_interval进行打印，Train与TrainEvaluate阶段在epoch之后会进行indicator与loss的mean allreduce然后log，step都为recorder.step
      关于summary： Train阶段会根据summary_interval进行summary，Train与TrainEvaluate阶段epoch之后都会对相关数据的mean allreduce进行summary，step都为recorder.step
    '''
    hvd = horovod.horovod(args)
    assert util.train_stage(args)
    recorder.epoch = epoch if stage == util.Stage.Train else recorder.epoch
    prefix = 'train' if stage == util.Stage.Train else 'evaluate'
    loss_scalar_collection = util.ScalarCollection() if util.train_stage(args) else None
    indicator_scalar_collection = util.ScalarCollection() if util.train_stage(args) else None
    def accumulation_fix(index):
        return np.ceil(index / args.accumulation_time)
    with torch.no_grad() if stage == util.Stage.Evaluate else util.nothing() as t:
        backbone.train() if stage == util.Stage.Train else backbone.eval()
        backend.train() if stage == util.Stage.Train else backend.eval()
        decode.train() if stage == util.Stage.Train else decode.eval()
        data_loader.sampler.set_epoch(epoch) if stage == util.Stage.Train else None

        logger.debug('start to {} epoch'.format(prefix))
        for batch_idx, datas in enumerate(data_loader):
            datas = [data.cuda() for data in datas] if util.iscuda(args) else datas
            recorder.step += accumulation_fix(1) if stage == util.Stage.Train else 0
            logger.debug('batch {}'.format(prefix))
            # TODO: data to cuda
            #img = torch.from_numpy(img).cuda();gt_box = torch.from_numpy(gt_box).cuda();gt_class = torch.from_numpy(gt_class).cuda();gt_obj = torch.from_numpy(gt_obj).cuda();radiance_factor = torch.from_numpy(radiance_factor).cuda()
            backbone_input = fit_data_to_input(datas)
            # time
            batch_start = time.time()
            # : run the backbone get the output TODO:
            logger.debug('run backbone')
            output = backbone(backbone_input)
            # : run the backend get the output
            logger.debug('run backend')
            output = backend(output)
            # : run the loss function get the ret
            logger.debug('run loss')
            losses = loss(datas, output)
            logger.debug('update loss to loss_scalar_collection')
            loss_scalar_collection.batch_update(losses)
            _loss = losses[loss.total_loss_name]
            # TODO: do some simple check
            #if _iou_loss.item() < -1.0 or _iou_loss.item() > 1.0:
            #    logger.warning('giou wrong')
            #if np.isnan(_loss.item()) or np.isinf(_loss.item()) is True:
            #    logger.warning('loss in train inf occured!')
            # : run the indicator function to get the indicators
            logger.debug('run indicator')
            indicators = indicator(datas, output)
            indicator_scalar_collection.batch_update(indicators)
            # use the accumulation backward
            logger.debug('run accumulated optimization')
            accumulated_opt.append(_loss, optimization, force_accumulation=None \
                if len(data_loader) % args.accumulation_time == 0 or \
                    len(data_loader) - batch_idx - 1 > len(data_loader) % args.accumulation_time \
                        else len(data_loader) % args.accumulation_time) if stage == util.Stage.Train else None
            ## : run the backward
            #_loss.backward() if stage == util.Stage.Train else None
            ## : do the optimize
            #optimizer.step() if stage == util.Stage.Train else None
            ## do the training
            #logger.debug('zero grad') if util.train_stage(args) else None
            #optimizer.zero_grad() if util.train_stage(args) else None
            # time
            batch_time = time.time() - batch_start
            # while in util.Stage.Train or util.Stage.TrainEvaluate
            if (recorder.step % args.log_interval == 0 or recorder.step % args.summary_interval == 0) and stage == util.Stage.Train:
                reduced_indicator_current = {util.ScalarCollection.generate_current_reduce_name(k): util.all_reduce(v, util.ScalarCollection.generate_current_reduce_name(k), hvd) \
                    for k, v in loss_scalar_collection.current_indicators.items()}
                reduced_loss_current = {util.ScalarCollection.generate_current_reduce_name(k): util.all_reduce(v, util.ScalarCollection.generate_current_reduce_name(k), hvd) \
                    for k, v in indicator_scalar_collection.current_indicators.items()}
                pass
                reduced_indicator_current.update(reduced_loss_current)
                util.scalar_log(logger, prefix, reduced_indicator_current, recorder, int(accumulation_fix(batch_idx)), \
                    int(accumulation_fix(len(data_loader)))) if recorder.step % args.log_interval == 0 and hvd.rank() == 0 and stage == util.Stage.Train else None
            if recorder.step % args.summary_interval == 0 and stage == util.Stage.Train:
                #gt_obj_ft = gt_obj.sum(1).gt(0.)
                ## reduce the target_indicator
                if hvd.rank() == 0:
                    # :decode
                    pre_output = decode(fit_data_to_input(datas), output)
                    gt_output = decode(datas)
                    # extract the target data
                    # TODO: do the summary use pre_output and gt_output
                    #pv = PointVisual();rv = RectangleVisual(2)
                    #result_visual(pv, rv, img_numpy, boxes, classes, indexes, '{}_pre'.format(prefix), 'pre', step)
                    #result_visual(pv, rv, img_numpy, gt_boxes, gt_classes, gt_indexes, '{}_gt'.format(prefix), 'gt', step)
                    # add target indicator to sumary
                    [writer.add_scalar('{}/{}'.format(prefix, k,), v, global_step=recorder.step) \
                            for k, v in reduced_indicator_current.items()]
                    [writer.add_scalar('{}/{}'.format(prefix, k,), v, global_step=recorder.step) \
                            for k, v in reduced_loss_current.items()]
                    pass
                pass
            pass
            logger.debug('batch {} end'.format(prefix))
            pass
        # : do the summary of this epoch
        reduced_indicator_epoch_average = {util.ScalarCollection.generate_epoch_average_reduce_name(k): util.all_reduce(v, util.ScalarCollection.generate_epoch_average_reduce_name(k), hvd) \
            for k, v in indicator_scalar_collection.epoch_average.items()}
        reduced_loss_epoch_average = {util.ScalarCollection.generate_epoch_average_reduce_name(k): util.all_reduce(v, util.ScalarCollection.generate_epoch_average_reduce_name(k), hvd) \
            for k, v in loss_scalar_collection.epoch_average.items()}
        # : do the log of this epoch
        if hvd.rank() == 0:
            reduced_indicator_epoch_average.update(reduced_loss_epoch_average)
            util.scalar_log(logger, prefix, reduced_indicator_epoch_average, recorder, None, None)
            [writer.add_scalar('{}/{}'.format(prefix, k, v, global_step=recorder.step)) for k, v in reduced_indicator_epoch_average.items()]
            [writer.add_scalar('{}/{}'.format(prefix, k, v, global_step=recorder.step)) for k, v in reduced_loss_epoch_average.items()]
        pass
    return reduced_indicator_epoch_average