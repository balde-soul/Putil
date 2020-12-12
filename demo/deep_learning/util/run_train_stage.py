# coding=utf-8
import torch
import time
from Putil.demo.deep_learning.base.util import train_stage
from Putil.demo.deep_learning.base.util import evaluate_stage
from Putil.demo.deep_learning.base.util import test_stage
from Putil.demo.deep_learning.base.util import Stage


class nothing():
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    pass


class ScalarCollection:
    def __init__(self, moving_epsilon=0.1):
        self._moving_average = dict()
        self._epoch_indicator = dict()
        self._current_indicator = dict()
        self._moving_epsilon = moving_epsilon
        pass

    def batch_update(self, indicators):
        if len(self._moving_average.keys()) == 0:
            self._moving_average.clear()
            self._epoch_indicator.clear()
            self._current_indicator.clear()
            for k, v in indicators.items():
                self._moving_average[k] = 0.
                self._epoch_indicator[k] = list()
                self._current_indicator[k] = 0.
                pass
            pass
        for k, v in indicators.items():
            self._current_indicator[k] = v
            self._epoch_indicator[k].append(v)
            self._moving_average[k] = self._moving_average[k] * self._moving_epsilon + (1 - self._moving_epsilon) * v
            pass
        pass

    @property
    def moving_average(self):
        return self._moving_average

    @property
    def epoch_average(self):
        return {k: v.mean() for k, v in self._epoch_indicator.items()}

    @property
    def current_indicators(self):
        return self._current_indicator

    @staticmethod
    def generate_epoch_average_reduce_name(base_name):
        return 'epoch_mean_{}'.format(base_name)

    @staticmethod
    def generate_current_reduce_name(base_name):
        return 'current_{}'.format(base_name)

    @staticmethod
    def generate_moving_reduce_name(base_name):
        return 'moving_'.format(base_name)
    pass


def all_reduce(value, name):
    if type(val).__name__ != 'Tensor':
        val = torch.tensor(val)
    avg_tensor = hvd.allreduce(val, name=name)
    return avg_tensor.item()


def train_stage_common(args, stage, epoch, fit_data_to_input, backbone, backend, decode, fit_decode_to_result, loss, optimization, indicator, statistic_indicator, data_loader, recorder, logger):
    assert train_stage(args)
    get_input = GetInput(args)
    prefix = 'train' if stage == Stage.Train else 'evaluate'
    dataset = data_loader.dataset
    loss_scalar_collection = ScalarCollection() if train_stage(args) else None
    indicator_scalar_collection = ScalarCollection() if train_stage(args) else None
    def scalar_log(indicators):
        logger.info('{0} epoch: {4}|{5} [{1}/{2}]: {3}'.format(prefix, \
            step % len(data_loader), len(data_loader), ['{}:{} '.format(k, v) for k, v in indicators.items()]), \
                epoch, step)
    with torch.no_grad() if stage == Stage.Evaluate else nothing() as t:
        backbone.train() if stage == Stage.Train else backbone.eval()
        backend.train() if stage == Stage.Train else backend.eval()
        decode.train() if stage == Stage.Train else decode.eval()
        data_loader.data_sampler.set_epoch(epoch) if stage == Stage.Train else None

        logger.debug('start to {} epoch'.format(prefix))
        for batch_idx, datas in enumerate(data_loader):
            recorder.step += 1 if stage == Stage.Train else 0; step = recorder.step
            logger.debug('batch {}'.format(prefix))
            # TODO: data to cuda
            #img = torch.from_numpy(img).cuda();gt_box = torch.from_numpy(gt_box).cuda();gt_class = torch.from_numpy(gt_class).cuda();gt_obj = torch.from_numpy(gt_obj).cuda();radiance_factor = torch.from_numpy(radiance_factor).cuda()
            backbone_input = fit_data_to_input(datas)
            # time
            batch_start = time.time()
            # do the training
            logger.debug('zero grad') if train_stage(args) else None
            optimizer.zero_grad() if train_stage(args) else None
            # : run the backbone get the output TODO:
            logger.debug('run backbone')
            output = backbone(backbone_input)
            # : run the backend get the output
            logger.debug('run backend')
            output = backend(output)
            # : run the loss function get the ret
            logger.debug('run loss') if train_stage(args) else None
            losses = loss(datas, output) if train_stage(args) else None
            logger.debug('update loss to loss_scalar_collection')
            loss_scalar_collection.batch_update(losses) if train_stage(args) else None
            _loss = losses[loss.total_loss_name]
            # TODO: do some simple check
            #if _iou_loss.item() < -1.0 or _iou_loss.item() > 1.0:
            #    logger.warning('giou wrong')
            #if np.isnan(_loss.item()) or np.isinf(_loss.item()) is True:
            #    logger.warning('loss in train inf occured!')
            # : run the indicator function to get the indicators
            logger.debug('run indicator') if train_stage(args) else None
            indicators = indicator((datas, output)) if train_stage(args) else None
            indicator_scalar_collection.batch_update(indicators) if train_stage(args) else None
            # : run the backward
            _loss.backward() if stage == Stage.Train else None
            # : do the optimize
            optimizer.step() if stage == Stage.Train else None
            # time
            batch_time = time.time() - batch_start
            # while in Stage.Train or Stage.TrainEvaluate
            if step % args.log_interval == 0 or (step % args.summary_interval == 0 and stage == Stage.Train):
                reduced_indicator_current = {ScalarCollection.generate_current_reduce_name(k): all_reduce(v, ScalarCollection.generate_current_reduce_name(k)) \
                    for k, v in loss_scalar_collection.current_indicators()}
                reduced_loss_current = {ScalarCollection.generate_current_reduce_name(k): all_reduce(v, ScalarCollection.generate_current_reduce_name(k)) \
                    for k, v in indicator_scalar_collection.current_indicators.items()}
            scalar_log(reduced_indicator_current.update(reduced_loss_current)) if step % args.log_interval == 0 else None
            if step % args.summary_interval == 0:
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
                    [writer.add_scalar('{}/{}'.format(prefix, k), all_reduce(v), global_step=step) for k, v in reduced_indicator_current.items()]
                    [writer.add_scalar('{}/{}'.format(prefix, k), all_reduce(v), global_step=step) for k, v in reduced_loss_current.items()]
                    pass
                pass
            pass
            logger.debug('batch {} end'.format(prefix))
            pass
        # : do the log of this epoch
        scalar_log(indicator_epoch_average.update(loss_epoch_average))
        # : do the summary of this epoch
        reduced_indicator_epoch_average = {ScalarCollection.generate_epoch_average_reduce_name(k): all_reduce(v, ScalarCollection.generate_epoch_average_reduce_name(k)) \
            for k, v in indicator_scalar_collection.epoch_average.items()}
        reduced_loss_epoch_average = {ScalarCollection.generate_epoch_average_reduce_name(k): all_reduce(v, ScalarCollection.generate_epoch_average_reduce_name(k)) \
            for k, v in loss_scalar_collection.epoch_average.items()}
        if hvd.rank() == 0:
            [writer.add_scalar('{}/{}'.format(prefix, k), all_reduce(v), global_step=step) for k, v in reduced_indicator_epoch_average.items()]
            [writer.add_scalar('{}/{}'.format(prefix, k), all_reduce(v), global_step=step) for k, v in reduce_loss_epoch_average.items()]
        pass
    return reduced_indicator_epoch_average.update(reduced_loss_epoch_average)