# coding=utf-8
import base.logger as plog
import base.project_base as ppb

DeepLearningBaseLogger = plog.PutilLogConfig('deep_learning_base').logger()
DeepLearningBaseLogger.setLevel(plog.DEBUG)
DLBaseArgsLogger = DeepLearningBaseLogger.getChild('DLBaseArgs')
DLBaseArgsLogger.setLevel(plog.DEBUG)


class SummaryManager:
    def __init__(self, summary_frequent):
        pass

    def summary_or_not(self, epoch, step, evaluate, train):
        pass
    pass

class DLBaseArgs(ppb.ProjectArg):
    def __init__(self, parser=None, train=True, evaluate=True, test=True, *args, **kwargs):
        ppb.ProjectArg.__init__(self, parser, args, kwargs)
        self._parser.add_argument('--device', dest='DLBaseArgsDevice', type=list, action='store', default=[0], 
        help='the device (list) used in tensorflow model runing, use DLBaseArgsDevice to get the arg')
        self._parser.add_argument('--batch_size', dest='DLBaseArgsBatchSize', type=list, action='store', default=[16],
        help='the batch size (list) for every device, use DLBaseArgsBatchSize to get the arg')
        self._parser.add_argument('--summary_frequent', dest='DLBaseArgsSummaryFrequent', type=int, action='store', default=32,
        help='the frequent base on the batch for make summary')
        if train:
            self._parser.add_argument('--epoch', dest='DLBaseArgsEpoch', type=int, action='store', 
            help='the training(int) epoch, use DLBaseArgsEpoch to get the arg')
            pass
        if evaluate:
            self._parser.add_argument('--evaluate_frequent', dest='DLBaseArgEvaluateRate', type=int, action='store',
            help='the frequent(int) base on train epoch for evaluating, use DLBaseArgEvaluateRate to get the arg')
            pass
        if test:
            self._parser.add_argument('--test', dest='TF')
            pass
        pass
    pass