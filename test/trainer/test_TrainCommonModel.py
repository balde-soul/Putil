# coding = utf-8

from optparse import OptionParser
import Putil.loger as plog
from colorama import Fore
import functools
import Putil.test.test_helper as th

parser = OptionParser(usage='usage %prog [options] arg1 arg2')
level_default = 'Debug'
parser.add_option(
    '--level',
    action='store',
    dest='Level',
    type=str,
    default=level_default,
    help='specify the log level for the app'
         'default: {0}'.format(level_default)
)
parser.add_option(
    '--test_train_with_update',
    action='store_true',
    default=False,
    dest='TestTrainWithUpdate',
    help='set this flag while you want to test TrainWithUpdate'
)
(options, args) = parser.parse_args()
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")

root_logger = plog.PutilLogConfig('tf/test/trainer/test_TrainCommonModel').logger()
root_logger.setLevel(plog.DEBUG)
TestTrainWithUpdateLogger = root_logger.getChild('TestTrainWithUpdate')
TestTrainWithUpdateLogger.setLevel(plog.DEBUG)
ModelLogger = root_logger

import Putil.trainer.TrainCommonModel as bmodel
import Putil.tf.param.learning_update as lru
import Putil.tf.param.auto_save as asa
import Putil.tf.param.early_stop as es
import numpy as np


# todo:
class Model(bmodel.TrainCommonModelBaseWithUpdate):
    def __init__(self):
        bmodel.TrainCommonModelBaseWithUpdate.__init__(self, {}, ModelLogger)
        self._val_acc = [
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            0.0, 1.0, 2.0, 3.0, 4.0,
            11.0, 0.0, 1.0, 2.0, 3.0,
            4.0, 0.0, 1.0, 2.0, 3.0,
        ]
        self._acc_batch_collection = []
        self._acc_epoch_mean = None
        self._locv = 0
        self._train_loss = [
            0.1, 0.2, 0.3, 0.4, 0.5
        ]
        self._lr_control = lru.LrUpdate().UseDefaultDecider(
            max=True,
            interval=5,
            epsilon=1.0,
            cooldown=3
        ).SetIndicatorGet(self._acc).CheckReduceLROnPlateau()
        self._save_control = asa.ImproveSave(5).UseDefaultDecider(
            max=True
        ).SetIndicatorGet(self.Acc).CheckAutoSave()
        self._es_control = es.ValAccStop().UseDefaultDecider(
            max=True,
            interval=6,
            threshold=1.0
        ).CheckEarlyStop()
        pass

    def Acc(self):
        return self._acc

    def __task_placeholder(self):
        Model.debug('-->__task_placeholder')
        Model.info()
        Model.debug('__task_placeholder-->')
        pass

    def re_init(self):
        self._locv = 0
        pass

    def TrainEpochUpdate(self):
        Model.debug('-->TrainEpochUpdate')
        Model.debug('TrainEpochUpdate-->')
        pass

    def TrainBatchUpdate(self):
        Model.debug('-->TrainBatchUpdate')
        Model.debug('TrainBatchUpdate-->')
        pass

    def ValBatchUpdate(self):
        Model.debug('-->ValBatchUpdate')
        Model.debug('ValBatchUpdate-->')
        pass

    def ValEpochUpdate(self):
        Model.debug('-->ValEpochUpdate')
        self._acc_epoch_mean = np.mean(self._acc_batch_collection)
        self._acc_batch_collection = []
        Model.debug('ValEpochUpdate-->')
        pass

    pass

# todo:
class IndexGenerator:
    def __init__(self):
        pass

