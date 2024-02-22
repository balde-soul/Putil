# coding=utf-8
import abc
from colorama import Fore
import Putil.base.logger_base as plogb
import tensorflow as tf
import Putil.loger as plog
import Putil.tf.model_helper as tfmh
import os
import six

root_logger = plog.PutilLogConfig('train/TraubCommonModel').logger()
root_logger.setLevel(plog.DEBUG)
TrainCommonModelBaseLogger = root_logger.getChild('TrainCommonModelBase')
TrainCommonModelBaseLogger.setLevel(plog.DEBUG)
TrainCommonModelBaseWithUpdateLogger = root_logger.getChild("logger=TrainCommonModelBaseLogger")
TrainCommonModelBaseWithUpdateLogger.setLevel(plog.DEBUG)


@six.add_metaclass(abc.ABCMeta)
class TrainCommonModelBaseAbs(object):
    @abc.abstractmethod
    def re_init(self):
        """
        re init the model for another cv, and flash the env for cv , including but not limited to:
            cv_summary_path_flash, cv_model_save_path_flash, cv_batch_re_init, cv_train_opt_flash,
            what ever you want to re init you should do it in this method
        :return:
        """
        pass

    @abc.abstractmethod
    def TrainCV(self, data):
        """
        train one batch base on the data
        :param data:
        :return:
        """
        pass

    @abc.abstractmethod
    def Val(self, data):
        """
        val one batch base on the data
        :param data:
        :return:
        """
        pass

    @abc.abstractmethod
    def TrainEpochUpdate(self):
        """
        an abstractmethod for update the trainer on train epoch, this should be complement in the exact model
        :return:
        """
        pass

    @abc.abstractmethod
    def ValEpochUpdate(self):
        """
        an abstractmethod for update the trainer on val epoch, this should be complement in the exact model
        :return:
        """
        pass

    @abc.abstractmethod
    def TrainBatchUpdate(self):
        """
        an abstractmethod for update the trainer on train batch, this should be complement in the exact model
        :return:
        """
        pass

    @abc.abstractmethod
    def ValBatchUpdate(self):
        """
        an abstractmethod for update the trainer on val batch, this should be complement in the exact model
        :return:
        """
        pass

    @abc.abstractmethod
    @property
    def TrainResultReflect(self):
        """
        an abstract method for offering the key of result dict in training
        :return:list
        """
        pass

    @abc.abstractmethod
    @property
    def ValResultReflect(self):
        """
        an abstract method for offering the key of result dict in val
        :return: list
        """
        pass
    pass


ParamTrainingDefault = {
    'SummaryBatch': 32,
    'DisplayBatch': 32,
    "MovingDecay": 0.999,
}


# this class is for train_common.model_cv
# without TrainerUpdate
@six.add_metaclass(abc.ABCMeta)
class TrainCommonModelBase(plogb.LoggerBase, TrainCommonModelBaseAbs):
    def __init__(self, param, logger=TrainCommonModelBaseLogger):
        plogb.LoggerBase.__init__(self, logger)
        self._param_training = param['Training']
        self._logger = logger
        self._output_dict = dict()
        self._placeholder = dict()
        self._sess = None
        self._step = None
        self._batch = 0
        self._opt = None
        self._ema = None
        self._grads_and_vars = None
        self._train_op = None
        self._writer = None
        self._graph_saved = False
        self._train_summary_op = None
        self._val_summary_op = None
        self._result_val = None
        self._result_train = None
        self.DictLog(self._param_training, plog.INFO, Fore.GREEN)
        pass

    def re_init(self):
        self._logger.info(Fore.GREEN + 'set batch == 0' + Fore.RESET)
        # : reset the batch
        self._batch = 0
        # : clean the graph
        self._logger.info(Fore.GREEN + 'clean the graph' + Fore.RESET)
        tf.reset_default_graph()
        # : build the placeholder
        self._logger(Fore.GREEN + 'make the placeholder' + Fore.RESET)
        self._placeholder = self.__task_placeholder()
        self.DictLog(self._placeholder, plog.INFO)
        # : rebuild the model
        self._logger.info(Fore.GREEN + 'rebuild the model' + Fore.RESET)
        self._build_with_loss()
        # : init the session and step
        self._logger.info(Fore.GREEN + 'init the session and step' + Fore.RESET)
        self._sess = tf.Session()
        self._step = tf.Variable(0, trainable=False)
        # : build the optimizer
        self._logger.info(Fore.GREEN + 'build the optimizer' + Fore.RESET)
        self._opt = self.__build_opt()
        # : apply moving average and mv_op
        self._logger.info(Fore.GREEN + 'build moving average object' + Fore.RESET)
        self._ema = tf.train.ExponentialMovingAverage(self._param_training['MovingDecay'], self._step)
        # : calculate the gradient and apply the gradient
        self._logger.info(Fore.GREEN + 'calculate the gradient of the trainable variances '
                                       'in collection GraphKeys.TRAINABLE_VARIABLES' + Fore.RESET)
        self._grads_and_vars = self._opt.compute_gradients(self.LOSS)
        with tf.control_dependencies([self._ema.apply(tf.trainable_variables())]):
            self._train_op = self._opt.apply_gradients(self._grads_and_vars, self._step)
            pass
        self._sess.run(tf.global_variables_initializer())
        if self._graph_saved is False:
            tfmh.save_graph_and_pause(self._param_training['SummaryPath'])
            self._graph_saved = True
            pass
        summary = self.__make_summary
        self._train_summary_op = summary[0]
        self._val_summary_op = summary[1]
        # : build the writer depend on dynamic path, check the path
        i = 0
        while True:
            path = os.path.join(self._param_training['SummaryPath'], str(i))
            if os.path.exists(path):
                i += 1
                continue
            else:
                self._logger.info(Fore.GREEN + 'set the writer operate path: {0}'.format(path) + Fore.RESET)
                self._writer = tf.summary.FileWriter(path)
                break
                pass
            pass
        pass

    def TrainCV(self, data):
        self._make_train_feed(data)
        self._result_train = self.__run_train()
        self._batch += 1
        if self._batch % self._param_training['SummaryBatch'] == 0:
            self._writer.add_summary(self._summary, self._batch)
            pass
        if self._batch % self._param_training['DisplayBatch'] == 0:
            self.__display_indicator()
            pass
        return self._result_train
        pass

    def Val(self, data):
        self.__make_train_feed(data)
        self._result_val = self.__run_val()
        return self._result_val
        pass

    def TrainEpochUpdate(self, cv_trainer):
        self._logger.info(Fore.LIGHTRED_EX +
                          'this method is empty , trainer would not update!!!!!!'
                          + Fore.RESET)
        pass

    def TrainBatchUpdate(self, cv_trainer):
        self._logger.info(Fore.LIGHTRED_EX +
                          'this method is empty , trainer would not update!!!!!!'
                          + Fore.RESET)
        pass

    def ValBatchUpdate(self, cv_trainer):
        self._logger.info(Fore.LIGHTRED_EX +
                          'this method is empty , trainer would not update!!!!!!'
                          + Fore.RESET)
        pass

    def ValEpochUpdate(self, cv_trainer):
        self._logger.info(Fore.LIGHTRED_EX +
                          'this method is empty , trainer would not update!!!!!!'
                          + Fore.RESET)
        pass

    @property
    @abc.abstractmethod
    def __make_summary(self):
        """
        an abstract method for building the summary operation
        :return:the summary operation for train, the summary operation for val
        """
        pass

    @abc.abstractmethod
    @property
    def LOSS(self):
        """
        an abstract method for getting the loss node
        :return: the loss node
        """
        pass

    @abc.abstractmethod
    def __build_opt(self):
        """
        an abstract method for building the optimizer
        :return: operation the optimizer
        """
        pass

    @abc.abstractmethod
    def __build_with_loss(self):
        """
        an abstract method for building the model for train in cv
        :return:
        """
        pass

    @abc.abstractmethod
    def __task_placeholder(self):
        """
        an abstract method to make the place holder for the model
        completement the self._placeholder
        :return: a dict: the placeholder
        """
        pass

    @abc.abstractmethod
    def __run_train(self):
        """
        an abstractmethod for training the model in one batch
            0: complement the self._summary and other indicators what you want to get
        :return: return the train result which you want to record and use in the estimate with type dict
        """
        pass

    @abc.abstractmethod
    def __run_val(self):
        """
        an abstractmethod for validity checking the model in one batch
            0: complement the self._summary and other indicators what you want to get
        :return: return the val result which you want to record and use in the estimate with type dict
        """
        pass

    @abc.abstractmethod
    def __make_train_feed(self, data):
        """
        an abstractmethod for making the feed dict for training
        :param data:
        :return: None
        """
        pass

    @abc.abstractmethod
    def __make_val_feed(self, data):
        """
        an abstractmethod for making the feed dict for validity checking
        :param data:
        :return:
        """
        pass

    @abc.abstractmethod
    def __display_indicator(self):
        """
        an abstractmethod for displaying the indicator
            0: display the indicator what you wan to show
        :return:
        """
        pass

    @abc.abstractmethod
    def __return_train_indicators(self):
        """
        an abstractmethod for returning indicators to the trainer in type: dict
        :return: dict()
        """
        pass

    @abc.abstractmethod
    def __return_val_indicators(self):
        """
        an abstractmethod for returning indicators to the trainer in type: dict
        :return: dict()
        """
        pass

    @property
    def OutputDict(self):
        return self._output_dict
        pass
    pass


# this class is for train_common.model_cv_trainer_update
# anything change while in training can be write in the TrainerUpdate
# like lr update, size update
# reset the Update method to abs
@abc.abstractmethod
class TrainCommonModelBaseWithUpdate(TrainCommonModelBase):
    def __init__(self, param, logger=TrainCommonModelBaseWithUpdateLogger):
        TrainCommonModelBase.__init__(param, logger)
        self._stop = False
        pass

    def re_init(self):
        TrainCommonModelBase.re_init(self)
        self._stop = False

    @abc.abstractmethod
    def TrainBatchUpdate(self, cv_trainer):
        pass

    @abc.abstractmethod
    def TrainEpochUpdate(self, cv_trainer):
        pass

    @abc.abstractmethod
    def ValEpochUpdate(self, cv_trainer):
        pass

    @abc.abstractmethod
    def ValBatchUpdate(self, cv_trainer):
        pass

    @property
    def Stop(self):
        return self._stop

