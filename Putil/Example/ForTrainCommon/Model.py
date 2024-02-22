# coding=utf-8
import Putil.loger as plog
import tensorflow as tf
from colorama import Fore
import os

# todo: config the root logger name
RootLoggerName = 'Root'
# todo: config the root logger basic log level
RootLoggerBasicLevel = plog.DEBUG
# todo: config the model logger name
ModelClassLoggerName = 'ModelClass'
# todo: config the model logger basic log level
ModelClassLoggerLevel = plog.DEBUG

root_logger = plog.PutilLogConfig(RootLoggerName).logger()
root_logger.setLevel(RootLoggerBasicLevel)
model_logger = root_logger.getChild(ModelClassLoggerName).logger()
model_logger.setLevel(ModelClassLoggerLevel)


class Model:
    def __init__(self, param):
        # todo: to extract the param to some special param part as what you want
        self._param = param
        self._param_opt = param['Opt']
        self._param_training = param['Training']
        # todo: specify the train output keys
        self._train_result_reflect = []
        # todo: specify the val output keys
        self._val_result_reflect = []
        # todo: configure the placeholder dict
        self._placeholder = dict()
        # todo: configure the pro part
        self._pro = None
        # todo: configure the loss value
        self._loss = None
        self._train_op = None
        # : use the model train batch record, configure in the re_init
        self._batch = 0
        # : the sess, configure in the re_init
        self._sess = None
        # : the train step , configure in the re_init
        self._step = None
        # : the summary writer, every cv hold a specified summary writer
        self._summary_writer = None
        pass

    # todo: according to the "BaseModel" param you should offer several type model build
    def __build_with_loss(self):
        # todo:
        if self._param['BaseModel'] == "__real_model_build_1":
            self.__real_model_build_1()
            pass
        # todo:
        elif self._param['BaseModel'] == "__real_model_build_2":
            self.__real_model_build_2()
            pass
        else:
            # todo: raise error information
            raise ValueError('')
        pass

    # todo: make the placeholder the model need in train or val
    def __make_placeholder(self):
        self._placeholder = None
        pass

    # : display the placeholder information
    def __display_placeholder(self):
        info = '-->placehold:\n'
        for i in self._placeholder.keys():
            info += i + ', '
        model_logger.info(
            Fore.GREEN + info
        )
        pass

    # todo: the real model build method
    def __real_model_build_1(self):
        # todo: build the model pro
        self._pro = None
        # todo: build the model loss
        self._loss = None
        # todo: or other you want
        pass

    # todo: the real model build method
    def __real_model_build_2(self):
        pass

    # todo: to build different opt according to the self._param_opt
    def __build_opt(self):
        # todo:
        if self._param_opt['Opt'] == "Adam":
            self.__real_opt_build_1()
            pass
        # todo:
        elif self._param_opt['Opt'] == "ADelta":
            self.__real_opt_build_2()
            pass
        # todo:
        else:
            # todo: while other unsupported type raise error
            raise ValueError()
        pass

    # todo: real opt builder, configure the self._train_op
    def __real_opt_build_1(self):
        self._train_op = None
        pass

    # todo: real opt builder, configure the self._train_op
    def __real_opt_build_2(self):
        self._train_op = None
        pass

    def re_init(self):
        self._batch = 0
        # : reset the model
        tf.reset_default_graph()
        # : generate placeholder
        self.__make_placeholder()
        # : display the placeholder information
        self.__display_placeholder()
        # : rebuild the model
        self.__build_with_loss()
        # : init the session and step
        self._sess = tf.Session()
        self._step = tf.Variable(0, trainable=False)
        # : generate the opt
        self.__build_opt()

        # todo: build up the summary

        # : make the summary path for every re_init
        i = 0
        while True:
            path = os.path.join(self._param_training['SummaryPath'], str(i))
            if os.path.exists(path):
                i += 1
                continue
            else:
                self._summary_writer = tf.summary.FileWriter(path)
                break
                pass
            pass
        pass
        pass

    # todo: configure the feed in data train
    def __make_train_feed(self, data):
        feed = dict()
        for i in data.keys():
            feed[self._placeholder[i]] = data[i]
            pass
        feed[self._placeholder['training']] = True
        return feed
        pass

    # todo: configure the feed in data in val
    def __make_val_feed(self, data):
        feed = dict()
        for i in data.keys():
            feed[self._placeholder[i]] = data[i]
            pass
        feed[self._placeholder['training']] = False
        return feed
        pass

    def TrainCV(self, data):
        # : get the feed dict from the data
        feed = self.__make_train_feed(data)
        # todo: training
        self._sess.run([], feed_dict=feed)
        self._batch += 1

        if self._batch % self._param_training['DisplayBatch'] == 0:
            # todo: display in display batch
            pass

        if self._batch % self._param_training['SummaryBatch'] == 0:
            # todo: save the summary
            self._summary_writer.add_summary()
            pass
        return {}
        pass

    def Val(self, data):
        # : feed the placeholder
        feed = self.__make_val_feed(data)
        # todo: tun the loss and train and ***
        self._sess.run([], feed_dict=feed)
        # : return the result want to estimate: ValResultReflect
        return {}
        pass

    @property
    def TrainResultReflect(self):
        return self._train_result_reflect
        pass

    @property
    def ValResultReflect(self):
        return self._val_result_reflect
        pass
    pass

