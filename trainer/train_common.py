# coding=utf-8
from colorama import Fore
import numpy as np
import Putil.estimate.cv_estimate as cv_est
import Putil.tf.model_helper as mh
import sys
import os
import json
import Putil.loger as plog

root_logger = plog.PutilLogConfig("TrainCommonRoot").logger()
root_logger.setLevel(plog.DEBUG)
TrainCommonLog = root_logger.getChild("TrainCommonLog")


class TrainCommon:
    def __init__(self):
        self.epoch = None
        self.batch = None
        self.val_epoch = None

        self.training_epoch = None
        self.valing_epoch = None

        self.val_batch_result = None
        self.train_batch_result = None
        self.val_epoch_result_collection = None
        self.train_epoch_result_collection = None
        pass

    def model_run(self):
        pass

    def model_train(self):
        pass

    # support for different kinds of data type which can not train together, but can train in the same model
    # support for cv estimate :which kinds of data type should has the same cv num
    # support for specify batch epoch
    # support estimate visual cross epoch
    def model_cv(
            self,
            model,
            cv_collection,
            index_to_data,
            epoch,
            val_epoch_step,
            batch,
            save_path,
            **options):
        """

        :param model:
        :param cv_collection:
        :param index_to_data:
        :param epoch:
        :param batch:
        :return:
        """
        # use to fix the data_generator keys to the model data feed keys{'gen_key': 'feed_key'}
        gen_data_feed_data_reflect = options.pop('gdfdr', None)
        # use to fix the result gen keys to the estimate wanted data keys{'result_key': 'estimate_key'}
        gen_result_estimate_result_reflect = options.pop('grerr', None)
        assert len(cv_collection) == len(index_to_data), \
            TrainCommonLog.error(Fore.RED + 'cv_collection should has the same length with index_to_data')
        assert (False in [i in index_to_data.keys() for i in cv_collection.keys()]) is False, \
            TrainCommonLog.error(Fore.RED + 'cv_collection should has thesame keys with index_to_data')
        # all cv result save space
        result_collection = dict()
        val_result_collection = dict()
        for _cv_type in cv_collection.keys():
            result_collection[_cv_type] = list()
            val_result_collection[_cv_type] = list()
            pass

        # cross estimate calculation
        cv = 0
        while True:
            cv += 1
            # : generator cv index generator in every dat type one by one
            _index_generator = dict()
            TrainCommonLog.info("start generate cv_gen cross every type in the cv_step")
            for _cv_type in cv_collection.keys():

                # explain: every _index_generator is {'train': this_cv_train_index_gen, 'val': this_cv_val_index_gen}
                try:
                    _index_generator[_cv_type] = cv_collection[_cv_type].__next__()
                    # append a dict to collect the train result while every cv_step came
                    result_collection[_cv_type].append(dict())
                    val_result_collection[_cv_type].append(dict())
                except StopIteration:
                    print(Fore.RED + 'cv : {0} finish'.format(_cv_type))
                    break
                    pass

                # set every result from model train as a list follow epoch step
                if gen_result_estimate_result_reflect is not None:
                    for _model_result_name in model.TrainResultReflect:
                        result_collection[_cv_type][-1][gen_result_estimate_result_reflect[_model_result_name]] = list()
                    for _model_result_name in model.ValResultReflect:
                        val_result_collection[_cv_type][-1][
                            gen_result_estimate_result_reflect[_model_result_name]] = list()
                    pass
                    pass
                else:
                    for _model_result_name in model.TrainResultReflect:
                        result_collection[_cv_type][-1][_model_result_name] = list()
                    for _model_result_name in model.ValResultReflect:
                        val_result_collection[_cv_type][-1][_model_result_name] = list()
                    pass
                pass
            TrainCommonLog.info("extract successful")

            # check if there are any cv in all type which do not finished
            # yes continue cv this type, no break and finish the while cv estimate
            if len(_index_generator.keys()) == 0:
                print(Fore.GREEN + 'total cv finish')
                break
                pass

            # every cv should re init the model
            TrainCommonLog.info("re init the model")
            model.re_init()
            TrainCommonLog.info("re init successful")

            # : set the data to feed(every kind of cv use the same feed_data_list)
            _train_data_batch = dict()
            _val_data_batch = dict()
            # fill the batch data use what index_to_data generates
            TrainCommonLog.info('prepare the struct for store the data for thiw')
            for _cv_type in cv_collection.keys():
                _train_data_batch[_cv_type] = dict()
                _val_data_batch[_cv_type] = dict()
                if gen_data_feed_data_reflect is not None:
                    for data_name in index_to_data[_cv_type].DataListName:
                        _train_data_batch[_cv_type][gen_data_feed_data_reflect[data_name]] = list()
                        _val_data_batch[_cv_type][gen_data_feed_data_reflect[data_name]] = list()
                        pass
                    pass
                else:
                    for data_name in index_to_data[_cv_type].DataListName:
                        _train_data_batch[_cv_type][data_name] = list()
                        _val_data_batch[_cv_type][data_name] = list()
                        pass
                    pass
                pass

            # : display the data keys which is wanted to feed
            info = 'Train:\n'
            for i in _train_data_batch.keys():
                info += '--->'
                info += str(i)
                info += ':\n'
                for j in _train_data_batch[i].keys():
                    info += str(j)
                    info += ', '
                    pass
                info += '\n'
                pass
            info_v = 'Val:\n'
            for i in _train_data_batch.keys():
                info_v += '--->'
                info_v += str(i)
                info_v += ':\n'
                for j in _train_data_batch[i].keys():
                    info_v += str(j)
                    info_v += ', '
                    pass
                info_v += '\n'
                pass
            TrainCommonLog.info(
                Fore.YELLOW + 'from train_common data feed to Model(actually the model used):\n'
                              '{0}\n{1}'.format(info, info_v))

            train_epoch = list()
            val_epoch = list()

            # : train all epoch in one cv with all data type
            TrainCommonLog.info("start {0} train".format(epoch))
            for _epoch in range(0, epoch):
                # batch_train_result collection
                epoch_result_with_cross_batch = dict()
                for _type_name in cv_collection.keys():
                    epoch_result_with_cross_batch[_type_name] = dict()
                    if gen_result_estimate_result_reflect is not None:
                        for _model_result_name in model.TrainResultReflect:
                            epoch_result_with_cross_batch[_type_name][
                                gen_result_estimate_result_reflect[_model_result_name]] = list()
                            pass
                        pass
                    else:
                        for _model_result_name in model.TrainResultReflect:
                            epoch_result_with_cross_batch[_type_name][_model_result_name] = list()
                            pass
                        pass
                    pass

                _all_type_batch_done = dict()
                # : the flag for all cv_type epoch done : initializer
                for _key in cv_collection.keys():
                    _all_type_batch_done[_key] = False
                    pass

                # : train epoch batch by batch in all kind of cv
                while True:
                    # while any kind of data type do not finish its epoch in this cv
                    if (False in _all_type_batch_done.values()) is False:
                        # : while one epoch done ,collect the mean result follow the batch to the result_collection
                        for _result_type_cv in epoch_result_with_cross_batch.keys():
                            for _result_type in epoch_result_with_cross_batch[_result_type_cv].keys():
                                result_collection[_result_type_cv][-1][_result_type].append(
                                    np.mean(epoch_result_with_cross_batch[_result_type_cv][_result_type])
                                )
                                pass
                            pass
                        break
                        pass

                    # : train batch by batch in all kind of cv
                    for _cv_type in cv_collection.keys():
                        if _all_type_batch_done[_cv_type] is not True:
                            # generate batch data and update the statue of epoch done or not in every kind of cv
                            for _batch in range(0, batch):
                                # require index_generator_in_cv yield {'data': data, 'total': total_or_not}
                                _train_data_index = _index_generator[_cv_type]['train'].__next__()

                                # set this cv_type epoch done
                                if _train_data_index['total']:
                                    _all_type_batch_done[_cv_type] = True
                                    pass

                                # index to real data
                                _train_data_one = index_to_data[_cv_type].index_to_data(_train_data_index['data'])
                                # generate batch
                                if gen_data_feed_data_reflect is not None:
                                    for _data_name in _train_data_one.keys():
                                        _train_data_batch[_cv_type][gen_data_feed_data_reflect[_data_name]].append(
                                            _train_data_one[_data_name])
                                        pass
                                    pass
                                else:
                                    for _data_name in _train_data_one.keys():
                                        _train_data_batch[_cv_type][_data_name].append(_train_data_one[_data_name])
                                        pass
                                    pass
                                pass

                            # use _train_data_batch to train the model, wanted!!: model train return a dict
                            try:
                                _train_result = model.TrainCV(_train_data_batch[_cv_type])
                            except Exception as e:
                                TrainCommonLog.error(
                                    Fore.RED +
                                    'train model exception\n{0}'.format(
                                        e))
                                # set the step for estimate
                                for _cv_type in cv_collection.keys():
                                    val_result_collection[_cv_type][-1]['step'] = val_epoch
                                    result_collection[_cv_type][-1]['step'] = train_epoch
                                    pass
                                pass
                                for _cv_type in result_collection.keys():
                                    cv_est.mutual_exclusion_cv_estimate(
                                        result_collection[_cv_type],
                                        val_result_collection[_cv_type],
                                        result_save=save_path,
                                        prefix=_cv_type,
                                    )
                                    pass
                                sys.exit()
                                pass

                            # : release the memory
                            for input_type in _train_data_batch[_cv_type].keys():
                                _train_data_batch[_cv_type][input_type] = list()
                                pass

                            # collect result to the batch collection
                            if gen_result_estimate_result_reflect is not None:
                                for _result_name in _train_result.keys():
                                    epoch_result_with_cross_batch[_cv_type][
                                        gen_result_estimate_result_reflect[_result_name]].append(
                                        _train_result[_result_name])
                                pass
                            else:
                                for _result_name in _train_result.keys():
                                    epoch_result_with_cross_batch[_cv_type][_result_name].append(
                                        _train_result[_result_name])
                                pass
                            pass
                        else:
                            pass
                        pass
                    pass

                # : train one epoch display
                epoch_info = 'cv: {1} train_epoch: {0}\n'.format(_epoch, cv)
                for _cv_type in result_collection.keys():
                    epoch_info += '-->cv_type: {0}\n'.format(_cv_type)
                    for _wanted in result_collection[_cv_type][-1].keys():
                        epoch_info += ' --<{0}: {1}\n'.format(_wanted, result_collection[_cv_type][-1][_wanted][-1])
                        pass
                    pass
                train_epoch.append(_epoch)
                print(Fore.RED + epoch_info)

                # val
                if _epoch % val_epoch_step == 0:
                    # batch_train_result collection
                    val_result_with_cross_batch = dict()
                    for _type_name in cv_collection.keys():
                        val_result_with_cross_batch[_type_name] = dict()
                        if gen_result_estimate_result_reflect is not None:
                            for _model_result_name in model.ValResultReflect:
                                val_result_with_cross_batch[_type_name][
                                    gen_result_estimate_result_reflect[_model_result_name]] = list()
                                pass
                            pass
                        else:
                            for _model_result_name in model.ValResultReflect:
                                val_result_with_cross_batch[_type_name][_model_result_name] = list()
                                pass
                            pass
                        pass
                    _val_all_type_batch_done = dict()
                    for _key in cv_collection.keys():
                        _val_all_type_batch_done[_key] = False
                        pass
                    while True:
                        # while any kind of data type do not finish its epoch in this cv
                        if (False in _val_all_type_batch_done.values()) is False:
                            # : while one epoch done ,collect the mean result follow the batch to the result_collection
                            for _result_type_cv in val_result_with_cross_batch.keys():
                                for _result_type in val_result_with_cross_batch[_result_type_cv].keys():
                                    val_result_collection[_result_type_cv][-1][_result_type].append(
                                        np.mean(val_result_with_cross_batch[_result_type_cv][_result_type])
                                    )
                                    pass
                                pass
                            break
                            pass
                        for _cv_type in cv_collection.keys():
                            if _val_all_type_batch_done[_cv_type] is not True:
                                for _batch in range(0, batch):
                                    _val_data_index = _index_generator[_cv_type]['val'].__next__()
                                    if _val_data_index['total']:
                                        _val_all_type_batch_done[_cv_type] = True
                                        pass
                                    _val_data_one = index_to_data[_cv_type].index_to_data(_val_data_index['data'])
                                    if gen_data_feed_data_reflect is not None:
                                        for _data_name in _val_data_one.keys():
                                            _val_data_batch[_cv_type][gen_data_feed_data_reflect[_data_name]].append(
                                                _val_data_one[_data_name])
                                            pass
                                        pass
                                    else:
                                        for _data_name in _val_data_one.keys():
                                            _val_data_batch[_cv_type][_data_name].append(_val_data_one[_data_name])
                                            pass
                                        pass
                                    pass
                                try:
                                    val_result = model.Val(_val_data_batch[_cv_type])
                                except:
                                    print(Fore.RED + 'val model exception')
                                    # set the step for estimate
                                    for _cv_type in cv_collection.keys():
                                        val_result_collection[_cv_type][-1]['step'] = val_epoch
                                        result_collection[_cv_type][-1]['step'] = train_epoch
                                        pass
                                    pass
                                    for _cv_type in result_collection.keys():
                                        cv_est.mutual_exclusion_cv_estimate(
                                            result_collection[_cv_type],
                                            val_result_collection[_cv_type],
                                            result_save=save_path,
                                            prefix=_cv_type,
                                        )
                                        pass
                                    sys.exit()
                                    pass

                                # : release the memory
                                for input_type in _val_data_batch[_cv_type].keys():
                                    _val_data_batch[_cv_type][input_type] = list()
                                    pass

                                if gen_result_estimate_result_reflect is not None:
                                    for _result_name in val_result.keys():
                                        val_result_with_cross_batch[_cv_type][
                                            gen_result_estimate_result_reflect[_result_name]].append(
                                            val_result[_result_name])
                                else:
                                    for _result_name in val_result.keys():
                                        val_result_with_cross_batch[_cv_type][_result_name].append(
                                            val_result[_result_name])
                                        pass
                                    pass
                                pass
                            pass
                        pass

                    val_epoch.append(_epoch)
                    pass

                # : val one epoch display
                epoch_info = 'cv: {1} val_epoch: {0}\n'.format(_epoch, cv)
                for _cv_type in val_result_collection.keys():
                    epoch_info += '-->cv_type: {0}\n'.format(_cv_type)
                    for _wanted in val_result_collection[_cv_type][-1].keys():
                        epoch_info += ' --<{0}: {1}\n'.format(_wanted, val_result_collection[_cv_type][-1][_wanted][-1])
                        pass
                    pass
                print(Fore.RED + epoch_info)

                pass
            # set the step for estimate
            for _cv_type in cv_collection.keys():
                val_result_collection[_cv_type][-1]['step'] = val_epoch
                result_collection[_cv_type][-1]['step'] = train_epoch
                pass
            pass
        pass

        # : save estimate result
        for _cv_type in result_collection.keys():
            try:
                cv_est.mutual_exclusion_cv_estimate(
                    result_collection[_cv_type],
                    val_result_collection[_cv_type],
                    result_save=save_path,
                    prefix=_cv_type,
                )
                pass
            except:
                # : id save failed , we want some effect data save
                print(Fore.RED + 'generate {0} estimate figure failed, save cv data in {1}'.format(cv, save_path))
                for cv in range(0, len(result_collection[_cv_type])):
                    path = os.path.join(os.path.join(save_path, _cv_type), cv)
                    print(Fore.RED + '{0}'.format(path))
                    with open(path, 'w') as fp:
                        info = json.dumps(result_collection[_cv_type][cv])
                        fp.write(info)
                        fp.close()
                        pass
                    pass
                pass
            pass
        pass

    def model_cv_trainer_update(
            self,
            model,
            cv_collection,
            index_to_data,
            epoch,
            val_epoch_step,
            batch,
            save_path,
            **options):
        """

               :param model:
               :param cv_collection:
               :param index_to_data:
               :param epoch:
               :param batch:
               :return:
               """
        self.val_epoch = val_epoch_step
        # use to fix the data_generator keys to the model data feed keys{'gen_key': 'feed_key'}
        gen_data_feed_data_reflect = options.pop('gdfdr', None)
        # use to fix the result gen keys to the estimate wanted data keys{'result_key': 'estimate_key'}
        gen_result_estimate_result_reflect = options.pop('grerr', None)
        assert len(cv_collection) == len(index_to_data), \
            TrainCommonLog.error(Fore.RED + 'cv_collection should has the same length with index_to_data')
        assert (False in [i in index_to_data.keys() for i in cv_collection.keys()]) is False, \
            TrainCommonLog.error(Fore.RED + 'cv_collection should has thesame keys with index_to_data')
        # all cv result save space
        result_collection = dict()
        val_result_collection = dict()
        for _cv_type in cv_collection.keys():
            result_collection[_cv_type] = list()
            val_result_collection[_cv_type] = list()
            pass

        # cross estimate calculation
        cv = 0
        while True:
            cv += 1
            # : generator cv index generator in every dat type one by one
            _index_generator = dict()
            TrainCommonLog.info("start generate cv_gen cross every type in the cv_step")
            for _cv_type in cv_collection.keys():

                # explain: every _index_generator is {'train': this_cv_train_index_gen, 'val': this_cv_val_index_gen}
                try:
                    _index_generator[_cv_type] = cv_collection[_cv_type].__next__()
                    # append a dict to collect the train result while every cv_step came
                    result_collection[_cv_type].append(dict())
                    val_result_collection[_cv_type].append(dict())
                except StopIteration:
                    print(Fore.RED + 'cv : {0} finish'.format(_cv_type))
                    break
                    pass

                # set every result from model train as a list follow epoch step
                if gen_result_estimate_result_reflect is not None:
                    for _model_result_name in model.TrainResultReflect:
                        result_collection[_cv_type][-1][gen_result_estimate_result_reflect[_model_result_name]] = list()
                    for _model_result_name in model.ValResultReflect:
                        val_result_collection[_cv_type][-1][
                            gen_result_estimate_result_reflect[_model_result_name]] = list()
                    pass
                    pass
                else:
                    for _model_result_name in model.TrainResultReflect:
                        result_collection[_cv_type][-1][_model_result_name] = list()
                    for _model_result_name in model.ValResultReflect:
                        val_result_collection[_cv_type][-1][_model_result_name] = list()
                    pass
                pass
            TrainCommonLog.info("extract successful")

            # check if there are any cv in all type which do not finished
            # yes continue cv this type, no break and finish the while cv estimate
            if len(_index_generator.keys()) == 0:
                print(Fore.GREEN + 'total cv finish')
                break
                pass

            # every cv should re init the model
            TrainCommonLog.info("re init the model")
            model.re_init()
            self.batch = 0
            self.epoch = 0
            TrainCommonLog.info("re init successful")

            # : set the data to feed(every kind of cv use the same feed_data_list)
            _train_data_batch = dict()
            _val_data_batch = dict()
            # fill the batch data use what index_to_data generates
            TrainCommonLog.info('prepare the struct for store the data for thiw')
            for _cv_type in cv_collection.keys():
                _train_data_batch[_cv_type] = dict()
                _val_data_batch[_cv_type] = dict()
                if gen_data_feed_data_reflect is not None:
                    for data_name in index_to_data[_cv_type].DataListName:
                        _train_data_batch[_cv_type][gen_data_feed_data_reflect[data_name]] = list()
                        _val_data_batch[_cv_type][gen_data_feed_data_reflect[data_name]] = list()
                        pass
                    pass
                else:
                    for data_name in index_to_data[_cv_type].DataListName:
                        _train_data_batch[_cv_type][data_name] = list()
                        _val_data_batch[_cv_type][data_name] = list()
                        pass
                    pass
                pass

            # : display the data keys which is wanted to feed
            info = 'Train:\n'
            for i in _train_data_batch.keys():
                info += '--->'
                info += str(i)
                info += ':\n'
                for j in _train_data_batch[i].keys():
                    info += str(j)
                    info += ', '
                    pass
                info += '\n'
                pass
            info_v = 'Val:\n'
            for i in _train_data_batch.keys():
                info_v += '--->'
                info_v += str(i)
                info_v += ':\n'
                for j in _train_data_batch[i].keys():
                    info_v += str(j)
                    info_v += ', '
                    pass
                info_v += '\n'
                pass
            TrainCommonLog.info(
                Fore.YELLOW + 'from train_common data feed to Model(actually the model used):\n'
                              '{0}\n{1}'.format(info, info_v))

            self.training_epoch = list()
            self.valing_epoch = list()

            # : train all epoch in one cv with all data type
            TrainCommonLog.info("start {0} train".format(epoch))
            for _epoch in range(0, epoch):
                if model.Stop is True:
                    TrainCommonLog.info(Fore.LIGHTRED_EX + 'model request for stop' + Fore.RESET)
                    break
                self.epoch = _epoch
                # batch_train_result collection
                epoch_result_with_cross_batch = dict()
                for _type_name in cv_collection.keys():
                    epoch_result_with_cross_batch[_type_name] = dict()
                    if gen_result_estimate_result_reflect is not None:
                        for _model_result_name in model.TrainResultReflect:
                            epoch_result_with_cross_batch[_type_name][
                                gen_result_estimate_result_reflect[_model_result_name]] = list()
                            pass
                        pass
                    else:
                        for _model_result_name in model.TrainResultReflect:
                            epoch_result_with_cross_batch[_type_name][_model_result_name] = list()
                            pass
                        pass
                    pass

                _all_type_batch_done = dict()
                # : the flag for all cv_type epoch done : initializer
                for _key in cv_collection.keys():
                    _all_type_batch_done[_key] = False
                    pass

                # : train epoch batch by batch in all kind of cv
                while True:
                    # while any kind of data type do not finish its epoch in this cv
                    if (False in _all_type_batch_done.values()) is False:
                        # : while one epoch done ,collect the mean result follow the batch to the result_collection
                        for _result_type_cv in epoch_result_with_cross_batch.keys():
                            for _result_type in epoch_result_with_cross_batch[_result_type_cv].keys():
                                result_collection[_result_type_cv][-1][_result_type].append(
                                    np.mean(epoch_result_with_cross_batch[_result_type_cv][_result_type])
                                )
                                pass
                            pass
                        break
                        pass

                    # : train batch by batch in all kind of cv
                    for _cv_type in cv_collection.keys():
                        if _all_type_batch_done[_cv_type] is not True:
                            # generate batch data and update the statue of epoch done or not in every kind of cv
                            for _batch in range(0, batch):
                                # require index_generator_in_cv yield {'data': data, 'total': total_or_not}
                                _train_data_index = _index_generator[_cv_type]['train'].__next__()

                                # set this cv_type epoch done
                                if _train_data_index['total']:
                                    _all_type_batch_done[_cv_type] = True
                                    pass

                                # index to real data
                                _train_data_one = index_to_data[_cv_type].index_to_data(_train_data_index['data'])
                                # generate batch
                                if gen_data_feed_data_reflect is not None:
                                    for _data_name in _train_data_one.keys():
                                        _train_data_batch[_cv_type][gen_data_feed_data_reflect[_data_name]].append(
                                            _train_data_one[_data_name])
                                        pass
                                    pass
                                else:
                                    for _data_name in _train_data_one.keys():
                                        _train_data_batch[_cv_type][_data_name].append(_train_data_one[_data_name])
                                        pass
                                    pass
                                pass

                            # use _train_data_batch to train the model, wanted!!: model train return a dict
                            self.batch += 1
                            try:
                                _train_result = model.TrainCV(_train_data_batch[_cv_type])
                                model.TrainBatchUpdate(self)
                            except Exception as e:
                                TrainCommonLog.error(
                                    Fore.RED +
                                    'train model exception\n{0}'.format(
                                        e))
                                # set the step for estimate
                                for _cv_type in cv_collection.keys():
                                    val_result_collection[_cv_type][-1]['step'] = self.valing_epoch
                                    result_collection[_cv_type][-1]['step'] = self.training_epoch
                                    pass
                                pass
                                for _cv_type in result_collection.keys():
                                    cv_est.mutual_exclusion_cv_estimate(
                                        result_collection[_cv_type],
                                        val_result_collection[_cv_type],
                                        result_save=save_path,
                                        prefix=_cv_type,
                                    )
                                    pass
                                sys.exit()
                                pass

                            # : release the memory
                            for input_type in _train_data_batch[_cv_type].keys():
                                _train_data_batch[_cv_type][input_type] = list()
                                pass

                            # collect result to the batch collection
                            if gen_result_estimate_result_reflect is not None:
                                for _result_name in _train_result.keys():
                                    epoch_result_with_cross_batch[_cv_type][
                                        gen_result_estimate_result_reflect[_result_name]].append(
                                        _train_result[_result_name])
                                pass
                            else:
                                for _result_name in _train_result.keys():
                                    epoch_result_with_cross_batch[_cv_type][_result_name].append(
                                        _train_result[_result_name])
                                pass
                            pass
                        else:
                            pass
                        pass
                    pass

                # : train one epoch display
                epoch_info = 'cv: {1} train_epoch: {0}\n'.format(_epoch, cv)
                for _cv_type in result_collection.keys():
                    epoch_info += '-->cv_type: {0}\n'.format(_cv_type)
                    for _wanted in result_collection[_cv_type][-1].keys():
                        epoch_info += ' --<{0}: {1}\n'.format(_wanted, result_collection[_cv_type][-1][_wanted][-1])
                        pass
                    pass
                self.train_epoch_result_collection = result_collection
                self.training_epoch.append(_epoch)
                print(Fore.RED + epoch_info)

                # model Update while one train epoch done
                model.TrainEpochUpdate(self)

                # val
                if _epoch % val_epoch_step == 0:
                    # batch_train_result collection
                    val_result_with_cross_batch = dict()
                    for _type_name in cv_collection.keys():
                        val_result_with_cross_batch[_type_name] = dict()
                        if gen_result_estimate_result_reflect is not None:
                            for _model_result_name in model.ValResultReflect:
                                val_result_with_cross_batch[_type_name][
                                    gen_result_estimate_result_reflect[_model_result_name]] = list()
                                pass
                            pass
                        else:
                            for _model_result_name in model.ValResultReflect:
                                val_result_with_cross_batch[_type_name][_model_result_name] = list()
                                pass
                            pass
                        pass
                    _val_all_type_batch_done = dict()
                    for _key in cv_collection.keys():
                        _val_all_type_batch_done[_key] = False
                        pass
                    while True:
                        # while any kind of data type do not finish its epoch in this cv
                        if (False in _val_all_type_batch_done.values()) is False:
                            # : while one epoch done ,collect the mean result follow the batch to the result_collection
                            for _result_type_cv in val_result_with_cross_batch.keys():
                                for _result_type in val_result_with_cross_batch[_result_type_cv].keys():
                                    val_result_collection[_result_type_cv][-1][_result_type].append(
                                        np.mean(val_result_with_cross_batch[_result_type_cv][_result_type])
                                    )
                                    pass
                                pass
                            break
                            pass
                        for _cv_type in cv_collection.keys():
                            if _val_all_type_batch_done[_cv_type] is not True:
                                for _batch in range(0, batch):
                                    _val_data_index = _index_generator[_cv_type]['val'].__next__()
                                    if _val_data_index['total']:
                                        _val_all_type_batch_done[_cv_type] = True
                                        pass
                                    _val_data_one = index_to_data[_cv_type].index_to_data(_val_data_index['data'])
                                    if gen_data_feed_data_reflect is not None:
                                        for _data_name in _val_data_one.keys():
                                            _val_data_batch[_cv_type][gen_data_feed_data_reflect[_data_name]].append(
                                                _val_data_one[_data_name])
                                            pass
                                        pass
                                    else:
                                        for _data_name in _val_data_one.keys():
                                            _val_data_batch[_cv_type][_data_name].append(_val_data_one[_data_name])
                                            pass
                                        pass
                                    pass
                                try:
                                    val_result = model.Val(_val_data_batch[_cv_type])
                                    model.ValBatchUpdate(self)
                                except:
                                    print(Fore.RED + 'val model exception')
                                    # set the step for estimate
                                    for _cv_type in cv_collection.keys():
                                        val_result_collection[_cv_type][-1]['step'] = self.valing_epoch
                                        result_collection[_cv_type][-1]['step'] = self.training_epoch
                                        pass
                                    pass
                                    for _cv_type in result_collection.keys():
                                        cv_est.mutual_exclusion_cv_estimate(
                                            result_collection[_cv_type],
                                            val_result_collection[_cv_type],
                                            result_save=save_path,
                                            prefix=_cv_type,
                                        )
                                        pass
                                    sys.exit()
                                    pass

                                # : release the memory
                                for input_type in _val_data_batch[_cv_type].keys():
                                    _val_data_batch[_cv_type][input_type] = list()
                                    pass

                                if gen_result_estimate_result_reflect is not None:
                                    for _result_name in val_result.keys():
                                        val_result_with_cross_batch[_cv_type][
                                            gen_result_estimate_result_reflect[_result_name]].append(
                                            val_result[_result_name])
                                else:
                                    for _result_name in val_result.keys():
                                        val_result_with_cross_batch[_cv_type][_result_name].append(
                                            val_result[_result_name])
                                        pass
                                    pass
                                pass
                            pass
                        pass

                    self.val_epoch_result_collection = val_result_collection
                    self.valing_epoch.append(_epoch)
                    pass

                # : val one epoch display
                epoch_info = 'cv: {1} val_epoch: {0}\n'.format(_epoch, cv)
                for _cv_type in val_result_collection.keys():
                    epoch_info += '-->cv_type: {0}\n'.format(_cv_type)
                    for _wanted in val_result_collection[_cv_type][-1].keys():
                        epoch_info += ' --<{0}: {1}\n'.format(_wanted, val_result_collection[_cv_type][-1][_wanted][-1])
                        pass
                    pass
                print(Fore.RED + epoch_info)

                # model Update while ine val epoch done
                model.ValEpochUpdate(self)

                pass
            # set the step for estimate
            for _cv_type in cv_collection.keys():
                val_result_collection[_cv_type][-1]['step'] = self.valing_epoch
                result_collection[_cv_type][-1]['step'] = self.training_epoch
                pass
            pass
        pass

        # : save estimate result
        for _cv_type in result_collection.keys():
            try:
                cv_est.mutual_exclusion_cv_estimate(
                    result_collection[_cv_type],
                    val_result_collection[_cv_type],
                    result_save=save_path,
                    prefix=_cv_type,
                )
                pass
            except:
                # : id save failed , we want some effect data save
                print(Fore.RED + 'generate {0} estimate figure failed, save cv data in {1}'.format(cv, save_path))
                for cv in range(0, len(result_collection[_cv_type])):
                    path = os.path.join(os.path.join(save_path, _cv_type), cv)
                    print(Fore.RED + '{0}'.format(path))
                    with open(path, 'w') as fp:
                        info = json.dumps(result_collection[_cv_type][cv])
                        fp.write(info)
                        fp.close()
                        pass
                    pass
                pass
            pass
        pass

    def this_cv_this_epoch_val_result(self, wanted):
        multi_cv_type = []
        [multi_cv_type.append(self.val_epoch_result_collection[i][-1][wanted][-1])
         for i in self.val_epoch_result_collection.keys()]
        return np.mean(multi_cv_type)
        pass

    def this_cv_this_epoch_train_result(self, wanted):
        multi_cv_type = []
        [multi_cv_type.append(self.train_epoch_result_collection[i][-1][wanted][-1])
         for i in self.train_epoch_result_collection.keys()]
        return np.mean(multi_cv_type)
        pass

    def total_train(
            self,
            model,
            cv_collection,
            index_to_data,
            epoch,
            val_epoch_step,
            batch,
            **options
    ):
        pass

    pass
