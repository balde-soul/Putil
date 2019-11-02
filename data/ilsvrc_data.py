# coding=utf-8
import sys
import cv2
from PIL import Image
import threading
import pandas as pd
import numpy as np
import os
import copy
import queue
import multiprocessing
from multiprocessing import Manager
from multiprocessing import Pool
from multiprocessing import cpu_count
multiprocessing.freeze_support()
import Putil.data.common_data as pcd
import Putil.base.logger as plog

ILSVRCDataLogger = plog.PutilLogConfig('ILSVRCData').logger()
ILSVRCDataLogger.setLevel(plog.DEBUG)
ILSVRCLogger = ILSVRCDataLogger.getChild('ILSVRC')
ILSVRCLogger.setLevel(plog.DEBUG)
ILSVRCStatisticLogger = ILSVRCLogger.getChild('ILSVRCStatistic')
ILSVRCStatisticLogger.setLevel(plog.DEBUG)


def ifp_listening(ifp, queue):
    while True:
        msg = queue.get()
        if msg == 'end':
            ifp.write('killed')
            break
        ifp.write(msg)
        ifp.flush()
        pass
    ifp.close()
    pass


def ifp_write(queue, msg):
    queue.put(msg)
    pass


def read_image_information(class_dir, sample_list, image_info_queue, ifp_queue):
    for sample_element in sample_list:
        sample_dir = os.path.join(class_dir, sample_element)
        try:
            im = Image.open(sample_dir)
            width, height = im.size
            channel = im.layers
            image_info_queue.put(
                [False, {'image_name': sample_element, 'height': height, 'width': width, 'channel': channel}])
            del im
        except Exception as ex:
            ifp_write(ifp_queue, '{0} failed {1}\n'.format(sample_dir, ex.args))
            pass
        pass
    image_info_queue.put([True, {}])
    pass


def deal_with_class(ilsvrc_train_root, classes, ifp_queue):
    df_for_label = pd.DataFrame(columns=['class_dir', 'reflect_name'])
    df_for_sample = pd.DataFrame(columns=['class', 'image_name', 'height', 'width', 'channel'])
    cla = 0
    while cla < len(classes):
        class_element = classes[cla]
        try:
            print('deal with {0}'.format(class_element))
            df_for_label = df_for_label.append({'class_dir': class_element, 'reflect_name': class_element}, ignore_index=True)
            class_dir = os.path.join(ilsvrc_train_root, class_element)
            sample_list = os.listdir(class_dir)
            # add to queue
            image_info_queue = queue.Queue()
            read_thread = threading.Thread(target=read_image_information,
                                           args=(class_dir, sample_list, image_info_queue, ifp_queue))
            read_thread.start()
            base_dict = {'class': class_element}
            sample_ = list()
            while True:
                element = image_info_queue.get()
                if element[0] is False:
                    base_dict.update(element[1])
                    sample_.append(copy.deepcopy(base_dict))
                    pass
                else:
                    break
                    pass
                pass

            read_thread.join()
            df_for_sample = df_for_sample.append(sample_, ignore_index=True)
            del sample_
            pass
        except Exception as ex:
            ifp_write(ifp_queue, '{0}\n'.format(ex.args))
            pass
        cla += 1
        print('pod: {0}, deal {1}, remain: {2}'.format(os.getpid(), cla, len(classes) - cla))
        pass
    print('done:{0}'.format(classes))
    return df_for_sample, df_for_label


def deal_with_ilsvrc(ilsvrc_train_root, info_save_to, sample_save_to, label_save_to, process_amount):
    class_list = os.listdir(ilsvrc_train_root)
    # seperate class_list to process_amount parts
    seperate_class_list = []
    if process_amount > len(class_list):
        process_amount = len(class_list)
    else:
        pass
    base_len = len(class_list) // process_amount
    end_len = len(class_list) % process_amount + base_len
    start = 0
    length = base_len
    for i in range(0, process_amount):
        seperate_class_list.append(class_list[start: length])
        start = start + base_len
        if i != process_amount - 2:
            length = start + base_len
            pass
        else:
            length = start + end_len
    assert(sum([len(i) for i in seperate_class_list]) == len(class_list))

    ifp_queue = Manager().Queue()
    process_list = []
    pool = Pool(processes=process_amount)

    with open(info_save_to, 'w') as ifp:
        pool.apply_async(ifp_listening, args=(ifp, ifp_queue))
        for scl in seperate_class_list:
            # process = pool.apply_async(test, args=(1,))
            process = pool.apply_async(deal_with_class, args=(ilsvrc_train_root, scl, ifp_queue))
            # process.start()
            process_list.append(process)
            pass
        pool.close()
        pool.join()
        pass

    [ILSVRCStatisticLogger.info(process.get()) for process in process_list]

    sample_pd_collection = []
    label_pd_collection = []
    for pl in process_list:
        s, la = pl.get()
        sample_pd_collection.append(s)
        label_pd_collection.append(la)
        pass

    label_pd = pd.concat(label_pd_collection, ignore_index=True)
    sample_pd = pd.concat(sample_pd_collection, ignore_index=True)

    label_pd.to_csv(label_save_to)
    sample_pd.to_csv(sample_save_to)
    pass


class ILSVRC(pcd.CommonData):
    def __init__(self, statistic_file, information_save_to='', load_truncate=None, k_fold=[0.9, 0.1], subset_class_amount=1000, data_drop_rate=0.0, **kwargs):
        '''
        :param statistic_file: the statistic file which generate by ILSVRC_statistic
        :param load_truncate: load a part of the statistic file
        :param split: the split of the train and evaluate: split for train
        :param subset_class_amount: the amount of the target class which use to ge the sub dataset base on the class
        :param data_drop_rate: the rate of the sample which would be dropped in every class , used to get the sub dataset base on the sample amount
        :param device_batch: batch size in every device
        '''
        pcd.CommonData.__init__(self)
        # a file which contain all the sample of the ImageNet
        self._statistic_file = statistic_file
        ILSVRCLogger.info(self._statistic_file)
        if os.path.exists(self._statistic_file):
            pass
        else:
            path_split = os.path.split(self._statistic_file)
            save_to = path_split[0]
            statistic_sample = path_split[1]
            statistic_label = 'statistic_label.csv'
            run_time_message = 'statistic_info.txt'
            process_amount = cpu_count()
            assert os.path.exists(save_to) is True, ILSVRCDataLogger.fatal('path : {0} does not exist'.format(save_to))
            ILSVRCDataLogger.info('statistic file: {0} does not exist, run statistic process, save statistic information to {1}'.format(self._statistic_file, save_to))
            ilsvrc_train_root = input('please input the ilsvrc_train root dir: ')
            ILSVRC.ilsvrc_statistic(ilsvrc_train_root, save_to, statistic_sample, statistic_label, run_time_message, process_amount)
            pass

        # the batch of the device list
        # self._device_batch = device_batch
        self._device_batch = [4]
        ILSVRCLogger.info(self._device_batch)

        # the subset of the total dataset, represent the class amount, which should be smaller than 1000
        assert(subset_class_amount <= 1000), print('subset amount should be lower than 1000')
        assert(isinstance(subset_class_amount, int)), print('subset amount should be a int')
        assert max(k_fold) < 1.0, ILSVRCDataLogger.fatal('max of k_fold should be smaller than 1.0 vs. {0}'.format(k_fold))
        assert min(k_fold) > 0.0, ILSVRCDataLogger.fatal('max of k_fold should be larger an 0.0 vs. {0}'.format(k_fold))
        assert sum(k_fold) == 1.0, ILSVRCDataLogger.fatal('sum of k_fold should be equal to 1.0 vs. {0}'.format(k_fold))
        self._subset = subset_class_amount
        ILSVRCLogger.info('subset: {0}'.format(self._subset))
        # the split for the train
        self._k_fold = k_fold
        ILSVRCLogger.info('split: {0}'.format(self._split))
        self._data_drop_rate = data_drop_rate
        ILSVRCLogger.info('data drop rate: {0}'.format(self._data_drop_rate))

        # : load the statistic file
        _dpd = pd.read_csv(self._statistic_file, nrows=load_truncate) if load_truncate is not None else pd.read_csv(
            self._statistic_file)
        _dpd = _dpd.drop(columns=['Unnamed: 0'])
        ILSVRCLogger.info('total sample amount: {0}'.format(_dpd.shape[0]))

        # : deal with the subset
        process = _dpd.drop_duplicates(subset='class', keep='first')
        classes = np.array(process['class'])
        np.random.shuffle(classes)

        # collection the class name
        subset_class_name_collection = classes[0: self._subset]
        ILSVRCLogger.info('sub set class : \n{0}'.format(subset_class_name_collection))
        self._class_reflect = dict()
        [self._class_reflect.update({class_name: np.where(subset_class_name_collection == class_name)[0][0]}) for
         class_name in subset_class_name_collection]
        ILSVRCLogger.info('name class reflect: \n{0}'.format(self._class_reflect))

        self._subset_df = _dpd[_dpd['class'].isin(subset_class_name_collection)]
        ILSVRCLogger.info('sub set class amount: {0}'.format(self._subset_df.index.size))
        del _dpd

        # : deal with the train valuate split
        self._fold_df = []
        for sub_class in self._class_reflect.items():
            # get the target class data
            target_class = self._subset_df[self._subset_df['class'] == sub_class[0]]
            ILSVRCLogger.debug('amount of class {0}: {1}'.format(sub_class[0], target_class.index.size))
            target_class_index = np.array(target_class.index)
            np.random.shuffle(target_class_index)
            # drop sample
            target_class_index = target_class_index[0: np.floor(target_class_index.size * (1.0 - self._data_drop_rate)).astype('int64')]
            # calc the size of every fold
            k_fold_step_amount = np.floor(np.multiply(self._k_fold, target_class_index.size)).astype(np.int64)
            if len(self._fold_df) != self._k_fold:
                for ele in range(0, self._k_fold):
                    target_index = target_class_index[sum(k_fold_step_amount[0: ele]): sum(k_fold_step_amount[0: ele]) + k_fold_step_amount[ele]]
                    self._fold_df.append(self._subset_df.loc[target_index])
                    pass
                pass
            else:
                for ele in range(0, self._k_fold):
                    target_index = target_class_index[sum(k_fold_step_amount[0: ele]): sum(k_fold_step_amount[0: ele]) + k_fold_step_amount[ele]]
                    self._fold_df[ele].append(self._subset_df.loc[target_index])
                    pass
                pass
            pass
        del self._subset_df

        self._train_field = list(self._train_df.index)
        np.random.shuffle(self._train_field)
        assert len(self._train_field) != 0, print('train field is zero')
        ILSVRCLogger.info('train field: {0}'.format(len(self._train_field)))
        self._evaluate_field = list(self._evaluate_df.index)
        np.random.shuffle(self._evaluate_field)
        assert len(self._evaluate_field) != 0, print('evaluate field is zero')
        ILSVRCLogger.info('evaluate field: {0}'.format(len(self._evaluate_field)))

        # the train and evaluate reader function : set the class amount
        # self._train_reader = read_train_image_label_func(len(self._class_reflect.items()))
        # self._evaluate_reader = read_evaluate_image_label_func(len(self._class_reflect.items()))

        # self._train_reader = DefaultTrainReader()
        # self._evaluate_reader = DefaultEvaluateReader()

        '''
        the second type function
        '''
        self._data_fold = None
        self._data_field = None
        self._ilvrc_root = None  # str
        self._target_type_is_train = None   # bool
        self._data_reset_mutex = threading.Condition()
        pass

    def data_fold(self):
        pass

    def __restart_data(self, data_fold, ilsvrc_root, shuffle):
        self._data_reset_mutex.acquire()
        self._data_fold = data_fold
        self._ilvrc_root = ilsvrc_root
        self._data_field = list()
        for fold in data_fold:
            self._data_field += list(self._fold_df[fold].index)
            pass
        np.random.shuffle(self._data_field) if shuffle is True else None
        self._data_reset_mutex.notify_all()
        self._data_reset_mutex.release()
        pass

    def _restart_process(self, restart_param):
        assert 'data_fold' in restart_param
        assert 'ilsvrc_root' in restart_param
        assert 'shuffle' in restart_param
        data_fold = restart_param['data_fold']
        ilsvrc_root = restart_param['ilsvrc_root']
        shuffle = restart_param['shuffle'].value
        size = restart_param['size'].value
        self.__restart_data(data_fold, ilsvrc_root, shuffle)
        pass

    def __read_data(self, index):
        element = self._data_field.loc[index]

        # get the data
        class_name = element['class']
        image_name = element['image_name']
        image_path = os.path.join(os.path.join(self._ilvrc_root, class_name), image_name)
        class_id = self._class_reflect[class_name]
        class_amount = self._subset

        img = cv2.imread(image_path)
        if img is None:
            ILSVRCLogger.error('image read : {0} failed'.format(image_path))
            sys.exit()
            pass
        img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pass
        image = np.asanyarray(img.astype('float32'))
        # normalization
        # sum = np.sum(img, axis=-1, keepdims=True, dtype=np.float32)
        # image = image / (sum + 1e-32)
        image = image / 255.0
        label = np.zeros(shape=[class_amount], dtype=np.float32)
        label[class_id] = 1.0
        return np.array([image]), np.array([label])
        pass

    def _generate_from_one_sample(self):
        self._generate_from_specified(self._index)
        self._index += 1
        pass

    def _generate_from_specified(self, index):
        data = self.__read_data(index)
        return data
        pass

    def _data_set_field(self):
        return self._data_field
        pass

    def _status_update(self):
        self._epoch_done = True if self._index == len(self._data_field) else False
        pass

    @staticmethod
    def ilsvrc_statistic(ilsvrc_train_root, save_to, statistic_sample, statistic_label, run_time_message, process_amount):
        deal_with_ilsvrc(ilsvrc_train_root, os.path.join(save_to, run_time_message), os.path.join(save_to, statistic_sample), os.path.join(save_to, statistic_label), process_amount)
        pass
    pass


pcd.CommonDataManager.register('ILSVRC', ILSVRC)
