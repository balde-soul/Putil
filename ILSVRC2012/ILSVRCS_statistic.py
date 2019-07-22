# coding=utf-8
import os
import sys
# === import project path ===
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
# ===========================
import base.default_excutable_argument as dea
import pandas as pd
from PIL import Image
import os
import copy
import queue
import threading
import multiprocessing
from multiprocessing import Manager
from multiprocessing import Pool
from multiprocessing import cpu_count
multiprocessing.freeze_support()

ILSVRC_train_root = '/data/ILSVRC2012/train'
save_to = '/data/process_data/caojihua/ILSVRC/'

#   collect information into a PD
statistic_sample = 'statistic_sample.csv'
statistic_label = 'statistic_label.csv'
run_time_message = 'statistic_info.txt'
process_amount = cpu_count()

argument = dea.Argument()
parser = argument.parser

parser.add_argument(
    '--train_root',
    action='store',
    dest='TrainRoot',
    type=str,
    default='',
    help='if this flag is set, the program just test train in perform'
)

args = parser.parse_args()

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


def deal_with_class(classes, ifp_queue):
    df_for_label = pd.DataFrame(columns=['class_dir', 'reflect_name'])
    df_for_sample = pd.DataFrame(columns=['class', 'image_name', 'height', 'width', 'channel'])
    l = 0
    while l < len(classes):
        class_element = classes[l]
        try:
            print('deal with {0}'.format(class_element))

            df_for_label = df_for_label.append({'class_dir': class_element, 'reflect_name': class_element},
                                                  ignore_index=True)

            class_dir = os.path.join(ILSVRC_train_root, class_element)
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
        l += 1
        print('pod: {0}, deal {1}, remain: {2}'.format(os.getpid(), l, len(classes) - l))
        pass
    print('done:{0}'.format(classes))
    return df_for_sample, df_for_label

def deal_with_ilsvrc(info_save_to, sample_save_to, label_save_to):

    global process_amount

    class_list = os.listdir(ILSVRC_train_root)

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
            process = pool.apply_async(deal_with_class, args=(scl, ifp_queue))
            # process.start()
            process_list.append(process)
            pass
        pool.close()
        pool.join()
        pass

    sample_pd_collection = []
    label_pd_collection = []
    for pl in process_list:
        s, l = pl.get()
        sample_pd_collection.append(s)
        label_pd_collection.append(l)
        pass

    label_pd = pd.concat(label_pd_collection, ignore_index=True)
    sample_pd = pd.concat(sample_pd_collection, ignore_index=True)

    label_pd.to_csv(label_save_to)
    sample_pd.to_csv(sample_save_to)
    pass

if __name__ == '__main__':
    deal_with_ilsvrc(os.path.join(save_to, run_time_message), os.path.join(save_to, statistic_sample),
                     os.path.join(save_to, statistic_label))

