# coding=utf-8
import Putil.data.ilsvrc_data as ilsvrcd
from multiprocessing import cpu_count


def test_statistic():
    ilsvrc_train_root = '/data/Public_Data/ILSVRC2012/train'
    save_to = 'test/test_generation/data/ilsvrc_data/statistic'
    statistic_sample = 'test_sample.csv'
    statistic_label = 'test_label.csv'
    run_time_message = 'test_run_time_message.txt'
    process_amount = cpu_count()
    ilsvrcd.ILSVRC.ilsvrc_statistic(ilsvrc_train_root, save_to, statistic_sample, statistic_label, run_time_message, process_amount)
    pass


def test_ilsvrc():
    ilsvrcd.ILSVRC(
    pass


if __name__ == '__main__':
    pass
