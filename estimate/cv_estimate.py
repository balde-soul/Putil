# coding=utf-8

import matplotlib.pyplot as plt
import os
from optparse import OptionParser
import numpy as np
import visual.matplotlib_plot as pltt
from colorama import Fore

parser = OptionParser(usage="usage:%prog [options] arg1 arg2")

parser.add_option(
    '--tmece',
    '--test_mutual_exclusion_cv_estimate',
    action='store_true',
    default=False,
    dest='TestMutualExclusionCvEstimate',
    help='set it for test function: mutual_exclusion_cv_estimate '
)


# todo: estimate for mutual exclusion
def mutual_exclusion_cv_estimate(train, val, result_save, **options):
    """
    
    :param train: list mean len(train) cv estimate, every item in train is dict={'loss':, 'step':}
    :param val: list mean len(val) cv estimate, every item in train is dict={'loss':, 'step':}
    :param result_save:
    :return: 
    """
    save_flag_prefix = options.pop('prefix', '')
    save_flag_suffix = options.pop('suffix', '')
    print(Fore.GREEN + 'use prefix: ' + save_flag_prefix) \
        if save_flag_prefix != '' else print(Fore.REd + 'not use prefix')
    print(Fore.GREEN + 'use suffix: ' + save_flag_suffix) \
        if save_flag_suffix != '' else print(Fore.RED + 'not use suffix')
    as_step = options.pop('as_step', None)
    if as_step is None:
        as_step = 'step'
        pass
    print(Fore.GREEN + 'set {0} as step'.format(as_step))
    for k in train:
        assert as_step in k.keys(), print(Fore.RED + 'key {0} is not in this train').format(as_step)
        pass
    for k in val:
        assert as_step in k.keys(), print(Fore.RED + 'key {0} is not in this val').format(as_step)
    wanted = options.pop('plot_choice', None)
    if wanted is None:
        wanted = list()
        [[wanted.append(j) for j in i.keys()] for i in train]
        wanted.remove('step')
        [[wanted.append(j) for j in i.keys()] for i in val]
        wanted.remove('step')
    if os.path.exists(result_save):
        pass
    else:
        os.makedirs(result_save)
        pass
    cv_count = 0
    _pltt = pltt.random_type()
    for cv in zip(train, val):
        support = list(cv[0].keys())
        support.remove(as_step)
        for name in support:
            if name in wanted:
                train_type = _pltt.type_gen()
                while _pltt.no_repeat(train_type) is False:
                    train_type = _pltt.type_gen()
                    pass
                plt.plot(cv[0][as_step], cv[0][name], train_type, label="$train-{0}$".format(name))
                pass
            else:
                pass
            pass
        support = list(cv[1].keys())
        support.remove(as_step)
        for name in support:
            if name in wanted:
                val_type = _pltt.type_gen()
                while _pltt.no_repeat(val_type) is False:
                    val_type = _pltt.type_gen()
                    pass
                plt.plot(cv[1][as_step], cv[1][name], val_type, label="$val-{0}$".format(name))
            else:
                pass
            pass
        plt.title('estimate for cv : {0}'.format(cv_count))
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(
            os.path.join(
                result_save,
                '{1}_cv_{0}_estimate_{2}.png'.format(
                    cv_count, save_flag_prefix, save_flag_suffix)))
        cv_count += 1
        plt.close()
    pass


def __test_mutual_exclusion_cv_estimate():
    """
    test
    :return: 
    """
    train = list()
    train_dict = dict()
    train_dict['loss'] = np.random.random_sample(100) * 2
    train_dict['score'] = np.random.random_sample(100) * 3
    train_dict['step'] = np.linspace(1, 100, 100)
    train.append(train_dict)
    train.append(train_dict)
    val = list()
    val_dict = dict()
    val_dict['loss'] = np.random.random_sample(20)
    val_dict['step'] = np.linspace(5, 100, 20)
    val.append(val_dict)
    val.append(val_dict)
    mutual_exclusion_cv_estimate(train, val, result_save='D:/test/')
    pass


def __test(**options):
    """
    test
    :param options: 
    :return: 
    """
    if options.pop('TestMutualExclusionCvEstimate'):
        __test_mutual_exclusion_cv_estimate()
        pass


if __name__ == '__main__':
    (options, args) = parser.parse_args()
    if options.TestMutualExclusionCvEstimate:
        TestMutualExclusionCvEstimate = True
        pass
    else:
        TestMutualExclusionCvEstimate = False
        pass
    __test(
        TestMutualExclusionCvEstimate=True
    )
    pass
