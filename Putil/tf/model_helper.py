# coding=utf-8
from colorama import Fore
import tensorflow as tf
import sys
import Putil.loger as plog
import os

root_logger = plog.PutilLogConfig("ModelHelper").logger()
root_logger.setLevel(plog.DEBUG)
SaveGraphAndPauseLogger = root_logger.getChild("SaveGraphAndPause")
SaveGraphAndPauseLogger.setLevel(plog.DEBUG)
GPUOrCPULogger = root_logger.getChild("GPUOrCPU")
GPUOrCPULogger.setLevel(plog.DEBUG)
SavePathProcessLogger = root_logger.getChild("SavePathProcess")
SavePathProcessLogger.setLevel(plog.DEBUG)


# : after build model with loss successful to save a graph and ask to continue run or not
def save_graph_and_pause(
        summary_save_path=None
):
    if os.path.exists(summary_save_path):
        pass
    else:
        os.makedirs(summary_save_path)
    if (summary_save_path is not None):
        writer = tf.summary.FileWriter(summary_save_path, tf.Session().graph)
        # ask for continue or not
        while True:
            go_run = input(Fore.RED +
                           'until now the model has been built and Arch has been saved ,would you want to '
                           'continue the next operates(y/n) : ')
            if go_run == 'y' or go_run == 'Y':
                return writer
            elif go_run == 'n' or go_run == 'N':
                writer.close()
                sys.exit()
                pass
            else:
                SaveGraphAndPauseLogger.info(Fore.RED + 'input error')
                pass
            pass
        pass
    pass


def GPUorCPU(cpu):
    if cpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        GPUOrCPULogger.info(Fore.GREEN + 'use cpu')
        pass
    else:
        GPUOrCPULogger.info(Fore.GREEN + 'use gpu')
        pass
    pass


def SavePathProcess(base_summary_path='./'):
    import time
    import os
    date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    summary_path = os.path.join(os.path.join(base_summary_path, date), 'summary')
    SaveGraphAndPauseLogger.info(Fore.GREEN + 'summary path : {0}'.format(summary_path))
    weight_path = os.path.join(os.path.join(base_summary_path, date), 'weight')
    SaveGraphAndPauseLogger.info(Fore.GREEN + 'weight path : {0}'.format(weight_path))
    return summary_path, weight_path
    pass


