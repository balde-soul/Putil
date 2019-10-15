# coding=utf-8
import sys
import traceback
import numpy as np
import Putil.base.logger as plog

plog.PutilLogConfig.config_log_level(stream=plog.DEBUG)
plog.PutilLogConfig.config_handler(plog.stream_method)
logger = plog.PutilLogConfig('TesCommonData').logger()
logger.setLevel(plog.DEBUG)
MainLogger = logger.getChild('Main')
MainLogger.setLevel(plog.DEBUG)

import Putil.data.common_data as pcd
import multiprocessing
import Putil.test.data.test_common_data_unit as tbase


if __name__ == '__main__':
    manager_common_data = pcd.CommonDataManager()
    manager_common_data.start()
    data = manager_common_data.TestCommonData()

    manager = multiprocessing.Manager()
    pool = multiprocessing.Pool()

    dpq = pcd.DataPutProcess(data, manager, pool)
    pool.close()

    dq = dpq.DataQueue()

    restart_param = dict()

    restart_param['device_batch'] = [1]
    restart_param['critical_process'] = 'random_fill'
    dpq.restart(**restart_param)

    # pool.join()
    # print(dpq.queue_process_ret.get())

    count = 0
    while dpq.has_next:
        data = dq.get()
        assert len(data) == 1
        assert data[0].shape[0] == 1
        count += 1
        pass
    assert count == 100

    restart_param['device_batch'] = [1]
    restart_param['critical_process'] = 'random_fill'
    dpq.restart(**restart_param)
    count = 0
    while dpq.has_next:
        dq.get()
        count += 1
        pass
    assert count == 100

    dpq.stop_generation()
    pool.join()
    print(dpq.queue_process_ret().get())
    # while dq.empty() is False or dpq.EpochDoneFlag.value is False:
    #     print('get')
    #     print(dq.get())
    pass
