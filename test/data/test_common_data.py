# coding=utf-8
import Putil.base.logger as plog

plog.PutilLogConfig.config_log_level(stream=plog.DEBUG)
plog.PutilLogConfig.config_handler(plog.stream_method)
logger = plog.PutilLogConfig('TesCommonData').logger()
logger.setLevel(plog.DEBUG)
MainLogger = logger.getChild('Main')
MainLogger.setLevel(plog.DEBUG)

import Putil.data.common_data as pcd
import multiprocessing


class TestCommonData(pcd.CommonData):
    def __init__(self):
        pcd.CommonData.__init__(self)
        self._field = list(range(0, 100))
        pass

    def _restart_process(self, restart_param):
        self._field = list(range(0, 100))
        pass

    def _generate_from_one_sample(self):
        target = self._field.pop()
        return target
        pass

    def _status_update(self):
        self._epoch_done = True if len(self._field) == 0 else False
        pass
    pass


pcd.CommonDataManager.register('TestCommonData', TestCommonData, proxytype=pcd.CommonDataProxy)


if __name__ == '__main__':
    manager_common_data = pcd.CommonDataManager()
    manager_common_data.start()
    data = manager_common_data.TestCommonData()

    manager = multiprocessing.Manager()
    pool = multiprocessing.Pool()

    dpq = pcd.DataPutProcess(data, manager, pool)
    pool.close()

    dq = dpq.DataQueue

    # pool.join()
    # print(dpq.queue_process_ret.get())

    restart_param = dict()
    dpq.restart(**restart_param)
    count = 0
    while dpq.has_next:
        # print('s')
        dq.get()
        count += 1
        pass
    assert count == 100
    dpq.restart(**restart_param)
    count = 0
    while dpq.has_next:
        dq.get()
        count += 1
        pass
    assert count == 100
    dpq.stop_generation()
    pool.join()
    print(dpq.queue_process_ret.get())
    # while dq.empty() is False or dpq.EpochDoneFlag.value is False:
    #     print('get')
    #     print(dq.get())
    pass
