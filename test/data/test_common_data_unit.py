# coding=utf-8
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


class TestCommonData(pcd.CommonData):
    def __init__(self):
        pcd.CommonData.__init__(self)
        self._field = list(range(0, 100))
        self._index = 0
        pass

    def _restart_process(self, restart_param):
        self._index = 0
        pass

    def _generate_from_one_sample(self):
        data = self._field[self._index]
        self._index += 1
        return np.array([[data]])
        pass

    def _generate_from_specified(self, index):
        data = self._field[index]
        return np.array([[data]])
        pass

    def _data_set_field(self):
        return list(range(0, 100))
        pass

    def _status_update(self):
        self._epoch_done = True if self._index == len(self._field) else False
        pass
    pass


pcd.CommonDataManager.register('TestCommonData', TestCommonData)


if __name__ == '__main__':
    # manager_common_data = pcd.CommonDataManager()
    # manager_common_data.start()
    # data = manager_common_data.TestCommonData()

    data = TestCommonData()

    restart_param = dict()

    test_time = 10
    for i in range(0, test_time):
        restart_param['device_batch'] = [1]
        restart_param['critical_process'] = 'random_fill'

        data.restart_data(restart_param)

        assert data.generate_epoch_done() is False
        count = 0
        while data.generate_epoch_done() is False:
            get = data.generate_data()
            assert len(get) == 1
            assert get[0].shape == (1, 1)
            count += 1
            pass
        assert count == 100

        restart_param['device_batch'] = [11]
        restart_param['critical_process'] = 'random_fill'

        data.restart_data(restart_param)

        assert data.generate_epoch_done() is False
        count = 0
        while data.generate_epoch_done() is False:
            get = data.generate_data()
            assert len(get) == 1
            assert get[0].shape == (11, 1)
            count += 1
            pass
        assert count == 10

        restart_param['device_batch'] = [3, 5]
        restart_param['critical_process'] = 'random_fill'

        data.restart_data(restart_param)

        assert data.generate_epoch_done() is False
        count = 0
        while data.generate_epoch_done() is False:
            get = data.generate_data()
            assert len(get) == 2
            assert get[0].shape == (3, 1)
            assert get[1].shape == (5, 1)
            count += 1
            pass
        assert count == 13

        restart_param['device_batch'] = [3]
        restart_param['critical_process'] = 'allow_low'

        data.restart_data(restart_param)

        assert data.generate_epoch_done() is False
        count = 0
        while data.generate_epoch_done() is False:
            get = data.generate_data()
            assert len(get) == 1
            count += 1
            if count == 34:
                assert get[0].shape == (1, 1)
            pass
        assert count == 34

        restart_param['device_batch'] = [3, 5]
        restart_param['critical_process'] = 'allow_low'

        data.restart_data(restart_param)

        assert data.generate_epoch_done() is False
        count = 0
        while data.generate_epoch_done() is False:
            get = data.generate_data()
            assert len(get) == 2
            count += 1
            if count == 13:
                assert get[0].shape == (2, 1)
                assert get[1].shape == (2, 1)
            pass
        assert count == 13
