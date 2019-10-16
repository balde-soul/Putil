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
        self._data_field = list(range(0, 100))
        pass

    def _restart_process(self, restart_param):
        pass

    def _generate_from_specified(self, index):
        data = self._data_field[index]
        return np.array([[data]])
        pass

    def _data_set_field(self):
        return list(range(0, 100))
        pass

    def _inject_operation(self, inject_param):
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
            assert get[0].datas().shape == (1, 1)
            assert get[0].indexs().shape == (1,)
            assert get[0].indexs()[0].index_info().type() == 'normal'
            assert get[0].indexs()[0].data_range() == [0, 1]
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
            assert get[0].datas().shape == (11, 1)
            assert get[0].indexs().shape == (11,)
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
            assert get[0].datas().shape == (3, 1)
            assert get[1].datas().shape == (5, 1)
            assert get[0].indexs().shape == (3,)
            assert get[1].indexs().shape == (5,)
            if count == 12:
                assert get[0].indexs()[2].index_info().type() == 'random_fill'
                assert get[1].indexs()[3].index_info().type() == 'random_fill'
                assert get[1].indexs()[4].index_info().type() == 'random_fill'
                pass
            else:
                assert get[0].indexs()[-1].index_info().type() == 'normal', print(get[0].indexs()[-1].index_info().type(), count)
                assert get[1].indexs()[-1].index_info().type() == 'normal'
                pass
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
            if count == 33:
                assert get[0].indexs()[-1].index_info().type() == 'normal', print(get[0].indexs()[-1].index_info().type())
                assert get[0].datas().shape == (1, 1), print(get[0].datas().shape)
            count += 1
            pass
        assert count == 34

        restart_param['device_batch'] = [1, 2]
        restart_param['critical_process'] = 'allow_low'

        data.restart_data(restart_param)

        assert data.generate_epoch_done() is False
        count = 0
        while data.generate_epoch_done() is False:
            get = data.generate_data()
            assert len(get) == 2
            if count == 33:
                assert get[0].datas().shape == (1, 1), print(get[0].datas().shape)
                assert get[1].datas().shape == (1, 1), print(get[1].datas().shape)
                assert get[0].indexs()[0].index_info().type() == 'normal', print(get[0].indexs()[0].index_info().type())
                assert get[1].indexs()[0].index_info().type() == 'allow_low', print(get[1].indexs()[0].index_info().type())
                pass
            else:
                assert get[0].datas().shape == (1, 1), print(get[0].datas().shape)
                assert get[1].datas().shape == (2, 1), print(get[1].datas().shape)
            count += 1
            pass
        assert count == 34
