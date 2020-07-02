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
        return {'data': np.array([data]), 'label': np.array([data])}

    def _data_set_field(self):
        return list(range(0, 100))

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
            got_data = data.generate_data()
            assert len(got_data) == 1, print(len(data))
            for d in got_data:
                for k, v in d.items():
                    get = v
                    assert get.datas().shape == (1, 1)
                    assert get.indexs().shape == (1,)
                    assert get.indexs()[0].index_info().type() == 'normal'
                    assert get.indexs()[0].data_range() == [0, 1]
            count += 1
            pass
        assert count == 100

        restart_param['device_batch'] = [11]
        restart_param['critical_process'] = 'random_fill'

        data.restart_data(restart_param)

        assert data.generate_epoch_done() is False
        count = 0
        while data.generate_epoch_done() is False:
            got_data = data.generate_data()
            assert len(got_data) == 1
            for d in got_data:
                for k, v in d.items():
                    get = v
                    assert get.datas().shape == (11, 1)
                    assert get.indexs().shape == (11,)
            count += 1
            pass
        assert count == 10

        restart_param['device_batch'] = [3, 5]
        restart_param['critical_process'] = 'random_fill'

        data.restart_data(restart_param)

        assert data.generate_epoch_done() is False
        count = 0
        while data.generate_epoch_done() is False:
            got_data = data.generate_data()
            assert len(got_data) == 2
            for index, d in enumerate(got_data):
                for k, v in d.items():
                    get = v
                    assert get.datas().shape == (restart_param['device_batch'][index], 1), print(get.datas().shape)
                    assert get.indexs().shape == (restart_param['device_batch'][index],)
                    if count == 12 and index == 1:
                        assert get.indexs()[2].index_info().type() == 'random_fill'
                        assert get.indexs()[3].index_info().type() == 'random_fill'
                        assert get.indexs()[4].index_info().type() == 'random_fill'
                        pass
                    elif count == 12 and index == 0:
                        assert get.indexs()[2].index_info().type() == 'random_fill'
                        pass
                    else:
                        assert get.indexs()[-1].index_info().type() == 'normal', print(get.indexs()[-1].index_info().type(), count)
                        assert get.indexs()[-1].index_info().type() == 'normal'
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
            got_data = data.generate_data()
            assert len(got_data) == 1
            if count == 33:
                for index, d in enumerate(got_data):
                    for k, v in d.items():
                        get = v
                        assert get.indexs()[-1].index_info().type() == 'normal', print(get.indexs()[-1].index_info().type())
                        assert get.datas().shape == (1, 1), print(get[0].datas().shape)
            count += 1
            pass
        assert count == 34

        restart_param['device_batch'] = [1, 2]
        restart_param['critical_process'] = 'allow_low'

        data.restart_data(restart_param)

        assert data.generate_epoch_done() is False
        count = 0
        while data.generate_epoch_done() is False:
            got_data = data.generate_data()
            assert len(got_data) == 2
            if count == 33:
                for index, d in enumerate(got_data):
                    for k, v in d.items():
                        get = v
                        assert get.datas().shape == (1, 1), print(get.datas().shape)
                        assert get.datas().shape == (1, 1), print(get.datas().shape)
                        assert get.indexs()[0].index_info().type() == 'normal' if index == 0 else 'allow_low', print(get.indexs()[0].index_info().type())
                        pass
                    pass
                pass
            else:
                for index, d in enumerate(got_data):
                    for k, v in d.items():
                        get = v
                        assert get.datas().shape == (1 if index == 0 else 2, 1), print(get.datas().shape)
                        pass
                    pass
                pass
            count += 1
            pass
        assert count == 34
