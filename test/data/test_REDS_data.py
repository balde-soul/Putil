# coding=utf-8
import Putil.data.REDS_data as REDSD

redsd = REDSD.REDSData(root='/data2/process_data/caojihua/data/VSR/REDS/')

restart_param = dict()
test_time = 10
for i in range(0, test_time):
    restart_param['device_batch'] = [1]
    restart_param['critical_process'] = 'random_fill'

    restart_param['generate_type'] = 'sequence'
    restart_param['data_type'] = 'train'
    restart_param['shuffle'] = True
    restart_param['sequence_len'] = 2
    restart_param['data_name'] = 'train_sharp'

    redsd.restart_data(restart_param)

    assert redsd.generate_epoch_done() is False
    count = 0
    while redsd.generate_epoch_done() is False:
        got_data = redsd.generate_data()
        assert len(got_data) == 1, print(len(got_data))
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

    redsd.restart_redsd(restart_param)

    assert redsd.generate_epoch_done() is False
    count = 0
    while redsd.generate_epoch_done() is False:
        got_redsd = redsd.generate_redsd()
        assert len(got_redsd) == 1
        for d in got_redsd:
            for k, v in d.items():
                get = v
                assert get.redsds().shape == (11, 1)
                assert get.indexs().shape == (11,)
        count += 1
        pass
    assert count == 10

    restart_param['device_batch'] = [3, 5]
    restart_param['critical_process'] = 'random_fill'

    redsd.restart_redsd(restart_param)

    assert redsd.generate_epoch_done() is False
    count = 0
    while redsd.generate_epoch_done() is False:
        got_redsd = redsd.generate_redsd()
        assert len(got_redsd) == 2
        for index, d in enumerate(got_redsd):
            for k, v in d.items():
                get = v
                assert get.redsds().shape == (restart_param['device_batch'][index], 1), print(get.redsds().shape)
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

    redsd.restart_redsd(restart_param)

    assert redsd.generate_epoch_done() is False
    count = 0
    while redsd.generate_epoch_done() is False:
        got_redsd = redsd.generate_redsd()
        assert len(got_redsd) == 1
        if count == 33:
            for index, d in enumerate(got_redsd):
                for k, v in d.items():
                    get = v
                    assert get.indexs()[-1].index_info().type() == 'normal', print(get.indexs()[-1].index_info().type())
                    assert get.redsds().shape == (1, 1), print(get[0].redsds().shape)
        count += 1
        pass
    assert count == 34

    restart_param['device_batch'] = [1, 2]
    restart_param['critical_process'] = 'allow_low'

    redsd.restart_redsd(restart_param)

    assert redsd.generate_epoch_done() is False
    count = 0
    while redsd.generate_epoch_done() is False:
        got_redsd = redsd.generate_redsd()
        assert len(got_redsd) == 2
        if count == 33:
            for index, d in enumerate(got_redsd):
                for k, v in d.items():
                    get = v
                    assert get.redsds().shape == (1, 1), print(get.redsds().shape)
                    assert get.redsds().shape == (1, 1), print(get.redsds().shape)
                    assert get.indexs()[0].index_info().type() == 'normal' if index == 0 else 'allow_low', print(get.indexs()[0].index_info().type())
                    pass
                pass
            pass
        else:
            for index, d in enumerate(got_redsd):
                for k, v in d.items():
                    get = v
                    assert get.redsds().shape == (1 if index == 0 else 2, 1), print(get.redsds().shape)
                    pass
                pass
            pass
        count += 1
        pass
    assert count == 34
    pass
pass