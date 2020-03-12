import Putil.path
import os


def test_touch_dir():
    one_step = os.path.split(os.path.realpath(__file__))[0]
    one_step = os.path.join(one_step, 'test_touch_dir')
    two_step = os.path.join(one_step, 'secode_step')
    assert os.path.exists(one_step) is False, print('waiting for complete')
    Putil.path.touch_dir(one_step)
    assert os.path.exists(one_step) is True, print('waiting for complete')
    os.rmdir(one_step)
    assert os.path.exists(two_step) is False, print('waiting for complete')
    Putil.path.touch_dir(two_step)
    assert os.path.exists(two_step) is True, print('waiting for complete')
    os.rmdir(two_step)
    os.rmdir(one_step)

    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    one_step = 'test_touch_dir_with_relative'
    two_step = os.path.join(one_step, 'secode_step')
    assert os.path.exists(one_step) is False, print('waiting for complete')
    Putil.path.touch_dir(one_step)
    assert os.path.exists(one_step) is True, print('waiting for complete')
    os.rmdir(one_step)
    assert os.path.exists(two_step) is False, print('waiting for complete')
    Putil.path.touch_dir(two_step)
    assert os.path.exists(two_step) is True, print('waiting for complete')
    os.rmdir(two_step)
    os.rmdir(one_step)

    one_step = './test_touch_dir_with_relative'
    two_step = os.path.join(one_step, 'secode_step')
    assert os.path.exists(one_step) is False, print('waiting for complete')
    Putil.path.touch_dir(one_step)
    assert os.path.exists(one_step) is True, print('waiting for complete')
    os.rmdir(one_step)
    assert os.path.exists(two_step) is False, print('waiting for complete')
    Putil.path.touch_dir(two_step)
    assert os.path.exists(two_step) is True, print('waiting for complete')
    os.rmdir(two_step)
    os.rmdir(one_step)

    for_executa_dir = './for_parent'
    assert os.path.exists(one_step) is False, print('waiting for complete')
    Putil.path.touch_dir(for_executa_dir)
    os.chdir(for_executa_dir)
    one_step = '../test_touch_dir_with_relative_father_fold'
    two_step = os.path.join(one_step, 'secode_step')
    assert os.path.exists(one_step) is False, print('waiting for complete')
    Putil.path.touch_dir(one_step)
    assert os.path.exists(one_step) is True, print('waiting for complete')
    os.rmdir(one_step)
    assert os.path.exists(two_step) is False, print('waiting for complete')
    Putil.path.touch_dir(two_step)
    assert os.path.exists(two_step) is True, print('waiting for complete')
    os.rmdir(two_step)
    os.rmdir(one_step)
    os.chdir('../')
    os.rmdir(for_executa_dir)
    pass


if __name__ == '__main__':
    test_touch_dir()
    pass