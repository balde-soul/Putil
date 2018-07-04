# coding = utf-8

import sklearn as skl
import sklearn.model_selection as sms
import numpy as np
from colorama import Fore

def check_mutual_exclusion_param_equal(*now, **params):
    """
    判断是否存在参数一致的单元
    :param now:目前类型中所有
    :param params:
    :return:
    """
    n_cross = params.pop('n_cross')
    for param_dict, index in zip(now, list(range(0, len(now)))):
        if n_cross == param_dict['n_cross']:
            return False, index
        else:
            pass
        pass
    return True, len(now)

def pack_param(all_param_dict, **options):
    """
    打包_type类型的参数集为tuple（顺序强制统一）
    :param all_param_dict:
    :param options:
    :return:
    """
    _type = options.pop('type')
    amount = len(all_param_dict[_type].keys())
    param_dict = all_param_dict[_type]
    param = list()
    [param.append(param_dict[i]) for i in range(0, amount)]
    return param

# 数据生成，随机打乱
# 可以反向索引
# 单源数据，多种类数据生成
class CrossSplit:
    """
    使用sklearn.model_selection
    n折互斥分割

    """
    def __init__(self, *arrays):
        """
        Split arrays or matrices into random train and test subsets

        Quick utility that wraps input validation and
        ``next(ShuffleSplit().split(X, y))`` and application to input data
        into a single call for splitting (and optionally subsampling) data in a
        oneliner.

        Read more in the :ref:`User Guide <cross_validation>`.

        Parameters
        ----------
        *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse
            matrices or pandas dataframes.

        test_size : float, int, None, optional
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. By default, the value is set to 0.25.
            The default will change in version 0.21. It will remain 0.25 only
            if ``train_size`` is unspecified, otherwise it will complement
            the specified ``train_size``.

        train_size : float, int, or None, default None
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        shuffle : boolean, optional (default=True)
            Whether or not to shuffle the data before splitting. If shuffle=False
            then stratify must be None.

        stratify : array-like or None (default is None)
            If not None, data is split in a stratified fashion, using this as
            the class labels.

        Returns
        -------
        splitting : list, length=2 * len(arrays)
            List containing train-test split of inputs.

            .. versionadded:: 0.16
                If the input is sparse, the output will be a
                ``scipy.sparse.csr_matrix``. Else, output type is the same as the
                input type.

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = np.arange(10).reshape((5, 2)), range(5)
        >>> X
        array([[0, 1],
               [2, 3],
               [4, 5],
               [6, 7],
               [8, 9]])
        >>> list(y)
        [0, 1, 2, 3, 4]

        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.33, random_state=42)
        ...
        >>> X_train
        array([[4, 5],
               [0, 1],
               [6, 7]])
        >>> y_train
        [2, 0, 3]
        >>> X_test
        array([[2, 3],
               [8, 9]])
        >>> y_test
        [1, 4]

        >>> train_test_split(y, shuffle=False)
        [[0, 1, 2], [3, 4]]
        """
        self._Element = len(arrays)
        if self._Element == 0:
            raise ValueError("At least one array required as input")
        assert False not in [type(i) == np.ndarray for i in arrays], \
            'element of arrays must be numpy.ndarray'
        self._Array = arrays
        self._Data = dict()
        self._Data['mutual_exclusion'] = dict()
        self._Param = dict()
        self._Param['mutual_exclusion'] = dict()
        pass

    @property
    def Array(self):
        return self._Array

    @property
    def Element(self):
        return self._Element

    @property
    def Data(self):
        return self._Data

    @property
    def Param(self):
        return self._Param

    def __mutual_exclusion(self, **options):
        """
        axis=0, 分割第一维度
        :param options:
        :return:
        """
        n_cross = options.pop('n_cross')
        new_index = options.pop('new_index', None)
        # generate new_index
        if new_index is None:
            new_index = len(self._Param['mutual_exclusion'].keys())
            pass
        # update: _Param collection
        self._Param['mutual_exclusion'][new_index] = dict()
        self._Param['mutual_exclusion'][new_index]['n_cross'] = n_cross
        print('>>>>>>>>>>>>cross split database use n_cross: ', n_cross, '>>>>>>>>>>>>')
        if options:
            raise TypeError("Invalid parameters passed: %s" % str(options))
        test_size = self._Array[0].shape[0] / n_cross
        assert np.ceil(test_size) == test_size, 'array data should be {0} cross exact division'.format(n_cross)
        remain = self.Array
        data = dict()
        temp = None
        for i in range(0, n_cross - 1):
            temp = sms.train_test_split(*remain, test_size=int(test_size))
            remain = list()
            data[i] = list()
            for j, index in zip(temp, range(0, len(temp))):
                if index % 2 == 0:
                    remain.append(j)
                else:
                    data[i].append(j)
                pass
            remain = tuple(remain)
        data[n_cross - 1] = list(remain)
        self._Data['mutual_exclusion'][new_index] = data
        return new_index
        pass

    #  数据分割统一调用函数，判断是否重新分割
    def __deal_data(self, func, flags, **options):
        if flags is True:
            func(**options)
        else:
            pass
        pass

    def gen_mutual_exclusion(self, **options):
        n_cross = options.pop('n_cross', 'default')
        #标志：是否强制新建分割数据
        force_re_split = options.pop('force_re_split', False)
        if n_cross == 'default':
            print('unspecified n_cross, use default: 5')
            n_cross = 5
            pass
        #   检查是否存在同等参数的数据分割，存在则不再进行分割，而是使用存在数已经分割的数据
        if force_re_split is True:
            flags = True
            index = None
            self.__deal_data(self.__mutual_exclusion(), flags=flags, n_cross=n_cross, new_index=index)
        else:
            flags, index = check_mutual_exclusion_param_equal(*pack_param(self._Param, type='mutual_exclusion'), n_cross=n_cross)
            self.__deal_data(self.__mutual_exclusion, flags=flags, n_cross=n_cross, new_index=index)
        #  循环参数数据
        max = len(list(self._Data['mutual_exclusion'][index].keys()))
        total = False
        while True:
            for i in range(0, max):
                if i == max - 1:
                    total = True
                yield  self._Data['mutual_exclusion'][index][i], total
                pass
            pass
        pass


def __test_mutual_exclusion():
    data = np.reshape(np.array(list(range(0, 1000)), dtype=np.float32), [250, 4])
    data2 = np.reshape(np.array(list(range(0, 250)), dtype=np.float32), [250, 1])
    data3 = np.reshape(np.array(list(range(0, 2500)), dtype=np.float32), [250, 10])
    split = CrossSplit(data, data2, data3)
    gen1 = split.gen_mutual_exclusion(n_cross=10)
    for i in range(0, 30):
        data, total = gen1.__next__()
        if i == 9:
            assert total == True, Fore.RED + 'stop error'
            break
    assert len(split.Param['mutual_exclusion'].keys()) == 1, 'one Param append error'
    assert split.Param['mutual_exclusion'][0]['n_cross'] == 10, 'one Param update error'
    assert len(split.Data['mutual_exclusion'].keys()) == 1, 'one Data append error'
    gen2 = split.gen_mutual_exclusion(n_cross=5)
    gen2.__next__()
    assert len(split.Param['mutual_exclusion'].keys()) == 2, 'two Param append error'
    assert split.Param['mutual_exclusion'][1]['n_cross'] == 5, 'two Param update error'
    assert len(split.Data['mutual_exclusion'].keys()) == 2, 'two Data append error'
    # todo: more test
    assert False not in []
    pass

if __name__ == '__main__':
    __test_mutual_exclusion()
    pass
