import multiprocessing as mlp


def func(param):
    print(param)
    pass


manager = mlp.Manager()
pool = mlp.Pool()

empty_param = manager.dict()
empty_dict_test = pool.apply_async(func, args=(empty_param,))

normal_type_param = manager.dict()
normal_type_param['a'] = [0, 1]
normal_type_test = pool.apply_async(func, args=(normal_type_param, ))

manager_type_param = manager.dict()
manager_type_param['a'] = manager.list()
manager_type_test = pool.apply_async(func, args=(manager_type_param, ))

pool.close()
pool.join()
print(empty_dict_test.get())
print(normal_type_test.get())
print(manager_type_test.get())
