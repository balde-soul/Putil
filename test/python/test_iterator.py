# coding=utf-8


'''
the mechanism of the iterator:
使用迭代器时会调用__iter__获取到迭代器
然后调用迭代器的__next__方法进行迭代
'''


class Iteratorable:
    def __init__(self, contain):
        self._contain = contain
        self._i = 0
        pass

    def __next__(self):
        if self._i < len(self._contain):
            item = self._contain[self._i]
            self._i += 1
            return item
        else:
            self._i = 0
            raise StopIteration
        pass

    def __iter__(self):
        return self
        pass
    pass


a = Iteratorable(list(range(0, 100)))
for i in a:
    print(i)
for i in a:
    print(i)
    pass

a = 'asdasdasd'
for i in a:
    print(i)
for i in a:
    print(i)
