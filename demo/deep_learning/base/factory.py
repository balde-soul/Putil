# coding=utf-8

class Factory:
    def __init__(self):
        pass

    def arg_factory():
        print('r')
        pass
    arg_factory = staticmethod(arg_factory)


class t(Factory):
    def __init__(self):
        Factory.__init__(self)
        pass


t.arg_factory()