import Putil.data.split as split


class a:
    def __init__(self):
        self._t = None
        pass

    def get_t(self):
        return self._t

    def set_t(self, val):
        self._t = val
        pass
    t = property(get_t, set_t)
b = a()
b.t = 9
pass
