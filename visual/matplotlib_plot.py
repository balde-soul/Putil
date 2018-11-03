# coding=utf-8
import random


class random_type:
    def __init__(self):
        self._line_type = ['-', '--', ':']
        self._color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self._marker = [
            '.', ',', 'o', 'v', '^', '>', '<',
            '1', '2', '3', '4', 's', 'p', '*',
            'h', 'H', '+', 'x', 'D', 'd', '|', '_'
        ]
        self._len_l = len(self._line_type)
        self._len_co = len(self._color)
        self._len_ma = len(self._marker)
        self._used = list()
        pass

    def type_gen(self, **options):
        color = options.pop('color', None)
        marker = options.pop('marker', None)
        line = options.pop('line', None)
        if line is not None:
            pass
        else:
            line = self._line_type[random.choice(list(range(0, self._len_l)))]
            pass
        if color is not None:
            pass
        else:
            color = self._color[random.choice(list(range(0, self._len_co)))]
            pass
        if marker is not None:
            pass
        else:
            marker = self._marker[random.choice(list(range(0, self._len_ma)))]
        type = '{0}{1}{2}'.format(color, marker, line)
        self._used.append(type)
        return type
        pass

    def no_repeat(self, type):
        if type in self._used:
            return True
        else:
            return False
        pass
    pass
