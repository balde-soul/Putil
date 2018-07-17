# coding=utf-8
import random
from optparse import OptionParser

parser = OptionParser(usage="usage:%prog [options] arg1 arg2")

parser.add_option(
    '--trt',
    '--TestRandomType',
    action='store_true',
    default=False,
    dest='TestRandomType',
    help='set this if you wan to test: random_type'
)


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


def __test_random_type():
    type = random_type()
    for i in range(0, 20):
        _type = type.type_gen(color='r')
        assert _type[0] == 'r'
        pass
    for i in range(0, 20):
        _type = type.type_gen(marker='o')
        assert _type[1] == 'o'
        pass
    for i in range(0, 20):
        _type = type.type_gen(line='--')
        assert _type[2:] == '--'
        pass
    pass


def __test(**options):
    test_random_type = options.pop('test_random_type', False)
    __test_random_type() if test_random_type else None
    __test_random_type()


if __name__ == '__main__':
    (options, args) = parser.parse_args()
    __test(
        test_random_type=options.TestRandomType
    )
    pass
