import Putil.base.logger as plog
from functools import wraps
from abc import ABCMeta, abstractmethod
_logger = plog.PutilLogConfig('Putil.base').logger()
_logger.setLevel(plog.DEBUG)
singlecallLogger = _logger.getChild('singlecall')
singlecallLogger.setLevel(plog.DEBUG)
EnvManagerLogger = _logger.getChild('EnvManager')
EnvManagerLogger.setLevel(plog.DEBUG)
from argparse import Namespace

class singlecall:
    class HasCalled(Exception):
        pass

    def __init__(self, raise_exception=True):
        self._state = dict()
        self._state['iscalled'] = False
        self._raise_exception = raise_exception
        pass

    def __call__(self, func):
        @wraps(func)
        def call(*args, **kwargs):
            if self._state['iscalled'] is False:
                singlecallLogger.info('{0} would be called'.format(func.__qualname__))
                self._state['iscalled']= True
                return func(*args, **kwargs)
            else:
                singlecallLogger.info('{0} has been called'.format(func.__qualname__))
                if self._raise_exception:
                    raise singlecall.HasCalled('{0} has been called'.format(func.__qualname__))
                else:
                    None
                pass
            pass
        return call
    pass


class EnvManager(Namespace):
    def __init__(self, **kwargs):
        self._c_attr_keys = list(kwargs.keys())
        Namespace.__init__(self, **kwargs)
        pass

    def switch(self, mode):
        pass

    def update(self, var):
        if isinstance(var, dict):
            self._c_attr_keys +=  list(var.keys())
            for k, v in var.items():
                setattr(self, k, v)
                pass
            pass
        elif isinstance(var, EnvManager):
            for k in var._c_attr_keys:
                setattr(self, k, eval('var.{0}'.format(k)))
                pass
            pass
        elif var is None:
            EnvManagerLogger.info('var is None, nothing updated')
        else:
            raise RuntimeError('not implement')
            pass
        pass
    pass