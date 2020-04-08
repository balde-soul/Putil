# coding=utf-8
# Putil
import Putil.base.logger as plog

def dict_pack(d):
    '''
    this function pack the args into a string with format: key1-value1_key2-value2
    '''
    origin = ''
    for k, v in d.items():
        origin = '{0}{1}-{2}'.format('' if origin == '' else '{0}_'.format(origin), k, v)
        pass
    return origin
    pass


def dict_log(d, logger):
    for k, v in d.items():
        logger.info(plog.info_color('\t{0}: {1}'.format(k, v)))
        pass
    pass