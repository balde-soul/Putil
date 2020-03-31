# coding=utf-8
import Putil.base.logger as plog


def args_pack(args):
    '''
    this function pack the args into a string with format: key1-value1_key2-value2
    '''
    collection = args.__dict__
    origin = ''
    for k, v in collection.items():
        origin = '{0}{1}-{2}'.format('' if origin == '' else '{0}_'.format(origin), k, v)
        pass
    return origin
    pass


def args_log(args, logger):
    collection = args.__dict__
    logger.info(plog.info_color('args:'))
    for k, v in collection.items():
        logger.info(plog.info_color('\t{0}: {1}'.format(k, v)))
        pass
    pass