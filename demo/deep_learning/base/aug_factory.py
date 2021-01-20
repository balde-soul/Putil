# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('aug_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.aug as standard
from util import aug as project
from importlib import reload
reload(standard)
reload(project)


def aug_factory(args, property_type, **kwargs):
    '''
     @brief
     @note
     @param[in]
      args
      aug_source: the aug source follow the format source_one-source_two-source-three-...
      aug_name: the aug name follow the format aug_one-aug_two-aug_three-...
    '''
    def _combine(source, name, target):
        target = '{} {}.{}'.format(target, source, name)
        return True
    target = ''
    [_combine(source, name, target) for source, name in zip(args.aug_sources, args.aug_names)]
    logger.info('augs: {}'.format(target))
    augs = list()
    for aug_source, aug_name in zip(args.aug_sources, args.aug_names):
        augs.append(eval('{}.{}(args)'.format(aug_source, aug_name)))
        pass
    return augs

def aug_arg_factory(parser, sources, names):
    #import pdb; pdb.set_trace()
    for aug_source, aug_name in zip(sources, names):
        eval('{}.{}Arg(parser)'.format(aug_source, aug_name))