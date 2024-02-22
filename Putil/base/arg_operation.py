# coding=utf-8
import os
import json
import argparse


##@brief save the arg(type: Namespace) to a file
# @note save \'unserized_obj\' while unserized object exist in the arg
# @param[in] args the namespace
# @param[in] file the full path of the target file
def args_save(args, file):
    with open(file, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, default=lambda unserized_obj: 'unserized_obj', check_circular=True, sort_keys=True)
    pass


def args_extract(file):
    args = argparse.Namespace()
    with open(file, 'r') as fp:
        args.__dict__ = json.load(fp)
    return args 


def args_merge(*args, **kwargs):
    '''
     @brief merge the args
     @note 小索引的args值优先
     @param[in] kwargs
      'invert_order': bool
       if invert_order is True, the args would be reverted
       if invert_order is not True, the order of args would be kept
    '''
    if len(args) == 0:
        return None
    else:
        invert_order = kwargs.get('invert_order', False)
        arg_dicts = [arg.__dict__ for arg in (args if invert_order is False else args[::-1])][::-1]
        [arg_dicts[0].update(ad) for index, ad in enumerate(arg_dicts)]
        return argparse.Namespace(**arg_dicts[0])