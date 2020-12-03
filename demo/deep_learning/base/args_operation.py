# coding=utf-8
import os
import json
import argparse


def args_save(args, file):
    print(args)
    with open(file, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4)
    pass


def args_extract(file):
    args = argparse.Namespace()
    with open(file, 'r') as fp:
        args.__dict__ = json.load(fp)
    return args 