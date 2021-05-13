# coding=utf-8

from Putil.demo.deep_learning.base import backbone 
import argparse

parser = argparse.ArgumentParser()

backbone.resnetArg(parser)

args = parser.parse_args()
args.backbone_arch = '18'
args.backbone_downsample_rate = 3
args.backbone_pretrained = True
args.backbone_weight_path = './test/demo/deep_learning/base/checkpoints'

backbone = backbone.resnet(args)()
backbone.cuda(2)
print(backbone)