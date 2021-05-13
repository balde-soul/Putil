# coding=utf-8

from Putil.demo.deep_learning.base import backbone 
import argparse

parser = argparse.ArgumentParser()

backbone.vggArg(parser)

args = parser.parse_args()
args.backbone_arch = 'vgg11'
args.backbone_downsample_rate = 16
args.backbone_pretrained = True
args.backbone_weight_path = './test/demo/deep_learning/base/checkpoints'

bb = backbone.vgg(args)()
bb.cuda(2)
print(bb)
