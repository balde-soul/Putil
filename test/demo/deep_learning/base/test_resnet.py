# coding=utf-8

import torch
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
backbone.cuda()
#print(backbone)
data = torch.zeros([32, 3, 512, 512], dtype=torch.float32)
data = data.cuda()
out = backbone(data)