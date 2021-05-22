# coding=utf-8
import torch
import argparse

from Putil.base import logger as plog
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_log_level(stream=plog.DEBUG)
plog.PutilLogConfig.config_format(plog.FormatRecommend)
logger = plog.PutilLogConfig('test_mobilenet').logger()
logger.setLevel(plog.DEBUG)

from Putil.demo.deep_learning.base.backbone import mobilenet, mobilenetArg

parser = argparse.ArgumentParser()

mobilenetArg(parser)

args = parser.parse_args()
args.backbone_arch = 'v1'
args.backbone_downsample_rate = 3
args.backbone_pretrained = True
args.backbone_weight_path = './test/demo/deep_learning/base/checkpoints'

backbone = mobilenet(args)()
backbone.cuda()
#print(backbone)
data = torch.zeros([32, 3, 512, 512], dtype=torch.float32)
data = data.cuda()
out = backbone(data)
