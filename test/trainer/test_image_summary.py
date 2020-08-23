# coding=utf-8

import os

path = os.path.abspath(__file__)

import torch
from tensorboardX import SummaryWriter

from Putil.trainer.image_summary import ImageSummary

writer = SummaryWriter(logdir=os.path.join())
IS = ImageSummary()