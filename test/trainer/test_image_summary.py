# coding=utf-8

import os
import numpy as np

path = os.path.split(os.path.abspath(__file__))[0]

import torch
from tensorboardX import SummaryWriter


writer = SummaryWriter(logdir=os.path.join(path, 'test_summary_result/test_image_summary_result'))
r = 5
for i in range(100):
    writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                    'xcosx':i*np.cos(i/r),
                                    'tanx': np.tan(i/r)}, i)
    writer.add_scalars('run_14h', {'xsinxshift':i*np.sin((i + 1)/r)}, i)
    writer.add_scalar('run_14h/tyes', torch.tensor(i * np.sin(i/ (r+1))), i)
writer.close()