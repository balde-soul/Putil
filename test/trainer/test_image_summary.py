# coding=utf-8

import os
from PIL import Image
import numpy as np

path = os.path.split(os.path.abspath(__file__))[0]


import torch
from tensorboardX import SummaryWriter

summary_dir = os.path.join(path, 'test_summary_result/test_image_summary_result')
#contents = os.listdir(summary_dir)
#for content in contents:
#    if content in ['.gitignore', '.', '..']:
#        pass
#    else:
#        os.rmdir(os.path.join(summary_dir, content))
#        pass
from Putil.trainer.image_summary import torch_rectangle_image_summary as ImageSummary


writer = SummaryWriter(logdir=summary_dir)

image = Image.open('./test/test_used_data/000000177842.jpg')
image_array = np.array(image)

images_array = np.stack([image_array, image_array, image_array, image_array, image_array])
images_array = np.transpose(images_array, [0, 3, 1, 2])
images_tensor = torch.from_numpy(images_array)

ImageSummary(writer, 'test', np.transpose(images_tensor.detach().cpu().numpy(), (0, 2, 3, 1)), \
    [[[20, 20, 50, 50]], [[20, 20, 50, 50]], [[10, 10, 30, 30]], [[10, 10, 40, 40]], [[10, 10, 50, 50]]], \
        0)