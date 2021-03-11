'''
Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
Alex Kendall|University of Cambridge|agk34@cam.ac.uk
Yarin Gal|University of Cambridge|yg279@cam.ac.uk
Roberto Cipolla|University of Cambridge|rc10001@cam.ac.uk
'''
#coding=utf-8
import torch 
from torch import nn


class UncertaintyLoss(nn.Module):
    def __init__(self, v_num):
        super(UncertaintyLoss, self).__init__()
        sigma = torch.ones(v_num) / v_num
        self.sigma = nn.Parameter(sigma)
        self.v_num = v_num

    def forward(self, **input):
        loss = 0
        for index, (_input_name, _input_value) in enumerate(input.items()):
            loss += _input_value / (2 * self.sigma[index] ** 2)
        loss += torch.log(self.sigma.pow(2).prod())
        return loss, {k: self.sigma[index] for index, (k, v) in enumerate(input.items())}