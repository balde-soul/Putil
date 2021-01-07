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
        sigma = torch.randn(v_num)
        self.sigma = nn.Parameter(sigma)
        self.v_num = v_num

    def forward(self, *input):
        loss = 0
        for i in range(self.v_num):
            loss += input[i] / (2 * self.sigma[i] ** 2)
        loss += torch.log(self.sigma.pow(2).prod())
        return loss