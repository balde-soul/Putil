# coding=utf-8


class DefaultModel(Model):
    def __init__(self, args):
        Model.__init__(self, args)
    
    def forward(self, *input):
        raise NotImplementedError('DefaultModel is not implemented')