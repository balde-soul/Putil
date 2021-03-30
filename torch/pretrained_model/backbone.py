# coding = utf-8


class Backbone:
    def __init__(self, downsample_rate, features, input_shape):
        self._downsample_rate = downsample_rate
        self._features = features
        self._input_shape = input_shape
        pass

    ##@brief 返回各个分辨率重采样的feature_map
    # @note
    # @return dict
    def feature_map(self):
        pass
    pass