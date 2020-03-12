# coding=utf-8
import numpy as np
import Putil.base.logger as plog
import PutilEnvSet as pes


logger = plog.PutilLogConfig('psnr').logger()
logger.setLevel(plog.DEBUG)
PSNRLogger = logger.getChild('PSNR')
PSNRLogger.setLevel(plog.DEBUG)


class psnr:
    def __init__(self, max_value):
        self._max_value = max_value
        pass

    def calc(self, gt, fake, combined_dim):
        '''
        '''
        assert gt.shape == fake.shape, PSNRLogger.error('gt shape vs. fake shape: {0} vs. {1}'.format(gt.shape, fake.shape))
        gt = gt.astype(np.float64)
        fake = fake.astype(np.float64)
        mse = np.mean((gt - fake) ** 2, axis=combined_dim)
        pass
    pass
