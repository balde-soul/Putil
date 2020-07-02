# coding=utf-8
import Putil.base.logger as plog

plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_log_level(stream=plog.DEBUG)
plog.PutilLogConfig.config_format(plog.Format)


logger = plog.PutilLogConfig('TestCOCODataUnit').logger()
logger.setLevel(plog.DEBUG)

import Putil.data.coco as coco

coco.COCOData.statistic()
