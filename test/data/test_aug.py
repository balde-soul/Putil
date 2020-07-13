# coding=utf-8
#In[]
import Putil.base.logger as plog

plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_log_level(stream=plog.DEBUG)
test_aug_logger = plog.PutilLogConfig('test_aug').logger()
test_aug_logger.setLevel(plog.DEBUG)

from Putil.data.aug import AugNode
from Putil.data.aug import AugFunc 
import Putil.data.aug as PAug
import pickle

pickle.dumps(PAug.AugFuncNoOp())

class gain(AugFunc):
    def __init__(self):
        AugFunc.__init__(self)
        pass

    def __call__(self, *args):
        return args[0] + 1, args[1]

    def _generate_name(self):
        return 'gain'

    def _generate_doc(self):
        return 'gain one'
    pass

root = AugNode(gain())
c1 = root.add_child(AugNode(gain()))
c2 = root.add_child(AugNode(gain()))
c3 = root.add_child(AugNode(gain()))
c31 = c3.add_child(AugNode(gain()))
c32 = c3.add_child(AugNode(gain()))
c33 = c3.add_child(AugNode(gain()))
root.freeze_node()
try:
    c33 = c3.add_child(AugNode(gain()))
except Exception as e:
    pass
test_aug_logger.debug(len(root))
for f in root:
    test_aug_logger.debug(f.name)
    test_aug_logger.debug(f.doc)
    test_aug_logger.debug(f.func(0, 0))

for f in c3:
    test_aug_logger.debug(f.func(0, 0))

test_aug_logger.debug(len(root))
for f in root:
    test_aug_logger.debug(f.func(0, 0))

for f in c3:
    test_aug_logger.debug(f.func(0, 0))

test_aug_logger.debug(len(root))
for f in root:
    test_aug_logger.debug(f.func(0, 0))

for f in c3:
    test_aug_logger.debug(f.func(0, 0))
