# coding=utf-8
import Putil.base.logger as plog

plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_log_level(stream=plog.DEBUG)

from Putil.data.aug import AugNode
from Putil.data.aug import AugFunc 

class gain(AugFunc):
    def __init__(self):
        AugFunc.__init__(self)
        def g(x):
            return [x[0] + 1, x[1]]
            pass
        self._func = g
        pass

    def _generate_name(self):
        return 'gain'

    def _generate_doc(self):
        return 'gain one'
    pass

root = AugNode(gain())
c1 = root.add_child(gain())
c2 = root.add_child(gain())
c3 = root.add_child(gain())
c31 = c3.add_child(gain())
c32 = c3.add_child(gain())
c33 = c3.add_child(gain())
root.freeze_node()
print(len(root))
for f in root:
    print(f([0, 0]))

for f in c3:
    print(f([0, 0]))

print(len(root))
for f in root:
    print(f([0, 0]))

for f in c3:
    print(f([0, 0]))

print(len(root))
for f in root:
    print(f([0, 0]))

for f in c3:
    print(f([0, 0]))
