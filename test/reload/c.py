from importlib import reload
import Putil.test.reload.b as b
reload(b)

def k():
    print('b in c: {}'.format(id(b.Type)))