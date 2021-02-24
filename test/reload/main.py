from importlib import reload
from Putil.test.reload import a
from Putil.test.reload import c
Type = a.Type
bT = a.b.Type
print('load Type in a: {}'.format(id(Type)))
print('load Type in b: {}'.format(id(bT)))
c.k()
reload(a)
print('load Type in a: {}'.format(id(Type)))
c.k()
Type = a.Type
print('load Type in a: {}'.format(id(Type)))
c.k()