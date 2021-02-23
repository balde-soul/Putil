from importlib import reload
from Putil.test.reload import a
Type = a.Type
print('load Type in a: {}'.format(id(Type)))
reload(a)
print('load Type in a: {}'.format(id(Type)))
Type = a.Type
print('load Type in a: {}'.format(id(Type)))