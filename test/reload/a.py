from importlib import reload
from Putil.test.reload import b

Type = b.Type
print('import b: {}'.format(id(Type)))

reload(b)
Type = b.Type
print('reload b in a: {}'.format(id(Type)))