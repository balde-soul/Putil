# coding=utf-8
import copy
import queue
from abc import ABCMeta, abstractmethod, ABC
import Putil.base.logger as plog

root_logger = plog.PutilLogConfig('Aug').logger()
root_logger.setLevel(plog.DEBUG)
AugNodeLogger = root_logger.getChild('AugNodeLogger')
AugNodeLogger.setLevel(plog.DEBUG)

class AugFunc(metaclass=ABCMeta):
    def __init__(self):
        pass

    def _generate_name(self):
        return ''

    def _generate_doc(self):
        return ''

    @property
    def func(self):
        return self

    @property
    def name(self):
        return self._generate_name()

    @property
    def doc(self):
        return self._generate_doc()

    @property
    def param(self):
        return 'not implemented'

    @abstractmethod
    def __call__(self, *args):
        raise NotImplementedError("this func is not implement")
    pass

class AugFuncNoOp(AugFunc):
    def __init__(self):
        AugFunc.__init__(self)
        pass

    def _generate_name(self):
        return 'no_op'

    def _generate_doc(self):
        return 'do nothing'

    @property
    def func(self):
        return self

    def __call__(self, *args):
        return args
    pass

def no_op(*args):
    return args

class AugFuncSum(AugFunc):
    def __init__(self, func_list):
        AugFunc.__init__(self)
        self._func_list = copy.deepcopy(func_list)
        pass

    def __call__(self, *args):
        result = args
        for func in self._func_list:
            result = func(*result)
            pass
        return result
    
    @property
    def func(self):
        return self

    def _generate_name(self):
        name = ''
        for fl in self._func_list:
            name = '{0}-{1}'.format(name, fl.name)
            pass
        return name
    
    def _generate_doc(self):
        doc = ''
        for fl in self._func_list:
            cell = 'name: {0}; param: {1}; doc: {2}'.format(fl.name, fl.param, fl.doc)
            doc = '{0}\n{1}'.format(doc, cell)
            pass
        return doc 
    pass

class AugNode:
    '''
    '''
    def __init__(self, aug_func):
        self._children = list()
        self._aug_func = aug_func
        self._parent = None
        self._freezed = False

        self._leaf_nodes = list()
        self._funcs = list()
        pass

    @property
    def aug_func(self):
        return self._aug_func

    def __len__(self):
        if self._freezed is False:
            child_len = 0
            if len(self._children) == 0:
                return 1
            else:
                for child in self._children:
                    child_len += len(child)
                    pass
                pass
            pass
        else:
            child_len = len(self._funcs)
            pass
        return child_len
    
    def __getitem__(self, index):
        return self._funcs[index]

    def child_amount(self):
        return len(self._children)

    def children(self):
        return self._children

    def child(self, index):
        return self._children[index]
    
    @property
    def parent(self):
        return self._parent
    
    def set_parent(self, parent):
        self._check_freezed()
        self._parent = parent
        pass

    def add_child(self, child_node):
        self._check_freezed()
        self._children.append(child_node)
        self._children[-1].set_parent(self)
        return child_node

    def _check_freezed(self):
        if self._freezed:
            raise ValueError("node has been freezed")

    def freeze_node(self, generate_funcs=True):
        self._freezed = True
        self._leaf_nodes = list()
        if len(self._children) == 0:
            self._leaf_nodes.append(self)
        for child in self._children:
            child.freeze_node()
            self._leaf_nodes += child._leaf_nodes
            pass
        
        self._funcs, self._funcs_list_collection = AugNode.generate_func_sum(self._leaf_nodes, self) if generate_funcs is True else None
        return self

    @property
    def LeafNodes(self):
        return self._leaf_nodes

    @staticmethod
    def generate_func_sum(leaf_nodes, root):
        _funcs = list()
        func_list_collection = list()
        count = 0
        for ln in leaf_nodes:
            func_list = list()
            depth = 0
            while ln is not None:
                depth += 1
                func_list.append(ln.aug_func)
                if ln == root:
                    break
                ln = ln.parent
                pass
            func_list.reverse()
            func_list_collection.append(func_list)
            AugNodeLogger.debug('leaf_nodes depth: {0}'.format(depth))
            pass
            _funcs.append(AugFuncSum(func_list))
            count += 1
            pass
        AugNodeLogger.debug('leaf count: {0}'.format(count))
        return _funcs, func_list_collection

    @staticmethod
    def empty_node():
        return AugNode(no_op)
    pass