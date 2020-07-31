import multiprocessing as mp
import Putil.trainer.multiprocess_auto_save as mas
from multiprocessing.managers import BaseManager

class TM(BaseManager):
    pass

TM.register('T', mas.T)
print(__name__)


if __name__ == '__main__':

    mp.set_start_method('spawn')
    #tm = TM()
    #tm.start()
    #t = tm.T()
    t = 0
    
    BaseManager.register('get_t', callable=lambda: t)
    manager = BaseManager(address=('', 18888), authkey=b'test')
    manager.start()
else:
    BaseManager.register('get_t')
    manager = BaseManager(address=('127.0.0.1', 18888), authkey=b'test')
    manager.connect()
    pass

def main():
    #mp.Process(target=master)
    pass

def master():
    t = manager.get_t()
    print(t)
    pass

def worker():
    pass


if __name__ == '__main__':
    main()