import multiprocessing as mp
import Putil.trainer.multiprocess_auto_save as mas
from multiprocessing.managers import BaseManager


if __name__ == 'main':
    BaseManager.register('T', mas.T)
    manager = BaseManager(address=('127.0.0.1', 4000), authkey=b'test')
    manager.start()

    t = manager.T()

def main():
    mp.spawn()
    mp.spawn.spawn_main()
    pass


if __name__ == '__main__':
    main()