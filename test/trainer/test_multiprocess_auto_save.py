import multiprocessing as mp
import Putil.trainer.multiprocess_auto_save as mas
from multiprocessing.managers import BaseManager


BaseManager.register('T', mas.T)
manager = BaseManager()
manager.start()

t = manager.T()

def main():
    mp.spawn()
    mp.spawn.spawn_main()
    pass


if __name__ == '__main__':
    main()