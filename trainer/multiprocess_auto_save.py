import multiprocessing as mp

class T:
    def __init__(self):
        self._l = list()
        self._lock = mp.Lock()
        pass

    def put(self, i):
        self._lock.acquire()
        self._l.append(i)
        print(len(self._l))
        self._lock.release