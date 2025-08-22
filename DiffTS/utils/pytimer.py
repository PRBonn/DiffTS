import time
import torch


def timeit(f):
    def wrap(*args, **kargs):
        time1 = time.time()
        ret = f(*args, **kargs)
        time2 = time.time()
        print(
            "{:s} function took {:.3f} ms".format(
                f.__name__, (time2 - time1) * 1000.0
            )
        )

        return ret

    return wrap


class Timer():
    def __init__(self):
        self.start = None
        self.duration = None
        self.tic()
        self.init_t = self.start

    def tic(self):
        self.start = time.time()
        return self.start

    def toc(self, prefix='', verbose=True):
        self.duration = time.time()-self.start
        if verbose:
            print(f'{prefix} {self.duration:.5f}s')
        return self.duration

    def tocTic(self, prefix='', verbose=True):
        self.toc(prefix, verbose)
        self.tic()
        return self.duration


TIMER = Timer()


def tic():
    return TIMER.tic()


def toc(prefix='', verbose=True):
    return TIMER.toc(prefix=prefix, verbose=verbose)


def tocTic(prefix='', verbose=True):
    return TIMER.tocTic(prefix=prefix, verbose=verbose)


class TorchTimer():
    def __init__(self, verbose=True):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.duration = 0
        self.tic()
        self.init_t = self.start
        self.verbose = verbose

    def tic(self):
        self.start.record()

    def toc(self, prefix=''):
        self.end.record()
        torch.cuda.synchronize()
        self.duration = self.start.elapsed_time(self.end)/1000
        if self.verbose:
            print(f'{prefix} {self.duration:.5f}s')
        return self.duration

    def tocTic(self, prefix=''):
        self.toc(prefix)
        self.tic()
        return self.duration
