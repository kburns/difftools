

import numpy as np


class ChebyshevExtrema(object):

    def __init__(self, N):

        # Store inputs
        self.N = N

    def grid(self):

        if not hasattr(self, '_grid'):
            i = np.arange(self.N + 1)
            x = np.cos(np.pi * i / self.N)
            self._grid = x

        return self._grid

    def cardinal

class ChebyshevRoots(object):

    def __init__(self, N):

        # Store inputs
        self.N = N

    def grid(self):

        if not hasattr(self, '_grid'):
            i = np.arange(self.N)
            x = np.cos(np.pi * (2. * i - 1.) / (2. * self.N))
            self._grid = x

        return self._grid





