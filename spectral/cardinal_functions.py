"""Spectral cardinal functions and grids."""


import numpy as np
from basis_functions import Chebyshev


class ChebyshevExtrema(object):

    def __init__(self, N):

        # Store inputs
        self.N = N

        # Connect to basis functions
        self.basis = Chebyshev()

    def grid(self):

        if not hasattr(self, '_grid'):
            i = np.arange(self.N + 1)
            x = np.cos(np.pi * i / self.N)
            self._grid = x

        return self._grid

    def evaluate(self, j, x):
        """
        Evaluate Chebyshev polynomials.

        Parameters
        ----------
        j : int
            Degree of Chebyshev polynomial
        x : float or array
            Locations for evaluation

        """

        m = np.arange(self.N + 1)
        Tm_xj = self.basis.evaluate(m, self.grid()[j])
        Tm_x = self.basis.evaluate(m, x)



        t = np.arccos(x)
        Tn = np.cos(n * t)

        return Tn


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

