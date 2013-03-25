"""
Series representations.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np
import scipy.linalg as linalg


class TruncatedSeries(object):
    """Truncated series class."""

    def __init__(self, basis, range=(-1., 1.)):
        """
        An object representing a function as a truncated series.

        Parameters
        ----------
        basis : basis object
            Spectral basis for representation
        range : tuple of floats
            (start, end) of domain.

        """

        # Store inputs
        self.basis = basis
        self.range = range

        # Setup parameters
        self.size = self.basis.size
        self.coefficients = np.zeros(self.size, dtype=np.complex128)

        # Transform to problem grid
        self._grid_scale = (range[1] - range[0]) / 2.
        self._grid_shift = (range[1] + range[0]) / 2.
        self._basis_coord = lambda x: (x - self._grid_shift) / self._grid_scale
        self._problem_coord = lambda x: self._grid_shift + x * self._grid_scale
        self.grid = self._problem_coord(basis.grid)

    def duplicate(self):
        """Return a copy series object."""

        copy = self.__class__(self.basis, self.range)
        copy.coefficients[:] = self.coefficients[:]

        return copy

    def evaluate(self, x):
        """
        Evaluate series.

        Parameters
        ----------
        x : array of floats
            Locations for evaluation

        """

        x = self._basis_coord(x)

        # Add terms in series
        out = np.zeros_like(x, dtype=np.complex128)
        for j in xrange(self.size):
            out += self.basis.evaluate(j, x) * self.coefficients[j]

        return out

    def derivative(self, p, x):
        """
        Evaluate derivatives of series.

        Parameters
        ----------
        p : int
            Derivative order
        x : array of floats
            Locations for evaluation

        """

        x = self._basis_coord(x)

        # Add terms in series
        out = np.zeros_like(x, dtype=np.complex128)
        for j in xrange(self.size):
            out += self.basis.derivative(p, j, x) * self.coefficients[j]

        return out

    def expand_points(self, arr):
        """
        Expand a functioned as evaluated at grid points.

        Parameters
        ----------
        arr : array of floats
            Function evaluated over basis grid.

        """

        LHS = self.basis.evalmatrix(self.basis.grid)
        RHS = arr

        self.coefficients[:] = linalg.solve(a=LHS, b=RHS)

    def expand_function(self, f):
        """
        Expand a function as evaluated at grid points.

        Parameters
        ----------
        f : function(x)
            Function to expand

        """

        x = self._problem_coord(self.basis.grid)
        arr = f(x)

        self.expand_points(arr)

    def E(self, series):
        """
        Wrapper around basis evalmatrix function.

        Parameters
        ----------
        series : series object
            Series whose basis grid will be used for evaluation points.

        """

        E = self.basis.evalmatrix(series.basis.grid)

        return E

    def D(self, p, series):
        """
        Wrapper around basis diffmatrix function.

        Parameters
        ----------
        p : int
            Derivative order
        series : series object
            Series whose basis grid will be used for evaluation points.

        """

        D = self.basis.diffmatrix(p, series.basis.grid)
        D /= self._grid_scale ** p

        return D

