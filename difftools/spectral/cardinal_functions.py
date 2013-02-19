"""
Spectral cardinal functions and grids.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np
import basis_functions


class CardinalBase(object):
    """Base class for cardinal bases."""

    def __init__(self, size, functions):
        """
        Construct basis object.

        Parameters
        ----------
        size : int
            Number of points in grid.

        """

        # Store inputs
        self.size = size
        self.functions = functions

        # Construct grid
        self.grid = self._construct_grid()

    def _construct_grid(self):

        pass

    def _construct_diff_matrix(self, p):

        D = np.empty((self.size, self.size))
        for i in xrange(self.size):
            for j in xrange(self.size):
                D[i, j] = self.derivative(p, j, i)

        return D

    def derivative(self, p, j, i):

        pass

    def diffmatrix(self, p):
        """
        Construct p-th order differentiation matrix.

        Parameters
        ----------
        p : int
            Derivative order

        """

        return self._construct_diff_matrix(p)

    def evaluate(self, j, i):
        """
        Evaluate Cardinal functions on the grid: Cj(xi).

        Parameters
        ----------
        j : int
            Basis index
        i : int
            Grid index

        """

        if i == j:
            return 1.
        else:
            return 0.


class ChebyshevExtrema(CardinalBase):
    """Chebysehv polynomials: extrema & endpoints ("Gauss-Lobatto") grid."""

    basis = basis_functions.Chebyshev

    def _construct_grid(self):

        self.N = self.size - 1
        i = np.arange(self.N + 1)
        x = np.cos(np.pi * i / self.N)

        return x

    def derivative(self, p, j, i):
        """
        Evaluate Cardinal derivatives on the grid: Cj_p(xi).

        Parameters
        ----------
        p : int
            Derivative order
        n : int
            Degree / basis index
        i : int
            Grid index

        """

        N = self.N
        xj = self.grid[j]
        xi = self.grid[i]

        # First derivative
        if i == j:
            if i == 0:
                Cj_x = (1. + 2.*N**2) / 6.
            elif i == N:
                Cj_x = -(1. + 2.*N**2) / 6.
            else:
                Cj_x = -xj / (2. * (1. - xj**2))
        else:
            pi = pj = 1.
            if (pi == 0) or (pi == N):
                pi = 2.
            if (pj == 0) or (pj == N):
                pj = 2.
            Cj_x = (-1) ** (i + j) * pi / (pj * (xi - xj))
        if p == 1:
            return Cj_x

        # Higher derivatives
        raise ValueError("Higher order derivatives not implemented.")

    def diffmatrix(self, p):
        """
        Construct p-th order differentiation matrix.

        Parameters
        ----------
        p : int
            Derivative order

        """

        D1 = self._construct_diff_matrix(1)
        Dp = np.identity(self.size)
        for i in xrange(p):
            Dp = np.dot(D1, Dp)
        return Dp

    def evaluate_off_grid(self, j, x):
        """
        Evaluate Cardinal functions off the grid: Cj(x).

        Parameters
        ----------
        j : int
            Degree of Chebyshev polynomial
        x : float
            Location for evaluation

        """

        N = self.N
        xj = self.grid[j]

        if x in self.grid:
            if x == xj:
                Cj = 1.
            else:
                Cj = 0.
        else:
            Cj = np.zeros_like(x)
            for m in xrange(N + 1):
                Tm_xj = self.basis.evaluate(m, xj)
                Tm_x = self.basis.evaluate(m, x)
                if (m == 0) or (m == N):
                    Cj += Tm_xj * Tm_x / 2.
                else:
                    Cj += Tm_xj * Tm_x
            if (j == 0) or (j == N):
                Cj *= 1. / N
            else:
                Cj *= 2. / N

        return Cj


class ChebyshevRoots(object):
    """Chebysehv polynomials: interior or "roots" grid."""

    basis = basis_functions.Chebyshev

    def _construct_grid(self):

        i = np.arange(self.N)
        x = np.cos(np.pi * (2.*i + 1.) / (2.*self.N))

        return x

    def derivative(self, p, j, i):
        """
        Evaluate Cardinal derivatives on the grid: Cj_p(xi).

        Parameters
        ----------
        p : int
            Derivative order
        n : int
            Degree / basis index
        i : int
            Grid index

        """

        N = self.N
        xj = self.grid[j]
        xi = self.grid[i]

        # First derivative
        if i == j:
            Cj_x = 0.5 * xj / (1. - xj**2)
        else:
            scratch = math.sqrt((1. - xj**2) / (1. - xi**2))
            Cj_x = (-1) ** (i + j) * scratch / (xi - xj)
        if p == 1:
            return Cj_x

        # Second derivative
        if i == j:
            Cj_xx = xj**2 / (1. - xj**2)**2 - (N**2 - 1.) / (3. * (1. - xj**2))
        else:
            Cj_xx = Cj_x * (xi / (1. - xi**2) - 2. / (xi - xj))
        if p == 2:
            return Cj_xx

        # Higher derivatives
        raise ValueError("Higher order derivatives not implemented.")

    def evaluate_off_grid(self, j, x):
        """
        Evaluate Cardinal functions off the grid: Cj(x).

        Parameters
        ----------
        j : int
            Degree of Chebyshev polynomial
        x : float
            Location for evaluation

        """

        N = self.N
        xj = self.grid[j]

        if x in self.grid:
            if x == xj:
                Cj = 1.
            else:
                Cj = 0.
        else:
            TN_x = self.basis.evaluate(N, x)
            TN_x_xj = self.basis.derivative(1, N, xj)
            Cj = TN_x / TN_x_xj / (x - xj)

        return Cj

