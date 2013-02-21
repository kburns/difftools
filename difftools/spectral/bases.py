"""
Basis sets.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np
import math


class _BasisBase(object):
    """Basis base class."""

    def __init__(self, size):
        """
        An object defining evaluation and derivatives in a spectral basis.

        Parameters
        ----------
        size : int
            Number of elements to include in the basis.

        """

        # Store inputs
        self.size = size

        # Construct collocation grid
        self._construct_grid()

    def _construct_diff_matrix(self, p):
        # VECTORIZE
        # DEPRICATE?

        Dp = np.empty((self.size, self.size))
        for i in xrange(self.size):
            for j in xrange(self.size):
                Dp[i, j] = self.derivative(p, j, i, index=True)

        return Dp

    def diffmatrix(self, p):
        # DEPRICATE?

        Dp = self._construct_diff_matrix(p)

        return Dp

    def evalmatrix(self):
        # VECTORIZE
        # DEPRICATE?

        E = np.empty((self.size, self.size))
        for i in xrange(self.size):
            for j in xrange(self.size):
                E[i, j] = self.evaluate(j, i, index=True)

        return E


class _ChebyshevPolynomialBase(_BasisBase):
    """Chebyshev polynomial basis base class."""

    def evaluate(self, j, x, index=False):
        """
        Evaluate a basis element.

        Parameters
        ----------
        j : int
            Basis element index
        x : float, int, or array
            Location / grid index for evaluation
        index : bool
            True if x is a grid index.

        """

        # Fetch grid points
        if index:
            x = self.grid[x]

        # Chebyshev polynomials
        t = np.arccos(x)
        Tj = np.cos(j * t)

        return Tj

    def derivative(self, p, j, x, index=False):
        """
        Evaluate derivatives of a basis element.

        Parameters
        ----------
        p : int
            Derivative order
        j : int
            Basis element index
        x : float, int, or array
            Location / grid index for evaluation
        index : bool
            True if x is a grid index.

        """

        # Fetch grid points
        if index:
            x = self.grid[x]

        # Split interior and exterior derivatives
        # VECTORIZE
        if x in (-1., 1.):
            return self._endpoint_derivative(p, j, x)
        else:
            return self._interior_derivative(p, j, x)

    def _interior_derivative(self, p, j, x):

        # Compute simple arrays
        t = np.arccos(x)
        cos_t = x
        sin_t = np.sin(t)
        cos_jt = np.cos(j * t)
        sin_jt = np.sin(j * t)

        # Chebyshev polynomials
        Tj = cos_jt

        # First derivative
        Tj_t = -sin_jt * j
        Tj_x = -Tj_t / sin_t
        if p == 1:
            return Tj_x

        # Second derivative
        Tj_tt = -cos_jt * j**2
        Tj_xt = -(Tj_tt + Tj_x * cos_t) / sin_t
        Tj_xx = -Tj_xt / sin_t
        if p == 2:
            return Tj_xx

        # Third derivative
        Tj_ttt = sin_jt * j**3
        Tj_xtt = -(Tj_ttt + 2 * Tj_xt * cos_t) / sin_t + Tj_x
        Tj_xxt = -(Tj_xtt + Tj_xx * cos_t) / sin_t
        Tj_xxx = -Tj_xxt / sin_t
        if p == 3:
            return Tj_xxx

        # Higher derivatives
        raise ValueError("Higher order derivatives not implemented.")

    def _endpoint_derivative(self, p, j, x):

        # p-th derivative
        C = 1.
        for k in xrange(p):
            C *= (j**2 - k**2) / (2.*k + 1.)
        Tj_xp = C * x ** (j + p)

        return Tj_xp


class ChebyshevExtremaPolynomials(_ChebyshevPolynomialBase):
    """Chebyshev polynomial basis on the extrema & endpoints grid."""

    def _construct_grid(self):

        self.N = self.size - 1
        i = np.arange(self.N + 1)
        self.grid = np.cos(np.pi * i / self.N)


class ChebyshevRootsPolynomials(_ChebyshevPolynomialBase):
    """Chebyshev polynomial basis on the roots grid."""

    def _construct_grid(self):

        self.N = self.size
        i = np.arange(self.N)
        self.grid = np.cos(np.pi * (2.*i + 1.) / (2.*self.N))


class ChebyshevExtremaCardinals(ChebyshevExtremaPolynomials):

    def _poly_evaluate(self, *args, **kwargs):

        return ChebyshevExtremaPolynomials.evaluate(self, *args, **kwargs)

    def _poly_derivative(self, *args, **kwargs):

        return ChebyshevExtremaPolynomials.derivative(self, *args, **kwargs)

    def evaluate(self, j, x, index=False):
        """
        Evaluate Cardinal functions off the grid: Cj(x).

        Parameters
        ----------
        j : int
            Degree of Chebyshev polynomial
        x : float
            Location for evaluation

        """

        # Index case
        if index:
            if x == j:
                return 1.
            else:
                return 0.

        # Position case
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
                Tm_xj = self._poly_evaluate(m, xj, index=False)
                Tm_x = self._poly_evaluate(m, x, index=False)
                if (m == 0) or (m == N):
                    Cj += Tm_xj * Tm_x / 2.
                else:
                    Cj += Tm_xj * Tm_x
            if (j == 0) or (j == N):
                Cj *= 1. / N
            else:
                Cj *= 2. / N

        return Cj

    def derivative(self, p, j, x, index=False):
        """
        p : int
            Derivative order
        n : int
            Degree / basis index
        i : int
            Grid index

        """

        if index:
            i = x
        else:
            raise NotImplementedError("Evaluation only implemented on grid points.")

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

        D1 = self._construct_diff_matrix(1)
        Dp = np.identity(self.size)
        for i in xrange(p):
            Dp = np.dot(D1, Dp)
        return Dp


class ChebyshevRootsCardinals(_BasisBase):
    pass

