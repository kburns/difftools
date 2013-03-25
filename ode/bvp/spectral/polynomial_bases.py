"""
Polynomial basis sets.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np


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
        self.grid = self._construct_grid()

    def diffmatrix(self, p, x):
        """
        Construct p-th order differentiation matrix.

        Parameters
        ----------
        p : int
            Derivative order
        x : array of floats
            Locations for evaluation

        """

        Dp = np.empty((self.size, x.size))
        for j in xrange(self.size):
            Dp[j] = self.derivative(p, j, x)

        return Dp.T

    def evalmatrix(self, x):
        """
        Construct evaluation matrix.

        Parameters
        ----------
        x : array of floats
            Locations for evaluation

        """

        E = np.empty((self.size, x.size))
        for j in xrange(self.size):
            E[j] = self.evaluate(j, x)

        return E.T


class _ChebyshevPolynomialBase(_BasisBase):
    """Chebyshev polynomial basis base class."""

    def evaluate(self, j, x):
        """
        Evaluate a basis element.

        Parameters
        ----------
        j : int
            Basis element index
        x : array of floats
            Locations for evaluation

        """

        # Chebyshev polynomials
        t = np.arccos(x)
        Tj = np.cos(j * t)

        return Tj

    def derivative(self, p, j, x):
        """
        Evaluate derivatives of a basis element.

        Parameters
        ----------
        p : int
            Derivative order
        j : int
            Basis element index
        x : array of floats
            Locations for evaluation

        """

        # Split interior and exterior derivatives
        endpts = (np.abs(x) == 1.)
        Tj_xp = np.empty_like(x)
        Tj_xp[endpts] = self._endpoint_derivative(p, j, x[endpts])
        Tj_xp[~endpts] = self._interior_derivative(p, j, x[~endpts])

        return Tj_xp

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

        N = self.size - 1
        i = np.arange(N + 1)
        x = np.cos(np.pi * i / N)

        return x


class ChebyshevRootsPolynomials(_ChebyshevPolynomialBase):
    """Chebyshev polynomial basis on the roots grid."""

    def _construct_grid(self):

        N = self.size
        i = np.arange(N)
        x = np.cos(np.pi * (2.*i + 1.) / (2.*N))

        return x

