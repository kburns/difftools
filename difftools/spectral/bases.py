"""
Basis sets.

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

    def _construct_diff_matrix(self, p, x):

        Dp = np.empty((self.size, x.size))
        for j in xrange(self.size):
            Dp[j] = self.derivative(p, j, x)

        return Dp.T

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

        return self._construct_diff_matrix(p, x)

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


class _CardinalBase(_BasisBase):
    """Chebyshev cardinal basis base class."""

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

        # Construct polynomial basis object
        self.poly_basis = self._poly_basis_class(size)

        # Construct collocation grid
        self.grid = self.poly_basis._construct_grid()

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

        if x is self.grid:
            Cj = np.zeros_like(x)
            Cj[j] = 1.
        else:
            Cj = self._evaluate_off_grid(j, x)

        return Cj


class ChebyshevExtremaCardinals(_CardinalBase):
    """Chebyshev cardinal basis on the extrema & endpoints grid."""

    _poly_basis_class = ChebyshevExtremaPolynomials

    def _evaluate_off_grid(self, j ,x):

        Cj = np.zeros_like(x)
        xj = self.grid[j]
        N = self.size - 1

        for m in xrange(N + 1):
            Tm_xj = self.poly_basis.evaluate(m, xj)
            Tm_x = self.poly_basis.evaluate(m, x)
            if (m == 0) or (m == N):
                Cj += Tm_xj * Tm_x / 2.
            else:
                Cj += Tm_xj * Tm_x

        if (j == 0) or (j == N):
            Cj *= 1. / N
        else:
            Cj *= 2. / N

        return Cj

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

        if x is not self.grid:
            raise NotImplementedError("Evaluation only implemented on grid points.")

        if p != 1:
            raise ValueError("Higher order derivatives not implemented.")

        N = self.size - 1
        i = np.arange(N + 1)
        xj = self.grid[j]

        # i != j, avoiding divide-by-zero
        dx = x - xj
        dx[j] = 1.
        Cj_x = (-1) ** (i + j) / dx
        Cj_x[0] *= 2.
        Cj_x[N] *= 2.
        if (j == 0) or (j == N):
            Cj_x /= 2.

        # i == j
        if j == 0:
            Cj_x[j] = (1. + 2.*N**2) / 6.
        elif j == N:
            Cj_x[j] = -(1. + 2.*N**2) / 6.
        else:
            Cj_x[j] = -xj / (2. * (1. - xj**2))

        return Cj_x

    def diffmatrix(self, p, x):
        """
        Construct p-th order differentiation matrix.

        Parameters
        ----------
        p : int
            Derivative order

        """

        # Construct higher derivative matrices through matrix multiplication
        D1 = self._construct_diff_matrix(1, x)
        Dp = np.identity(self.size)
        for i in xrange(p):
            Dp = np.dot(D1, Dp)

        return Dp


class ChebyshevRootsCardinals(_BasisBase):
    """Chebyshev cardinal basis on the roots grid."""

    _poly_basis_class = ChebyshevRootsPolynomials

    def _evaluate_off_grid(self, j, x):

        N = self.size
        xj = self.grid[j]

        # Mask singularity at xj
        dx = x - xj
        singmask = np.where(dx == 0)
        dx[singmask] = 1.
        TN_x = self.basis.evaluate(N, x)
        TN_x_xj = self.basis.derivative(1, N, xj)
        Cj = TN_x / TN_x_xj / dx
        Cj[singmask] = 1.

        return Cj

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

        if x is not self.grid:
            raise NotImplementedError("Evaluation only implemented on grid points.")

        N = self.size
        i = np.arange(N)
        xj = self.grid[j]

        # Avoid divide-by-zeros
        dx = x - xj
        dx[j] = 1.

        # First derivative
        scratch = np.sqrt((1. - xj**2) / (1. - x**2))
        Cj_x = (-1) ** (i + j) * scratch / dx
        Cj_x[j] = 0.5 * xj / (1. - xj**2)

        if p == 1:
            return Cj_x

        # Second derivative
        Cj_xx = Cj_x * (x / (1. - x**2) - 2. / dx)
        Cj_xx[j] = xj**2 / (1. - xj**2)**2 - (N**2 - 1.) / (3. * (1. - xj**2))

        if p == 2:
            return Cj_xx

        # Higher derivatives
        raise ValueError("Higher order derivatives not implemented.")

