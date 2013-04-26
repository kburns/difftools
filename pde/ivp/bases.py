"""
Spectral basis sets.
Optimized using Numba.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np
import scipy.fftpack as fft
from numba import jit, double, int_, void


class ChebyshevExtremaPolynomials(object):
    """Chebyshev polynomial basis on the extrema & endpoints grid."""

    def __init__(self, size):

        # Parameters
        self.size = size
        self.N = size - 1
        self.dtype = np.float64

        # Grid
        i = np.arange(self.N + 1)
        self.grid = np.cos(np.pi * i / self.N)

        # Differentiation matrix
        self.diff_matrix = np.zeros((size, size), dtype=self.dtype)
        build_chebyshev_diff_matrix(self.grid, self.diff_matrix, self.N)

        # Math arrays
        self._math1 = np.zeros(size, dtype=self.dtype)
        self._math2 = np.zeros(size, dtype=self.dtype)

    def create_x_data(self):

        return np.zeros(self.size, dtype=self.dtype)

    def create_k_data(self):

        return np.zeros(self.size, dtype=self.dtype)

    def forward(self, xdata, kdata):

        # DCT with adjusted coefficients
        kdata[:] = fft.dct(xdata, type=1, norm=None)
        kdata /= self.N
        kdata[0] /= 2.
        kdata[-1] /= 2.

    def backward(self, kdata, xdata):

        # DCT with adjusted coefficients
        kdata[1:-1] /= 2.
        xdata[:] = fft.dct(kdata, type=1, norm=None)

    def x_diff(self, xdata, p):

        # References
        math1 = self._math1
        math2 = self._math2
        D = self.diff_matrix

        # Copy input
        math1[:] = xdata

        # Apply differentiation matrix p times
        for d in xrange(p):
            np.dot(D, math1, out=math2)
            if d < p - 1:
                (math1, math2) = (math2, math1)

        return math2

    def k_diff(self, kdata, p):

        # Referencess
        math1 = self._math1
        math2 = self._math2
        N = self.N

        # Copy input
        math1[:] = kdata

        # Apply recursion relation p times
        for d in xrange(p):
            chebyshev_diff_recursion(math1, math2, N)
            if d < p - 1:
                (math1, math2) = (math2, math1)

        return math2

    def interpolate(self, kdata, x):

        N = self.N
        theta = np.arccos(x)
        out = np.zeros_like(x)

        for n in xrange(N + 1):
            out += kdata[n] * np.cos(n * theta)

        return out


class ArcsinMappedCEP(ChebyshevExtremaPolynomials):

    def __init__(self, size, alpha):

        # Inherited initialization
        ChebyshevExtremaPolynomials.__init__(self, size)

        # Tranform grid
        self.alpha = alpha
        self._chebyshev_grid = CG = self.grid
        self.grid = np.arcsin(alpha * CG) / np.arcsin(alpha)

        # Modify differentiation matrix
        M = np.sqrt(1 - (alpha * CG)**2) * np.arcsin(alpha) / alpha
        self.diff_matrix = (self.diff_matrix.T * M).T

    def k_diff(self, kdata, p):

        k_deriv = self._math1
        x_deriv = self._math2
        alpha = self.alpha
        CG = self._chebyshev_grid

        M = np.sqrt(1 - (alpha * CG)**2) * np.arcsin(alpha) / alpha
        k_deriv[:] = ChebyshevExtremaPolynomials.k_diff(self, kdata, p)
        self.backward(k_deriv, x_deriv)

        return x_deriv * M

    def interpolate(self, kdata, x):

     CX = np.sin(np.arcsin(self.alpha) * x) / self.alpha
     out = ChebyshevExtremaPolynomials.interpolate(self, kdata, CX)

     return out


@jit(void(double[:], double[:,:], int_))
def build_chebyshev_diff_matrix(x, D, N):

    for i in xrange(N + 1):
        for j in xrange(N + 1):
            if i == j:
                if i == 0:
                    D[i, j] = (1. + 2.*N**2) / 6.
                elif i == N:
                    D[i, j] = -(1. + 2.*N**2) / 6.
                else:
                    xj = x[j]
                    D[i, j] = -xj / 2. / (1. - xj*xj)
            else:
                D[i, j] = (-1)**(i+j) / (x[i] - x[j])
                if (i == 0) or (i == N):
                    D[i, j] *= 2.
                if (j == 0) or (j == N):
                    D[i, j] /= 2.


@jit(void(double[:], double[:], int_))
def chebyshev_diff_recursion(a, b, N):

    b[N] = 0.
    b[N-1] = 2. * N * a[N]
    for i in xrange(N-2, 0, -1):
        b[i] = 2 * (i+1) * a[i+1] + b[i+2]
    b[0] = a[1] + b[2] / 2.

