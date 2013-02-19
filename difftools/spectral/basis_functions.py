"""
Utility classes defining spectral basis functions.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np
import math


class Chebyshev(object):

    @staticmethod
    def evaluate(self, n, x):
        """
        Evaluate Chebyshev polynomials.

        Parameters
        ----------
        n : int
            Degree of Chebyshev polynomial
        x : float or array
            Location for evaluation

        """

        # Chebyshev polynomials
        t = np.arccos(x)
        Tn = np.cos(n * t)

        return Tn

    @staticmethod
    def derivative(self, p, n, x):
        """
        Evaluate derivatives of Chebyshev polynomials.

        Parameters
        ----------
        p : int
            Derivative order
        n : int
            Degree of Chebyshev polynomial
        x : float or array
            Location for evaluation

        """

        # Handle endpoints
        if np.isscalar(x):
            if x in (-1., 1.):
                return self._endpoint_derivative(p, n, x)
            else:
                return self._interior_derivative(p, n, x)
        else:
            if (x[0] in (-1., 1.)) or (x[-1] in (-1., 1.)):
                first = self.derivative(p, n, x[0])
                mid = self.derivative(p, n, x[1:-1])
                last = self.derivative(p, n, x[-1])
                return np.concatenate(([first], mid, [last]))
            else:
                return self._interior_derivative(p, n, x)

    @staticmethod
    def _interior_derivative(self, p, n, x):

        # Compute simple arrays
        t = np.arccos(x)
        cos_t = x
        sin_t = np.sin(t)
        cos_nt = np.cos(n * t)
        sin_nt = np.sin(n * t)

        # Chebyshev polynomials
        Tn = cos_nt

        # First derivative
        Tn_t = -sin_nt * n
        Tn_x = -Tn_t / sin_t
        if p == 1:
            return Tn_x

        # Second derivative
        Tn_tt = -cos_nt * n**2
        Tn_xt = -(Tn_tt + Tn_x * cos_t) / sin_t
        Tn_xx = -Tn_xt / sin_t
        if p == 2:
            return Tn_xx

        # Third derivative
        Tn_ttt = sin_nt * n**3
        Tn_xtt = -(Tn_ttt + 2 * Tn_xt * cos_t) / sin_t + Tn_x
        Tn_xxt = -(Tn_xtt + Tn_xx * cos_t) / sin_t
        Tn_xxx = -Tn_xxt / sin_t
        if p == 3:
            return Tn_xxx

        # Higher derivatives
        raise ValueError("Higher order derivatives not implemented.")

    @staticmethod
    def _endpoint_derivative(self, p, n, x):

        C = 1.
        for k in xrange(p):
            C *= (n**2 - k**2) / (2.*k + 1.)
        Tn_xp = C * x ** (n + p)

        return Tn_xp

