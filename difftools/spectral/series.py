"""
Series representations.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np
import scipy.linalg as linalg


class TruncatedSeries(object):
    """Truncated series class."""

    def __init__(self, basis):
        """
        An object representing a function as a truncated series.

        Parameters
        ----------
        basis : basis object
            Spectral basis for representation

        """

        # Store inputs
        self.basis = basis

        # Setup parameters
        self.size = self.basis.size
        self.coefficients = np.zeros(self.size)

    def evaluate(self, x):
        """
        Evaluate series.

        Parameters
        ----------
        x : array of floats
            Locations for evaluation

        """

        # Add terms in series
        out = np.zeros_like(x)
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

        # Add terms in series
        out = np.zeros_like(x)
        for j in xrange(self.size):
            out += self.basis.derivative(p, j, x) * self.coefficients[j]

        return out

    def expand_function(self, f):
        """
        Expand a function as evaluated at grid points.

        Parameters
        ----------
        f : function(x)
            Function to expand

        """

        LHS = self.basis.evalmatrix()
        RHS = f(self.basis.grid)

        self.coefficients[:] = linalg.solve(a=LHS, b=RHS)

