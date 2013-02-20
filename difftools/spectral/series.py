"""
Series representations.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np
import scipy.linalg as linalg


class TruncatedSeries(object):

    def __init__(self, basis):
        """
        Create a truncated series representation of a function.

        Parameters
        ----------
        basis : basis object
            Finite basis for representation

        """

        # Store inputs
        self.basis = basis

        # Setup parameters
        self.size = self.basis.size
        self.coefficients = np.zeros(self.size)

    def evaluate(self, x, index=False):
        """
        Evaluate series.

        Parameters
        ----------
        x : float or int
            Location or grid index for evaluation
        index : bool
            True if x is grid index, False if x is location.

        """

        # Add terms in series
        # VECTORIZE
        scratch = np.empty(self.size)
        for j in xrange(self.size):
            scratch[j] = self.basis.evaluate(j, x, index=index)
        out = np.dot(scratch, self.coefficients)

        return out

    def derivative(self, p, x, index=False):
        """
        Evaluate derivative of series.

        Parameters
        ----------
        p : int
            Derivative order
        x : float or int
            Location or grid index for evaluation
        index : bool
            True if x is grid index, False if x is location.

        """

        # Add terms in series
        # VECTORIZE
        scratch = np.empty(self.size)
        for j in xrange(self.size):
            scratch[j] = self.basis.derivative(p, j, x, index=index)
        out = np.dot(scratch, self.coefficients)

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

