"""
Frameworks for solving various problems in differential equations.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np
import scipy.linalg as linalg


class _ProblemBase(object):
    """Problem solver base class."""

    def __init__(self, varlist):

        # Store inputs
        self.varlist = varlist
        self.syssize = np.sum([var.size for var in self.varlist])

        # Copy variable coefficients
        self.syscoeff = np.concatenate([var.coefficients for var in self.varlist])

        # Make variable attributes views into the problem vector
        start = 0
        for var in self.varlist:
            var.coefficients = self.syscoeff[start:start+var.size]
            start += var.size


class BoundaryValueProblem(object):
    """Linear boundary value problem solver."""

    def __init__(self, varlist):
        """
        Setup the boundary value problem

            LHS . u = RHS

        Parameters
        ----------
        varlist : list of series objects
            Variables in system.

        """

        # Inherited initialization
        _ProblemBase.__init__(self, varlist)

        # Construct matrices
        self.LHS = np.zeros((self.syssize, self.syssize))
        self.RHS = np.zeros(self.syssize)

    def solve(self):
        """Solve the BVP using a matrix solve."""

        # Matrix solve
        self.syscoeff[:] = linalg.solve(a=self.LHS, b=self.RHS)

        return self.varlist


class EigenProblem(object):
    """Linear eigenproblem solver."""

    def __init__(self, varlist):
        """
        Setup the eigenvalue problem

            LHS . u = eigval * RHS . u

        Parameters
        ----------
        varlist : list of series objects
            Variables in system.

        """

        # Inherited initialization
        _ProblemBase.__init__(self, varlist)

        # Construct matrices
        self.LHS = np.zeros((self.syssize, self.syssize))
        self.RHS = np.identity(self.syssize)

    def solve(self):
        """Solve the generalized eigenproblem."""

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = linalg.eig(a=self.LHS, b=self.RHS)

        # Sort by eigenvalue
        sorter = np.argsort(np.abs(eigvals))
        eigvals = eigvals[sorter]
        eigvecs = eigvecs[:, sorter].T

        return (eigvals, eigvecs)

