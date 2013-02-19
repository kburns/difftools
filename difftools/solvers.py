"""
Frameworks for solving various problems in differential equations.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np
import scipy.linalg as linalg


class BoundaryValueProblem(object):

    def __init__(self, varlist):
        """
        Setup the boundary value problem

            LHS . u = RHS

        Parameters
        ----------
        varlist : list of series objects
            Variables in system.

        """

        # Store inputs
        self.varlist = varlist

        # Construct matrices
        self.syssize = np.sum([var.size for var in self.varlist])
        self.LHS = np.zeros((self.syssize, self.syssize))
        self.RHS = np.zeros(self.syssize)

    def solve(self):
        """Solve the bvp using a matrix solve."""

        # Matrix solve
        u = linalg.solve(a=self.LHS, b=self.RHS)

        return u


class EigenProblem(object):

    def __init__(self, varlist):
        """
        Setup the eigenvalue problem

            LHS . u = eigval * RHS . u

        Parameters
        ----------
        varlist : list of series objects
            Variables in system.

        """

        # Store inputs
        self.varlist = varlist

        # Construct matrices
        self.syssize = np.sum([var.size for var in self.varlist])
        self.LHS = np.zeros((self.syssize, self.syssize))
        self.RHS = np.identity(self.syssize)

    def solve(self):
        """Solve the generalized eigenproblem."""

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = linalg.eig(a=self.LHS, b=self.RHS)

        # Sort by eigenvalue
        sorter = np.argsort(eigvals)
        eigvals = eigvals[sorter]
        eigvecs = eigvecs[:, sorter].T

        return (eigvals, eigvecs)

