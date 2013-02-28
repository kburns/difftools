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

        # Get start indeces and system size
        self.varstart = {}
        self.syssize = 0
        for var in self.varlist:
            self.varstart[var] = self.syssize
            self.syssize += var.size

        # Copy variable coefficients to system array
        self.syscoeff = np.concatenate([var.coefficients for var in self.varlist])

        # Make variable attributes views into the system array
        for var in self.varlist:
            S = self.varstart[var]
            N = var.size
            var.coefficients = self.syscoeff[S:S+N]

    def add_to_LHS(self, var1, var2, array):
        """
        var1        evaluation on var1 grid
        var2        evaluation of var2 basis
        array       array to add

        """

        S1 = self.varstart[var1]
        S2 = self.varstart[var2]
        N1 = var1.size
        N2 = var2.size

        self.LHS[S1:S1+N1, S2:S2+N2] += array

    def set_dirichlet_bc(self, var, x, value):

        if x not in var.basis.grid:
            raise ValueError("Boundary condition must be specified on a grid point.")

        i = np.where(var.basis.grid == x)[0][0]
        evalrow = var.basis.evalmatrix(var.basis.grid)[i]

        start = self.varstart[var]
        row = start + i

        self.LHS[row] = 0.
        self.LHS[row, start:start+var.size] = evalrow
        self.RHS[row] = value

    def set_neumann_bc(self, var, x, value):

        if x not in var.basis.grid:
            raise ValueError("Boundary condition must be specified on a grid point.")

        i = np.where(var.basis.grid == x)[0][0]
        diffrow = var.basis.diffmatrix(1, var.basis.grid)[i]

        start = self.varstart[var]
        row = start + i

        self.LHS[row] = 0.
        self.LHS[row, start:start+var.size] = diffrow
        self.RHS[row] = value


class BoundaryValueProblem(_ProblemBase):
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


class EigenProblem(_ProblemBase):
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

    def set_dirichlet_bc(self, var, x, value):

        if value != 0:
            raise NotImplementedError("Non-homogeneous Dirichlet BC not implemented.")

        _ProblemBase.set_dirichlet_bc(self, var, x, value)

    def set_neumann_bc(self, var, x, value):

        if value != 0:
            raise NotImplementedError("Non-homogeneous Neumann BC not implemented.")

        _ProblemBase.set_neumann_bc(self, var, x, value)

