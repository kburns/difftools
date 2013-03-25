"""
Matrix solvers for linear BVPs and EPs.

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
            start = self.varstart[var]
            end = start + var.size
            var.coefficients = self.syscoeff[start:end]

    def LHS_block(self, var1, var2):
        """
        Return subblock of LHS.

        Parameters
        ----------
        var1 : variable/series object
            Pick rows specifying var1 values/coefficients.
        var2 : variable/series object
            Pick columns depending on var2 values/coefficients.

        """

        start1 = self.varstart[var1]
        start2 = self.varstart[var2]
        end1 = start1 + var1.size
        end2 = start2 + var2.size
        subblock = self.LHS[start1:end1, start2:end2]

        return subblock

    def RHS_block(self, var1, var2=None):
        """
        Return subblock of RHS.

        Parameters
        ----------
        var1 : variable/series object
            Pick rows specifying var1 values/coefficients.
        var2 : variable/series object, optional
            Pick columns depending on var2 values/coefficients, if RHS is 2-dimensional.

        """

        start1 = self.varstart[var1]
        end1 = start1 + var1.size
        subblock = self.RHS[start1:end1]

        if var2:
            start2 = self.varstart[var2]
            end2 = start2 + var2.size
            subblock = subblock[:, start2:end2]

        return subblock

    def set_dirichlet_bc(self, var, x, value):
        """
        Implement a Dirichlet boundary condition using boundary bordering.

        Parameters
        ----------
        var : variable/series object
            Applicable variable
        x : float
            Location where BC is specified.
        value : float
            Value of var at x

        """

        if x not in var.grid:
            raise ValueError("Boundary condition must be specified on a grid point.")

        # Get x evaluation row
        i = np.where(var.grid == x)[0][0]
        evalrow = var.E(var)[i]

        # Repalce x equation with BC
        start = self.varstart[var]
        end = start + var.size
        row = start + i
        self.LHS[row] = 0.
        self.LHS[row, start:end] = evalrow
        self.RHS[row] = value

    def set_neumann_bc(self, var, x, value):
        """
        Implement a Neumann boundary condition using boundary bordering.

        Parameters
        ----------
        var : variable/series object
            Applicable variable
        x : float
            Location where BC is specified.
        value : float
            First derivative of var at x

        """

        if x not in var.grid:
            raise ValueError("Boundary condition must be specified on a grid point.")

        # Get x derivative row
        i = np.where(var.grid == x)[0][0]
        diffrow = var.D(1, var)[i]

        # Replace x equation with BC
        start = self.varstart[var]
        end = start + var.size
        row = start + i
        self.LHS[row] = 0.
        self.LHS[row, start:end] = diffrow
        self.RHS[row] = value


class BoundaryValueProblem(_ProblemBase):
    """Linear boundary value problem solver."""

    def __init__(self, varlist):
        """
        Setup the boundary value problem

            LHS . u = RHS

        Parameters
        ----------
        varlist : list of variable/series objects
            Variables in system

        """

        # Inherited initialization
        _ProblemBase.__init__(self, varlist)

        # Construct matrices
        self.LHS = np.zeros((self.syssize, self.syssize), dtype=np.complex128)
        self.RHS = np.zeros(self.syssize, dtype=np.complex128)

    def solve(self):
        """Solve the BVP using a matrix solve."""

        # Matrix solve
        self.syscoeff[:] = linalg.solve(a=self.LHS, b=self.RHS)

        return self.varlist


class EigenProblem(_ProblemBase):
    """Linear eigenproblem solver."""

    def __init__(self, varlist):
        """
        Setup the generalized eigenvalue problem

            LHS . u = eigval * RHS . u

        Parameters
        ----------
        varlist : list of variable/series objects
            Variables in system

        """

        # Inherited initialization
        _ProblemBase.__init__(self, varlist)

        # Construct matrices
        self.LHS = np.zeros((self.syssize, self.syssize), dtype=np.complex128)
        self.RHS = np.zeros((self.syssize, self.syssize), dtype=np.complex128)

    def solve(self):
        """Solve the generalized eigenproblem."""

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = linalg.eig(a=self.LHS, b=self.RHS)

        # Sort by eigenvalue
        sorter = np.argsort(np.abs(eigvals))
        eigvals = eigvals[sorter]
        eigvecs = eigvecs.T[sorter]

        return (eigvals, eigvecs)

    def set_dirichlet_bc(self, var, x, value):
        """
        Implement a Dirichlet boundary condition using boundary bordering.

        Parameters
        ----------
        var : variable/series object
            Applicable variable
        x : float
            Location where BC is specified.
        value : float
            Value of var at x

        """

        # Check homogeneity
        if value != 0:
            raise NotImplementedError("Non-homogeneous Dirichlet BC not implemented.")

        # Inherited method
        _ProblemBase.set_dirichlet_bc(self, var, x, value)

    def set_neumann_bc(self, var, x, value):
        """
        Implement a Neumann boundary condition using boundary bordering.

        Parameters
        ----------
        var : variable/series object
            Applicable variable
        x : float
            Location where BC is specified.
        value : float
            First derivative of var at x

        """

        # Check homogeneity
        if value != 0:
            raise NotImplementedError("Non-homogeneous Neumann BC not implemented.")

        # Inherited method
        _ProblemBase.set_neumann_bc(self, var, x, value)

