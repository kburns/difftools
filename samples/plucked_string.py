

import numpy as np
from difftools.ode.bvp.public import *


def plucked_string(res=256, range=(-1., 1.)):
    """
    Compute eigenmodes of a plucked string.

    Parameters
    ----------
    res : int
        Number of grid points
    range : tuple of floats
        (start, end) of domain.

    Notes
    -----
    Equations:
        y_xx = - L**2 * y

        y(-1) = 0
        y(1) = 0

    """

    # Setup
    basis = DoubleDirichletChebyshevExtremaPolynomials(res)
    y = TruncatedSeries(basis, range=range)
    EP = EigenProblem([y])

    # Operators
    EP.LHS[:] = y.D(2, y)
    EP.RHS[:] = y.E(y)

    # Solve
    eigvals, eigvecs = EP.solve()

    # Convert eigenvalues
    L = np.sqrt(-eigvals)

    # Construct eigenfunctions
    eigfuncs = []
    for i in xrange(eigvals.size):
        ef = y.duplicate()
        ef.coefficients = eigvecs[i]
        eigfuncs.append(ef)

    return (L, eigvals, eigfuncs)

