"""
Compute eigenfrequencies of a plucked string:

    u_xx = - L**2 * u

    u(-1) = 0
    u(1) = 0

"""


import numpy as np
from difftools.public import *


def main(res=128):

    # Setup
    basis = DoubleDirichletChebyshevExtremaPolynomials(res)
    y = TruncatedSeries(basis)
    EP = EigenProblem([y])

    # Operators
    EP.LHS = basis.diffmatrix(2, basis.grid)
    EP.RHS = basis.evalmatrix(basis.grid)

    # Solve
    eigvals, eigvecs = EP.solve()

    # Convert eigenvalues
    L = np.sqrt(-eigvals)

    # Construct eigenfunctions
    eigfuncs = []
    for i in xrange(eigvals.size):
        ef_i = TruncatedSeries(basis)
        ef_i.coefficients = eigvecs[i]
        eigfuncs.append(ef_i)

    return (L, eigvals, eigfuncs)

