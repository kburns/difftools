"""
Compute eigenfrequencies of a plucked string:

    y_xx = - L**2 * y

    y(-1) = 0
    y(1) = 0

"""


import numpy as np
from difftools.public import *


def plucked_string(res=256, range=(-1., 1.)):

    # Setup
    basis = DoubleDirichletChebyshevExtremaPolynomials(res)
    y = TruncatedSeries(basis, range=range)
    EP = EigenProblem([y])

    # Operators
    EP.LHS = y.diffmatrix(2, y)
    EP.RHS = y.evalmatrix(y)

    # Solve
    eigvals, eigvecs = EP.solve()

    # Convert eigenvalues
    L = np.sqrt(-eigvals)

    # Construct eigenfunctions
    eigfuncs = []
    for i in xrange(eigvals.size):
        ef_i = TruncatedSeries(basis, range=range)
        ef_i.coefficients = eigvecs[i]
        eigfuncs.append(ef_i)

    return (L, eigvals, eigfuncs)

