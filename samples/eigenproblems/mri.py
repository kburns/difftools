"""
Test eigenproblem: frequencies of a plucked string.

Solve

    F_zz = - K**2 * h(z) * F
    h(z) = (1 - (z/H)**2) ** m
    G = F_z / K

    F_z(-1) = 0
    F_z(1) = 0

"""


import numpy as np
from difftools.public import *


def main(res = 512, m=3./2.):

    # Setup
    basis = DoubleNeumannChebyshevExtremaPolynomials(res)
    F = TruncatedSeries(basis)
    EP = EigenProblem([F])

    # Stratification
    H = 1.
    h = lambda z: (1 - (z/H)**2) ** m

    # Operators
    EP.LHS = basis.diffmatrix(2, basis.grid)
    EP.RHS = basis.evalmatrix(basis.grid) * np.array([h(basis.grid)]).T

    # Solve
    eigvals, eigvecs = EP.solve()

    # Convert eigenvalues
    K = np.sqrt(-eigvals)

    # Construct eigenfunctions
    eigfuncs = []
    for i in xrange(eigvals.size):
        ef_i = TruncatedSeries(basis)
        ef_i.coefficients = eigvecs[i]
        eigfuncs.append(ef_i)

    return (K, eigvals, eigfuncs)

