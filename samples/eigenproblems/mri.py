"""
Test eigenproblem: frequencies of a plucked string.

Solve

    F_zz = - K**2 * h(z) * F

    F_z(-1) = 0
    F_z(1) = 0

    G = F_z / K

"""


import numpy as np
from difftools.public import *


# Setup
n_points = 500
basis = DoubleNeumannChebyshevExtremaPolynomials(n_points)
F = TruncatedSeries(basis)
EP = EigenProblem([F])

# Stratification
H = 1.
m = 3. / 2.
h = lambda z: (1 - (z/H)**2) ** m

# Operators
EP.LHS = basis.diffmatrix(2, basis.grid)
EP.RHS = basis.evalmatrix(basis.grid) * np.array([h(basis.grid)]).T

# Solve
eigvals, eigvecs = EP.solve()

# Convert eigenvalues
K = np.sqrt(-eigvals)

# Construct eigenfunctions
eigF = []
for i in xrange(eigvals.size):
    Fi = TruncatedSeries(basis)
    Fi.coefficients = eigfuncs[i]
    eigF.append(Fi)

