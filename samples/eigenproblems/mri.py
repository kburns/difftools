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
DNCEP = DoubleNeumannChebyshevExtremaPolynomials(n_points)
F = TruncatedSeries(DNCEP)
EP = EigenProblem([F])

# Stratification
H = 1.
m = 3. / 2.
h = lambda z: (1 - (z/H)**2) ** m

# Operators
EP.LHS = F.basis.diffmatrix(2, F.basis.grid)
EP.RHS = F.basis.evalmatrix(F.basis.grid) * np.array([h(F.basis.grid)]).T

# Solve
eigvals, eigvecs = EP.solve()

# Convert eigenvalues
K = np.sqrt(-eigvals)

# Construct eigenfunctions
eigF = []
for i in xrange(eigvals.size):
    Fi = TruncatedSeries(DNCEP)
    Fi.coefficients = eigfuncs[i]
    eigF.append(Fi)

