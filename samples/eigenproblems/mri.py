"""
Test eigenproblem: frequencies of a plucked string.

Solve

    F_zz = - K**2 * h(z) * F

    F_z(-1) = 0
    F_z(1) = 0

"""


import numpy as np
from difftools.public import *


# Setup
#basis = DoubleNeumannChebyshevExtremaPolynomials(100)
basis = ChebyshevExtremaCardinals(500)
F = TruncatedSeries(basis)
EP = EigenProblem([F])

H = 1.
m = 3. / 2.
h = lambda z: (1 - (z/H)**2) ** m

# Operators
EP.LHS = basis.diffmatrix(2, basis.grid)
EP.RHS = basis.evalmatrix(basis.grid) * np.array([h(basis.grid)]).T

# Boundary conditions
EP.set_neumann_bc(F, -1., 0.)
EP.set_neumann_bc(F, 1., 0.)

# Solve
eigvals, eigfuncs = EP.solve()

# Convert eigenvalues
K = np.sqrt(-eigvals)

