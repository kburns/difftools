"""
Test eigenproblem: frequencies of a plucked string.

Solve

    u_xx = - L**2 * u

    u(-1) = 0
    u(1) = 0

"""


import numpy as np
from difftools.public import *


# Setup
basis = ChebyshevExtremaCardinals(20)
y = TruncatedSeries(basis)
EP = EigenProblem([y])

# Operators
EP.LHS = basis.diffmatrix(2, basis.grid)
EP.RHS = basis.evalmatrix(basis.grid)
#EP.LHS_blocks[y][y][:, :] = y.basis.diffmatrix(2, basis.grid)
#EP.add_to_LHS(y, y, y.basis.diffmatrix(2, y.basis.grid))

# Boundary conditions
EP.set_dirichlet_bc(y, -1., 0.)
EP.set_dirichlet_bc(y, 1., 0.)

# Solve
eigvals, eigvecs = EP.solve()

# Convert eigenvalues
L = np.sqrt(-eigvals)

