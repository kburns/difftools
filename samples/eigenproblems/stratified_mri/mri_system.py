"""
Test eigenproblem: frequencies of a plucked string.

Solve

    F_z = K * G
    G_z = -K * h * F

    F_z(-1) = 0
    F_z(1) = 0

    G(-1) = 0
    G(1) = 0

"""


import numpy as np
from difftools.public import *


# Setup
DNCEP = DoubleNeumannChebyshevExtremaPolynomials(100)
DDCEP = DoubleDirichletChebyshevExtremaPolynomials(100)
CEP = ChebyshevExtremaPolynomials(100)
F = TruncatedSeries(DNCEP)
G = TruncatedSeries(DDCEP)
EP = EigenProblem([F, G])

H = 1.
m = 3. / 2.
h = lambda z: (1 - (z/H)**2) ** m

# Operators
EP.LHS_block(F, F)[:, :] = F.basis.diffmatrix(1, F.basis.grid)
EP.LHS_block(G, G)[:, :] = G.basis.diffmatrix(1, G.basis.grid)

EP.RHS_block(F, G)[:, :] = G.basis.evalmatrix(F.basis.grid)
EP.RHS_block(G, F)[:, :] = -F.basis.evalmatrix(G.basis.grid) * np.array([h(G.basis.grid)]).T

# Boundary conditions
# EP.set_dirichlet_bc(G, -1., 0.)
# EP.set_dirichlet_bc(G, 1., 0.)
# EP.set_neumann_bc(F, -1., 0.)
# EP.set_neumann_bc(F, 1., 0.)

# Solve
eigvals, eigfuncs = EP.solve()

# Convert eigenvalues
K = eigvals

