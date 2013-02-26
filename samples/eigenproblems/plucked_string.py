"""
Test EigenProblem: frequencies of a plucked string.

Solve
    u_xx = - L**2 * u

"""


import numpy as np
from difftools.public import *


#basis = ChebyshevExtremaCardinals(100)
basis = DirichletChebyshevExtremaPolynomials(100)
y = TruncatedSeries(basis)
EP = EigenProblem([y])

#EV = y.basis.evalmatrix(y.basis.grid)

#EP.add_to_LHS(y, y, y.basis.diffmatrix(2, y.basis.grid))
EP.LHS = basis.diffmatrix(2, basis.grid)
EP.RHS = basis.evalmatrix(basis.grid)

# EP.set_boundary_condition(y, y.basis.grid[0], EV[0])
# EP.set_boundary_condition(y, y.basis.grid[-1], EV[-1])

# EP.set_dirichlet({1.: 0., -1.: 0.})

# EP.add_to_LHS(y, y, 'D2')

# -or- EP.add_to_LHS(v, p, )
# EP.set_boundary_condition(u, 'left', 'dirichlet', 0.)
# EP.set_boundary_condition(u, 'right', 'dirichlet', 0.)
# EP.implement_boundary_bordering()

#EP.LHS += basis.diffmatrix(2)
# EP.LHS[0, :] = 0.
# EP.LHS[0, 0] = 1.
# EP.LHS[-1, :] = 0.
# EP.LHS[-1, -1] = 1.

# EP.RHS[0, :] = 0.
# EP.RHS[-1, :] = 0.

eigvals, eigfuncs = EP.solve()

L = np.sqrt(-eigvals)

