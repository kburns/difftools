"""
Test EigenProblem: frequencies of a plucked string.

Solve
    F_zz = - K**2 * h(z) * F

"""


import numpy as np
from difftools.public import *


basis = DoubleNeumannChebyshevExtremaPolynomials(100)
F = TruncatedSeries(basis)
EP = EigenProblem([F])

H = 1.
m = 3. / 2.
h = lambda z: (1 - (z/H)**2) ** m

EP.LHS = basis.diffmatrix(2, basis.grid)
EP.RHS = basis.evalmatrix(basis.grid) * np.array([h(basis.grid)]).T

eigvals, eigfuncs = EP.solve()

K = np.sqrt(-eigvals)

