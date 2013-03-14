

import numpy as np
from difftools.public import *


def stratified_mri(res=256, m=3./2.):
    """
    Compute MRI channel mode velocities in a vertically stratified polytropic
    accretion disc.

    Parameters
    ----------
    res : int
        Number of grid points
    m : float
        Polytropic index

    Notes
    -----
    Equations:
        F_zz = - K**2 * h(z) * F
        h(z) = (1 - (z/H)**2) ** m

        F_z(-1) = 0
        F_z(1) = 0

    Following:
        Latter, H. N., Fromang, S., Gressel, O., 2010. MNRAS, 406, 848.

    """

    # Setup
    basis = DoubleNeumannChebyshevExtremaPolynomials(res)
    F = TruncatedSeries(basis)
    EP = EigenProblem([F])

    # Stratification
    H = 1.
    h = lambda z: (1. - (z/H)**2) ** m

    # Operators
    EP.LHS[:] = F.D(2, F)
    EP.RHS[:] = F.E(F) * np.array([h(F.grid)]).T

    # Solve
    eigvals, eigvecs = EP.solve()

    # Convert eigenvalues
    K = np.sqrt(-eigvals)

    # Construct eigenfunctions
    eigfuncs = []
    for i in xrange(eigvals.size):
        ef = F.duplicate()
        ef.coefficients = eigvecs[i]
        eigfuncs.append(ef)

    return (K, eigvals, eigfuncs)

