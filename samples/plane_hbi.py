

import numpy as np
from difftools.ode.bvp.public import *


def plane_hbi(res=250, k=250.):
    """
    Compute HBI modes in a plane parallel model of the intracluster medium.

    Parameters
    ----------
    res : int
        Number of grid points
    k : float
        Horizontal wavenumber

    Notes
    -----
    Following:
        Latter, H. N., Kunz, M., 2012. MNRAS, 423, 1964.

    """

    column = lambda x: np.array([x]).T

    # Bases
    CEP = ChebyshevExtremaPolynomials(res)
    DDCEP = DoubleDirichletChebyshevExtremaPolynomials(res)
    DNCEP = DoubleNeumannChebyshevExtremaPolynomials(res)

    # Variables
    range = (0., 1.)
    r = TruncatedSeries(CEP, range=range)
    u = TruncatedSeries(DNCEP, range=range)
    w = TruncatedSeries(DDCEP, range=range)
    T = TruncatedSeries(DDCEP, range=range)
    A = TruncatedSeries(DNCEP, range=range)

    # Problem
    EP = EigenProblem([r, u, w, T, A])
    LHS = EP.LHS_block
    RHS = EP.RHS_block

    # Parameters
    Beta0 = 1e5
    G = 2.
    Kn0 = 1. / 1500.
    Re = 2.08 * G / Kn0
    Pe = 0.042 * G / Kn0
    zeta = 2.5 ** (7./2.) - 1.
    q = -2. / 7. * zeta

    # Equilibrium state
    T0 = lambda z: (1. + zeta*z) ** (2./7.)
    dz_lnT0 = lambda z: 2./7. * zeta / (1. + zeta*z)

    p0 = lambda z: np.exp(-7./5. * G/zeta * ((1. + zeta*z)**(5./7.) - 1.))
    dz_lnp0 = lambda z: -G * (1. + zeta*z)**(-2./7.)

    r0 = lambda z: p0(z) / T0(z)
    dz_lnr0 = lambda z: dz_lnp0(z) - dz_lnT0(z)

    # Density operators
    g = r.grid
    C1 = column(dz_lnr0(g))
    LHS(r, u)[:] = -1j * k * u.E(r)
    LHS(r, w)[:] = -C1*w.E(r) - w.D(1, r)
    RHS(r, r)[:] = r.E(r)

    # x velocity operators
    g = u.grid
    C1 = column(-1j * k * T0(g))
    C2 = column(2. / Beta0 / r0(g))
    C3 = column(-1j * k/Re * T0(g)**(5./2.) / r0(g))
    LHS(u, r)[:] = C1 * r.E(u)
    LHS(u, T)[:] = C1 * T.E(u)
    LHS(u, A)[:] = C2 * (k**2 * A.E(u) - A.D(2, u))
    LHS(u, u)[:] = -1./3. * C3 * 1j * k * u.E(u)
    LHS(u, w)[:] = 2./3. * C3 * w.D(1, u)
    RHS(u, u)[:] = u.E(u)

    # z velocity operators
    g = w.grid
    C1 = column(-T0(g))
    C2 = column(dz_lnp0(g))
    C3 = column(2./Re * T0(g)**(5./2.) / r0(g))
    C4 = column(5./2. * dz_lnT0(g))
    LHS(w, r)[:] = C1 * r.D(1, w)
    LHS(w, T)[:] = C1 * (C2*T.E(w) + T.D(1, w))
    LHS(w, w)[:] = 2./3. * C3 * (C4*w.D(1, w) + w.D(2, w))
    LHS(w, u)[:] = -1j/3. * k * C3 * (C4*u.E(w) + u.D(1, w))
    RHS(w, w)[:] = w.E(w)

    # Temperature operators
    g = T.grid
    C1 = column(3./2. * dz_lnT0(g))
    C2 = column(1. / Pe / p0(g))
    C3 = column(1. + zeta*g)
    LHS(T, u)[:] = -1j * k * u.E(T)
    LHS(T, w)[:] = -C1*w.E(T) - w.D(1, T)
    LHS(T, T)[:] = C2 * (2.*zeta*T.D(1, T) + C3*T.D(2, T))
    LHS(T, A)[:] = C2 * q * 1j * k * A.D(1, T)
    RHS(T, T)[:] = 3./2. * T.E(T)

    # Flux function operators
    LHS(A, u)[:] = -u.E(A)
    RHS(A, A)[:] =  A.E(A)

    # Solve
    eigvals, eigvecs = EP.solve()

    # Construct eigenfunctions
    eigfuncs = []
    for i in xrange(eigvals.size):
        ef = []
        for var in EP.varlist:
            start = EP.varstart[var]
            end = start + var.size
            evar = TruncatedSeries(var.basis, range=range)
            evar.coefficients = eigvecs[i, start:end]
            ef.append(evar)
        eigfuncs.append(ef)

    return (eigvals, eigfuncs)

