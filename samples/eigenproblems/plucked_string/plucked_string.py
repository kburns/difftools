"""
Test eigenproblem: frequencies of a plucked string.

Solve

    u_xx = - L**2 * u

    u(-1) = 0
    u(1) = 0

"""


import numpy as np
import matplotlib.pyplot as plt
from difftools.public import *


def plucked_string(n_points):

    # Setup
    basis = DoubleDirichletChebyshevExtremaPolynomials(n_points)
    y = TruncatedSeries(basis)
    EP = EigenProblem([y])

    # Operators
    EP.LHS = basis.diffmatrix(2, basis.grid)
    EP.RHS = basis.evalmatrix(basis.grid)

    # Solve
    eigvals, eigvecs = EP.solve()

    # Convert eigenvalues
    L = np.sqrt(-eigvals)

    return (eigvals, eigvecs, L)


if __name__ == '__main__':

    # Solve at two resolutions
    s1 = 16
    s2 = 32
    (ev1, _, _) = plucked_string(s1)
    (ev2, _, _) = plucked_string(s2)

    # Compare to expected eigenvalues
    ex1 = -(np.arange(1, ev1.size+1) * np.pi / 2.) ** 2
    ex2 = -(np.arange(1, ev2.size+1) * np.pi / 2.) ** 2
    err1 = np.abs(ev1 - ex1)
    err2 = np.abs(ev2 - ex2)

    # Plot
    plt.figure(1)
    plt.clf()
    plt.semilogy(np.arange(1, ev1.size+1), err1, 'o-k', label='n=%i' %s1)
    plt.semilogy(np.arange(1, ev2.size+1), err2, 'o-k', mfc='w', mew=1, label='n=%i' %s2)
    plt.axhline(0.01, ls='dashed', c='k')
    plt.legend(loc='lower right')
    plt.xlabel(r'Mode number $j$')
    plt.ylabel(r'$j$-th eigvenvalue abs. error')
    plt.savefig('abs_error.png', dpi=200)

