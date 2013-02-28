

import numpy as np
from bases import _BasisBase, ChebyshevExtremaPolynomials


class DoubleNeumannChebyshevExtremaPolynomials(_BasisBase):

    _full_basis_class = ChebyshevExtremaPolynomials

    def __init__(self, size):

        # Store inputs
        self.size = size - 2

        # Construct full basis object
        self.full_basis = self._full_basis_class(size)

        # Construct collocation grid
        self.grid = self.full_basis._construct_grid()[1:-1]

    def evaluate(self, j, x):

        k = j + 2.
        Tj = self.full_basis.evaluate(j, x)
        Tk = self.full_basis.evaluate(k, x)

        if j == 0:
            Pj = np.ones_like(x)
        elif j%2 == 0:
            n = j / 2.
            Pj = Tj - (n**2 / (n + 1.)**2) * Tk
        else:
            Pj = Tj - (j / k)**2 * Tk

        return Pj

    def derivative(self, p, j, x):

        k = j + 2.
        Tj_xp = self.full_basis.derivative(p, j, x)
        Tk_xp = self.full_basis.derivative(p, k, x)

        if j == 0:
            Pj_xp = np.zeros_like(x)
        elif j%2 == 0:
            n = j / 2.
            Pj_xp = Tj_xp - (n**2 / (n + 1.)**2) * Tk_xp
        else:
            Pj_xp = Tj_xp - (j / k)**2 * Tk_xp

        return Pj_xp


class DoubleDirichletChebyshevExtremaPolynomials(_BasisBase):

    _full_basis_class = ChebyshevExtremaPolynomials

    def __init__(self, size):

        # Store inputs
        self.size = size - 2

        # Construct full basis object
        self.full_basis = self._full_basis_class(size)

        # Construct collocation grid
        self.grid = self.full_basis._construct_grid()[1:-1]

    def evaluate(self, j, x):

        Tj = self.full_basis.evaluate(j + 2, x)

        if j%2 == 0:
            Pj = Tj - 1.
        else:
            Pj = Tj - x

        return Pj

    def derivative(self, p, j, x):

        Tj_xp = self.full_basis.derivative(p, j + 2, x)

        if j%2 == 0:
            Pj_xp = Tj_xp
        else:
            if p == 1:
                Pj_xp = Tj_xp - 1.
            else:
                Pj_xp = Tj_xp

        return Pj_xp

    def convert_to_full_basis(self):

        self.full_basis.coefficients[0] = -np.sum(self.coefficients[0::2])
        self.full_basis.coefficients[1] = -np.sum(self.coefficients[1::2])
        self.full_basis.coefficients[2:] = self.coefficients

        return self.full_basis

