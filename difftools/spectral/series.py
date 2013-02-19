

import numpy as np


class TruncatedSeries(object):

    def __init__(self, basis):

        self.basis = basis

        self.size = self.basis.size
        self.coefficients = np.zeros(self.size)

    def evaluate(self, x, index=False):

        scratch = np.empty(self.size)
        for j in xrange(self.size):
            scratch[j] = self.basis.evaluate(j, x, index=index)
        out = np.dot(scratch, self.coefficients)

        return out

    def derivative(self, p, x, index=False):

        scratch = np.empty(self.size)
        for j in xrange(self.size):
            scratch[j] = self.basis.derivative(p, j, x, index=index)
        out = np.dot(scratch, self.coefficients)

        return out

    def _set_to_grid_function(self, f):

        LHS = self.basis.evalmatrix()
        RHS = f(self.basis.grid)

        self.coefficients[:] = linalg.solve(a=LHS, b=RHS)
