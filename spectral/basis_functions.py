

import numpy as np


class Chebyshev(object):

    def value(self, n, x):

        theta = np.arccos(x)
        cos_nt = np.cos(n * theta)

        T = cos_nt
        return T

    def derivative(self, deriv, n, x):

        theta = np.arccos(x)

        if deriv == 1:
            sin_t = np.sin(theta)
            sin_nt = np.sin(n * theta)
            out = n * sin_nt / sin_t

        elif deriv == 2:
            sin_t = np.sin(theta)
            sin_nt = np.sin(n * theta)
            cos_t = x
            cos_nt = np.cos(n * theta)
            out = n * sin_nt * cos_t / sin_t ** 3 - n ** 2 * cos_nt / sin_t ** 2

        else:
            raise ValueError("Higher order derivatives not implemented.")

        return out

    def endderivative(self, deriv, n, x):

        if x not in (-1., 1.):
            raise ValueError("Must evaluate at an endpoint.")
