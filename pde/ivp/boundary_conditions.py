"""
Boundary condition enforcers.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np


class Dirichlet(object):

    def __init__(self, var, i, vi, space):

        # Store inputs
        self.var = var
        self.i = i
        self.vi = vi
        self.space = space

        if space == 'xspace':
            self.enforce = self._x_enforce
        elif space == 'kspace':
            N = var.size - 1
            n = np.arange(N + 1)
            c = np.ones(N + 1)
            c[0] = 2
            c[N] = 2

            self.Ei = np.cos(n * i * np.pi / N)
            self._update_const = 2 * self.Ei / c / c[i] / N
            self._update = np.zeros(var.size)

            self.enforce = self._k_enforce
        else:
            "'space' must be 'xspace' or 'kspace'"

    def _x_enforce(self):

        self.var['xspace'][self.i] = self.vi

    def _k_enforce(self):

        # Place references
        update = self._update
        kdata = self.var['kspace']

        # Calculate update
        delta_j = self.vi - np.dot(self.Ei, kdata)
        update[:] = delta_j
        update *= self._update_const

        # Apply update
        kdata += update


class DoubleNeumann(object):

    def __init__(self, var, i, j, vi, k, l, vk, space):

        # Store inputs
        self.var = var
        self.i = i
        self.j = j
        self.vi = vi
        self.k = k
        self.l = l
        self.vk = vk
        self.space = space

        # Setup math
        N = var.size - 1
        n = np.arange(N + 1)
        c = np.ones(N + 1)
        c[0] = 2
        c[N] = 2

        self.Ei = np.cos(n * i * np.pi / N)
        self.Ek = np.cos(n * k * np.pi / N)

        xi = var.basis.grid[i]
        xj = var.basis.grid[j]
        xk = var.basis.grid[k]
        xl = var.basis.grid[l]

        self.dji = cardinal_deriv(j, i, xj, xi, N) * var._diff_scale
        self.dli = cardinal_deriv(l, i, xl, xi, N) * var._diff_scale
        self.djk = cardinal_deriv(j, k, xj, xk, N) * var._diff_scale
        self.dlk = cardinal_deriv(l, k, xl, xk, N) * var._diff_scale

        if space == 'xspace':
            self.enforce = self._x_enforce
        elif space == 'kspace':
            Ej = np.cos(n * j * np.pi / N)
            self.uj_const = 2 * Ej / c / c[j] / N
            El = np.cos(n * l * np.pi / N)
            self.ul_const = 2 * El / c / c[l] / N

            self.update = np.zeros(var.size)

            self.enforce = self._k_enforce
        else:
            "'space' must be 'xspace' or 'kspace'"

    def _x_enforce(self):

        var = self.var

        diff = self.var.differentiate(1, 'xspace')
        ei = self.vi - diff[self.i]
        ek = self.vk - diff[self.k]
        delta_l = (ek - ei * self.djk / self.dji) / (self.dlk - self.dli * self.djk / self.dji)
        delta_j = (ei - self.dli * delta_l) / self.dji

        var['xspace'][self.j] += delta_j
        var['xspace'][self.l] += delta_l

    def _k_enforce(self):

        var = self.var
        update = self.update

        diff = var.differentiate(1, 'kspace')
        ei = self.vi - np.dot(self.Ei, diff)
        ek = self.vk - np.dot(self.Ek, diff)
        delta_l = (ek - ei * self.djk / self.dji) / (self.dlk - self.dli * self.djk / self.dji)
        delta_j = (ei - self.dli * delta_l) / self.dji

        update[:] = delta_j * self.uj_const + delta_l * self.ul_const

        var['kspace'] += update


def cardinal_deriv(j, i, xj, xi, N):

    if i == j:
        if i == 0:
            dji = (1 + 2 * N**2) / 6.
        elif i == N:
            dji = -(1 + 2 * N**2) / 6.
        else:
            dji = -xj / 2 / (1 - xj**2)
    else:
        dji = (-1)**(i+j) / (xi - xj)
        if (i == 0) or (i == N):
            dji *= 2
        if (j == 0) or (j == N):
            dji /= 2

    return dji










# class Neumann(object):

#     def __init__(self, var, i, j, value, space):

#         # Store inputs
#         self.var = var
#         self.value = value

#         # Retrieve grid indeces
#         xi = x
#         xj = x_enforce
#         if xj is None:
#             xj = xi
#         if (xi not in var.grid) or (xj not in var.grid):
#             raise ValueError("Boundary condition must be specified on a grid point.")
#         i = np.where(var.grid == xi)[0][0]
#         j = np.where(var.grid == xj)[0][0]
#         xi = var._basis_coord(xi)
#         xj = var._basis_coord(xj)

#         # Setup math
#         N = var.size - 1
#         n = np.arange(N + 1)
#         c = np.ones(N + 1)
#         c[0] = 2
#         c[N] = 2

#         self.Ei = np.cos(n * i * np.pi / N)

#         if i == j:
#             if i == 0:
#                 dji = (1 + 2 * N**2) / 6.
#             elif i == N:
#                 dji = -(1 + 2 * N**2) / 6.
#             else:
#                 dji = -xj / 2 / (1 - xj**2)
#         else:
#             dji = (-1)**(i+j) * c[i] / c[j] / (xi - xj)

#         self.dji = cardinal_deriv(j, i, xj, xi, N) * var._diff_scale

#         Ej = np.cos(n * j * np.pi / N)
#         self.update_const = 2 * Ej / c / c[j] / N

#         # Allocate arrays
#         self.update = np.zeros(var.size)

#     def enforce(self):

#         # Place references
#         update = self.update
#         var = self.var

#         # Calculate update
#         epsilon_i = self.value - np.dot(self.Ei, var.differentiate(1, 'kspace'))
#         delta_j = epsilon_i / self.dji
#         update[:] = delta_j
#         update *= self.update_const

#         # Apply update
#         var['kspace'] += update
