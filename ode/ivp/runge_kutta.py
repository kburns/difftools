"""
Runge-Kutta integrators.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np


class _ExplicitRungeKutta(object):
    """Explicit Runge-Kutta base class."""

    def __init__(self, system_size, dtype=None):
        """
        Explicit Runge-Kutta integrator.

        Parameters
        ----------
        system_size : int
            System size
        dtype : data-type
            System data-type

        """

        # Allocate arrays for evaluation
        self._u0 = np.zeros(system_size, dtype=dtype)
        self._k = np.zeros((system_size, self.c.size), dtype=dtype)

    def integrate(self, f, t0, u0, dt):
        """
        Advance system by one time step.

        Parameters
        ----------
        f : function(t, u)
            System derivative function
        t0 : float
            Time
        u0 : 1d array
            System
        dt : float
            Time step

        """

        # Copy initial values, in case f modifies u0
        self._u0[:] = u0

        # Calculate stages
        for i in xrange(self.c.size):
            ti = t0 + dt * self.c[i]
            ui = self._u0 + dt * np.dot(self._k, self.a[i])
            self._k[:, i] = f(ti, ui)

        # Calculate step
        t1 = t0 + dt
        u1 = self._u0 + dt * np.dot(self._k, self.b)
        new_dt = dt

        return (t1, u1, new_dt)


class _EmbeddedRungeKutta(_ExplicitRungeKutta):
    """Embedded Runge-Kutta base class."""

    def integrate(self, f, t0, u0, dt):
        """
        Advance system by one time step.

        Parameters
        ----------
        f : function(t, u)
            System derivative function
        t0 : float
            Time
        u0 : 1d array
            System
        dt : float
            Time step

        """

        # Copy initial values, in case f modifies u0
        self._u0[:] = u0

        # Calculate stages
        for i in xrange(self.c.size):
            ti = t0 + dt * self.c[i]
            ui = self._u0 + dt * np.dot(self._k, self.a[i])
            self._k[:, i] = f(ti, ui)

        # Calculate error
        error = dt * np.dot(self._k, self.b_err - self.b)
        max_error = np.abs(error).max()

        # Assume  max_error = C * dt ** (order + 1)
        # Calculate new_dt such that  new_max_error = tolerance * new_dt
        C = max_error / dt ** (self.order + 1)
        new_dt = (self.tolerance / C) ** (1. / self.order)

        # Restrict new_dt to reasonable changes
        new_dt = min(max(new_dt, 0.1 * dt), 2 * dt)

        # Calculate step or rerun
        if max_error <= self.tolerance * dt:
            t1 = t0 + dt
            u1 = self._u0 + dt * np.dot(self._k, self.b)
            return (t1, u1, new_dt)
        else:
            return self.integrate(f, t0, self._u0, new_dt)


class Euler(_ExplicitRungeKutta):
    """First-order forward Euler method."""

    name = 'Euler'
    order = 1

    # Butcher tableau
    a = np.array([[0.]])
    b = np.array([1.])
    c = np.array([0.])


class RK4(_ExplicitRungeKutta):
    """Classic fourth-order Runge-Kutta method."""

    name = 'RK4'
    order = 4

    # Butcher tableau
    a = np.array([[0.,  0.,  0., 0.],
                  [0.5, 0.,  0., 0.],
                  [0.,  0.5, 0., 0.],
                  [0.,  0.,  1., 0.]])
    b = np.array([1., 2., 2., 1.]) / 6.
    c = np.array([0., 0.5, 0.5, 1.])


class RKCK(_EmbeddedRungeKutta):
    """Cash-Karp 4/5 embedded Runge-Kutta method."""

    name = 'RKCK'
    order = 4

    # Butcher tableau
    a = np.array([[0., 0., 0., 0., 0., 0.],
                  [1./5., 0., 0., 0., 0., 0.],
                  [3./40., 9./40., 0., 0., 0., 0.],
                  [3./10., -9./10., 6./5., 0., 0., 0.],
                  [-11./54., 5./2., -70./27., 35./27., 0., 0.],
                  [1631./55296., 175./512., 575./13824., 44275./110592., 253./4096., 0.]])
    b = np.array([37./378., 0., 250./621., 125./594., 0., 512./1771.])
    b_err = np.array([2825./27648., 0., 18575./48384., 13525./55296., 277./14336., 1./4.])
    c = np.array([0., 1./5., 3./10., 3./5., 1., 7./8.])


class RKDP(_EmbeddedRungeKutta):
    """Dormand-Prince 5/4 embedded Runge-Kutta method."""

    name = 'RKDP'
    order = 4

    # Butcher tableau
    a = np.array([[0., 0., 0., 0., 0., 0., 0.],
                  [1./5., 0., 0., 0., 0., 0., 0.],
                  [3./40., 9./40., 0., 0., 0., 0., 0.],
                  [44./45., -56./15., 32./9., 0., 0., 0., 0.],
                  [19372./6561., -25360./2187., 64448./6561., -212./729., 0., 0., 0.],
                  [9017./3168., -355./33., 46732./5247., 49./176., -5103./18656., 0., 0.],
                  [35./384., 0., 500./1113., 125./192., -2187./6784., 11./84., 0.]])
    b = np.array([35./384., 0., 500./1113., 125./192., -2187./6784., 11./84., 0.])
    b_err = np.array([5179./57600., 0., 7571./16695., 393./640., -92097./339200., 187./2100., 1./40.])
    c = np.array([0., 1./5., 3./10., 4./5., 8./9., 1., 1.])

