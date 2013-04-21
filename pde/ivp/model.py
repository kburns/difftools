"""
Model framework for IVPs.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np


class InitialValueProblem(object):

    def __init__(self, data, deriv, rhs, bc=[], intspace='xspace', time=0.):

        # Store inputs
        self.data = data
        self.deriv = deriv
        self.rhs = rhs
        self.boundary_conditions = bc
        self.intspace = intspace
        self._t = time

        # Get start indeces and system size
        startlist = []
        syssize = 0
        for var in data:
            startlist.append(syssize)
            syssize += var.size

        # Copy variable data to system array
        self._u = np.concatenate([var[intspace] for var in data])
        self._du = np.concatenate([var[intspace] for var in deriv])

        # Make variable attributes views into the system array
        for i in xrange(len(data)):
            start = startlist[i]
            end = start + data[i].size
            if intspace == 'xspace':
                data[i].data = data[i]._xdata = self._u[start:end]
                deriv[i].data = deriv[i]._xdata = self._du[start:end]
            elif intspace == 'kspace':
                data[i].data = data[i]._kdata = self._u[start:end]
                deriv[i].data = deriv[i]._kdata = self._du[start:end]

    @property
    def time(self):
        return self._t

    def _stop_condition(self):
        return False

    def _f(self, t, u):
        """Wrapper for interfacing with timestepper."""

        for var in self.data + self.deriv:
            var.require_space(self.intspace)

        self._t = t
        self._u[:] = u

        for bc in self.boundary_conditions:
            bc.enforce()

        self.rhs(self.data, self.deriv)

        for var in self.data + self.deriv:
            var.require_space(self.intspace)

        return self._du

    def pre_timestep(self):

        for var in self.data + self.deriv:
            var.require_space(self.intspace)

    def post_timestep(self):

        for bc in self.boundary_conditions:
            bc.enforce()
