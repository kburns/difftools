"""
Model class definitions.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np


class Model(object):
    """
    Class for models, i.e. collections of fields and methods for computing
    time derivatives.

    """

    def __init__(self, bodies=[], forces=[], time=0.):

        # Store inputs
        self.bodies = bodies
        self.forces = forces
        self._t = time

        # Copy positions and velocities into model vectors
        positions = np.concatenate([body.position for body in self.bodies])
        velocities = np.concatenate([body.velocity for body in self.bodies])

        # Caste as a system of first-order equations
        self._rv = self._u = np.concatenate([positions, velocities])
        self._va = np.zeros_like(self._rv)

        # Make body object attributes views of the model vector
        n = len(self.bodies)
        self.n_bodies = n
        for i, body in enumerate(self.bodies):
            body.position = self._rv[3*i:3*(i+1)]
            body.velocity = self._rv[3*(n+i):3*(n+i+1)]
            body.acceleration = self._va[3*(n+i):3*(n+i+1)]

    def __getitem__(self, key):

        # Look for body with specified name
        for body in self.bodies:
            if body.name == key:
                return body

        # If none are found, raise exception
        raise KeyError(key)

    @property
    def time(self):
        return self._t

    def _f(self, t, u):
        """Wrapper for interfacing with timestepper."""

        self._t = t
        self._u[:] = u
        self.compute_forces()

        return self._va

    def compute_forces(self):
        """
        Compute all included forces to determine instantaneous acceleration of
        the included bodies.

        """

        # Copy velocities, zero accelerations
        n = self.n_bodies
        self._va[:3*n] = self._rv[3*n:]
        self._va[3*n:] = 0.

        # Compute all forces
        for force in self.forces:
            force.compute()

