"""
Timestepping framework.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np
import time


class TimeStepper:

    def __init__(self, model, integrator_class):
        """
        Timestepper for controlling time-evolution of a model.

        Parameters
        ----------
        model : object
            Model
        integrator_class : class
            Integrator class.

        Notes
        -----
        The model object must provide the following methods/attributes:

        _t : float
            Time
        _u : 1d array
            System
        _f(t, u) : function
            Derivative function
        _stop_condition() : function
            Halting function

        """

        # Store inputs
        self.model = model
        self.integrator = integrator_class(model._u.size)

        # Default parameters
        self.stop_sim_time = 1
        self.stop_wall_time = 1
        self.stop_iteration = 1

        # Instantiation time
        self.start_time = time.time()
        self.iteration = 0

    @property
    def ok(self):

        if self.model._stop_condition():
            print('Model stop condition satisfied.')
            return False
        elif self.model._t >= self.stop_sim_time:
            print('Simulation stop time reached.')
            return False
        elif (time.time() - self.start_time) >= self.stop_wall_time:
            print('Wall stop time reached.')
            return False
        elif self.iteration >= self.stop_iteration:
            print('Stop iteration reached.')
            return False
        else:
            return True

    def step(self, dt):
        """Advance system by one iteration/timestep."""

        # Run integrator
        (t1, u1, dt) = self.integrator.integrate(self.model._f,
                                                 self.model._t,
                                                 self.model._u,
                                                 dt)

        # Update model
        self.model._t = t1
        self.model._u[:] = u1

        # Update iteration
        self.iteration += 1

        return dt

