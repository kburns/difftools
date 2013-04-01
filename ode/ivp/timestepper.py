"""
Timestepping framework.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


import numpy as np
import time


class TimeStepper(object):

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
        self.dt = 0.01
        self.sim_stop_time = 1.
        self.wall_stop_time = 60.
        self.stop_iteration = 100.

        # Instantiation time
        self.start_time = time.time()
        self.iteration = 0

    @property
    def ok(self):

        if self.model._stop_condition():
            ok_flag = False
            print 'Model stop condition satisfied.'
        elif self.model._t >= self.sim_stop_time:
            ok_flag = False
            print 'Simulation stop time reached.'
        elif (time.time() - self.start_time) >= self.wall_stop_time:
            ok_flag = False
            print 'Wall stop time reached.'
        elif self.iteration >= self.stop_iteration:
            ok_flag = False
            print 'Stop iteration reached.'
        else:
            ok_flag = True

        return ok_flag

    def advance(self):
        """Advance system by one time step."""

        # Run integrator
        (t1, u1, dt) = self.integrator.integrate(self.model._f,
                                                 self.model._t,
                                                 self.model._u,
                                                 self.dt)

        # Update model
        self.model._t = t1
        self.model._u[:] = u1
        self.dt = dt

        # Aim for final time
        if self.model._t + self.dt > self.sim_stop_time:
            self.dt = self.sim_stop_time - self.model._t

        self.iteration += 1

