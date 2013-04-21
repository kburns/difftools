"""
Series representations.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


class TruncatedSeries(object):
    """Truncated series class."""

    def __init__(self, basis, range=(-1., 1.)):
        """
        An object representing a function as a truncated series.

        Parameters
        ----------
        basis : basis object
            Spectral basis for representation
        range : tuple of floats
            (start, end) of domain.

        """

        # Store inputs
        self.basis = basis
        self.range = range
        self.size = self.basis.size

        # Setup data containers
        self._xdata = self.basis.create_x_data()
        self._kdata = self.basis.create_k_data()

        # Initial space
        self.current_space = 'xspace'
        self.data = self._xdata

        # Setup grid transforms
        radius = (range[1] - range[0]) / 2.
        center = (range[1] + range[0]) / 2.
        self._diff_scale = 1. / radius
        self._basis_coord = lambda x: (x - center) / radius
        self._problem_coord = lambda x: center + (x * radius)
        self.grid = self._problem_coord(basis.grid)

    def __getitem__(self, space):

        self.require_space(space)

        return self.data

    def __setitem__(self, space, data):

        if space == 'xspace':
            self.data = self._xdata
        elif space == 'kspace':
            self.data = self._kdata
        else:
            raise KeyError("'space' must be 'xspace' or 'kspace'")

        self.data[:] = data
        self.current_space = space

    def require_space(self, space):

        if self.current_space != space:
            if space == 'xspace':
                self.backward()
            elif space == 'kspace':
                self.forward()
            else:
                raise ValueError("'space' must be 'kspace' or 'xspace'")

    def forward(self):

        if self.current_space == 'kspace':
            raise ValueError('Cannot perform forward transform from kspace.')

        self.basis.forward(self._xdata, self._kdata)
        self.data = self._kdata
        self.current_space = 'kspace'

    def backward(self):

        if self.current_space == 'xspace':
            raise ValueError('Cannot perform backward transform from xspace.')

        self.basis.backward(self._kdata, self._xdata)
        self.data = self._xdata
        self.current_space = 'xspace'

    def differentiate(self, p, space):

        if space == 'xspace':
            deriv = self.basis.x_diff(self['xspace'], p)
        elif space == 'kspace':
            deriv = self.basis.k_diff(self['kspace'], p)
        else:
            raise ValueError("'space' must be 'kspace' or 'xspace'")

        deriv *= self._diff_scale ** p

        return deriv

