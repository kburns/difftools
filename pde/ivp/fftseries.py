

import numpy as np
import scipy.fftpack as fft


class ChebyshevSeries(object):

    def __init__(self, size, grid='extrema'):

        # Store inputs
        self.size = size
        self.grid = grid

        # Setup tranforms
        if grid == 'extrema':
            self.grid = self._extrema_grid()
            self._fwd = self._extrema_fwd
            self._bwd = self._extrema_bwd
        elif grid == 'roots':
            self.grid = self._roots_grid()
            self._fwd = self._roots_fwd
            self._bwd = self._roots_bwd
        else:
            raise ValueError("'grid' must be 'extrema' or 'roots'")

        # Setup data containers
        self.xdata = np.zeros(size)
        self.kdata = np.zeros(size)
        self._deriv = np.zeros(size)

        self.current_space = 'xspace'
        self.data = self.xdata

    def __getitem__(self, space):

        self.require_space(space)
        return self.data

    def __setitem__(self, space, data):

        if space == 'xspace':
            self.data = self.xdata
        elif space == 'kspace':
            self.data = self.kdata
        else:
            raise KeyError("space must be 'kspace' or 'xspace'")

        self.data[:] = data
        self.current_space = space

    def require_space(self, space):

        if space != self.current_space:
            if space == 'kspace':
                self.forward()
            elif space == 'xspace':
                self.backward()
            else:
                raise ValueError("space must be 'kspace' or 'xspace'")

    def _extrema_grid(self):

        N = self.size - 1
        i = np.arange(N + 1)
        x = np.cos(np.pi * i / N)

        return x

    def _extrema_fwd(self):

        self.kdata[:] = fft.dct(self.xdata, type=1, norm=None)
        self.kdata /= self.size - 1.
        self.kdata[0] /= 2.
        self.kdata[-1] /= 2.

    def _extrema_bwd(self):

        self.kdata[1:-1] /= 2.
        self.xdata[:] = fft.dct(self.kdata, type=1, norm=None)

    def _roots_grid(self):

        N = self.size
        i = np.arange(N)
        x = np.cos(np.pi * (2.*i + 1.) / (2.*N))

        return x

    def _roots_fwd(self):

        self.kdata[:] = fft.dct(self.xdata, type=2, norm=None)
        self.kdata /= self.size
        self.kdata[0] /= 2.

    def _roots_bwd(self):

        self.kdata[1:] /= 2.
        self.xdata[:] = fft.dct(self.kdata, type=3, norm=None)

    def forward(self):

        if self.current_space == 'kspace':
            raise ValueError('Cannot go forward from kspace')

        self._fwd()
        self.data = self.kdata
        self.current_space = 'kspace'

    def backward(self):

        if self.current_space == 'xspace':
            raise ValueError('Cannot go backward from xspace')

        self._bwd()
        self.data = self.xdata
        self.current_space = 'xspace'

    def differentiate(self, p):

        self.require_space('kspace')
        a = self.kdata
        b = self._deriv
        N = self.size - 1

        for derivative in xrange(p):
            b[N] = 0.
            b[N-1] = 2 * N * a[N]
            for i in xrange(N-2, 0, -1):
                b[i] = 2 * (i+1) * a[i+1] + b[i+2]
            b[0] = a[1] + b[2] / 2.

        return b





