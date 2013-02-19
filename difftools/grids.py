

class GridBase(object):

    def __init__(self, size, functions):

        self.size = size
        self.basis = self.bases[functions]


class ChebyshevExtrema(GridBase):

    bases = {'polynomial': polynomials.Chebyshev,
             'cardinal': cardinals.ChebyshevExtrema}

    def _construct_grid
