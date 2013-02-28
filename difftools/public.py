"""
Public API for difftools package.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


from spectral.bases import (ChebyshevExtremaPolynomials,
                            ChebyshevExtremaCardinals,
                            ChebyshevRootsPolynomials,
                            ChebyshevRootsCardinals)
from spectral.recombined_bases import (DirichletChebyshevExtremaPolynomials,
                                       DoubleNeumannChebyshevExtremaPolynomials)
from spectral.series import TruncatedSeries
from solvers import BoundaryValueProblem, EigenProblem

