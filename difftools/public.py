"""
Public API for difftools package.

Author: Keaton J. Burns <keaton.burns@gmail.com>

"""


from spectral.polynomial_bases import (ChebyshevExtremaPolynomials,
                                       ChebyshevRootsPolynomials)
from spectral.cardinal_bases import (ChebyshevExtremaCardinals,
                                     ChebyshevRootsCardinals)
from spectral.recombined_bases import (DoubleDirichletChebyshevExtremaPolynomials,
                                       DoubleNeumannChebyshevExtremaPolynomials)
from spectral.series import TruncatedSeries
from solvers import BoundaryValueProblem, EigenProblem

