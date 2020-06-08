"""FInite element Automatic Tabulator -- supports constructing and
evaluating arbitrary order Lagrange and many other elements.
Simplices in one, two, and three dimensions are supported."""

import pkg_resources

# Import finite element classes
from FIAT.finite_element import FiniteElement, CiarletElement   # noqa: F401

from .elements import (Argyris, Bernstein, Bell, QuinticArgyris,
                       BrezziDouglasMarini, BrezziDouglasFortinMarini,
                       DiscontinuousLagrange, DiscontinuousTaylor,
                       DiscontinuousRaviartThomas, Serendipity, DPC,
                       CubicHermite, Lagrange, GaussLobattoLegendre,
                       GaussLegendre, GaussRadau, Morley, Nedelec,
                       NedelecSecondKind, P0, RaviartThomas,
                       CrouzeixRaviart, Regge, HellanHerrmannJohnson,
                       Bubble, FacetBubble, TensorProductElement,
                       EnrichedElement, NodalEnrichedElement,
                       DiscontinuousElement, HDivTrace, MixedElement,
                       RestrictedElement, QuadratureElement)    # noqa: F401

from .elements import supported_elements, extra_elements        # noqa: F401

# Important functionality
from FIAT.reference_element import ufc_cell, ufc_simplex        # noqa: F401
from FIAT.quadrature import make_quadrature, create_quadrature  # noqa: F401
from FIAT.elements.hdivcurl import Hdiv, Hcurl                  # noqa: F401

__version__ = pkg_resources.get_distribution("fenics-fiat").version
