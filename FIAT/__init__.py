"""FInite element Automatic Tabulator -- supports constructing and
evaluating arbitrary order Lagrange and many other elements.
Simplices in one, two, and three dimensions are supported."""

import pkg_resources

# Import finite element classes
from FIAT.finite_element import FiniteElement, CiarletElement  # noqa: F401
from FIAT.elements.argyris import Argyris
from FIAT.elements.bernstein import Bernstein
from FIAT.elements.bell import Bell
from FIAT.elements.argyris import QuinticArgyris
from FIAT.elements.brezzi_douglas_marini import BrezziDouglasMarini
from FIAT.elements.brezzi_douglas_fortin_marini import BrezziDouglasFortinMarini
from FIAT.elements.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.elements.discontinuous_taylor import DiscontinuousTaylor
from FIAT.elements.discontinuous_raviart_thomas import DiscontinuousRaviartThomas
from FIAT.elements.serendipity import Serendipity
from FIAT.elements.discontinuous_pc import DPC
from FIAT.elements.hermite import CubicHermite
from FIAT.elements.lagrange import Lagrange
from FIAT.elements.gauss_lobatto_legendre import GaussLobattoLegendre
from FIAT.elements.gauss_legendre import GaussLegendre
from FIAT.elements.gauss_radau import GaussRadau
from FIAT.elements.morley import Morley
from FIAT.elements.nedelec import Nedelec
from FIAT.elements.nedelec_second_kind import NedelecSecondKind
from FIAT.elements.p0 import P0
from FIAT.elements.raviart_thomas import RaviartThomas
from FIAT.elements.crouzeix_raviart import CrouzeixRaviart
from FIAT.elements.regge import Regge
from FIAT.elements.hellan_herrmann_johnson import HellanHerrmannJohnson
from FIAT.elements.bubble import Bubble, FacetBubble
from FIAT.elements.tensor_product import TensorProductElement
from FIAT.elements.enriched import EnrichedElement
from FIAT.elements.nodal_enriched import NodalEnrichedElement
from FIAT.elements.discontinuous import DiscontinuousElement
from FIAT.elements.hdiv_trace import HDivTrace
from FIAT.elements.mixed import MixedElement                       # noqa: F401
from FIAT.elements.restricted import RestrictedElement             # noqa: F401
from FIAT.quadrature_element import QuadratureElement     # noqa: F401

# Important functionality
from FIAT.reference_element import ufc_cell, ufc_simplex  # noqa: F401
from FIAT.quadrature import make_quadrature               # noqa: F401
from FIAT.quadrature_schemes import create_quadrature     # noqa: F401
from FIAT.hdivcurl import Hdiv, Hcurl                     # noqa: F401

__version__ = pkg_resources.get_distribution("fenics-fiat").version

# List of supported elements and mapping to element classes
supported_elements = {"Argyris": Argyris,
                      "Bell": Bell,
                      "Bernstein": Bernstein,
                      "Brezzi-Douglas-Marini": BrezziDouglasMarini,
                      "Brezzi-Douglas-Fortin-Marini": BrezziDouglasFortinMarini,
                      "Bubble": Bubble,
                      "FacetBubble": FacetBubble,
                      "Crouzeix-Raviart": CrouzeixRaviart,
                      "Discontinuous Lagrange": DiscontinuousLagrange,
                      "S": Serendipity,
                      "DPC": DPC,
                      "Discontinuous Taylor": DiscontinuousTaylor,
                      "Discontinuous Raviart-Thomas": DiscontinuousRaviartThomas,
                      "Hermite": CubicHermite,
                      "Lagrange": Lagrange,
                      "Gauss-Lobatto-Legendre": GaussLobattoLegendre,
                      "Gauss-Legendre": GaussLegendre,
                      "Gauss-Radau": GaussRadau,
                      "Morley": Morley,
                      "Nedelec 1st kind H(curl)": Nedelec,
                      "Nedelec 2nd kind H(curl)": NedelecSecondKind,
                      "Raviart-Thomas": RaviartThomas,
                      "Regge": Regge,
                      "EnrichedElement": EnrichedElement,
                      "NodalEnrichedElement": NodalEnrichedElement,
                      "TensorProductElement": TensorProductElement,
                      "BrokenElement": DiscontinuousElement,
                      "HDiv Trace": HDivTrace,
                      "Hellan-Herrmann-Johnson": HellanHerrmannJohnson}

# List of extra elements
extra_elements = {"P0": P0,
                  "Quintic Argyris": QuinticArgyris}
