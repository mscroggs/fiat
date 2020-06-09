from .argyris import Argyris
from .bernstein import Bernstein
from .bell import Bell
from .argyris import QuinticArgyris
from .brezzi_douglas_marini import BrezziDouglasMarini
from .brezzi_douglas_fortin_marini import BrezziDouglasFortinMarini
from .discontinuous_lagrange import DiscontinuousLagrange
from .discontinuous_taylor import DiscontinuousTaylor
from .discontinuous_raviart_thomas import DiscontinuousRaviartThomas
from .serendipity import Serendipity
from .discontinuous_pc import DPC
from .hermite import CubicHermite
from .lagrange import Lagrange
from .gauss_lobatto_legendre import GaussLobattoLegendre
from .gauss_legendre import GaussLegendre
from .gauss_radau import GaussRadau
from .morley import Morley
from .nedelec import Nedelec
from .nedelec_second_kind import NedelecSecondKind
from .p0 import P0
from .raviart_thomas import RaviartThomas
from .crouzeix_raviart import CrouzeixRaviart
from .regge import Regge
from .hellan_herrmann_johnson import HellanHerrmannJohnson
from .bubble import Bubble, FacetBubble
from .tensor_product import TensorProductElement
from .enriched import EnrichedElement
from .nodal_enriched import NodalEnrichedElement
from .discontinuous import DiscontinuousElement
from .hdiv_trace import HDivTrace
from .mixed import MixedElement                       # noqa: F401
from .restricted import RestrictedElement             # noqa: F401
from .quadrature_element import QuadratureElement     # noqa: F401

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
