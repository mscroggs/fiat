from .quadrature import make_quadrature             # noqa: F401
from .quadrature_schemes import create_quadrature   # noqa: F401
# TODO: Give these two functions more distinct names

from .quadrature import (QuadratureRule, make_tensor_product_quadrature,
                         GaussJacobiQuadratureLineRule,
                         GaussLobattoLegendreQuadratureLineRule,
                         GaussLegendreQuadratureLineRule,
                         RadauQuadratureLineRule, CollapsedQuadratureTriangleRule,
                         CollapsedQuadratureTetrahedronRule,
                         UFCTetrahedronFaceQuadratureRule)             # noqa: F401
