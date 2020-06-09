# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by David A. Ham (david.ham@imperial.ac.uk), 2015

from FIAT import functional, quadrature
from FIAT.abstract_element import DualSet, CiarletElement
from FIAT.polynomials import ONPolynomialSet


class GaussLegendreDualSet(DualSet):
    """The dual basis for 1D discontinuous elements with nodes at the
    Gauss-Legendre points."""
    def __init__(self, ref_el, degree):
        entity_ids = {0: {0: [], 1: []},
                      1: {0: list(range(0, degree+1))}}
        lr = quadrature.GaussLegendreQuadratureLineRule(ref_el, degree+1)
        nodes = [functional.PointEvaluation(ref_el, x) for x in lr.pts]

        super().__init__(nodes, ref_el, entity_ids)


class GaussLegendre(CiarletElement):
    """1D discontinuous element with nodes at the Gauss-Legendre points."""
    def __init__(self, ref_el, degree):
        if ref_el.shape != "LINE":
            raise ValueError("Gauss-Legendre elements are only defined in one dimension.")
        poly_set = ONPolynomialSet(ref_el, degree)
        dual = GaussLegendreDualSet(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super().__init__(poly_set, dual, degree, formdegree)
