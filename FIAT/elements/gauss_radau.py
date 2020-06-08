# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Robert C. Kirby (robert_kirby@baylor.edu), 2020


from FIAT import functional, quadrature
from FIAT.finite_element import CiarletElement, DualSet
from FIAT.polynomials import ONPolynomialSet
from FIAT.reference_element import LINE


class GaussRadauDualSet(DualSet):
    """The dual basis for 1D discontinuous elements with nodes at the
    Gauss-Radau points."""
    def __init__(self, ref_el, degree, right=True):
        # Do DG connectivity because it's bonkers to do one-sided assembly even
        # though we have an endpoint in the point set!
        entity_ids = {0: {0: [], 1: []},
                      1: {0: list(range(0, degree+1))}}
        lr = quadrature.RadauQuadratureLineRule(ref_el, degree+1, right)
        nodes = [functional.PointEvaluation(ref_el, x) for x in lr.pts]

        super().__init__(nodes, ref_el, entity_ids)


class GaussRadau(CiarletElement):
    """1D discontinuous element with nodes at the Gauss-Radau points."""
    def __init__(self, ref_el, degree):
        if ref_el.shape != LINE:
            raise ValueError("Gauss-Radau elements are only defined in one dimension.")
        poly_set = ONPolynomialSet(ref_el, degree)
        dual = GaussRadauDualSet(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super().__init__(poly_set, dual, degree, formdegree)
