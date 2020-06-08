# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import functional
from FIAT.finite_element import DualSet, CiarletElement
from FIAT.polynomials import ONPolynomialSet
from .p0 import P0


class DiscontinuousLagrangeDualSet(DualSet):
    """The dual basis for Lagrange elements.  This class works for
    simplices of any dimension.  Nodes are point evaluation at
    equispaced points.  This is the discontinuous version where
    all nodes are topologically associated with the cell itself"""

    def __init__(self, ref_el, degree):
        entity_ids = {}
        nodes = []

        # make nodes by getting points
        # need to do this dimension-by-dimension, facet-by-facet
        top = ref_el.get_topology()

        cur = 0
        for dim in sorted(top):
            entity_ids[dim] = {}
            for entity in sorted(top[dim]):
                pts_cur = ref_el.make_points(dim, entity, degree)
                nodes_cur = [functional.PointEvaluation(ref_el, x)
                             for x in pts_cur]
                nnodes_cur = len(nodes_cur)
                nodes += nodes_cur
                entity_ids[dim][entity] = []
                cur += nnodes_cur

        entity_ids[dim][0] = list(range(len(nodes)))

        super().__init__(nodes, ref_el, entity_ids)


class HigherOrderDiscontinuousLagrange(CiarletElement):
    """The discontinuous Lagrange finite element.  It is what it is."""

    def __init__(self, ref_el, degree):
        poly_set = ONPolynomialSet(ref_el, degree)
        dual = DiscontinuousLagrangeDualSet(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super().__init__(poly_set, dual, degree, formdegree)


def DiscontinuousLagrange(ref_el, degree):
    if degree == 0:
        return P0(ref_el)
    else:
        return HigherOrderDiscontinuousLagrange(ref_el, degree)
