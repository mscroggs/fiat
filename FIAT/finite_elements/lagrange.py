# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import functional
from FIAT.abstract_element import DualSet, CiarletElement
from FIAT.polynomials import ONPolynomialSet


class LagrangeDualSet(DualSet):
    """The dual basis for Lagrange elements.  This class works for
    simplices of any dimension.  Nodes are point evaluation at
    equispaced points."""

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
                entity_ids[dim][entity] = list(range(cur, cur + nnodes_cur))
                cur += nnodes_cur

        super().__init__(nodes, ref_el, entity_ids)


class Lagrange(CiarletElement):
    """The Lagrange finite element.  It is what it is."""

    def __init__(self, ref_el, degree):
        poly_set = ONPolynomialSet(ref_el, degree)
        dual = LagrangeDualSet(ref_el, degree)
        formdegree = 0  # 0-form
        super().__init__(poly_set, dual, degree, formdegree)
