# Copyright (C) 2005 The University of Chicago
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Robert C. Kirby
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

from FIAT import functional
from FIAT.finite_element import DualSet, CiarletElement
from FIAT.polynomials import ONPolynomialSet
import numpy as np


class P0Dual(DualSet):
    def __init__(self, ref_el):
        entity_ids = {}
        nodes = []
        vs = np.array(ref_el.get_vertices())
        bary = tuple(np.average(vs, 0))

        nodes = [functional.PointEvaluation(ref_el, bary)]
        entity_ids = {}
        top = ref_el.get_topology()
        for dim in sorted(top):
            entity_ids[dim] = {}
            for entity in sorted(top[dim]):
                entity_ids[dim][entity] = []

        entity_ids[dim] = {0: [0]}

        super().__init__(nodes, ref_el, entity_ids)


class P0(CiarletElement):
    def __init__(self, ref_el):
        poly_set = ONPolynomialSet(ref_el, 0)
        dual = P0Dual(ref_el)
        degree = 0
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super().__init__(poly_set, dual, degree, formdegree)
