# Copyright (C) 2010 Marie E. Rognes
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Marie E. Rognes <meg@simula.no> based on original
# implementation by Robert C. Kirby.
#
# Last changed: 2010-01-28

from FIAT.polynomials import ONPolynomialSet
from FIAT import functional
from FIAT.finite_element import DualSet, CiarletElement


def _initialize_entity_ids(topology):
    entity_ids = {}
    for (i, entity) in list(topology.items()):
        entity_ids[i] = {}
        for j in entity:
            entity_ids[i][j] = []
    return entity_ids


class CrouzeixRaviartDualSet(DualSet):
    """Dual basis for Crouzeix-Raviart element (linears continuous at
    boundary midpoints)."""

    def __init__(self, cell, degree):

        # Get topology dictionary
        d = cell.get_spatial_dimension()
        topology = cell.get_topology()

        # Initialize empty nodes and entity_ids
        entity_ids = _initialize_entity_ids(topology)
        nodes = [None for i in list(topology[d - 1].keys())]

        # Construct nodes and entity_ids
        for i in topology[d - 1]:

            # Construct midpoint
            x = cell.make_points(d - 1, i, d)[0]

            # Degree of freedom number i is evaluation at midpoint
            nodes[i] = functional.PointEvaluation(cell, x)
            entity_ids[d - 1][i] += [i]

        # Initialize super-class
        super().__init__(nodes, cell, entity_ids)


class CrouzeixRaviart(CiarletElement):
    """The Crouzeix-Raviart finite element:

    K:                 Triangle/Tetrahedron
    Polynomial space:  P_1
    Dual basis:        Evaluation at facet midpoints
    """

    def __init__(self, cell, degree):

        # Crouzeix Raviart is only defined for polynomial degree == 1
        if not (degree == 1):
            raise Exception("Crouzeix-Raviart only defined for degree 1")

        # Construct polynomial spaces, dual basis and initialize
        # FiniteElement
        space = ONPolynomialSet(cell, 1)
        dual = CrouzeixRaviartDualSet(cell, 1)
        super().__init__(space, dual, 1)
