# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import quadrature, functional
from FIAT.abstract_element import DualSet, CiarletElement
from FIAT.polynomials import (ONPolynomialSet, PolynomialSet,
                              polynomial_set_union_normalized,
                              expansions)
import numpy as np
from itertools import chain


def RTSpace(ref_el, deg):
    """Constructs a basis for the the Raviart-Thomas space
    (P_k)^d + P_k x"""
    sd = ref_el.get_spatial_dimension()

    vec_Pkp1 = ONPolynomialSet(ref_el, deg + 1, (sd,))

    dimPkp1 = expansions.polynomial_dimension(ref_el, deg + 1)
    dimPk = expansions.polynomial_dimension(ref_el, deg)
    dimPkm1 = expansions.polynomial_dimension(ref_el, deg - 1)

    vec_Pk_indices = list(chain(*(range(i * dimPkp1, i * dimPkp1 + dimPk)
                                  for i in range(sd))))
    vec_Pk_from_Pkp1 = vec_Pkp1.take(vec_Pk_indices)

    Pkp1 = ONPolynomialSet(ref_el, deg + 1)
    PkH = Pkp1.take(list(range(dimPkm1, dimPk)))

    Q = quadrature.make_quadrature(ref_el, 2 * deg + 2)

    # have to work on this through "tabulate" interface
    # first, tabulate PkH at quadrature points
    Qpts = np.array(Q.get_points())
    Qwts = np.array(Q.get_weights())

    zero_index = tuple([0 for i in range(sd)])

    PkH_at_Qpts = PkH.tabulate(Qpts)[zero_index]
    Pkp1_at_Qpts = Pkp1.tabulate(Qpts)[zero_index]

    PkHx_coeffs = np.zeros((PkH.get_num_members(),
                            sd,
                            Pkp1.get_num_members()), "d")

    for i in range(PkH.get_num_members()):
        for j in range(sd):
            fooij = PkH_at_Qpts[i, :] * Qpts[:, j] * Qwts
            PkHx_coeffs[i, j, :] = np.dot(Pkp1_at_Qpts, fooij)

    PkHx = PolynomialSet(ref_el, deg, deg + 1,
                         vec_Pkp1.get_expansion_set(),
                         PkHx_coeffs, vec_Pkp1.get_dmats())

    return polynomial_set_union_normalized(vec_Pk_from_Pkp1, PkHx)


class RTDualSet(DualSet):
    """Dual basis for Raviart-Thomas elements consisting of point
    evaluation of normals on facets of codimension 1 and internal
    moments against polynomials"""

    def __init__(self, ref_el, degree):
        entity_ids = {}
        nodes = []

        sd = ref_el.get_spatial_dimension()
        t = ref_el.get_topology()

        # codimension 1 facets
        for i in range(len(t[sd - 1])):
            pts_cur = ref_el.make_points(sd - 1, i, sd + degree)
            for j in range(len(pts_cur)):
                pt_cur = pts_cur[j]
                f = functional.PointScaledNormalEvaluation(ref_el, i, pt_cur)
                nodes.append(f)

        # internal nodes.  Let's just use points at a lattice
        if degree > 0:
            cpe = functional.ComponentPointEvaluation
            pts = ref_el.make_points(sd, 0, degree + sd)
            for d in range(sd):
                for i in range(len(pts)):
                    l_cur = cpe(ref_el, d, (sd,), pts[i])
                    nodes.append(l_cur)

            # Q = quadrature.make_quadrature(ref_el, 2 * ( degree + 1 ))
            # qpts = Q.get_points()
            # Pkm1 = ONPolynomialSet(ref_el, degree - 1)
            # zero_index = tuple([0 for i in range(sd)])
            # Pkm1_at_qpts = Pkm1.tabulate(qpts)[zero_index]

            # for d in range(sd):
            #     for i in range(Pkm1_at_qpts.shape[0]):
            #         phi_cur = Pkm1_at_qpts[i, :]
            #         l_cur = functional.IntegralMoment(ref_el, Q, phi_cur, (d,), (sd,))
            #         nodes.append(l_cur)

        # sets vertices (and in 3d, edges) to have no nodes
        for i in range(sd - 1):
            entity_ids[i] = {}
            for j in range(len(t[i])):
                entity_ids[i][j] = []

        cur = 0

        # set codimension 1 (edges 2d, faces 3d) dof
        pts_facet_0 = ref_el.make_points(sd - 1, 0, sd + degree)
        pts_per_facet = len(pts_facet_0)
        entity_ids[sd - 1] = {}
        for i in range(len(t[sd - 1])):
            entity_ids[sd - 1][i] = list(range(cur, cur + pts_per_facet))
            cur += pts_per_facet

        # internal nodes, if applicable
        entity_ids[sd] = {0: []}
        if degree > 0:
            num_internal_nodes = expansions.polynomial_dimension(ref_el,
                                                                 degree - 1)
            entity_ids[sd][0] = list(range(cur, cur + num_internal_nodes * sd))

        super().__init__(nodes, ref_el, entity_ids)


class RaviartThomas(CiarletElement):
    """The Raviart-Thomas finite element"""

    def __init__(self, ref_el, q):

        degree = q - 1
        poly_set = RTSpace(ref_el, degree)
        dual = RTDualSet(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension() - 1  # (n-1)-form
        super().__init__(poly_set, dual, degree, formdegree,
                         mapping="contravariant piola")
