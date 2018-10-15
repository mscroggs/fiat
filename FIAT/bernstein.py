# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Miklós Homolya
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FIAT.  If not, see <https://www.gnu.org/licenses/>.

import itertools
import math

import numpy
import sympy

from FIAT.finite_element import FiniteElement
from FIAT.dual_set import DualSet
from FIAT.polynomial_set import mis


class BernsteinDualSet(DualSet):
    """The dual basis for Bernstein elements."""

    def __init__(self, ref_el, degree):
        # Initialise data structures
        topology = ref_el.get_topology()
        entity_ids = {dim: {entity_i: []
                            for entity_i in entities}
                      for dim, entities in topology.items()}

        # Calculate inverse topology
        inverse_topology = {vertices: (dim, entity_i)
                            for dim, entities in topology.items()
                            for entity_i, vertices in entities.items()}

        # Generate triangular barycentric indices
        dim = ref_el.get_spatial_dimension()
        alphas = [(degree - sum(beta),) + tuple(reversed(beta))
                  for beta in itertools.product(range(degree + 1), repeat=dim)
                  if sum(beta) <= degree]

        # Fill data structures
        nodes = []
        for i, alpha in enumerate(alphas):
            vertices, = numpy.nonzero(alpha)
            entity_dim, entity_i = inverse_topology[tuple(vertices)]
            entity_ids[entity_dim][entity_i].append(i)

            # Leave nodes unimplemented for now
            nodes.append(None)

        super(BernsteinDualSet, self).__init__(nodes, ref_el, entity_ids)


class Bernstein(FiniteElement):
    """A finite element with Bernstein polynomials as basis functions."""

    def __init__(self, ref_el, degree):
        dual = BernsteinDualSet(ref_el, degree)
        k = 0  # 0-form
        super(Bernstein, self).__init__(ref_el, dual, degree, k)

    def degree(self):
        """The degree of the polynomial space."""
        return self.get_order()

    def value_shape(self):
        """The value shape of the finite element functions."""
        return ()

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points.

        :arg order: The maximum order of derivative.
        :arg points: An iterable of points.
        :arg entity: Optional (dimension, entity number) pair
                     indicating which topological entity of the
                     reference element to tabulate on.  If ``None``,
                     default cell-wise tabulation is performed.
        """
        # Transform points to reference cell coordinates
        ref_el = self.get_reference_element()
        if entity is None:
            entity = (ref_el.get_spatial_dimension(), 0)

        entity_dim, entity_id = entity
        entity_transform = ref_el.get_entity_transform(entity_dim, entity_id)
        cell_points = list(map(entity_transform, points))

        # Construct Cartesian to Barycentric coordinate mapping
        vs = numpy.asarray(ref_el.get_vertices())
        B2R = numpy.vstack([vs.T, numpy.ones(len(vs))])
        R2B = numpy.linalg.inv(B2R)

        dim = ref_el.get_spatial_dimension()
        X = sympy.symbols('X Y Z')[:dim]
        B = R2B.dot(X + (1,))

        # Generate triangular barycentric indices
        deg = self.degree()
        etas = [(deg - sum(beta),) + tuple(reversed(beta))
                for beta in itertools.product(range(deg + 1), repeat=dim)
                if sum(beta) <= deg]

        result = {}
        for D in range(order + 1):
            for alpha in mis(dim, D):
                table = numpy.zeros((len(etas), len(cell_points)))
                for i, eta in enumerate(etas):
                    for j, point in enumerate(cell_points):
                        c = math.factorial(deg)
                        for k in eta:
                            c = c // math.factorial(k)
                        b = c * (B ** eta).prod()
                        e = sympy.diff(b, *zip(X, alpha))
                        table[i, j] = e.subs(dict(zip(X, point))).evalf()
                result[alpha] = table
        return result
