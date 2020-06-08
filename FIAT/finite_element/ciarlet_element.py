# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2014
# Modified by Thomas H. Gibson (t.gibson15@imperial.ac.uk), 2016

import numpy as np

from FIAT.polynomial_set import PolynomialSet
from FIAT.quadrature.quadrature_schemes import create_quadrature
from .finite_element import FiniteElement


class CiarletElement(FiniteElement):
    """Class implementing Ciarlet's abstraction of a finite element
    being a domain, function space, and set of nodes.

    Elements derived from this class are nodal finite elements, with a nodal
    basis generated from polynomials encoded in a `PolynomialSet`.
    """

    def __init__(self, poly_set, dual, order, formdegree=None, mapping="affine", ref_el=None):
        ref_el = ref_el or poly_set.get_reference_element()
        super(CiarletElement, self).__init__(ref_el, dual, order, formdegree, mapping)

        # build generalized Vandermonde matrix
        old_coeffs = poly_set.get_coeffs()
        dualmat = dual.to_riesz(poly_set)

        shp = dualmat.shape
        if len(shp) > 2:
            num_cols = np.prod(shp[1:])

            A = np.reshape(dualmat, (dualmat.shape[0], num_cols))
            B = np.reshape(old_coeffs, (old_coeffs.shape[0], num_cols))
        else:
            A = dualmat
            B = old_coeffs

        V = np.dot(A, np.transpose(B))
        self.V = V

        Vinv = np.linalg.inv(V)

        new_coeffs_flat = np.dot(np.transpose(Vinv), B)

        new_shp = tuple([new_coeffs_flat.shape[0]] + list(shp[1:]))
        new_coeffs = np.reshape(new_coeffs_flat, new_shp)

        self.poly_set = PolynomialSet(ref_el,
                                      poly_set.get_degree(),
                                      poly_set.get_embedded_degree(),
                                      poly_set.get_expansion_set(),
                                      new_coeffs,
                                      poly_set.get_dmats())

    def degree(self):
        "Return the degree of the (embedding) polynomial space."
        return self.poly_set.get_embedded_degree()

    def get_nodal_basis(self):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        return self.poly_set

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        return self.poly_set.get_coeffs()

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
        if entity is None:
            entity = (self.ref_el.get_spatial_dimension(), 0)

        entity_dim, entity_id = entity
        transform = self.ref_el.get_entity_transform(entity_dim, entity_id)
        return self.poly_set.tabulate(list(map(transform, points)), order)

    def value_shape(self):
        "Return the value shape of the finite element functions."
        return self.poly_set.get_shape()

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        return self.get_nodal_basis().get_dmats()

    def get_num_members(self, arg):
        "Return number of members of the expansion set."
        return self.get_nodal_basis().get_expansion_set().get_num_members(arg)

    @staticmethod
    def is_nodal():
        """True if primal and dual bases are orthogonal. If false,
        dual basis is not implemented or is undefined.

        All implementations/subclasses are nodal including this one.
        """
        return True


def entity_support_dofs(elem, entity_dim):
    """Return the map of entity id to the degrees of freedom for which the
    corresponding basis functions take non-zero values

    :arg elem: FIAT finite element
    :arg entity_dim: Dimension of the cell subentity.
    """
    if not hasattr(elem, "_entity_support_dofs"):
        elem._entity_support_dofs = {}
    cache = elem._entity_support_dofs
    try:
        return cache[entity_dim]
    except KeyError:
        pass

    ref_el = elem.get_reference_element()
    dim = ref_el.get_spatial_dimension()

    entity_cell = ref_el.construct_subelement(entity_dim)
    quad = create_quadrature(entity_cell, max(2*elem.degree(), 1))
    weights = quad.get_weights()

    eps = 1.e-8  # Is this a safe value?

    result = {}
    for f in elem.entity_dofs()[entity_dim].keys():
        entity_transform = ref_el.get_entity_transform(entity_dim, f)
        points = list(map(entity_transform, quad.get_points()))

        # Integrate the square of the basis functions on the facet.
        vals = np.double(elem.tabulate(0, points)[(0,) * dim])
        # Ints contains the square of the basis functions
        # integrated over the facet.
        if elem.value_shape():
            # Vector-valued functions.
            ints = np.dot(np.einsum("...ij,...ij->...j", vals, vals), weights)
        else:
            ints = np.dot(vals**2, weights)

        result[f] = [dof for dof, i in enumerate(ints) if i > eps]

    cache[entity_dim] = result
    return result
