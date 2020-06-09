# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2014
# Modified by Lizao Li (lzlarryli@gmail.com), 2016
# Modified by Matthew Scroggs (mws48@cam.ac.uk), 2020
"""
Abstract class and particular implementations of finite element
reference simplex geometry/topology.

Provides an abstract base class and particular implementations for the
reference simplex geometry and topology.
The rest of FIAT is abstracted over this module so that different
reference element geometry (e.g. a vertex at (0,0) versus at (-1,-1))
and orderings of entities have a single point of entry.

Currently implemented are UFC and Default Line, Triangle and Tetrahedron.
"""
from itertools import chain, product, count
from functools import reduce
from collections import defaultdict
import operator
import numpy as np
from .reference_element import Cell
from .simplex import UFCInterval, Point, DefaultLine


class TensorProductCell(Cell):
    """A cell that is the product of FIAT cells."""

    def __init__(self, *cells):
        # Vertices
        vertices = tuple(tuple(chain(*coords))
                         for coords in product(*[cell.get_vertices()
                                                 for cell in cells]))

        # Topology
        shape = tuple(len(c.get_vertices()) for c in cells)
        topology = {}
        for dim in product(*[cell.get_topology().keys()
                             for cell in cells]):
            topology[dim] = {}
            topds = [cell.get_topology()[d]
                     for cell, d in zip(cells, dim)]
            for tuple_ei in product(*[sorted(topd)for topd in topds]):
                tuple_vs = list(product(*[topd[ei]
                                          for topd, ei in zip(topds, tuple_ei)]))
                vs = tuple(np.ravel_multi_index(np.transpose(tuple_vs), shape))
                topology[dim][tuple_ei] = vs
            # flatten entity numbers
            topology[dim] = dict(enumerate(topology[dim][key]
                                           for key in sorted(topology[dim])))

        super().__init__("TENSORPRODUCT", vertices, topology)
        self.cells = tuple(cells)

    def _key(self):
        return self.cells

    @staticmethod
    def _split_slices(lengths):
        n = len(lengths)
        delimiter = [0] * (n + 1)
        for i in range(n):
            delimiter[i + 1] = delimiter[i] + lengths[i]
        return [slice(delimiter[i], delimiter[i+1])
                for i in range(n)]

    def get_dimension(self):
        """Returns the subelement dimension of the cell, a tuple of
        dimensions for each cell in the product."""
        return tuple(c.get_dimension() for c in self.cells)

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: dimension in each "direction" (tuple)
        """
        return TensorProductCell(*[c.construct_subelement(d)
                                   for c, d in zip(self.cells, dimension)])

    def get_entity_transform(self, dim, entity_i):
        """Returns a mapping of point coordinates from the
        `entity_i`-th subentity of dimension `dim` to the cell.

        :arg dim: subelement dimension (tuple)
        :arg entity_i: entity number (integer)
        """
        # unravel entity_i
        shape = tuple(len(c.get_topology()[d])
                      for c, d in zip(self.cells, dim))
        alpha = np.unravel_index(entity_i, shape)

        # entity transform on each subcell
        sct = [c.get_entity_transform(d, i)
               for c, d, i in zip(self.cells, dim, alpha)]

        slices = TensorProductCell._split_slices(dim)

        def transform(point):
            return list(chain(*[t(point[s])
                                for t, s in zip(sct, slices)]))
        return transform

    def volume(self):
        """Computes the volume in the appropriate dimensional measure."""
        return np.prod([c.volume() for c in self.cells])

    def compute_reference_normal(self, facet_dim, facet_i):
        """Returns the unit normal in infinity norm to facet_i of
        subelement dimension facet_dim."""
        assert len(facet_dim) == len(self.get_dimension())
        indicator = np.array(self.get_dimension()) - np.array(facet_dim)
        (cell_i,), = np.nonzero(indicator)

        n = []
        for i, c in enumerate(self.cells):
            if cell_i == i:
                n.extend(c.compute_reference_normal(facet_dim[i], facet_i))
            else:
                n.extend([0] * c.get_spatial_dimension())
        return np.asarray(n)

    def contains_point(self, point, epsilon=0):
        """Checks if reference cell contains given point
        (with numerical tolerance)."""
        lengths = [c.get_spatial_dimension() for c in self.cells]
        assert len(point) == sum(lengths)
        slices = TensorProductCell._split_slices(lengths)
        return reduce(operator.and_,
                      (c.contains_point(point[s], epsilon=epsilon)
                       for c, s in zip(self.cells, slices)),
                      True)


class UFCQuadrilateral(Cell):
    """This is the reference quadrilateral with vertices
    (0.0, 0.0), (0.0, 1.0), (1.0, 0.0) and (1.0, 1.0)."""

    def __init__(self):
        product = TensorProductCell(UFCInterval(), UFCInterval())
        pt = product.get_topology()

        verts = product.get_vertices()
        topology = flatten_entities(pt)

        super().__init__("QUADRILATERAL", verts, topology)

        self.product = product
        self.unflattening_map = compute_unflattening_map(pt)

    def get_dimension(self):
        """Returns the subelement dimension of the cell.  Same as the
        spatial dimension."""
        return self.get_spatial_dimension()

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        if dimension == 2:
            return self
        elif dimension == 1:
            return UFCInterval()
        elif dimension == 0:
            return Point()
        else:
            raise ValueError("Invalid dimension: %d" % (dimension,))

    def get_entity_transform(self, dim, entity_i):
        """Returns a mapping of point coordinates from the
        `entity_i`-th subentity of dimension `dim` to the cell.

        :arg dim: entity dimension (integer)
        :arg entity_i: entity number (integer)
        """
        d, e = self.unflattening_map[(dim, entity_i)]
        return self.product.get_entity_transform(d, e)

    def volume(self):
        """Computes the volume in the appropriate dimensional measure."""
        return self.product.volume()

    def compute_reference_normal(self, facet_dim, facet_i):
        """Returns the unit normal in infinity norm to facet_i."""
        assert facet_dim == 1
        d, i = self.unflattening_map[(facet_dim, facet_i)]
        return self.product.compute_reference_normal(d, i)

    def contains_point(self, point, epsilon=0):
        """Checks if reference cell contains given point
        (with numerical tolerance)."""
        return self.product.contains_point(point, epsilon=epsilon)


class UFCHexahedron(Cell):
    """This is the reference hexahedron with vertices
    (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0),
    (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0) and (1.0, 1.0, 1.0)."""

    def __init__(self):
        product = TensorProductCell(UFCInterval(), UFCInterval(), UFCInterval())
        pt = product.get_topology()

        verts = product.get_vertices()
        topology = flatten_entities(pt)

        super().__init__("HEXAHEDRON", verts, topology)

        self.product = product
        self.unflattening_map = compute_unflattening_map(pt)

    def get_dimension(self):
        """Returns the subelement dimension of the cell.  Same as the
        spatial dimension."""
        return self.get_spatial_dimension()

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        if dimension == 3:
            return self
        elif dimension == 2:
            return UFCQuadrilateral()
        elif dimension == 1:
            return UFCInterval()
        elif dimension == 0:
            return Point()
        else:
            raise ValueError("Invalid dimension: %d" % (dimension,))

    def get_entity_transform(self, dim, entity_i):
        """Returns a mapping of point coordinates from the
        `entity_i`-th subentity of dimension `dim` to the cell.

        :arg dim: entity dimension (integer)
        :arg entity_i: entity number (integer)
        """
        d, e = self.unflattening_map[(dim, entity_i)]
        return self.product.get_entity_transform(d, e)

    def volume(self):
        """Computes the volume in the appropriate dimensional measure."""
        return self.product.volume()

    def compute_reference_normal(self, facet_dim, facet_i):
        """Returns the unit normal in infinity norm to facet_i."""
        assert facet_dim == 2
        d, i = self.unflattening_map[(facet_dim, facet_i)]
        return self.product.compute_reference_normal(d, i)

    def contains_point(self, point, epsilon=0):
        """Checks if reference cell contains given point
        (with numerical tolerance)."""
        return self.product.contains_point(point, epsilon=epsilon)


def is_hypercube(cell):
    if isinstance(cell, (DefaultLine, UFCInterval, UFCQuadrilateral, UFCHexahedron)):
        return True
    elif isinstance(cell, TensorProductCell):
        return reduce(lambda a, b: a and b, [is_hypercube(c) for c in cell.cells])
    else:
        return False


def flatten_reference_cube(ref_el):
    """This function flattens a Tensor Product hypercube to the corresponding UFC hypercube"""
    flattened_cube = {2: UFCQuadrilateral(), 3: UFCHexahedron()}
    if np.sum(ref_el.get_dimension()) <= 1:
        # Just return point/interval cell arguments
        return ref_el
    else:
        # Handle cases where cell is a quad/cube constructed from a tensor product or
        # an already flattened element
        if is_hypercube(ref_el):
            return flattened_cube[np.sum(ref_el.get_dimension())]
        else:
            raise TypeError('Invalid cell type')


def flatten_entities(topology_dict):
    """This function flattens topology dict of TensorProductCell and entity_dofs dict of TensorProductElement"""

    flattened_entities = defaultdict(list)
    for dim in sorted(topology_dict.keys()):
        flat_dim = tuple_sum(dim)
        flattened_entities[flat_dim] += [v for k, v in sorted(topology_dict[dim].items())]

    return {dim: dict(enumerate(entities))
            for dim, entities in flattened_entities.items()}


def compute_unflattening_map(topology_dict):
    """This function returns unflattening map for the given tensor product topology dict."""

    counter = defaultdict(count)
    unflattening_map = {}

    for dim, entities in sorted(topology_dict.items()):
        flat_dim = tuple_sum(dim)
        for entity in entities:
            flat_entity = next(counter[flat_dim])
            unflattening_map[(flat_dim, flat_entity)] = (dim, entity)

    return unflattening_map


def tuple_sum(tree):
    """
    This function calculates the sum of elements in a tuple, it is needed to handle nested tuples in TensorProductCell.
    Example: tuple_sum(((1, 0), 1)) returns 2
    If input argument is not the tuple, returns input.
    """
    if isinstance(tree, tuple):
        return sum(map(tuple_sum, tree))
    else:
        return tree
