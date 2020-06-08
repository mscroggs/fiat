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
POINT = 0
LINE = 1
TRIANGLE = 2
TETRAHEDRON = 3
QUADRILATERAL = 11
HEXAHEDRON = 111
TENSORPRODUCT = 99


class Cell(object):
    """Abstract class for a reference cell.  Provides accessors for
    geometry (vertex coordinates) as well as topology (orderings of
    vertices that make up edges, facecs, etc."""

    def __init__(self, shape, vertices, topology):
        """The constructor takes a shape code, the physical vertices expressed
        as a list of tuples of numbers, and the topology of a cell.

        The topology is stored as a dictionary of dictionaries t[i][j]
        where i is the dimension and j is the index of the facet of
        that dimension.  The result is a list of the vertices
        comprising the facet."""
        self.shape = shape
        self.vertices = vertices
        self.topology = topology

        # Given the topology, work out for each entity in the cell,
        # which other entities it contains.
        self.sub_entities = {}
        for dim, entities in topology.items():
            self.sub_entities[dim] = {}

            for e, v in entities.items():
                vertices = frozenset(v)
                sub_entities = []

                for dim_, entities_ in topology.items():
                    for e_, vertices_ in entities_.items():
                        if vertices.issuperset(vertices_):
                            sub_entities.append((dim_, e_))

                # Sort for the sake of determinism and by UFC conventions
                self.sub_entities[dim][e] = sorted(sub_entities)

        # Build connectivity dictionary for easier queries
        self.connectivity = {}
        for dim0, sub_entities in self.sub_entities.items():

            # Skip tensor product entities
            # TODO: Can we do something better?
            if isinstance(dim0, tuple):
                continue

            for entity, sub_sub_entities in sorted(sub_entities.items()):
                for dim1 in range(dim0+1):
                    d01_entities = filter(lambda x: x[0] == dim1, sub_sub_entities)
                    d01_entities = tuple(x[1] for x in d01_entities)
                    self.connectivity.setdefault((dim0, dim1), []).append(d01_entities)

    def _key(self):
        """Hashable object key data (excluding type)."""
        # Default: only type matters
        return None

    def __eq__(self, other):
        return type(self) == type(other) and self._key() == other._key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((type(self), self._key()))

    def get_shape(self):
        """Returns the code for the element's shape."""
        return self.shape

    def get_vertices(self):
        """Returns an iterable of the element's vertices, each stored as a
        tuple."""
        return self.vertices

    def get_spatial_dimension(self):
        """Returns the spatial dimension in which the element lives."""
        return len(self.vertices[0])

    def get_topology(self):
        """Returns a dictionary encoding the topology of the element.

        The dictionary's keys are the spatial dimensions (0, 1, ...)
        and each value is a dictionary mapping."""
        return self.topology

    def get_connectivity(self):
        """Returns a dictionary encoding the connectivity of the element.

        The dictionary's keys are the spatial dimensions pairs ((1, 0),
        (2, 0), (2, 1), ...) and each value is a list with entities
        of second dimension ordered by local dim0-dim1 numbering."""
        return self.connectivity

    def get_vertices_of_subcomplex(self, t):
        """Returns the tuple of vertex coordinates associated with the labels
        contained in the iterable t."""
        return tuple([self.vertices[ti] for ti in t])

    def get_dimension(self):
        """Returns the subelement dimension of the cell.  For tensor
        product cells, this a tuple of dimensions for each cell in the
        product.  For all other cells, this is the same as the spatial
        dimension."""
        raise NotImplementedError("Should be implemented in a subclass.")

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: `tuple` for tensor product cells, `int` otherwise
        """
        raise NotImplementedError("Should be implemented in a subclass.")

    def get_entity_transform(self, dim, entity_i):
        """Returns a mapping of point coordinates from the
        `entity_i`-th subentity of dimension `dim` to the cell.

        :arg dim: `tuple` for tensor product cells, `int` otherwise
        :arg entity_i: entity number (integer)
        """
        raise NotImplementedError("Should be implemented in a subclass.")
