# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2014
# Modified by Lizao Li (lzlarryli@gmail.com), 2016
# Modified by Matthew Scroggs (mws48@cam.ac.uk), 2020
import numpy as np
from .reference_cell import Cell
from . import POINT, LINE, TRIANGLE, TETRAHEDRON
from .lattice import make_lattice
from math import factorial


class Simplex(Cell):
    """Abstract class for a reference simplex."""

    def compute_normal(self, facet_i):
        """Returns the unit normal vector to facet i of codimension 1."""
        # Interval case
        if self.get_shape() == LINE:
            verts = np.asarray(self.vertices)
            v_i, = self.get_topology()[0][facet_i]
            n = verts[v_i] - verts[[1, 0][v_i]]
            return n / np.linalg.norm(n)

        # first, let's compute the span of the simplex
        # This is trivial if we have a d-simplex in R^d.
        # Not so otherwise.
        vert_vecs = [np.array(v)
                     for v in self.vertices]
        vert_vecs_foo = np.array([vert_vecs[i] - vert_vecs[0]
                                  for i in range(1, len(vert_vecs))])

        (u, s, vt) = np.linalg.svd(vert_vecs_foo)
        rank = len([si for si in s if si > 1.e-10])

        # this is the set of vectors that span the simplex
        spanu = u[:, :rank]

        t = self.get_topology()
        sd = self.get_spatial_dimension()
        vert_coords_of_facet = \
            self.get_vertices_of_subcomplex(t[sd-1][facet_i])

        # now I find everything normal to the facet.
        vcf = [np.array(foo)
               for foo in vert_coords_of_facet]
        facet_span = np.array([vcf[i] - vcf[0]
                               for i in range(1, len(vcf))])
        (uf, sf, vft) = np.linalg.svd(facet_span)

        # now get the null space from vft
        rankfacet = len([si for si in sf if si > 1.e-10])
        facet_normal_space = np.transpose(vft[rankfacet:, :])

        # now, I have to compute the intersection of
        # facet_span with facet_normal_space
        foo = linalg_subspace_intersection(facet_normal_space, spanu)

        num_cols = foo.shape[1]

        if num_cols != 1:
            raise Exception("barf in normal computation")

        # now need to get the correct sign
        # get a vector in the direction
        nfoo = foo[:, 0]

        # what is the vertex not in the facet?
        verts_set = set(t[sd][0])
        verts_facet = set(t[sd - 1][facet_i])
        verts_diff = verts_set.difference(verts_facet)
        if len(verts_diff) != 1:
            raise Exception("barf in normal computation: getting sign")
        vert_off = verts_diff.pop()
        vert_on = verts_facet.pop()

        # get a vector from the off vertex to the facet
        v_to_facet = np.array(self.vertices[vert_on]) \
            - np.array(self.vertices[vert_off])

        if np.dot(v_to_facet, nfoo) > 0.0:
            return nfoo
        else:
            return -nfoo

    def compute_tangents(self, dim, i):
        """Computes tangents in any dimension based on differences
        between vertices and the first vertex of the i:th facet
        of dimension dim.  Returns a (possibly empty) list.
        These tangents are *NOT* normalized to have unit length."""
        t = self.get_topology()
        vs = list(map(np.array, self.get_vertices_of_subcomplex(t[dim][i])))
        ts = [v - vs[0] for v in vs[1:]]
        return ts

    def compute_normalized_tangents(self, dim, i):
        """Computes tangents in any dimension based on differences
        between vertices and the first vertex of the i:th facet
        of dimension dim.  Returns a (possibly empty) list.
        These tangents are normalized to have unit length."""
        ts = self.compute_tangents(dim, i)
        return [t / np.linalg.norm(t) for t in ts]

    def compute_edge_tangent(self, edge_i):
        """Computes the nonnormalized tangent to a 1-dimensional facet.
        returns a single vector."""
        t = self.get_topology()
        (v0, v1) = self.get_vertices_of_subcomplex(t[1][edge_i])
        return np.array(v1) - np.array(v0)

    def compute_normalized_edge_tangent(self, edge_i):
        """Computes the unit tangent vector to a 1-dimensional facet"""
        v = self.compute_edge_tangent(edge_i)
        return v / np.linalg.norm(v)

    def compute_face_tangents(self, face_i):
        """Computes the two tangents to a face.  Only implemented
        for a tetrahedron."""
        if self.get_spatial_dimension() != 3:
            raise Exception("can't get face tangents yet")
        t = self.get_topology()
        (v0, v1, v2) = list(map(np.array,
                                self.get_vertices_of_subcomplex(t[2][face_i])))
        return (v1 - v0, v2 - v0)

    def compute_face_edge_tangents(self, dim, entity_id):
        """Computes all the edge tangents of any k-face with k>=1.
        The result is a array of binom(dim+1,2) vectors.
        This agrees with `compute_edge_tangent` when dim=1.
        """
        vert_ids = self.get_topology()[dim][entity_id]
        vert_coords = [np.array(x)
                       for x in self.get_vertices_of_subcomplex(vert_ids)]
        edge_ts = []
        for source in range(dim):
            for dest in range(source + 1, dim + 1):
                edge_ts.append(vert_coords[dest] - vert_coords[source])
        return edge_ts

    def make_points(self, dim, entity_id, order):
        """Constructs a lattice of points on the entity_id:th
        facet of dimension dim.  Order indicates how many points to
        include in each direction."""
        if dim == 0:
            return (self.get_vertices()[entity_id], )
        elif 0 < dim < self.get_spatial_dimension():
            entity_verts = \
                self.get_vertices_of_subcomplex(
                    self.get_topology()[dim][entity_id])
            return make_lattice(entity_verts, order, 1)
        elif dim == self.get_spatial_dimension():
            return make_lattice(self.get_vertices(), order, 1)
        else:
            raise ValueError("illegal dimension")

    def volume(self):
        """Computes the volume of the simplex in the appropriate
        dimensional measure."""
        return volume(self.get_vertices())

    def volume_of_subcomplex(self, dim, facet_no):
        vids = self.topology[dim][facet_no]
        return volume(self.get_vertices_of_subcomplex(vids))

    def compute_scaled_normal(self, facet_i):
        """Returns the unit normal to facet_i of scaled by the
        volume of that facet."""
        dim = self.get_spatial_dimension()
        v = self.volume_of_subcomplex(dim - 1, facet_i)
        return self.compute_normal(facet_i) * v

    def compute_reference_normal(self, facet_dim, facet_i):
        """Returns the unit normal in infinity norm to facet_i."""
        assert facet_dim == self.get_spatial_dimension() - 1
        n = Simplex.compute_normal(self, facet_i)  # skip UFC overrides
        return n / np.linalg.norm(n, np.inf)

    def get_entity_transform(self, dim, entity):
        """Returns a mapping of point coordinates from the
        `entity`-th subentity of dimension `dim` to the cell.

        :arg dim: subentity dimension (integer)
        :arg entity: entity number (integer)
        """
        topology = self.get_topology()
        celldim = self.get_spatial_dimension()
        codim = celldim - dim
        if dim == 0:
            # Special case vertices.
            i, = topology[dim][entity]
            vertex = self.get_vertices()[i]
            return lambda point: vertex
        elif dim == celldim:
            assert entity == 0
            return lambda point: point

        try:
            subcell = self.construct_subelement(dim)
        except NotImplementedError:
            # Special case for 1D elements.
            x_c, = self.get_vertices_of_subcomplex(topology[0][entity])
            return lambda x: x_c

        subdim = subcell.get_spatial_dimension()

        assert subdim == celldim - codim

        # Entity vertices in entity space.
        v_e = np.asarray(subcell.get_vertices())

        A = np.zeros([subdim, subdim])

        for i in range(subdim):
            A[i, :] = (v_e[i + 1] - v_e[0])
            A[i, :] /= A[i, :].dot(A[i, :])

        # Entity vertices in cell space.
        v_c = np.asarray(self.get_vertices_of_subcomplex(topology[dim][entity]))

        B = np.zeros([celldim, subdim])

        for j in range(subdim):
            B[:, j] = (v_c[j + 1] - v_c[0])

        C = B.dot(A)

        offset = v_c[0] - C.dot(v_e[0])

        return lambda x: offset + C.dot(x)

    def get_dimension(self):
        """Returns the subelement dimension of the cell.  Same as the
        spatial dimension."""
        return self.get_spatial_dimension()


# Backwards compatible name
ReferenceElement = Simplex


class UFCSimplex(Simplex):

    def get_facet_element(self):
        dimension = self.get_spatial_dimension()
        return self.construct_subelement(dimension - 1)

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return ufc_simplex(dimension)

    def contains_point(self, point, epsilon=0):
        """Checks if reference cell contains given point
        (with numerical tolerance)."""
        result = (sum(point) - epsilon <= 1)
        for c in point:
            result &= (c + epsilon >= 0)
        return result


class Point(Simplex):
    """This is the reference point."""

    def __init__(self):
        verts = ((),)
        topology = {0: {0: (0,)}}
        super(Point, self).__init__(POINT, verts, topology)


class DefaultLine(Simplex):
    """This is the reference line with vertices (-1.0,) and (1.0,)."""

    def __init__(self):
        verts = ((-1.0,), (1.0,))
        edges = {0: (0, 1)}
        topology = {0: {0: (0,), 1: (1,)},
                    1: edges}
        super(DefaultLine, self).__init__(LINE, verts, topology)

    def get_facet_element(self):
        raise NotImplementedError()


class UFCInterval(UFCSimplex):
    """This is the reference interval with vertices (0.0,) and (1.0,)."""

    def __init__(self):
        verts = ((0.0,), (1.0,))
        edges = {0: (0, 1)}
        topology = {0: {0: (0,), 1: (1,)},
                    1: edges}
        super(UFCInterval, self).__init__(LINE, verts, topology)


class DefaultTriangle(Simplex):
    """This is the reference triangle with vertices (-1.0,-1.0),
    (1.0,-1.0), and (-1.0,1.0)."""

    def __init__(self):
        verts = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0))
        edges = {0: (1, 2),
                 1: (2, 0),
                 2: (0, 1)}
        faces = {0: (0, 1, 2)}
        topology = {0: {0: (0,), 1: (1,), 2: (2,)},
                    1: edges, 2: faces}
        super(DefaultTriangle, self).__init__(TRIANGLE, verts, topology)

    def get_facet_element(self):
        return DefaultLine()


class UFCTriangle(UFCSimplex):
    """This is the reference triangle with vertices (0.0,0.0),
    (1.0,0.0), and (0.0,1.0)."""

    def __init__(self):
        verts = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        edges = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        faces = {0: (0, 1, 2)}
        topology = {0: {0: (0,), 1: (1,), 2: (2,)},
                    1: edges, 2: faces}
        super(UFCTriangle, self).__init__(TRIANGLE, verts, topology)

    def compute_normal(self, i):
        "UFC consistent normal"
        t = self.compute_tangents(1, i)[0]
        n = np.array((t[1], -t[0]))
        return n / np.linalg.norm(n)


class IntrepidTriangle(Simplex):
    """This is the Intrepid triangle with vertices (0,0),(1,0),(0,1)"""

    def __init__(self):
        verts = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        edges = {0: (0, 1),
                 1: (1, 2),
                 2: (2, 0)}
        faces = {0: (0, 1, 2)}
        topology = {0: {0: (0,), 1: (1,), 2: (2,)},
                    1: edges, 2: faces}
        super(IntrepidTriangle, self).__init__(TRIANGLE, verts, topology)

    def get_facet_element(self):
        # I think the UFC interval is equivalent to what the
        # IntrepidInterval would be.
        return UFCInterval()


class DefaultTetrahedron(Simplex):
    """This is the reference tetrahedron with vertices (-1,-1,-1),
    (1,-1,-1),(-1,1,-1), and (-1,-1,1)."""

    def __init__(self):
        verts = ((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0),
                 (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0))
        vs = {0: (0, ),
              1: (1, ),
              2: (2, ),
              3: (3, )}
        edges = {0: (1, 2),
                 1: (2, 0),
                 2: (0, 1),
                 3: (0, 3),
                 4: (1, 3),
                 5: (2, 3)}
        faces = {0: (1, 3, 2),
                 1: (2, 3, 0),
                 2: (3, 1, 0),
                 3: (0, 1, 2)}
        tets = {0: (0, 1, 2, 3)}
        topology = {0: vs, 1: edges, 2: faces, 3: tets}
        super(DefaultTetrahedron, self).__init__(TETRAHEDRON, verts, topology)

    def get_facet_element(self):
        return DefaultTriangle()


class IntrepidTetrahedron(Simplex):
    """This is the reference tetrahedron with vertices (0,0,0),
    (1,0,0),(0,1,0), and (0,0,1) used in the Intrepid project."""

    def __init__(self):
        verts = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        vs = {0: (0, ),
              1: (1, ),
              2: (2, ),
              3: (3, )}
        edges = {0: (0, 1),
                 1: (1, 2),
                 2: (2, 0),
                 3: (0, 3),
                 4: (1, 3),
                 5: (2, 3)}
        faces = {0: (0, 1, 3),
                 1: (1, 2, 3),
                 2: (0, 3, 2),
                 3: (0, 2, 1)}
        tets = {0: (0, 1, 2, 3)}
        topology = {0: vs, 1: edges, 2: faces, 3: tets}
        super(IntrepidTetrahedron, self).__init__(TETRAHEDRON, verts, topology)

    def get_facet_element(self):
        return IntrepidTriangle()


class UFCTetrahedron(UFCSimplex):
    """This is the reference tetrahedron with vertices (0,0,0),
    (1,0,0),(0,1,0), and (0,0,1)."""

    def __init__(self):
        verts = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        vs = {0: (0, ),
              1: (1, ),
              2: (2, ),
              3: (3, )}
        edges = {0: (2, 3),
                 1: (1, 3),
                 2: (1, 2),
                 3: (0, 3),
                 4: (0, 2),
                 5: (0, 1)}
        faces = {0: (1, 2, 3),
                 1: (0, 2, 3),
                 2: (0, 1, 3),
                 3: (0, 1, 2)}
        tets = {0: (0, 1, 2, 3)}
        topology = {0: vs, 1: edges, 2: faces, 3: tets}
        super(UFCTetrahedron, self).__init__(TETRAHEDRON, verts, topology)

    def compute_normal(self, i):
        "UFC consistent normals."
        t = self.compute_tangents(2, i)
        n = np.cross(t[0], t[1])
        return -2.0 * n / np.linalg.norm(n)


def default_simplex(spatial_dim):
    """Factory function that maps spatial dimension to an instance of
    the default reference simplex of that dimension."""
    if spatial_dim == 1:
        return DefaultLine()
    elif spatial_dim == 2:
        return DefaultTriangle()
    elif spatial_dim == 3:
        return DefaultTetrahedron()
    else:
        raise RuntimeError("Can't create default simplex of dimension %s." % str(spatial_dim))


def ufc_simplex(spatial_dim):
    """Factory function that maps spatial dimension to an instance of
    the UFC reference simplex of that dimension."""
    if spatial_dim == 0:
        return Point()
    elif spatial_dim == 1:
        return UFCInterval()
    elif spatial_dim == 2:
        return UFCTriangle()
    elif spatial_dim == 3:
        return UFCTetrahedron()
    else:
        raise RuntimeError("Can't create UFC simplex of dimension %s." % str(spatial_dim))


def volume(verts):
    """Constructs the volume of the simplex spanned by verts"""

    # use fact that volume of UFC reference element is 1/n!
    sd = len(verts) - 1
    ufcel = ufc_simplex(sd)
    ufcverts = ufcel.get_vertices()

    A, b = make_affine_mapping(ufcverts, verts)

    # can't just take determinant since, e.g. the face of
    # a tet being mapped to a 2d triangle doesn't have a
    # square matrix

    (u, s, vt) = np.linalg.svd(A)

    # this is the determinant of the "square part" of the matrix
    # (ie the part that maps the restriction of the higher-dimensional
    # stuff to UFC element
    p = np.prod([si for si in s if (si) > 1.e-10])

    return p / factorial(sd)


def make_affine_mapping(xs, ys):
    """Constructs (A,b) such that x --> A * x + b is the affine
    mapping from the simplex defined by xs to the simplex defined by ys."""

    dim_x = len(xs[0])
    dim_y = len(ys[0])

    if len(xs) != len(ys):
        raise Exception("")

    # find A in R^{dim_y,dim_x}, b in R^{dim_y} such that
    # A xs[i] + b = ys[i] for all i

    mat = np.zeros((dim_x * dim_y + dim_y, dim_x * dim_y + dim_y), "d")
    rhs = np.zeros((dim_x * dim_y + dim_y,), "d")

    # loop over points
    for i in range(len(xs)):
        # loop over components of each A * point + b
        for j in range(dim_y):
            row_cur = i * dim_y + j
            col_start = dim_x * j
            col_finish = col_start + dim_x
            mat[row_cur, col_start:col_finish] = np.array(xs[i])
            rhs[row_cur] = ys[i][j]
            # need to get terms related to b
            mat[row_cur, dim_y * dim_x + j] = 1.0

    sol = np.linalg.solve(mat, rhs)

    A = np.reshape(sol[:dim_x * dim_y], (dim_y, dim_x))
    b = sol[dim_x * dim_y:]

    return A, b


def linalg_subspace_intersection(A, B):
    """Computes the intersection of the subspaces spanned by the
    columns of 2-dimensional arrays A,B using the algorithm found in
    Golub and van Loan (3rd ed) p. 604.  A should be in
    R^{m,p} and B should be in R^{m,q}.  Returns an orthonormal basis
    for the intersection of the spaces, stored in the columns of
    the result."""

    # check that vectors are in same space
    if A.shape[0] != B.shape[0]:
        raise Exception("Dimension error")

    # A,B are matrices of column vectors
    # compute the intersection of span(A) and span(B)

    # Compute the principal vectors/angles between the subspaces, G&vL
    # p.604
    (qa, _ra) = np.linalg.qr(A)
    (qb, _rb) = np.linalg.qr(B)

    C = np.dot(np.transpose(qa), qb)

    (y, c, _zt) = np.linalg.svd(C)

    U = np.dot(qa, y)

    rank_c = len([s for s in c if np.abs(1.0 - s) < 1.e-10])

    return U[:, :rank_c]
