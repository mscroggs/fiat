import numpy as np


def lattice_iter(start, finish, depth):
    """Generator iterating over the depth-dimensional lattice of
    integers between start and (finish-1).  This works on simplices in
    1d, 2d, 3d, and beyond"""
    if depth == 0:
        return
    elif depth == 1:
        for ii in range(start, finish):
            yield [ii]
    else:
        for ii in range(start, finish):
            for jj in lattice_iter(start, finish - ii, depth - 1):
                yield jj + [ii]


def make_lattice(verts, n, interior=0):
    """Constructs a lattice of points on the simplex defined by verts.
    For example, the 1:st order lattice will be just the vertices.
    The optional argument interior specifies how many points from
    the boundary to omit.  For example, on a line with n = 2,
    and interior = 0, this function will return the vertices and
    midpoint, but with interior = 1, it will only return the
    midpoint."""

    vs = np.array(verts)
    hs = (vs - vs[0])[1:, :] / n

    m = hs.shape[0]
    result = [tuple(vs[0] + np.array(indices).dot(hs))
              for indices in lattice_iter(interior, n + 1 - interior, m)]

    return result
