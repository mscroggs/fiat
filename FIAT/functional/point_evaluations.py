# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# Modified 2020 by the same from Baylor University
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import sympy
import numpy as np
from .functional import Functional, index_iterator


class PointEvaluation(Functional):
    """Class representing point evaluation of scalar functions at a
    particular point x."""

    def __init__(self, ref_el, x):
        pt_dict = {x: [(1.0, tuple())]}
        super().__init__(ref_el, tuple(), pt_dict, {}, "PointEval")

    def __call__(self, fn):
        """Evaluate the functional on the function fn."""
        return fn(tuple(self.pt_dict.keys())[0])

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "u(%s)" % (','.join(x),)


class ComponentPointEvaluation(Functional):
    """Class representing point evaluation of a particular component
    of a vector function at a particular point x."""

    def __init__(self, ref_el, comp, shp, x):
        if len(shp) != 1:
            raise Exception("Illegal shape")
        if comp < 0 or comp >= shp[0]:
            raise Exception("Illegal component")
        self.comp = comp
        pt_dict = {x: [(1.0, (comp,))]}
        super().__init__(ref_el, shp, pt_dict, {},
                         "ComponentPointEval")

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "(u[%d](%s)" % (self.comp, ','.join(x))


class PointDerivative(Functional):
    """Class representing point partial differentiation of scalar
    functions at a particular point x."""

    def __init__(self, ref_el, x, alpha):
        dpt_dict = {x: [(1.0, tuple(alpha), tuple())]}
        self.alpha = tuple(alpha)
        self.order = sum(self.alpha)

        super().__init__(ref_el, tuple(), {}, dpt_dict, "PointDeriv")

    def __call__(self, fn):
        """Evaluate the functional on the function fn. Note that this depends
        on sympy being able to differentiate fn."""
        x = list(self.deriv_dict.keys())[0]

        X = sympy.DeferredVector('x')
        dX = np.asarray([X[i] for i in range(len(x))])

        dvars = tuple(d for d, a in zip(dX, self.alpha)
                      for count in range(a))

        return sympy.diff(fn(X), *dvars).evalf(subs=dict(zip(dX, x)))


class PointNormalDerivative(Functional):
    """Represents d/dn at a point on a facet."""
    def __init__(self, ref_el, facet_no, pt):
        n = ref_el.compute_normal(facet_no)
        self.n = n
        sd = ref_el.get_spatial_dimension()

        alphas = []
        for i in range(sd):
            alpha = [0] * sd
            alpha[i] = 1
            alphas.append(alpha)
        dpt_dict = {pt: [(n[i], tuple(alphas[i]), tuple()) for i in range(sd)]}

        super().__init__(ref_el, tuple(), {}, dpt_dict, "PointNormalDeriv")


class PointNormalSecondDerivative(Functional):
    """Represents d^/dn^2 at a point on a facet."""
    def __init__(self, ref_el, facet_no, pt):
        n = ref_el.compute_normal(facet_no)
        self.n = n
        sd = ref_el.get_spatial_dimension()
        tau = np.zeros((sd*(sd+1)//2,))

        alphas = []
        cur = 0
        for i in range(sd):
            for j in range(i, sd):
                alpha = [0] * sd
                alpha[i] += 1
                alpha[j] += 1
                alphas.append(tuple(alpha))
                tau[cur] = n[i]*n[j]
                cur += 1

        self.tau = tau
        self.alphas = alphas
        dpt_dict = {pt: [(n[i], alphas[i], tuple()) for i in range(sd)]}

        super().__init__(ref_el, tuple(), {}, dpt_dict, "PointNormalDeriv")


class PointNormalEvaluation(Functional):
    """Implements the evaluation of the normal component of a vector at a
    point on a facet of codimension 1."""

    def __init__(self, ref_el, facet_no, pt):
        n = ref_el.compute_normal(facet_no)
        self.n = n
        sd = ref_el.get_spatial_dimension()

        pt_dict = {pt: [(n[i], (i,)) for i in range(sd)]}

        shp = (sd,)
        super().__init__(ref_el, shp, pt_dict, {}, "PointNormalEval")


class PointEdgeTangentEvaluation(Functional):
    """Implements the evaluation of the tangential component of a
    vector at a point on a facet of dimension 1."""

    def __init__(self, ref_el, edge_no, pt):
        t = ref_el.compute_edge_tangent(edge_no)
        self.t = t
        sd = ref_el.get_spatial_dimension()
        pt_dict = {pt: [(t[i], (i,)) for i in range(sd)]}
        shp = (sd,)
        super().__init__(ref_el, shp, pt_dict, {}, "PointEdgeTangent")

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "(u.t)(%s)" % (','.join(x),)


class PointFaceTangentEvaluation(Functional):
    """Implements the evaluation of a tangential component of a
    vector at a point on a facet of codimension 1."""

    def __init__(self, ref_el, face_no, tno, pt):
        t = ref_el.compute_face_tangents(face_no)[tno]
        self.t = t
        self.tno = tno
        sd = ref_el.get_spatial_dimension()
        pt_dict = {pt: [(t[i], (i,)) for i in range(sd)]}
        shp = (sd,)
        super().__init__(ref_el, shp, pt_dict, {}, "PointFaceTangent")

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "(u.t%d)(%s)" % (self.tno, ','.join(x),)


class PointScaledNormalEvaluation(Functional):
    """Implements the evaluation of the normal component of a vector at a
    point on a facet of codimension 1, where the normal is scaled by
    the volume of that facet."""

    def __init__(self, ref_el, facet_no, pt):
        self.n = ref_el.compute_scaled_normal(facet_no)
        sd = ref_el.get_spatial_dimension()
        shp = (sd,)

        pt_dict = {pt: [(self.n[i], (i,)) for i in range(sd)]}
        super().__init__(ref_el, shp, pt_dict, {}, "PointScaledNormalEval")

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "(u.n)(%s)" % (','.join(x),)


class PointwiseInnerProductEvaluation(Functional):
    """
    This is a functional on symmetric 2-tensor fields. Let u be such a
    field, p be a point, and v,w be vectors. This implements the evaluation
    v^T u(p) w.

    Clearly v^iu_{ij}w^j = u_{ij}v^iw^j. Thus the value can be computed
    from the Frobenius inner product of u with wv^T. This gives the
    correct weights.
    """

    def __init__(self, ref_el, v, w, p):
        sd = ref_el.get_spatial_dimension()

        wvT = np.outer(w, v)

        pt_dict = {p: [(wvT[i][j], (i, j))
                       for i, j in index_iterator((sd, sd))]}

        shp = (sd, sd)
        super().__init__(ref_el, shp, pt_dict, {}, "PointwiseInnerProductEval")
