# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# Modified 2020 by the same from Baylor University
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
from collections import OrderedDict
from .functional import Functional


class IntegralMoment(Functional):
    """Functional representing integral of the input against some tabulated function f.

    :arg ref_el: a :class:`Cell`.
    :arg Q: a :class:`QuadratureRule`.
    :arg f_at_qpts: an array tabulating the function f at the quadrature
         points.
    :arg comp: Optional argument indicating that only a particular
         component of the input function should be integrated against f
    :arg shp: Optional argument giving the value shape of input functions.
    """

    def __init__(self, ref_el, Q, f_at_qpts, comp=tuple(), shp=tuple()):
        qpts, qwts = Q.get_points(), Q.get_weights()
        pt_dict = OrderedDict()
        self.comp = comp
        for i in range(len(qpts)):
            pt_cur = tuple(qpts[i])
            pt_dict[pt_cur] = [(qwts[i] * f_at_qpts[i], comp)]
        super().__init__(ref_el, shp, pt_dict, {}, "IntegralMoment")

    def __call__(self, fn):
        """Evaluate the functional on the function fn."""
        pts = list(self.pt_dict.keys())
        wts = np.array([foo[0][0] for foo in list(self.pt_dict.values())])
        result = np.dot([fn(p) for p in pts], wts)

        if self.comp:
            result = result[self.comp]
        return result


class IntegralMomentOfNormalDerivative(Functional):
    """Functional giving normal derivative integrated against some function on a facet."""

    def __init__(self, ref_el, facet_no, Q, f_at_qpts):
        n = ref_el.compute_normal(facet_no)
        self.n = n
        self.f_at_qpts = f_at_qpts
        self.Q = Q

        sd = ref_el.get_spatial_dimension()

        # map points onto facet

        fmap = ref_el.get_entity_transform(sd-1, facet_no)
        qpts, qwts = Q.get_points(), Q.get_weights()
        dpts = [fmap(pt) for pt in qpts]
        self.dpts = dpts

        dpt_dict = OrderedDict()

        alphas = [tuple([1 if j == i else 0 for j in range(sd)]) for i in range(sd)]
        for j, pt in enumerate(dpts):
            dpt_dict[tuple(pt)] = [(qwts[j]*n[i]*f_at_qpts[j], alphas[i], tuple()) for i in range(sd)]

        super().__init__(ref_el, tuple(),
                         {}, dpt_dict, "IntegralMomentOfNormalDerivative")


class FrobeniusIntegralMoment(Functional):

    def __init__(self, ref_el, Q, f_at_qpts):
        # f_at_qpts is num components x num_qpts
        if len(Q.get_points()) != f_at_qpts.shape[1]:
            raise Exception("Mismatch in number of quadrature points and values")

        # make sure that shp is same shape as f given
        shp = (f_at_qpts.shape[0],)

        qpts, qwts = Q.get_points(), Q.get_weights()
        pt_dict = {}
        for i in range(len(qpts)):
            pt_cur = tuple(qpts[i])
            pt_dict[pt_cur] = [(qwts[i] * f_at_qpts[j, i], (j,))
                               for j in range(f_at_qpts.shape[0])]

        super().__init__(ref_el, shp, pt_dict, {}, "FrobeniusIntegralMoment")


class IntegralMomentOfDivergence(Functional):
    def __init__(self, ref_el, Q, f_at_qpts):
        self.f_at_qpts = f_at_qpts
        self.Q = Q

        sd = ref_el.get_spatial_dimension()

        qpts, qwts = Q.get_points(), Q.get_weights()
        dpts = qpts
        self.dpts = dpts

        dpt_dict = OrderedDict()

        alphas = [tuple([1 if j == i else 0 for j in range(sd)]) for i in range(sd)]
        for j, pt in enumerate(dpts):
            dpt_dict[tuple(pt)] = [(qwts[j]*f_at_qpts[j], alphas[i], (i,)) for i in range(sd)]

        super().__init__(ref_el, tuple(),
                         {}, dpt_dict, "IntegralMomentOfDivergence")


class IntegralMomentOfTensorDivergence(Functional):
    """Like IntegralMomentOfDivergence, but on symmetric tensors."""

    def __init__(self, ref_el, Q, f_at_qpts):
        self.f_at_qpts = f_at_qpts
        self.Q = Q
        qpts, qwts = Q.get_points(), Q.get_weights()
        nqp = len(qpts)
        dpts = qpts
        self.dpts = dpts

        assert len(f_at_qpts.shape) == 2
        assert f_at_qpts.shape[0] == 2
        assert f_at_qpts.shape[1] == nqp

        sd = ref_el.get_spatial_dimension()

        dpt_dict = OrderedDict()

        alphas = [tuple([1 if j == i else 0 for j in range(sd)]) for i in range(sd)]
        for q, pt in enumerate(dpts):
            dpt_dict[tuple(pt)] = [(qwts[q]*f_at_qpts[i, q], alphas[j], (i, j)) for i in range(2) for j in range(2)]

        super().__init__(ref_el, tuple(),
                         {}, dpt_dict, "IntegralMomentOfTensorDivergence")
