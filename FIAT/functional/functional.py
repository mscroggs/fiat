# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# Modified 2020 by the same from Baylor University
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# functionals require:
# - a degree of accuracy (-1 indicates that it works for all functions
#   such as point evaluation)
# - a reference element domain
# - type information

from itertools import chain
import numpy as np

from FIAT import polynomial_set
from FIAT.helpers import index_iterator

class Functional(object):
    r"""Abstract class representing a linear functional.
    All FIAT functionals are discrete in the sense that
    they are written as a weighted sum of (derivatives of components of) their
    argument evaluated at particular points.

    :arg ref_el: a :class:`Cell`
    :arg target_shape: a tuple indicating the value shape of functions on
         the functional operates (e.g. if the function eats 2-vectors
         then target_shape is (2,) and if it eats scalars then
         target_shape is ()
    :arg pt_dict: A dict mapping points to lists of information about
         how the functional is evaluated.  Each entry in the list takes
         the form of a tuple (wt, comp) so that (at least if the
         deriv_dict argument is empty), the functional takes the form
         :math:`\ell(f) = \sum_{q=1}^{N_q} \sum_{k=1}^{K_q} w^q_k f_{c_k}(x_q)`
         where :math:`f_{c_k}` indicates a particular vector or tensor component
    :arg deriv_dict: A dict that is similar to `pt_dict`, although the entries
         of each list are tuples (wt, alpha, comp) with alpha a tuple
         of nonnegative integers corresponding to the order of partial
         differentiation in each spatial direction.
    :arg functional_type: a string labeling the kind of functional
         this is.
    """
    def __init__(self, ref_el, target_shape, pt_dict, deriv_dict,
                 functional_type):
        self.ref_el = ref_el
        self.target_shape = target_shape
        self.pt_dict = pt_dict
        self.deriv_dict = deriv_dict
        self.functional_type = functional_type
        if len(deriv_dict) > 0:
            per_point = list(chain(*deriv_dict.values()))
            alphas = [foo[1] for foo in per_point]
            self.max_deriv_order = max([sum(foo) for foo in alphas])
        else:
            self.max_deriv_order = 0

    def evaluate(self, f):
        """Obsolete and broken functional evaluation.

        To evaluate the functional, call it on the target function:

          functional(function)
        """
        raise AttributeError("To evaluate the functional just call it on a function.")

    def __call__(self, fn):
        raise NotImplementedError("Evaluation is not yet implemented for %s" % type(self))

    def get_point_dict(self):
        """Returns the functional information, which is a dictionary
        mapping each point in the support of the functional to a list
        of pairs containing the weight and component."""
        return self.pt_dict

    def get_reference_element(self):
        """Returns the reference element."""
        return self.ref_el

    def get_type_tag(self):
        """Returns the type of function (e.g. point evaluation or
        normal component, which is probably handy for clients of FIAT"""
        return self.functional_type

    def to_riesz(self, poly_set):
        r"""Constructs an array representation of the functional so
        that the functional may be applied to a function expressed in
        in terms of the expansion set underlying  `poly_set` by means
        of contracting coefficients.

        That is, `poly_set` will have members all expressed in the
        form :math:`p = \sum_{i} \alpha^i \phi_i`
        where :math:`\{\phi_i\}_{i}` is some orthonormal expansion set
        and :math:`\alpha^i` are coefficients.  Note: the orthonormal
        expansion set is always scalar-valued but if the members of
        `poly_set` are vector or tensor valued the :math:`\alpha^i`
        will be scalars or vectors.

        This function constructs a tensor :math:`R` such that the
        contraction of :math:`R` with the array of coefficients
        :math:`\alpha` produces the effect of :math:`\ell(f)`

        In the case of scalar-value functions, :math:`R` is just a
        vector of the same length as the expansion set, and
        :math:`R_i = \ell(\phi_i)`.  For vector-valued spaces,
        :math:`R_{ij}` will be :math:`\ell(e^i \phi_j)` where
        :math:`e^i` is the canonical unit vector nonzero only in one
        entry :math:`i`.
        """
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()
        nexp = es.get_num_members(ed)

        pt_dict = self.get_point_dict()

        pts = list(pt_dict.keys())
        npts = len(pts)

        bfs = es.tabulate(ed, pts)
        result = np.zeros(poly_set.coeffs.shape[1:], "d")

        # loop over points
        for j in range(npts):
            pt_cur = pts[j]
            wc_list = pt_dict[pt_cur]

            # loop over expansion functions
            for i in range(nexp):
                for (w, c) in wc_list:
                    result[c][i] += w * bfs[i, j]

        if self.deriv_dict:
            dpt_dict = self.deriv_dict

            # this makes things quicker since it uses dmats after
            # instantiation
            es_foo = polynomial_set.ONPolynomialSet(self.ref_el, ed)
            dpts = list(dpt_dict.keys())

            dbfs = es_foo.tabulate(dpts, self.max_deriv_order)

            ndpts = len(dpts)
            for j in range(ndpts):
                dpt_cur = dpts[j]
                wac_list = dpt_dict[dpt_cur]
                for i in range(nexp):
                    for (w, alpha, c) in wac_list:
                        result[c][i] += w * dbfs[tuple(alpha)][i, j]

        return result

    def tostr(self):
        return self.functional_type
