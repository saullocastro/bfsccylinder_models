"""Normalizations of the multi-mode Koiter coefficients of run_case.py

The models of bfsccylinder_models scale every Koiter mode so that its largest
nodal translation equals the thickness h. Every term of b_ijkl carries the
four mode amplitudes and its denominator d_i = phi20_i . u_i the square of
that of mode i, so the coefficients of the modes rescaled by s_i are,
exactly,

    b_ijkl*s_j*s_k*s_l/s_i and a_ijk*s_j*s_k/s_i

which lets the normalization be chosen after the run:

- nodal: the models, largest nodal translation equal to h;
- crest: crest of w, between the nodes as well, equal to h, s_i = 1/crest_w;
- rms: RMS of w over the surface equal to h, s_i = 1/rms_w;
- energy: lambda_i*|d_i| = 1, the normalization of the multi-mode analysis of
  Rahman (2009), Eq. (3.21), lambda_I q_I^T [dK_D + dK_G] q_I = 1, of which
  lambda_i*d_i is the counterpart in the models, see b_coefficients in
  koiter_tensors.py.

With the energy normalization the quadratic part of the reduced equations is
the same for every mode of a cluster, so the most imperfection sensitive
direction e, e.e = 1, and its coefficient b_min follow from the eigenvalue
problem b_ijkI e_i e_j e_k = b e_I of Salerno, Eq. (3.31) of Rahman (2009),
b_ijkl being made symmetric in all four indices as in his Section 3.2.1.
Unlike the coefficient of a single mode, b_min is invariant under the mixing
of modes with the same multiplier. b_min_t is b_min for the combined mode
sum_i e_i u_i rescaled so that its crest of w equals h, the amplitude with
which Rahman (2009) sets the imperfections of his multi-mode analyses
"""
from itertools import permutations

import numpy as np
from scipy.optimize import minimize


def rescaled(b, a, s):
    """b_ijkl and a_ijk of the modes multiplied by s, over trailing axes"""
    return (np.einsum('...ijkl,...j,...k,...l,...i->...ijkl', b, s, s, s, 1/s),
            np.einsum('...ijk,...j,...k,...i->...ijk', a, s, s, 1/s))


def energy_scales(lambda_d):
    """s_i such that lambda_i*|d_i| of the rescaled modes equals one"""
    return 1/np.sqrt(np.abs(np.asarray(lambda_d)))


def symmetrized(b):
    """b_ijkl averaged over the 24 permutations of its indices"""
    return sum(np.transpose(b, p) for p in permutations(range(4)))/24.


def min_direction(b, num_starts=40, seed=0):
    """b_min and e, the minimum of b_ijkl e_i e_j e_k e_l over e.e = 1

    Its stationary points are the solutions of b_ijkI e_i e_j e_k = b e_I,
    for the symmetrized b_ijkl
    """
    bs = symmetrized(np.asarray(b, dtype=np.float64))
    m = bs.shape[0]

    def f(x):
        e = x/np.linalg.norm(x)
        return np.einsum('ijkl,i,j,k,l', bs, e, e, e, e)

    def grad(x):
        n = np.linalg.norm(x)
        e = x/n
        g = 4*np.einsum('ijkl,j,k,l->i', bs, e, e, e)
        return (g - (g @ e)*e)/n

    rng = np.random.default_rng(seed)
    starts = list(np.eye(m)) + list(rng.standard_normal((num_starts, m)))
    best = min((minimize(f, x0, jac=grad, method='BFGS', tol=1.e-12)
                for x0 in starts), key=lambda r: r.fun)
    e = best.x/np.linalg.norm(best.x)
    #NOTE the sign of e is free; the largest component made positive
    if e[np.argmax(np.abs(e))] < 0:
        e = -e
    return float(best.fun), e
