"""Where the index non-symmetry of the multi-mode b_ijkl comes from

The multi-mode b_ijkl of the models are not index-symmetric, a known
limitation of doc/nlprebuck_implementation.tex. This study does NOT change
them; it takes them apart, on the Waters shell of
tests/test_koiter_cylinder_Waters_sanders.py with three Koiter modes, so that
j, k and l can all differ.

The models compute

    b_ijkl = -1/(6 lambda_i phi20_i.u_i) (phi4_ijkl
             + 3 X_ij,kl + 3 X_il,jk
             + lambda_i (a_iij P30_ikl + a_ijk P30_ili + a_ikl P30_iij)
             + phi200_ii lambda_i**2 (a_iij a_ikl + a_ijk a_ili + a_ikl a_iij))

with X_ab,cd = phi3_ab . u_cd and P30_abc = phi30_ab . u_c. The tensors are
captured from koiter_element_tensors and the output of the model, b_ijkl is
rebuilt from them, and compared with

- its asymmetry under every swap of two indices;
- the same formula with the two pairings 3 X_ij,kl + 3 X_il,jk replaced by
  the three pairings of mode i with j, k and l, 2 (X_ij,kl + X_ik,jl
  + X_il,jk), which is symmetric in j, k and l;
- b_ijkl symmetrized over j, k and l, which is all that enters the amplitude
  equations sum_jkl b_ijkl xi_j xi_k xi_l.

usage: python multimode_b_symmetry.py [ny] [num_modes]
"""
import os
import sys
from itertools import permutations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir))

import numpy as np
from composites import laminated_plate

import bfsccylinder_models.koiter_cylinder_sanders as model


def run(ny, m, NLprebuck):
    captured = {}
    koiter_element_tensors = model.koiter_element_tensors

    def capture(*args, **kwargs):
        captured['tensors'] = koiter_element_tensors(*args, **kwargs)
        return captured['tensors']

    model.koiter_element_tensors = capture
    try:
        L = 0.3556
        R = 0.20318603
        laminaprop = (127.629e9, 11.3074e9, 0.300235, 6.00257e9, 6.00257e9,
                      6.00257e9)
        prop = laminated_plate(stack=[45, -45, 0, 90, 90, 0, -45, 45],
                laminaprop=laminaprop, plyt=0.00012692375, offset=0, rho=1611)
        nx = int(ny*L/(2*np.pi*R))
        if nx % 2 == 0:
            nx += 1
        out = model.fkoiter_cyl_SS3(L, R, nx, ny, prop, num_eigvals=2*m,
                koiter_num_modes=m, NLprebuck=NLprebuck)
    finally:
        model.koiter_element_tensors = koiter_element_tensors
    return out, captured['tensors']


def sym_jkl(b):
    return sum(b.transpose((0,) + tuple(1 + np.array(p)))
               for p in permutations(range(3)))/6


def analyse(out, tensors, m):
    phi20, phi3, phi30, cst, phi200, phi4 = tensors
    k = out['koiter']
    U = np.column_stack([k['ui'][i] for i in range(m)])
    lam = np.array([k['lambda_i'][i] for i in range(m)])
    a = np.array([[[k['a_ijk'][(i, j, l)] for l in range(m)] for j in range(m)]
                  for i in range(m)])
    b = np.array([k['b_ijkl'][idx] for idx in np.ndindex(*(m,)*4)]
                 ).reshape((m,)*4)
    X = np.zeros((m,)*4)
    for (kk, ll), u in k['uij'].items():
        X[:, :, kk, ll] = np.tensordot(phi3, u, axes=(0, 0))
    P30 = np.tensordot(phi30, U, axes=(0, 0))
    d = np.array([phi20[:, i] @ U[:, i] for i in range(m)])

    def formula(pairings):
        out = np.zeros((m,)*4)
        for i, j, kk, ll in np.ndindex(*(m,)*4):
            out[i, j, kk, ll] = -1/(6*lam[i]*d[i])*(
                    phi4[i, j, kk, ll]
                    + pairings(i, j, kk, ll)
                    + lam[i]*(a[i, i, j]*P30[i, kk, ll]
                              + a[i, j, kk]*P30[i, ll, i]
                              + a[i, kk, ll]*P30[i, i, j])
                    + phi200[i, i]*lam[i]**2*(a[i, i, j]*a[i, kk, ll]
                                              + a[i, j, kk]*a[i, ll, i]
                                              + a[i, kk, ll]*a[i, i, j]))
        return out

    b_code = formula(lambda i, j, kk, ll: 3*X[i, j, kk, ll] + 3*X[i, ll, j, kk])
    b_three = formula(lambda i, j, kk, ll: 2*(X[i, j, kk, ll]
                                              + X[i, kk, j, ll]
                                              + X[i, ll, j, kk]))
    scale = np.abs(b).max()
    rel = lambda A, B: np.abs(A - B).max()/scale
    print('    max |b_ijkl|                                    %.3e' % scale)
    print('    max |a_ijk|                                     %.1e'
          % np.abs(a).max())
    print('    rebuilt b against the model                     %.1e'
          % rel(b_code, b))
    print('    phi4 against phi4 with any two indices swapped  %.1e'
          % (max(np.abs(phi4 - phi4.swapaxes(p, q)).max()
                 for p in range(4) for q in range(p))/np.abs(phi4).max()))
    for name, (p, q) in [('i<->j', (0, 1)), ('j<->k', (1, 2)),
                         ('k<->l', (2, 3)), ('j<->l', (1, 3))]:
        print('    b against b with %s swapped                   %.1e'
              % (name, rel(b, b.swapaxes(p, q))))
    print('    b against the three-pairing form                %.1e'
          % rel(b, b_three))
    print('    b symmetrized over j, k, l against three-pairing %.1e'
          % rel(sym_jkl(b), b_three))
    print('    three-pairing form, symmetric in j, k, l?       %.1e'
          % rel(b_three, sym_jkl(b_three)))
    #NOTE b_ijkl lambda_i phi20_i.u_i, the coefficients of the energy, which
    #     would be symmetric in all four indices for a potential
    A = sym_jkl(b)*(lam*d)[:, None, None, None]
    print('    lambda_i phi20_i.u_i sym_jkl(b) against i<->j   %.1e'
          % (np.abs(A - A.swapaxes(0, 1)).max()/np.abs(A).max()))
    #NOTE the entries of b that differ from the three-pairing form, the
    #     ones read out one at a time
    for idx in [(0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0)]:
        print('    b_%d%d%d%d model %+.6f  symmetrized %+.6f  three-pairing '
              '%+.6f' % (tuple(i + 1 for i in idx) + (b[idx], sym_jkl(b)[idx],
                                                      b_three[idx])))


def main():
    ny = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    for NLprebuck in [False, True]:
        out, tensors = run(ny, m, NLprebuck)
        print('\nWaters shell, ny=%d, %d Koiter modes, NLprebuck=%s'
              % (ny, m, NLprebuck))
        print('    load multipliers', out['load_mult'][:m])
        analyse(out, tensors, m)


if __name__ == '__main__':
    main()
