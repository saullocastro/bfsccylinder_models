"""How many Gauss points per direction does the element need?

- the number of zero eigenvalues of the constitutive stiffness matrix KC0 of
  a single element against nint; beyond those of the rigid body motions they
  are zero energy modes of an under-integrated element
- lambda_c and b_1111 of the two reference shells of the test suite, ny=40,
  against nint

Reported in: Section "Known limitations".
Runtime: about 10 minutes.
"""

import contextlib
import io
import os
import sys
import time
#NOTE the repository root goes FIRST on sys.path: an installed
#     bfsccylinder_models would otherwise shadow the working tree, silently
#     verifying a different version of the code than the one being edited
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir))

import numpy as np
from scipy.sparse import coo_matrix
from composites import laminated_plate
import bfsccylinder as vk
import bfsccylinder.sanders as sa
from bfsccylinder.quadrature import get_points_weights

import bfsccylinder_models.koiter_cylinder as vk_model
import bfsccylinder_models.koiter_cylinder_sanders as sa_model

DOF = 10
NINTS = (2, 3, 4, 5, 6)

# a fully populated laminate, as in tangent_consistency.py
A = np.array([[6.1e7, 2.0e7, 9.0e6],
              [2.0e7, 4.3e7, 7.0e6],
              [9.0e6, 7.0e6, 2.4e7]])
B = np.array([[3.1e4, 1.2e4, 5.0e3],
              [1.2e4, 2.2e4, 4.0e3],
              [5.0e3, 4.0e3, 1.4e4]])
D = np.array([[52.0, 17.0, 8.0],
              [17.0, 39.0, 6.0],
              [8.0, 6.0, 21.0]])

L = 0.3556
E11, E22, G12, nu12 = 127.629e9, 11.3074e9, 6.00257e9, 0.300235
STACK = (45, -45, 0, 90, 90, 0, -45, 45)
H = 0.00101539


def zero_modes(mod, cls, nint):
    elem = cls(nint)
    elem.n1, elem.n2, elem.n3, elem.n4 = 1, 2, 3, 4
    elem.c1, elem.c2, elem.c3, elem.c4 = 0, DOF, 2*DOF, 3*DOF
    elem.R, elem.lex, elem.ley = 0.068, 0.02, 0.015
    elem.init_k_KC0 = elem.init_k_KCNL = elem.init_k_KG = 0
    for nm, M in (('A', A), ('B', B), ('D', D)):
        for (a, b), idx in {(0, 0): '11', (0, 1): '12', (0, 2): '16',
                            (1, 1): '22', (1, 2): '26', (2, 2): '66'}.items():
            getattr(elem, nm + idx)[:, :] = M[a, b]
    points, weights = get_points_weights(nint=nint)
    r = np.zeros(mod.KC0_SPARSE_SIZE, dtype=mod.INT)
    c = np.zeros(mod.KC0_SPARSE_SIZE, dtype=mod.INT)
    v = np.zeros(mod.KC0_SPARSE_SIZE, dtype=mod.DOUBLE)
    mod.update_KC0(elem, points, weights, r, c, v)
    K = coo_matrix((v, (r, c)), shape=(4*DOF, 4*DOF)).toarray()
    ev = np.linalg.eigvalsh((K + K.T)/2)
    return int((np.abs(ev) <= 1e-10*np.abs(ev).max()).sum())


print('zero eigenvalues of KC0 of one element, %d degrees of freedom' % (4*DOF))
for name, mod, cls in (('von Karman', vk, vk.BFSCCylinder),
                       ('Sanders', sa, sa.BFSCCylinderSanders)):
    print('  %-11s %s' % (name, '   '.join('nint %d: %d' % (nint,
          zero_modes(mod, cls, nint)) for nint in NINTS)))
sys.stdout.flush()

prop = laminated_plate(stack=STACK,
        laminaprop=(E11, E22, nu12, G12, G12, G12), plyt=H/len(STACK))
for tag, model, R, Nxxunit, fx, num_eigvals in (
        ('Sun et al. 3.1, Sanders', sa_model, 0.2032, 20000., 1., 4),
        ('AW-CYL-1-1, von Karman', vk_model, 0.20318603, 10000., 1.5, 2)):
    ny = 40
    nx = int(fx*ny*L/(2*np.pi*R))
    nx += 1 - nx % 2
    Ncl = E11*H**2/(R*np.sqrt(3*(1 - nu12**2)))
    print('\n=== %s, nx %d ny %d' % (tag, nx, ny))
    res = {}
    for nint in NINTS:
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = model.fkoiter_cyl_SS3(L, R, nx, ny, prop,
                        nint=nint, num_eigvals=num_eigvals,
                        koiter_num_modes=1, Nxxunit=Nxxunit, NLprebuck=True)
        except Exception as e:
            print('  nint %d  failed, %s: %s' % (nint, type(e).__name__,
                                                str(e)[:70]))
            continue
        res[nint] = (out['load_mult'][0]*Nxxunit/Ncl,
                     out['koiter']['b_ijkl'][(0, 0, 0, 0)])
        print('  nint %d  lambda_c %.6f  b_1111 %+.6f  %.0fs'
              % (nint, res[nint][0], res[nint][1], time.time() - t0))
        sys.stdout.flush()
    ref = max(res)
    for nint in sorted(res):
        print('  nint %d against nint %d: lambda_c %+.2e  b_1111 %+.2e'
              % (nint, ref, res[nint][0]/res[ref][0] - 1,
                 res[nint][1]/res[ref][1] - 1))
