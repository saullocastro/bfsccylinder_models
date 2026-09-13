"""How many modes sit in the near-critical cluster of the reference cases?

That count is the minimum size a multi-mode Koiter expansion has to have: a
single-mode expansion about one member of a cluster cannot see the
interaction that determines b_ijkl.

Reported in: Table of Section "How many modes the expansion actually
needs".
Runtime: about 5 minutes.
"""

import os
import sys
#NOTE the repository root goes FIRST on sys.path: an installed
#     bfsccylinder_models would otherwise shadow the working tree, silently
#     verifying a different version of the code than the one being edited
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir))

import numpy as np
from composites import laminated_plate

from bfsccylinder_models.koiter_cylinder_sanders import (
        fkoiter_cyl_SS3 as f_sa)
from bfsccylinder_models.koiter_cylinder import fkoiter_cyl_SS3 as f_vk

CASES = []

# Sun et al. Section 3.1 / Arbocz AW-CYL-1-1, as in the test suite
E11, E22, G12, nu12 = 127.629e9, 11.3074e9, 6.00257e9, 0.300235
stack = (45, -45, 0, 90, 90, 0, -45, 45)
h = 0.00101539
prop = laminated_plate(stack=stack,
                       laminaprop=(E11, E22, nu12, G12, G12, G12),
                       plyt=h/len(stack))
ny = 40
L, R = 0.3556, 0.2032
nx = int(ny*L/(2*np.pi*R))
if nx % 2 == 0:
    nx += 1
CASES.append(('Sun 3.1 (Sanders)', f_sa, L, R, nx, ny, prop, 20000.))

L2, R2 = 0.3556, 0.20318603
nx2 = int(1.5*ny*L2/(2*np.pi*R2))
if nx2 % 2 == 0:
    nx2 += 1
CASES.append(('Arbocz AW-CYL-1-1 (vK)', f_vk, L2, R2, nx2, ny, prop, 10000.))

for tag, fn, L, R, nx, ny, prop, Nxxunit in CASES:
    for NL in (False, True):
        out = fn(L, R, nx, ny, prop, cg_x0=None, nint=4, num_eigvals=12,
                 koiter_num_modes=0, Nxxunit=Nxxunit, NLprebuck=NL)
        lm = out['load_mult']
        ev = out['eigvecs']
        print('\n=== %s  nx %d ny %d  NLprebuck=%s  Pcr %.1f' % (
              tag, nx, ny, NL, out['Pcr']))
        for k in range(len(lm)):
            w = ev[:, k].reshape(nx, ny, 10)[:, :, 6]
            P = (np.abs(np.fft.rfft(w, axis=1))**2).sum(axis=0)
            P = P/P.sum()
            n = int(np.argmax(P))
            Pa = (np.abs(np.fft.rfft(w, axis=0))**2).sum(axis=1)
            m = int(np.argmax(Pa[1:]) + 1)
            print('   %2d  mult %12.2f  ratio %.5f  n=%2d (%.2f)  m~%d'
                  % (k, lm[k], lm[k]/lm[0], n, P[n], m))
        within = {p: int((lm/lm[0] - 1 <= p).sum()) for p in (0.005, 0.01,
                                                              0.02, 0.05)}
        print('   modes within 0.5%%/1%%/2%%/5%% of critical: %s'
              % ' / '.join(str(within[p]) for p in (0.005, 0.01, 0.02, 0.05)))
