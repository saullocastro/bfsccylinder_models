"""Circumferential harmonic content of the lowest buckling modes.

koiter_num_modes=0 returns right after the eigenvalue analysis, so this is
cheap enough to sweep meshes with. Two studies: the steered design, theta
30 -> 60 deg, against the mesh, and that design next to two constant-angle
ones on the ny=60 mesh.

Reported in: Section "The buckling spectrum is design-dependent", whose
table is the second study.
Runtime: about 20 minutes.
"""

import os
import sys
#NOTE the repository root goes FIRST on sys.path: an installed
#     bfsccylinder_models would otherwise shadow the working tree, silently
#     verifying a different version of the code than the one being edited
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir))

import sys
import numpy as np

from bfsccylinder_models.koiter_cylinder_CTS_sanders import (
        fkoiter_cylinder_CTS_circum as f_sa)

L, R = 0.3, 0.136/2
E11, E22, nu12, G12 = 90e9, 7e9, 0.32, 4.4e9
h_tow, rho = 0.4e-3, 1611.
rCTS, param_n, c2_ratio = 0.05, 3, 0.5
NL = (len(sys.argv) > 1 and sys.argv[1] == 'nl')
print('# NLprebuck', NL)


def spectrum(ny, nxt, thetadeg_c1, thetadeg_c2):
    out = f_sa(L, R, rCTS, nxt, ny, E11, E22, nu12, G12, rho, h_tow,
               param_n, c2_ratio, thetadeg_c1, thetadeg_c2,
               num_eigvals=6, koiter_num_modes=0, Nxxunit=1.,
               idealistic_CTS=True, NLprebuck=NL)
    nx, ny_o = out['nx'], out['ny']
    lm = out['load_mult']
    ev = out['eigvecs']
    print('\n=== theta %g -> %g deg  ny %d  nx %d  Pcr %.2f  lambda_b %.4g'
          % (thetadeg_c1, thetadeg_c2, ny_o, nx, out['Pcr'], out['lambda_b']))
    for k in range(ev.shape[1]):
        w = ev[:, k].reshape(nx, ny_o, 10)[:, :, 6]
        # energy per circumferential harmonic, summed over axial stations
        P = (np.abs(np.fft.rfft(w, axis=1))**2).sum(axis=0)
        P = P/P.sum()
        top = np.argsort(P)[::-1][:3]
        print('   mode %d  mult %12.2f  ratio %.5f | n=%s  (axisym share %.3f)'
              % (k, lm[k], lm[k]/lm[0],
                 ', '.join('%d:%.2f' % (t, P[t]) for t in top), P[0]))
    sys.stdout.flush()


print('\n# the steered design against the mesh')
for ny, nxt in [(30, 3), (45, 4), (60, 5), (80, 7), (100, 9), (120, 11)]:
    spectrum(ny, nxt, 30., 60.)

#NOTE with thetadeg_c1 == thetadeg_c2 the model meshes the cylinder with a
#     uniform axial spacing of its own, and param_n is not used
print('\n# three designs on the ny=60 mesh')
for thetadeg_c1, thetadeg_c2 in [(45., 45.), (30., 60.), (60., 60.)]:
    spectrum(60, 5, thetadeg_c1, thetadeg_c2)
