"""Mesh convergence of the CTS Koiter model on a representative
variable-stiffness design from the dos Santos and Castro design space.

Reports, per mesh, the buckling load, the circumferential wave number n of
the critical mode, how many nodes the mesh puts on the 2n harmonic that the
second-order field carries, and b_1111.

Reported in: Table of Section "Measured convergence" (no argument) and
the table of Section "The same design with a non-linear pre-buckling
state" (argument "nl").
Runtime: about 12 minutes linear, about 35 minutes non-linear.
"""

import os
import sys
#NOTE the repository root goes FIRST on sys.path: an installed
#     bfsccylinder_models would otherwise shadow the working tree, silently
#     verifying a different version of the code than the one being edited
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir))

import sys
import time
import numpy as np

from bfsccylinder_models.koiter_cylinder_CTS_sanders import (
        fkoiter_cylinder_CTS_circum as f_sa)

# geometry and material of the filament-wound cylinder of the test suite
L = 0.3           # m
R = 0.136/2       # m
E11, E22, nu12, G12 = 90e9, 7e9, 0.32, 4.4e9
h_tow = 0.4e-3    # m
rho = 1611.

# a representative variable-stiffness design: theta 30 -> 60 deg, n = 3
rCTS = 0.05
param_n = 3
c2_ratio = 0.5
thetadeg_c1, thetadeg_c2 = 30., 60.

NLprebuck = (len(sys.argv) > 1 and sys.argv[1] == 'nl')

circ = 2*np.pi*R
print('# L %.4f  R %.5f  circ %.5f' % (L, R, circ))
print('# boundary layer 1/beta = %.3e m' % (
      np.sqrt(R*2*h_tow)/(3*(1 - nu12**2))**0.25))
print('# NLprebuck', NLprebuck)
print()

rows = []
#NOTE the linear study is carried to ny=120, the non-linear one stops at
#     ny=80, which is where it was run for the document: a non-linear
#     analysis at ny=120 costs hours and adds nothing, Pcr having already
#     settled to 0.5 per cent at ny=60
MESHES = [(30, 3), (45, 4), (60, 5), (80, 7), (100, 9), (120, 11)]
if NLprebuck:
    MESHES = MESHES[:4]

for ny, nxt in MESHES:
    t0 = time.time()
    out = f_sa(L, R, rCTS, nxt, ny, E11, E22, nu12, G12, rho, h_tow,
               param_n, c2_ratio, thetadeg_c1, thetadeg_c2,
               num_eigvals=4, koiter_num_modes=1, Nxxunit=1.,
               idealistic_CTS=True, NLprebuck=NLprebuck,
               #NOTE the default 12 is a constant-stiffness figure and is
               #     exhausted by this design: the ny=45 run stops at
               #     lambda_b/lambda_c = 0.9933 with a WARNING
               NLprebuck_maxiter=30)
    dt = time.time() - t0

    nx, ny_out = out['nx'], out['ny']
    DOF = 10
    w = out['eigvecs'][:, 0].reshape(nx, ny_out, DOF)[:, :, 6]
    imax = np.argmax(np.abs(w).max(axis=1))
    spec = np.abs(np.fft.rfft(w[imax]))
    n_circ = int(np.argmax(spec[1:]) + 1)

    # axial half-waves of the critical mode, along the generator of max crest
    jmax = np.argmax(np.abs(w[imax]))
    col = w[:, jmax]
    m_ax = int(np.argmax(np.abs(np.fft.rfft(col))[1:]) + 1)

    rows.append(dict(ny=ny_out, nx=nx, dof=DOF*nx*ny_out, n=n_circ, m=m_ax,
                     per2n=ny_out/(2.*n_circ), dx=L/(nx - 1),
                     dy=circ/ny_out, Pcr=out['Pcr'],
                     b=out['koiter']['b_ijkl'][(0, 0, 0, 0)],
                     lb=out['lambda_b'], mu=out['mu'][0], dt=dt))
    r = rows[-1]
    print('ROW ny %3d  nx %3d  DOF %7d  dx %.2fmm dy %.2fmm ar %.2f | '
          'n %2d  m %2d  nodes/2n-wave %.2f | Pcr %11.2f  b_1111 %10.5f | '
          '%.0fs' % (r['ny'], r['nx'], r['dof'], 1e3*r['dx'], 1e3*r['dy'],
                     r['dy']/r['dx'], r['n'], r['m'], r['per2n'], r['Pcr'],
                     r['b'], r['dt']))
    sys.stdout.flush()

print('\n# change with respect to the finest mesh')
fine = rows[-1]
for r in rows:
    print('#  ny %3d   Pcr %+7.2f%%   b_1111 %+8.2f%%'
          % (r['ny'], 100*(r['Pcr']/fine['Pcr'] - 1),
             100*(r['b']/fine['b'] - 1)))
