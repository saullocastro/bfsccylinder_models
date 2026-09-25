"""Simply supported edges fixed at the nodes or along the whole edge, on the
NASA shell AW-CYL-1-1 of Arbocz, Starnes and Nemeth (AIAA-2001-1392)

The shell, laminate, load and mesh rule of test_Arbocz_Starnes_2002 of
tests/test_koiter_cylinder_newton_raphson.py, von Karman (fkoiter_cyl_SS3 of
koiter_cylinder) or Sanders (koiter_cylinder_sanders) kinematics, NLprebuck,
one Koiter mode. Which edge condition is solved is set by the library on
PYTHONPATH: the one of the models, v = w = 0 along the whole edge (with the
axial translation removed by inertia relief since, which leaves lambda_c
and b unchanged), or an export of an earlier commit, v and w fixed at the
edge nodes only. Both are
SS-3, the condition of the references, as the mesh is refined: ANILISA with
rigorous nonlinear pre-buckling gives lambda_c = 0.328594 (n = 11) and
b = -0.37605 for that mode, STAGS-A lambda_c = 0.327759.

Prints one line "RESULT {json}" with ny, nx, the library, lambda_c, n,
b_1111 and lambda_b/lambda_c.

usage, with the library to test first on PYTHONPATH:

    python edge_condition_benchmark.py NY [sanders]
"""
import json
import sys

import numpy as np
from composites import laminated_plate

import bfsccylinder_models
from bfsccylinder_models.koiter_cylinder import fkoiter_cyl_SS3
from bfsccylinder_models.koiter_cylinder_sanders import (
    fkoiter_cyl_SS3 as fkoiter_cyl_SS3_sanders)

ny = int(sys.argv[1])
sanders = len(sys.argv) > 2 and sys.argv[2] == 'sanders'
L = 0.3556
R = 0.20318603
nx = int(1.5*ny*L/(2*np.pi*R))
if nx % 2 == 0:
    nx += 1
E11 = 127.629e9
E22 = 11.3074e9
G12 = 6.00257e9
nu12 = 0.300235
stack = [45, -45, 0, 90, 90, 0, -45, 45]
h = 0.00101539
prop = laminated_plate(stack=stack, laminaprop=(E11, E22, nu12, G12, G12, G12),
                       plyt=h/len(stack))
Nxxunit = 10000.
model = fkoiter_cyl_SS3_sanders if sanders else fkoiter_cyl_SS3
#NOTE eps1 = 0.0005 as the runs of the convergence study, so that the expansion point
#     is the same for both edge conditions
out = model(L, R, nx, ny, prop, nint=4, num_eigvals=4,
            koiter_num_modes=1, Nxxunit=Nxxunit, NLprebuck=True,
            NLprebuck_eps1=0.0005)
ref = E11*h**2/(R*np.sqrt(3*(1 - nu12**2)))
lambda_c = out['load_mult'][0]*Nxxunit/ref
#NOTE wave number of mode 0 from the w of the nodes of the middle station
circ = 2*np.pi*R
x = out['x']
xmid = x[np.argmin(abs(x - L/2))]
sel = np.isclose(x, xmid)
w = out['eigvecs'][6::10, 0][sel]
spec = abs(np.fft.rfft(w))
n = int(np.argmax(spec[1:]) + 1)
result = dict(ny=ny, nx=nx, kinematics='sanders' if sanders else 'donnell',
              library=bfsccylinder_models.__file__, lambda_c=float(lambda_c),
              n=n, b_1111=float(out['koiter']['b_ijkl'][(0, 0, 0, 0)]),
              lambda_ratio=float(out['lambda_b']/out['load_mult'][0]))
print('RESULT ' + json.dumps(result))
