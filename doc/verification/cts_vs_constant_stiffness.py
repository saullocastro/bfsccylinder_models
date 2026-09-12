"""CTS model in its constant-stiffness limit against the newton_raphson model,
with the nonlinear pre-buckling algorithm switched on. The two must agree to
round off: same mesh, same laminate, same algorithm.

Reported in: Section "The variable-stiffness CTS model".
Runtime: about 10 minutes.
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

import bfsccylinder_models.koiter_cylinder_CTS as cts_vk
import bfsccylinder_models.koiter_cylinder_CTS_sanders as cts_sa
import bfsccylinder_models.koiter_cylinder_newton_raphson as nr_vk
import bfsccylinder_models.koiter_cylinder_newton_raphson_sanders as nr_sa

L = 0.3
R = 0.136/2
ny = 30
E11, E22, nu12, G12 = 90e9, 7e9, 0.32, 4.4e9
tow_thick = 0.4e-3
rho = 1611.
rCTS = 0.2
Nxxunit = 1.

for cts, nr, name in [(cts_vk, nr_vk, 'von Karman'), (cts_sa, nr_sa, 'Sanders')]:
    for NLprebuck in (False, True):
        out1 = cts.fkoiter_cylinder_CTS_circum(L, R, rCTS, 2, ny, E11, E22,
                nu12, G12, rho, tow_thick, 0, 0, 45, 45, num_eigvals=5,
                koiter_num_modes=1, Nxxunit=Nxxunit, idealistic_CTS=True,
                NLprebuck=NLprebuck)

        nx = int(ny*L/(2*np.pi*R))
        if nx % 2 == 0:
            nx += 1
        prop = laminated_plate(stack=[45, -45],
                laminaprop=(E11, E22, nu12, G12, G12, G12), plyt=tow_thick,
                offset=tow_thick, rho=rho)
        out2 = nr.fkoiter_cyl_SS3(L, R, nx, ny, prop, cg_x0=None,
                num_eigvals=5, koiter_num_modes=1, Nxxunit=Nxxunit,
                NLprebuck=NLprebuck)

        b1 = out1['koiter']['b_ijkl'][(0, 0, 0, 0)]
        b2 = out2['koiter']['b_ijkl'][(0, 0, 0, 0)]
        rel = lambda a, b: abs(a - b)/max(abs(a), abs(b), 1e-300)
        print('\n### %s, NLprebuck=%s' % (name, NLprebuck))
        print('  %-12s CTS %20.12g   NR %20.12g   rel %.3e'
              % ('Pcr', out1['Pcr'], out2['Pcr'], rel(out1['Pcr'], out2['Pcr'])))
        print('  %-12s CTS %20.12g   NR %20.12g   rel %.3e'
              % ('lambda_b', out1['lambda_b'], out2['lambda_b'],
                 rel(out1['lambda_b'], out2['lambda_b'])))
        print('  %-12s CTS %20.12g   NR %20.12g   rel %.3e'
              % ('mu[0]', out1['mu'][0], out2['mu'][0],
                 rel(out1['mu'][0], out2['mu'][0])))
        print('  %-12s CTS %20.12g   NR %20.12g   rel %.3e'
              % ('b_1111', b1, b2, rel(b1, b2)))
        print('  %-12s CTS %20.12g   NR %20.12g   rel %.3e'
              % ('volume', out1['volume'], out2['volume'],
                 rel(out1['volume'], out2['volume'])))
