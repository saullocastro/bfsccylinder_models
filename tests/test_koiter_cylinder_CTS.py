import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np

from bfsccylinder_models.models import koiter_cylinder_CTS_circum

def test_2_runs_in_seq():
    L = 1.1 # m
    R = 0.3 # m

    ny = 40

    E11 = 90e9
    E22 = 7e9
    nu12 = 0.32
    G12 = 4.4e9
    tow_thick = 0.4e-3
    rho = 1611 # kg/m3
    rCTS = 0.2

    nxt = 1
    param_n = 2
    param_f = 0.1
    thetadeg_c = 10
    thetadeg_s = 30

    out = koiter_cylinder_CTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12,
            rho, tow_thick, param_n, param_f, thetadeg_c, thetadeg_s,
            clamped=True,
            koiter_num_modes=1)
    tmp = out['eigvals']
    out = koiter_cylinder_CTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12,
            rho, tow_thick, param_n, param_f, thetadeg_c, thetadeg_s, clamped=True,
            cg_x0=out['cg_x0'],
            lobpcg_X=out['lobpcg_X'],
            koiter_num_modes=1)
    np.allclose(tmp, out['eigvals'])

if __name__ == '__main__':
    test_2_runs_in_seq()

