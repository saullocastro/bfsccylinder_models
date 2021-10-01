import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np

from bfsccylinder_models.models import linBuck_VAFW
from bfsccylinder_models.vatfunctions import func_VAT_P_x
from bfsccylinder_models.koiter_cylinder_CTS import fkoiter_cylinder_CTS_circum

def test_pm45():
    L = 0.3 # m
    R = 0.136/2 # m

    ny = 40

    E11 = 90e9
    E22 = 7e9
    nu12 = 0.32
    G12 = 4.4e9
    tow_thick = 0.4e-3
    rho = 1611 # kg/m3

    rCTS = 0.2

    nxt = 2
    param_n = 0
    param_f = 0
    thetadeg_c = 45
    thetadeg_s = 50

    out = fkoiter_cylinder_CTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12,
            rho, tow_thick, param_n, param_f, thetadeg_c, thetadeg_s,
            clamped=True, num_eigvals=2, koiter_num_modes=0)
    print('cylinder_CTS eigvals', out['eigvals'])

    desvars = [[45, 45, 45]]
    nx = int(ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1
    out = linBuck_VAFW(L, R, nx, ny, E11, E22, nu12, G12, rho, tow_thick,
            desvars, func_VAT_P_x, clamped=True)

    print('linBuck_VAFW eigvals', out['eigvals'])


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
    print(tmp)
    out = koiter_cylinder_CTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12,
            rho, tow_thick, param_n, param_f, thetadeg_c, thetadeg_s, clamped=True,
            cg_x0=out['cg_x0'],
            lobpcg_X=out['lobpcg_X'],
            koiter_num_modes=1)
    np.allclose(tmp, out['eigvals'])

if __name__ == '__main__':
    test_pm45()

