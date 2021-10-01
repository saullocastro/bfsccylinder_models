import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np

from bfsccylinder_models.models import linBuck_VAFW
from bfsccylinder_models.vatfunctions import func_VAT_P_x

def test_2_runs_in_seq():
    L = 0.3 # m
    R = 0.136/2 # m

    ny = 40 # circumferential
    nx = int(ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1

    E11 = 90e9
    E22 = 7e9
    nu12 = 0.32
    G12 = 4.4e9
    tow_thick = 0.4e-3
    rho = 1611 # kg/m3
    theta_VP_1 = 45.4
    theta_VP_2 = 86.5
    theta_VP_3 = 85.8
    desvars = [[theta_VP_1, theta_VP_2, theta_VP_3]]
    out = linBuck_VAFW(L, R, nx, ny, E11, E22, nu12, G12, rho, tow_thick,
            desvars, func_VAT_P_x, clamped=True)
    theta_VP_1 = 55.4
    theta_VP_2 = 76.5
    theta_VP_3 = 75.8
    desvars = [[theta_VP_1, theta_VP_2, theta_VP_3]]
    out = linBuck_VAFW(L, R, nx, ny, E11, E22, nu12, G12, rho, tow_thick,
            desvars, func_VAT_P_x, clamped=True, cg_x0=out['cg_x0'],
            lobpcg_X=out['lobpcg_X'])
    print(out)

def test_Z33():
    L = 0.510 # m
    R = 0.250 # m
    ny = 40 # circumferential
    nx = int(ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1
    E11 = 145.5e9 #Pa
    E22 = 8.7e9 #Pa
    nu12 = 0.28
    G12 = 5.1e9 #Pa
    plyt = 0.125e-3
    rho = 1611 # kg/m3
    desvars = [
               [0, 0, 0],
               [19, 19, 19],
               [37, 37, 37],
               [45, 45, 45],
               [51, 51, 51],
              ]
    out = linBuck_VAFW(L, R, nx, ny, E11, E22, nu12, G12, rho, plyt, desvars,
            func_VAT_P_x, clamped=True)
    print(out)

if __name__ == '__main__':
    test_Z33()

