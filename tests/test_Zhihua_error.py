import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np

from bfsccylinder_models.models import linBuck_VAFW
from bfsccylinder_models.vatfunctions import func_VAT_P_x

def test():
    L = 0.3 # m
    R = 0.15 # m
    ny = 30 # circumferential
    nx = int(ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1
    E11 = 90.e9 #Pa
    E22 = 7.e9 #Pa
    nu12 = 0.32
    G12 = 4.4e9 #Pa
    plyt = 0.4e-3
    rho = 1611 # kg/m3
    nint = 4
    desvars = [
              [38.5, 48.6, 59.2],
              [11.5, 38, 18.9],
              [51.4, 5.1, 42.6],
              ]
    out = linBuck_VAFW(L, R, nx, ny, E11, E22, nu12, G12,
            rho, plyt, desvars, func_VAT_P_x, clamped=True, nint=nint)
    print(out)

if __name__ == '__main__':
    test()

