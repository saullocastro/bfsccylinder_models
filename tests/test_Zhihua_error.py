import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np

from bfsccylinder_models.linbuck_VAFW import flinBuck_VAFW
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
    #NOTE the design that made the lobpcg eigen solver of the earlier,
    #     displacement controlled version of the model fail
    out = flinBuck_VAFW(L, R, nx, ny, E11, E22, nu12, G12,
            rho, plyt, desvars, func_VAT_P_x, nint=nint)
    print('Pcr', out['Pcr'])
    #NOTE regression value of the SS3 edges with inertia relief
    assert np.isclose(out['Pcr'], 357801.2187399378, rtol=1e-5)

if __name__ == '__main__':
    test()

