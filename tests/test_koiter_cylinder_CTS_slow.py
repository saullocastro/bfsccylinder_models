import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np
from composites import laminated_plate

from bfsccylinder_models.models import linBuck_VAFW
from bfsccylinder_models.vatfunctions import func_VAT_P_x
from bfsccylinder_models.koiter_cylinder_CTS_slow import fkoiter_cylinder_CTS_circum
from bfsccylinder_models.koiter_cylinder_newton_raphson import fkoiter_cyl_SS3

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

    load = 1000
    NLprebuck = False
    out = fkoiter_cylinder_CTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12,
            rho, tow_thick, param_n, param_f, thetadeg_c, thetadeg_s,
            num_eigvals=2, koiter_num_modes=1, load=load,
            NLprebuck=NLprebuck)
    print('cylinder_CTS eigvals', out['eigvals'])
    print('cylinder_CTS koiter', out['koiter'])

    nx = int(ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    plyt = tow_thick
    stack = [45, -45]
    prop = laminated_plate(stack=stack, laminaprop=laminaprop, plyt=plyt,
            offset=plyt)
    out = fkoiter_cyl_SS3(L, R, nx, ny, prop, cg_x0=None, lobpcg_X=None,
            num_eigvals=2, koiter_num_modes=1, load=load, NLprebuck=NLprebuck)

    print('fkoiter_cyl_SS3 eigvals', out['eigvals'])
    print('fkoiter_cyl_SS3 koiter', out['koiter'])

if __name__ == '__main__':
    test_pm45()

