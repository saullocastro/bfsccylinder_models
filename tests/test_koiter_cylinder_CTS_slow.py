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

    ny = 20

    E11 = 90e9
    E22 = 7e9
    nu12 = 0.32
    G12 = 4.4e9
    tow_thick = 0.4e-3
    rho = 1611 # kg/m3

    rCTS = 0.2

    nxt = 2
    param_n = 0
    c2_ratio = 0
    thetadeg_c1 = 45
    thetadeg_c2 = 45

    load = 1000
    NLprebuck = True
    out1 = fkoiter_cylinder_CTS_circum(L, R, rCTS, nxt, ny, E11, E22, nu12, G12,
            rho, tow_thick, param_n, c2_ratio, thetadeg_c1, thetadeg_c2,
            num_eigvals=2, koiter_num_modes=2, load=load, idealistic_CTS=True,
            NLprebuck=NLprebuck)
    print('cylinder_CTS eigvals', out1['eigvals'])
    print('cylinder_CTS koiter', out1['koiter'])

    nx = int(ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    plyt = tow_thick
    stack = [45, -45]
    prop = laminated_plate(stack=stack, laminaprop=laminaprop, plyt=plyt,
            offset=plyt, rho=rho)
    out2 = fkoiter_cyl_SS3(L, R, nx, ny, prop, cg_x0=None, num_eigvals=2,
            koiter_num_modes=2, load=load, NLprebuck=NLprebuck)

    print('fkoiter_cyl_SS3 eigvals', out2['eigvals'])
    print('fkoiter_cyl_SS3 koiter', out2['koiter'])

    assert np.allclose(out1['eigvals'], out2['eigvals'])
    assert np.isclose(out1['volume'], out2['volume'])
    assert np.isclose(out1['mass'], out2['mass'])

    #TODO I am unsure about the a factors
    #for k in out1['koiter']['a_ijk'].keys():
        #assert np.isclose(out1['koiter']['a_ijk'][k],
                          #out2['koiter']['a_ijk'][k],
                          #atol=1e-5)
    assert np.isclose(out1['koiter']['b_ijkl'][(0, 0, 0, 0)],
                      out2['koiter']['b_ijkl'][(0, 0, 0, 0)],
                      rtol=0.05)
    #TODO mixed modes are not correct
    assert np.isclose(out1['koiter']['b_ijkl'][(1, 1, 1, 1)],
                      out2['koiter']['b_ijkl'][(1, 1, 1, 1)],
                      rtol=0.05)

if __name__ == '__main__':
    test_pm45()

