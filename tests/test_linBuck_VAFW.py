import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np

from bfsccylinder_models.linbuck_VAFW import flinBuck_VAFW
from bfsccylinder_models.vatfunctions import func_VAT_P_x

#NOTE regression values of the SS3 edges with inertia relief, since the
#     model was changed from displacement controlled clamped edges; the
#     constant laminate is checked against koiter_cylinder in test_edges.py


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
    Pcr = []
    for thetas in ([45.4, 86.5, 85.8], [55.4, 76.5, 75.8]):
        out = flinBuck_VAFW(L, R, nx, ny, E11, E22, nu12, G12, rho,
                tow_thick, [thetas], func_VAT_P_x)
        Pcr.append(out['Pcr'])
    print('Pcr', Pcr)
    assert np.allclose(Pcr, [52827.015618484445, 42176.41489530511],
                       rtol=1e-5)


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
    out = flinBuck_VAFW(L, R, nx, ny, E11, E22, nu12, G12, rho, plyt, desvars,
            func_VAT_P_x)
    print('Pcr', out['Pcr'])
    assert np.isclose(out['Pcr'], 197096.99124887068, rtol=1e-5)
    assert np.all(np.diff(out['load_mult']) >= 0)


if __name__ == '__main__':
    test_2_runs_in_seq()
    test_Z33()
