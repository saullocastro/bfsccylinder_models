import sys
sys.path.append(r'..')
sys.path.append(r'../../bfsccylinder')

import numpy as np
from composites import laminated_plate

from bfsccylinder_models.koiter_cylinder_newton_raphson import fkoiter_cyl_SS3

def test_Waters_shell():
    #Arbocz, J., and Starnes, J. H., 2002, “On a High-Fidelity Hierarchical Approach to Buckling Load Calculations,” New Approaches to Structural Mechanics, Shells and Biological Structures, pp. 271–292.
    ny = 60
    NLprebuck = False
    Nxxunit = 1. # N/m

    L = 0.3556 # m
    R = 0.20318603 # m
    E11 = 127.629e9 # Pa
    E22 = 11.3074e9 # Pa
    G12 = 6.00257e9 # Pa
    nu12 = 0.300235
    rho = 1611 # kg/m3
    stack = [45, -45, 0, 90, 90, 0, -45, 45] #NOTE there is no different in inverting +- 45, significant different inverting 0 and 90 plies
    plyt = 0.00012692375 # m
    h = plyt*len(stack)

    aspect_ratio_nx_ny = 1
    nx = int(aspect_ratio_nx_ny*ny*L/(2*np.pi*R))
    if nx % 2 == 0:
        nx += 1
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    prop = laminated_plate(stack=stack, laminaprop=laminaprop, plyt=plyt,
            offset=0, rho=rho)
    out = fkoiter_cyl_SS3(L, R, nx, ny, prop, cg_x0=None, num_eigvals=2,
            koiter_num_modes=1, Nxxunit=Nxxunit, NLprebuck=NLprebuck)

    b_1111 = out['koiter']['b_ijkl'][(0, 0, 0, 0)]
    print('b_1111', b_1111)
    assert np.isclose(b_1111, -0.04801688128097509, rtol=0.02)

if __name__ == '__main__':
    test_Waters_shell()

